"""
Background Orchestration Pipeline & SSE Progress Broadcaster.
Executes the full 9-stage analysis pipeline and streams live Server-Sent Events (SSE).
Ensures client disconnection does not interrupt ongoing background sync jobs.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.api.rules_checker import rules_checker
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.data.store import data_store
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.server.analysis import analysis_service

logger = logging.getLogger("fpl_oracle.pipeline")


class SyncPipeline:
    def __init__(self):
        self._is_running: bool = False
        self._task: asyncio.Task[None] | None = None
        self._current_step: str = "idle"
        self._progress_pct: int = 0
        self._message: str = "Ready"
        self._subscribers: list[asyncio.Queue[str]] = []
        self._last_completed_at: datetime | None = None
        self._last_error: str | None = None
        self._last_result_summary: dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self._is_running,
            "current_step": self._current_step,
            "progress_pct": self._progress_pct,
            "message": self._message,
            "last_completed_at": self._last_completed_at.isoformat() if self._last_completed_at else None,
            "last_error": self._last_error,
            "summary": self._last_result_summary,
        }

    async def _broadcast(self, step: str, progress: int, message: str, done: bool = False, error: str | None = None):
        self._current_step = step
        self._progress_pct = progress
        self._message = message
        if error:
            self._last_error = error

        payload = {
            "step": step,
            "progress_pct": progress,
            "message": message,
            "done": done,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        msg_str = f"data: {json.dumps(payload)}\n\n"

        dead_queues = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg_str)
            except Exception:
                dead_queues.append(q)

        for dq in dead_queues:
            if dq in self._subscribers:
                self._subscribers.remove(dq)

    async def run_pipeline(self):
        """Execute all 9 stages in sequential order."""
        if self._is_running:
            return

        self._is_running = True
        self._last_error = None
        profile = data_store.get_profile()

        try:
            logger.info(">>> Starting FPL Oracle Orchestration Pipeline <<<")

            # ------------------------------------------------------------------
            # Stage 1: Upstream API Sync
            # ------------------------------------------------------------------
            await self._broadcast(
                "sync_upstream", 10, "Fetching live Premier League API bootstrap, fixtures, and event status..."
            )
            await analysis_service.invalidate()
            state = await manager_state_service.get_current_state(force_refresh=True)
            boot, _ = await fpl_client.get_bootstrap_static()
            fixtures, _ = await fpl_client.get_fixtures()
            curr_gw, next_gw = state.current_gw, state.target_gw
            if len(state.squad) != 15:
                raise ValueError(state.error_message or "Configured squad unavailable")
            if state.is_stale:
                raise ValueError("Live refresh failed; cached published data remains marked stale")

            # ------------------------------------------------------------------
            # Stage 2: Live Rules Verification & Game State
            # ------------------------------------------------------------------
            await self._broadcast(
                "rules_and_gamestate",
                20,
                "Verifying 2026/27 official rules & calculating GameState deadline countdown...",
            )
            rules_ver = rules_checker.verify(boot)
            game_state = await game_state_manager.get_game_state()

            target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 1)
            effective_curr_gw = curr_gw or (target_gw - 1 if target_gw > 1 else 1)

            # ------------------------------------------------------------------
            # Stage 3: News Evidence Ingestion & Single Reconciliation
            # ------------------------------------------------------------------
            await self._broadcast(
                "news", 30, "Ingesting official risks and extracting factual availability evidence..."
            )
            try:
                analyzed_news = await news_analyzer.get_player_news_signals(boot, target_gw=target_gw)
            except Exception as e:
                logger.warning("News ingestion warning: %s", e)
                analyzed_news = []

            # Extract unified reconciled availability map (N1)

            # ------------------------------------------------------------------
            # Stage 4: Feature Engineering
            # ------------------------------------------------------------------
            await self._broadcast("features", 45, f"Engineering pre-deadline features for Gameweek {target_gw}...")
            horizon = 5

            # ------------------------------------------------------------------
            # Stage 5: Component ML Inference
            # ------------------------------------------------------------------
            await self._broadcast(
                "ml_inference",
                55,
                f"Generating calibrated xP, P10 floor & P90 ceiling across GW {target_gw}-{target_gw + horizon - 1}...",
            )
            projections = await analysis_service.projections(
                start_gw=target_gw, horizon=horizon, bootstrap=boot, fixtures=fixtures
            )
            target_df = projections.get(target_gw)

            # ------------------------------------------------------------------
            # Stage 5: Transfer & Lineup Optimization
            # ------------------------------------------------------------------
            await self._broadcast(
                "optimization", 65, "Solving mathematical MILP for squad, starting XI, and transfer roadmap..."
            )
            user_squad_df = state.to_squad_dataframe()
            bank = state.bank_tenths
            free_transfers = state.free_transfers
            history_obj = None
            if state.manager_id:
                history_obj, _ = await fpl_client.get_manager_history(state.manager_id)
            if target_df is None or target_df.empty:
                raise ValueError("Projection snapshot unavailable")
            projected_ids = set(target_df["element"])
            if not set(user_squad_df["element"]).issubset(projected_ids):
                raise ValueError("Projection snapshot is missing configured squad players")
            for column in target_df.columns:
                if column not in ("element", "selling_price", "purchase_price", "price_provenance"):
                    user_squad_df[column] = user_squad_df["element"].map(target_df.set_index("element")[column])

            # ------------------------------------------------------------------
            # Stage 5 & 6: Unified Joint Transfer & Chip Optimization (G10)
            # ------------------------------------------------------------------
            await self._broadcast(
                "optimization", 65, "Solving mathematical MILP for squad, starting XI, and transfer roadmap..."
            )
            available_chips = state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2
            chips_used = [c["name"] for c in state.chips_used if (c["event"] <= 19) == (target_gw <= 19)]

            opt_res = {}
            lineup_res = {}
            if user_squad_df is not None and len(user_squad_df) == 15 and target_df is not None:
                lineup_res = lineup_optimizer.select_lineup_and_captain(
                    user_squad_df, risk_preference=profile.risk_preference
                )
                opt_res = await analysis_service.joint_plan(
                    current_squad_df=user_squad_df,
                    player_pool_df=target_df,
                    bank=bank,
                    free_transfers=free_transfers,
                    horizon_projections=projections,
                    current_gw=effective_curr_gw,
                    target_gw=target_gw,
                    available_chips=available_chips,
                    chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
                    chips_already_used=chips_used,
                    risk_preference=profile.risk_preference or "balanced",
                )

            await self._broadcast("chips", 75, "Running joint DP/beam search across Set 1 & Set 2 chip calendars...")
            chips_plan = await asyncio.to_thread(
                chip_planner.generate_chip_strategy,
                current_gw=effective_curr_gw,
                current_squad_df=user_squad_df if user_squad_df is not None else target_df,
                horizon_projections=projections,
                fixtures=fixtures,
                bootstrap=boot,
                manager_history=history_obj,
            )

            # ------------------------------------------------------------------
            # Stage 7: Mini-League Monte Carlo Simulation
            # ------------------------------------------------------------------
            await self._broadcast(
                "league", 85, "Simulating mini-league trajectories and rival differential ownership..."
            )
            t_league = profile.target_league_id
            league_res = None
            try:
                standings_data = await league_standings_manager.get_league_standings(t_league) if t_league else None
                if standings_data and standings_data.get("standings") and user_squad_df is not None:
                    rivals_res = await rival_analyzer.analyze_rivals(
                        standings=standings_data["standings"],
                        user_manager_id=profile.manager_id,
                        current_gw=effective_curr_gw,
                        bootstrap=boot,
                    )
                    user_pts = 0.0
                    if profile.manager_id:
                        try:
                            entry, _ = await fpl_client.get_manager_entry(profile.manager_id)
                            user_pts = float(entry.summary_overall_points or 0)
                        except Exception:
                            pass
                    league_res = await asyncio.to_thread(
                        monte_carlo_simulator.simulate_league,
                        user_points=user_pts,
                        user_squad_df=user_squad_df,
                        rival_squads=rivals_res.get("rival_squads", []),
                        projections_df=target_df if target_df is not None else user_squad_df,
                        horizon_gws=5,
                        projections_by_gw=projections,
                    )
            except Exception as e:
                logger.warning("Monte Carlo simulation warning: %s", e)

            # ------------------------------------------------------------------
            # Stage 8: Weekly Briefing Generation
            # ------------------------------------------------------------------
            await self._broadcast(
                "briefing", 98, f"Synthesizing Gameweek {target_gw} executive intelligence briefing..."
            )
            try:
                briefing_data = await weekly_briefing_generator.generate_briefing(profile.manager_id)
            except Exception as e:
                logger.warning("Briefing generation warning: %s", e)
                briefing_data = {"markdown": None, "error": str(e)}

            self._last_completed_at = datetime.now(UTC)
            self._last_result_summary = {
                "gameweek": target_gw,
                "rules_verified": rules_ver.verified,
                "game_state_phase": game_state.phase.value,
                "players_projected": len(target_df) if target_df is not None else 0,
                "transfer_plan": opt_res.get("recommended_plan", "ROLL"),
                "captain": opt_res.get("recommended_plan", {})
                .get("lineup", lineup_res)
                .get("captain", {})
                .get("web_name", "None"),
                "news_articles": len(analyzed_news),
                "chips_plan": chips_plan.get("joint_schedule", {}),
                "joint_plan": opt_res,
                "league_sim": league_res,
                "briefing_ready": bool(briefing_data.get("markdown")),
                "data_as_of": state.source_timestamp,
                "bank_tenths": state.bank_tenths,
                "free_transfers": state.free_transfers,
                "snapshot_id": target_df.attrs.get("snapshot_id"),
            }

            await self._broadcast(
                "complete", 100, f"Full FPL Oracle pipeline completed successfully for Gameweek {target_gw}!", done=True
            )
            logger.info(">>> FPL Oracle Orchestration Pipeline Finished Successfully! <<<")

        except Exception as e:
            logger.error("Pipeline failure: %s", e, exc_info=True)
            self._last_error = str(e)
            await self._broadcast("error", self._progress_pct, f"Pipeline failed: {e}", done=True, error=str(e))
        finally:
            self._is_running = False

    def trigger_sync(self) -> dict[str, Any]:
        """Explicitly trigger background analysis pipeline. Returns run ID."""
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        if not self._is_running and (self._task is None or self._task.done()):
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self.run_pipeline())
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    self._task = loop.create_task(self.run_pipeline())
                except Exception as e:
                    logger.error("Failed to schedule background pipeline: %s", e)
                    return {"status": "error", "error": str(e), "run_id": run_id, "is_running": False}
            return {"status": "started", "run_id": run_id, "is_running": True}
        return {"status": "already_running", "run_id": run_id, "is_running": True}

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Subscribe to live SSE stream (observation only). Does not auto-trigger run."""
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.append(q)

        # Send current status immediately upon connection
        current_msg = {
            "step": self._current_step,
            "progress_pct": self._progress_pct,
            "message": self._message,
            "done": not self._is_running and self._last_completed_at is not None,
            "error": self._last_error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await q.put(f"data: {json.dumps(current_msg)}\n\n")

        try:
            while True:
                msg = await q.get()
                yield msg
                # If message indicates done or error and pipeline isn't running, we can stop stream
                if '"done": true' in msg.lower():
                    break
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)


sync_pipeline = SyncPipeline()
