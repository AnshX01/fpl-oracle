"""
FastAPI REST API routes for FPL Oracle.
"""

import asyncio
import json
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.api.rules_checker import rules_checker
from fpl_oracle.briefing.decision_card import decision_card_generator, format_decision_card_markdown
from fpl_oracle.briefing.review import post_gameweek_reviewer
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.config import PLANNER_HORIZON
from fpl_oracle.data.fuzzy_match import fuzzy_matcher
from fpl_oracle.data.store import data_store
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.llm.agent import expert_agent
from fpl_oracle.llm.provider import get_llm_status
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.news.ingest import news_ingestion
from fpl_oracle.optimise.contingency import contingency_engine
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.server.analysis import analysis_service
from fpl_oracle.server.jobs import get_jobs_status, run_job_on_demand
from fpl_oracle.server.pipeline import sync_pipeline
from fpl_oracle.server.safe_json import SafeJSONResponse, safe_json_serialize

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", default_response_class=SafeJSONResponse)


class RollbackRequest(BaseModel):
    target_version: str


class ProfileUpdateRequest(BaseModel):
    manager_id: int | None = None
    target_league_id: int | None = None
    risk_preference: str | None = None
    llm_provider: str | None = None
    bank: float | None = None
    free_transfers: int | None = None
    manual_squad: list[int] | None = None


class ConfirmTeamRequest(BaseModel):
    gameweek: int
    player_ids: list[int]
    bank_tenths: int
    free_transfers: int
    available_chips: list[str]
    active_chip: str | None = None
    hit_cost: int
    selling_prices: dict[int, int]
    purchase_prices: dict[int, int]
    captain: int | None = None
    vice_captain: int | None = None
    bench: list[int] = []


@router.post("/team/followed")
async def confirm_followed_team():
    from fpl_oracle.domain.team_confirmation import team_confirmation

    state = await manager_state_service.get_current_state()
    boot, stale = await fpl_client.get_bootstrap_static()
    if stale:
        raise HTTPException(409, "Current player data unavailable; refresh first.")
    rec = team_confirmation.recommendation(state.manager_id)
    try:
        payload = team_confirmation.followed_payload(state, boot, rec)
    except (ValueError, KeyError) as error:
        raise HTTPException(409, str(error)) from None
    return await _confirm_team_payload(payload)


@router.post("/team/undo")
async def undo_followed_team():
    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.optimise.hit_ledger import hit_ledger
    from fpl_oracle.server.advice_job import advice_publisher

    state = await manager_state_service.get_current_state()
    record = team_confirmation.read(state.manager_id)
    if not record or not team_confirmation.locked(state):
        raise HTTPException(409, "No followed advice to undo for this deadline.")
    prior = record.get("undo_state")
    if prior:
        team_confirmation.write(state.manager_id, prior)
    else:
        team_confirmation.remove(state.manager_id)
    hit_ledger.undo_user_confirmation(state.manager_id or "manual", state.target_gw, prior)
    await advice_publisher.invalidate()
    await analysis_service.invalidate()
    return dict(status="undone", note="Local confirmation undone. Nothing changed on FPL.")


@router.get("/team/confirmation")
async def get_team_confirmation():
    from fpl_oracle.domain.team_confirmation import team_confirmation

    state = await manager_state_service.get_current_state()
    saved = team_confirmation.read(state.manager_id)
    boot, _ = await fpl_client.get_bootstrap_static()
    recommendation = team_confirmation.recommendation(state.manager_id)
    return safe_json_serialize(
        dict(
            state=state.model_dump(),
            players=[
                dict(element=e.id, name=e.web_name, price=e.now_cost, position=e.element_type) for e in boot.elements
            ],
            saved=saved,
            locked=team_confirmation.locked(state),
            recommendation=recommendation,
        )
    )


@router.post("/team/confirmation")
async def confirm_team(req: ConfirmTeamRequest):
    return await _confirm_team_payload(req.model_dump())


async def _confirm_team_payload(payload):
    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.optimise.hit_ledger import hit_ledger
    from fpl_oracle.server.advice_job import advice_publisher

    state = await manager_state_service.get_current_state()
    boot, stale = await fpl_client.get_bootstrap_static()
    if stale:
        raise HTTPException(409, "Current player data unavailable; refresh before confirming.")
    recommendation = team_confirmation.recommendation(state.manager_id)
    try:
        record = team_confirmation.confirm(state, boot, payload, recommendation)
    except ValueError as error:
        raise HTTPException(422, str(error)) from None
    if payload["hit_cost"]:
        hit_ledger.write(
            state.manager_id or "manual",
            dict(
                kind="taken",
                gameweek=payload["gameweek"],
                ins=record["actual_ins"],
                outs=record["actual_outs"],
                hit_cost=payload["hit_cost"],
                status="pending",
                source="user_confirmed",
                expected_gain=recommendation.get("expected_gain")
                if recommendation and record["comparison"] == "followed"
                else None,
                matched_recommendation=record["comparison"] == "followed",
            ),
        )
    hit_ledger._refreshes.clear()
    await advice_publisher.invalidate()
    await analysis_service.invalidate()
    return safe_json_serialize(record)


class MatchSquadRequest(BaseModel):
    raw_text: str


class ManualSquadRequest(BaseModel):
    player_ids: list[int]
    bank: float | None = 0.0
    free_transfers: int | None = 1


class OptimizeRequest(BaseModel):
    locked_in: list[int] | None = None
    locked_out: list[int] | None = None
    excluded_teams: list[int] | None = None
    custom_budget: float | None = None


class PanicRequest(BaseModel):
    query: str | None = None
    ruled_out_ids: list[int] | None = None
    duration: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = "default"


@router.get("/research/league-validation")
async def get_league_validation_status():
    from fpl_oracle.config import BASE_DIR
    from fpl_oracle.league.validation import validation_summary

    directory = BASE_DIR / "data" / "league_forward"
    scored = [json.loads(p.read_text()) for p in directory.glob("*.score.json")] if directory.exists() else []
    summary = validation_summary(scored)
    summary["frozen_forecast_count"] = len(list(directory.glob("*.index.json"))) if directory.exists() else 0
    summary["capture_window"] = (
        "Existing 15-minute job records once within 24h before deadline while local server is running"
    )
    summary["event_definition"] = (
        "Strict cumulative leader after target GW among fixed observed manager set; not entire league or season title odds"
    )
    return safe_json_serialize(summary)


@router.get("/research/chip-expiry")
async def get_chip_expiry_research():
    return safe_json_serialize(await analysis_service.expiry_sensitivity())


@router.post("/advice/start")
async def start_advice_publication():
    from fpl_oracle.server.advice_job import advice_publisher, profile_key

    return safe_json_serialize(advice_publisher.start(profile_key(data_store.get_profile())))


@router.get("/advice/status/{job_id}")
async def get_advice_publication(job_id: str):
    from fpl_oracle.server.advice_job import advice_publisher

    state = advice_publisher.public()
    if state.get("id") != job_id:
        raise HTTPException(status_code=404, detail="Advice job unavailable or superseded")
    return safe_json_serialize(state)


@router.get("/health")
async def get_health():
    game_state = await game_state_manager.get_game_state()
    cache_ages = cache_manager.get_all_cache_ages()
    model_status = projection_engine.get_model_status()
    news_status = news_ingestion.get_news_status()
    llm_status = get_llm_status()

    rules_ver = rules_checker.get_last_result()
    if rules_ver is None:
        try:
            boot, _ = await fpl_client.get_bootstrap_static()
            rules_ver = rules_checker.verify(boot)
        except Exception:
            pass

    return {
        "status": "healthy" if not fpl_client.is_stale_mode else "degraded",
        "api_reachability": not fpl_client.is_stale_mode,
        "rules_verification": rules_ver.to_dict()
        if rules_ver
        else {"verified": True, "rules_source": "cached", "mismatches": []},
        "season": game_state.season,
        "current_gameweek": game_state.current_gw,
        "next_gameweek": game_state.next_gw,
        "game_state": {
            "phase": game_state.phase.value,
            "deadline_time": game_state.deadline_time,
            "seconds_to_deadline": game_state.seconds_to_deadline,
            "is_live": game_state.is_live,
            "bonus_added": game_state.bonus_added,
            "leagues_updated": game_state.leagues_updated,
            "blank_gws": game_state.blank_gws,
            "double_gws": game_state.double_gws,
            "postponed_fixtures_count": game_state.postponed_fixtures_count,
        },
        "cache_age_seconds": cache_ages,
        "model": model_status,
        "news": news_status,
        "llm": llm_status,
        "is_stale": fpl_client.is_stale_mode,
        "stale": fpl_client.is_stale_mode,
        "last_sync": fpl_client.last_sync_time.isoformat() if fpl_client.last_sync_time else None,
        "data_as_of": game_state.data_as_of,
    }


@router.get("/game-state")
async def get_game_state_endpoint():
    state = await game_state_manager.get_game_state()
    return safe_json_serialize(state.model_dump())


@router.get("/profile")
def get_profile():
    p = data_store.get_profile()
    from fpl_oracle.config import app_config

    diag = app_config.get_redacted_status()
    return {
        "manager_id": p.manager_id,
        "manager_id_configured": diag["manager_id_configured"],
        "manager_id_redacted": diag["manager_id_redacted"],
        "target_league_id": p.target_league_id,
        "league_id_configured": diag["league_id_configured"],
        "league_id_redacted": diag["league_id_redacted"],
        "risk_preference": p.risk_preference,
        "llm_provider": p.llm_provider,
        "bank": p.bank,
        "free_transfers": p.free_transfers,
        "bank_override_enabled": p.bank_override_enabled,
        "ft_override_enabled": p.ft_override_enabled,
        "manual_squad": json.loads(p.manual_squad) if p.manual_squad else None,
        "status_diagnostic": diag,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@router.post("/profile")
def update_profile(req: ProfileUpdateRequest):
    kwargs: dict[str, Any] = {}
    if req.risk_preference is not None:
        kwargs["risk_preference"] = req.risk_preference
    if "bank" in req.model_fields_set:
        kwargs["bank"] = req.bank
    if "free_transfers" in req.model_fields_set:
        kwargs["free_transfers"] = req.free_transfers
    if req.manual_squad is not None:
        kwargs["manual_squad"] = req.manual_squad

    import math

    if req.bank is not None and (not math.isfinite(req.bank) or req.bank < 0):
        raise HTTPException(status_code=422, detail="Bank must be a finite nonnegative value")
    if req.free_transfers is not None and not 0 <= req.free_transfers <= 5:
        raise HTTPException(status_code=422, detail="Free transfers must be between 0 and 5")
    data_store.update_profile(**kwargs)
    return {"status": "success", "profile": get_profile()}


@router.get("/config/status")
def get_config_status():
    """Diagnostic status endpoint returning configured variables without exposing secrets."""
    from fpl_oracle.config import app_config

    return app_config.get_redacted_status()


@router.post("/sync/trigger")
async def trigger_sync_endpoint():
    """Trigger the background synchronization and analysis pipeline. Returns run ID."""
    return sync_pipeline.trigger_sync()


@router.get("/sync/stream")
async def get_sync_stream():
    """
    Server-Sent Events (SSE) stream broadcasting background analysis pipeline progress.
    Client disconnection does not terminate the underlying background analysis.
    """
    return StreamingResponse(
        sync_pipeline.subscribe(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sync/status")
def get_sync_status():
    """Return status and latest summary of the background synchronization pipeline."""
    return sync_pipeline.get_status()


@router.get("/squad/basic")
async def get_basic_squad(refresh: bool = False):
    state = await manager_state_service.get_current_state(force_refresh=refresh)
    squad = state.to_squad_dataframe()
    if len(squad) != 15 or squad["element"].nunique() != 15:
        return {
            "status": "unavailable",
            "starters": [],
            "bench": [],
            "reason": state.error_message or "No complete configured squad",
            "is_stale": state.is_stale,
        }
    # Basic data is published picks, never an optimized XI or point forecast.
    squad["now_cost"] = squad["now_cost"] / 10.0
    squad["selling_price"] = squad["selling_price"] / 10.0
    starters = squad[squad["is_starter"]].to_dict("records")
    bench = squad[~squad["is_starter"]].sort_values("bench_order").to_dict("records")
    return safe_json_serialize(
        {
            "status": "published",
            "starters": starters,
            "bench": bench,
            "captain": next((p for p in starters if p["is_captain"]), None),
            "vice_captain": next((p for p in starters if p["is_vice_captain"]), None),
            "team_confirmed": state.team_confirmed,
            "published_gameweek": state.current_gw,
            "total_squad_value": round(float(squad["now_cost"].sum()), 1),
            "total_selling_value": round(float(squad["selling_price"].sum()), 1),
            "total_team_value": round(float(squad["selling_price"].sum()) + state.bank_millions, 1),
            "bank_millions": state.bank_millions,
            "bank_source": state.bank_source,
            "free_transfers": state.free_transfers,
            "ft_source": state.ft_source,
            "is_stale": state.is_stale,
            "data_as_of": state.source_timestamp,
            "source_kind": state.mode.value,
            "target_gameweek": state.target_gw,
        }
    )


@router.get("/squad")
async def get_squad(manager_id: int | None = None):
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)

    fixtures_list, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    team_map = {t.id: t for t in boot.teams}

    horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures=fixtures_list)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) != 15:
        return {
            "status": "unavailable",
            "starters": [],
            "bench": [],
            "reason": "Configured squad unavailable",
            "is_stale": effective_state.is_stale,
        }

    # Attach current GW projections to user squad
    if not target_df.empty and "expected_points" in target_df.columns:
        xp_map = {int(r["element"]): float(r["expected_points"]) for _, r in target_df.iterrows()}
        p10_map = {int(r["element"]): float(r.get("p10", 0.0)) for _, r in target_df.iterrows()}
        p90_map = {int(r["element"]): float(r.get("p90", 0.0)) for _, r in target_df.iterrows()}
        defcon_map = {int(r["element"]): float(r.get("exp_defcon_pts", 0.0)) for _, r in target_df.iterrows()}

        user_squad_df["expected_points"] = user_squad_df["element"].map(xp_map).fillna(user_squad_df["expected_points"])
        user_squad_df["p10"] = user_squad_df["element"].map(p10_map).fillna(0.0)
        user_squad_df["p90"] = user_squad_df["element"].map(p90_map).fillna(0.0)
        user_squad_df["exp_defcon_pts"] = user_squad_df["element"].map(defcon_map).fillna(0.0)

    history = None
    if effective_state.manager_id:
        history, _ = await fpl_client.get_manager_history(effective_state.manager_id)
    joint = await analysis_service.joint_plan(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=float(effective_state.bank_tenths),
        free_transfers=int(effective_state.free_transfers),
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        available_chips=effective_state.chips_remaining_set_1
        if target_gw <= 19
        else effective_state.chips_remaining_set_2,
        chips_by_set={1: effective_state.chips_remaining_set_1, 2: effective_state.chips_remaining_set_2},
        chips_already_used=[c["name"] for c in effective_state.chips_used if (c["event"] <= 19) == (target_gw <= 19)],
    )
    lineup_res = joint["recommended_plan"]["lineup"]

    def get_next_fixtures(team_id: int):
        f_list = []
        for f in fixtures_list:
            if f.event and f.event >= target_gw and f.event < target_gw + 5:
                if f.team_h == team_id:
                    opp = team_map.get(f.team_a)
                    f_list.append(
                        {
                            "event": f.event,
                            "opp_short": opp.short_name if opp else "OPP",
                            "is_home": True,
                            "difficulty": f.team_h_difficulty or 3,
                        }
                    )
                elif f.team_a == team_id:
                    opp = team_map.get(f.team_h)
                    f_list.append(
                        {
                            "event": f.event,
                            "opp_short": opp.short_name if opp else "OPP",
                            "is_home": False,
                            "difficulty": f.team_a_difficulty or 3,
                        }
                    )
        return f_list[:5]

    recs = news_analyzer.get_reconciled_inputs(boot, target_gw)
    news_map = {
        eid: {
            "quote": r.source_quote,
            "source_url": r.source_url,
            "applied_to_production": r.applied_to_production,
            "expected_minutes_limit": r.expected_minutes_limit,
            "reconciliation_reason": r.reconciliation_reason,
        }
        for eid, r in recs.items()
    }

    starters_out = []
    for _, s in lineup_res["starters"].iterrows():
        elem_id = int(s["element"])
        sig = news_map.get(elem_id, {})
        starters_out.append(
            {
                "element": elem_id,
                "web_name": s["web_name"],
                "team": int(s["team"]),
                "team_short": team_map[s["team"]].short_name if s["team"] in team_map else "PL",
                "position": s["position"],
                "now_cost": round(float(s["value"]) / 10.0, 1),
                "purchase_price": round(float(s.get("purchase_price", s["value"])) / 10.0, 1),
                "selling_price": round(float(s.get("selling_price", s["value"])) / 10.0, 1),
                "expected_points": round(float(s["expected_points"]), 2),
                "p10": round(float(s.get("p10", 0.0)), 2),
                "p90": round(float(s.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(s.get("exp_defcon_pts", 0.0)), 2),
                "chance_of_playing": s.get("chance_of_playing", 100),
                "status": next((e.status for e in boot.elements if e.id == elem_id), "a"),
                "news": next((e.news for e in boot.elements if e.id == elem_id), ""),
                "news_quote": sig.get("quote", ""),
                "source_url": sig.get("source_url", ""),
                "news_mode": "gated_active"
                if sig.get("applied_to_production")
                else ("shadow" if sig.get("quote") else "official"),
                "expected_minutes_limit": sig.get("expected_minutes_limit"),
                "reconciliation_reason": sig.get("reconciliation_reason", ""),
                "is_captain": elem_id == lineup_res["captain"]["element"],
                "is_vice_captain": elem_id == lineup_res["vice_captain"]["element"],
                "next_fixtures": get_next_fixtures(int(s["team"])),
            }
        )

    bench_out = []
    for idx, (_, b) in enumerate(lineup_res["bench"].iterrows(), start=1):
        b_elem_id = int(b["element"])
        b_sig = news_map.get(b_elem_id, {})
        bench_out.append(
            {
                "element": b_elem_id,
                "web_name": b["web_name"],
                "team": int(b["team"]),
                "team_short": team_map[b["team"]].short_name if b["team"] in team_map else "PL",
                "position": b["position"],
                "now_cost": round(float(b["value"]) / 10.0, 1),
                "purchase_price": round(float(b.get("purchase_price", b["value"])) / 10.0, 1),
                "selling_price": round(float(b.get("selling_price", b["value"])) / 10.0, 1),
                "expected_points": round(float(b["expected_points"]), 2),
                "p10": round(float(b.get("p10", 0.0)), 2),
                "p90": round(float(b.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(b.get("exp_defcon_pts", 0.0)), 2),
                "chance_of_playing": b.get("chance_of_playing", 100),
                "status": next((e.status for e in boot.elements if e.id == b_elem_id), "a"),
                "news": next((e.news for e in boot.elements if e.id == b_elem_id), ""),
                "news_quote": b_sig.get("quote", ""),
                "source_url": b_sig.get("source_url", ""),
                "news_mode": "gated_active"
                if b_sig.get("applied_to_production")
                else ("shadow" if b_sig.get("quote") else "official"),
                "expected_minutes_limit": b_sig.get("expected_minutes_limit"),
                "reconciliation_reason": b_sig.get("reconciliation_reason", ""),
                "bench_order": idx,
                "next_fixtures": get_next_fixtures(int(b["team"])),
            }
        )

    squad_val = round(sum(s["now_cost"] for s in starters_out + bench_out), 1)
    sell_val = round(sum(s["selling_price"] for s in starters_out + bench_out), 1)
    team_val = round(sell_val + effective_state.bank_millions, 1)

    return safe_json_serialize(
        {
            "manager_id": effective_state.manager_id,
            "manager_name": effective_state.manager_name,
            "team_name": effective_state.team_name,
            "overall_points": effective_state.overall_points,
            "overall_rank": effective_state.overall_rank,
            "mode": effective_state.mode.value,
            "team_confirmed": effective_state.team_confirmed,
            "confirmation_as_of": effective_state.confirmation_as_of,
            "confirmation_comparison": effective_state.confirmation_comparison,
            "confidence": effective_state.confidence,
            "target_gameweek": target_gw,
            "bank_millions": effective_state.bank_millions,
            "bank": effective_state.bank_millions,
            "bank_source": effective_state.bank_source,
            "has_bank_override": effective_state.has_bank_override,
            "free_transfers": effective_state.free_transfers,
            "available_transfers": effective_state.free_transfers,
            "ft_source": effective_state.ft_source,
            "has_ft_override": effective_state.has_ft_override,
            "total_squad_value": squad_val,
            "total_selling_value": sell_val,
            "total_team_value": team_val,
            "chips_used": [c.get("name") for c in effective_state.chips_used],
            "formation": lineup_res["formation"],
            "captain": lineup_res["captain"],
            "vice_captain": lineup_res["vice_captain"],
            "starters": starters_out,
            "bench": bench_out,
            "starters_expected_points": lineup_res.get("starters_expected_points", 0.0),
            "captain_bonus_expected_points": lineup_res.get(
                "captain_bonus_expected_points", lineup_res["captain"]["expected_points"]
            ),
            "total_expected_points": lineup_res["total_gameweek_expected_points"],
            "stale": is_stale or effective_state.is_stale,
            "is_stale": is_stale or effective_state.is_stale,
            "data_as_of": effective_state.source_timestamp,
        }
    )


@router.post("/squad/match")
async def match_squad_names(req: MatchSquadRequest):
    boot, _ = await fpl_client.get_bootstrap_static()
    team_map = {t.id: t.short_name for t in boot.teams}
    res = fuzzy_matcher.parse_and_match_squad(req.raw_text, boot.elements, team_map)
    return safe_json_serialize(res)


@router.post("/squad/manual")
async def save_manual_squad(req: ManualSquadRequest):
    if len(req.player_ids) != 15:
        raise HTTPException(status_code=400, detail="A valid squad must contain exactly 15 players.")
    boot, _ = await fpl_client.get_bootstrap_static()
    elem_dict = {e.id: e for e in boot.elements}

    # Verify positions and 3-per-club rule
    pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    team_counts: dict[int, int] = {}
    for pid in req.player_ids:
        if pid not in elem_dict:
            raise HTTPException(status_code=400, detail=f"Player ID {pid} is invalid.")
        elem = elem_dict[pid]
        pos_counts[elem.element_type] += 1
        team_counts[elem.team] = team_counts.get(elem.team, 0) + 1
        if team_counts[elem.team] > 3:
            raise HTTPException(status_code=400, detail=f"Exceeded max 3 players for club {elem.team}.")

    if pos_counts[1] != 2 or pos_counts[2] != 5 or pos_counts[3] != 5 or pos_counts[4] != 3:
        raise HTTPException(status_code=400, detail="Invalid formation! Required: 2 GKP, 5 DEF, 5 MID, 3 FWD.")

    data_store.update_profile(manual_squad=req.player_ids, bank=req.bank, free_transfers=req.free_transfers)
    return safe_json_serialize(
        {"status": "success", "message": "Custom squad saved successfully.", "player_ids": req.player_ids}
    )


@router.get("/projections")
async def get_projections(position: str | None = None, team_id: int | None = None, horizon: int = Query(5, ge=1, le=8)):
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    _, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = await analysis_service.projections(target_gw, horizon, boot, fixtures)
    df = horizon_proj.get(target_gw, pd.DataFrame())

    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if team_id:
        df = df[df["team"] == team_id]

    results = []
    for _, r in df.sort_values(by="expected_points", ascending=False).head(50).iterrows():
        results.append(
            {
                "element": int(r["element"]),
                "web_name": r["web_name"],
                "team": int(r["team"]),
                "position": r["position"],
                "cost": round(r["value"] / 10.0, 1),
                "expected_points": round(float(r["expected_points"]), 2),
                "p10": round(float(r.get("p10", 0.0)), 2),
                "p90": round(float(r.get("p90", 0.0)), 2),
                "exp_defcon_pts": round(float(r.get("exp_defcon_pts", 0.0)), 2),
            }
        )

    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "horizon": horizon,
            "players": results,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.post("/optimize")
@router.post("/transfers")
async def run_optimizer(req: OptimizeRequest | None = None):
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)

    # News adjustment path: pass reconciled availabilities map
    horizon_proj = await analysis_service.projections(target_gw, PLANNER_HORIZON, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    # Get user squad with real selling prices and actual free transfers
    user_squad_df, bank_tenths, free_transfers = await _get_effective_user_squad(target_df, boot)

    locked_in = req.locked_in if req else None
    locked_out = req.locked_out if req else None
    excl_teams = req.excluded_teams if req else None

    # Manager history chips status
    manager_hist = None
    if effective_state.manager_id:
        try:
            manager_hist, _ = await fpl_client.get_manager_history(effective_state.manager_id)
        except Exception as e:
            logger.warning("Manager history fetch warning: %s", e)

    available_chips = (
        effective_state.chips_remaining_set_1 if target_gw <= 19 else effective_state.chips_remaining_set_2
    )
    chips_used = [c["name"] for c in effective_state.chips_used if (c["event"] <= 19) == (target_gw <= 19)]

    res = await analysis_service.joint_plan(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        available_chips=available_chips,
        chips_by_set={1: effective_state.chips_remaining_set_1, 2: effective_state.chips_remaining_set_2},
        chips_already_used=chips_used,
        locked_in_ids=locked_in,
        locked_out_ids=locked_out,
        excluded_team_ids=excl_teams,
    )
    res["stale"] = is_stale or effective_state.is_stale
    res["is_stale"] = res["stale"]
    res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    res["manager_state"] = {
        "mode": effective_state.mode.value,
        "bank_millions": effective_state.bank_millions,
        "bank_source": effective_state.bank_source,
        "free_transfers": effective_state.free_transfers,
        "ft_source": effective_state.ft_source,
    }
    return safe_json_serialize(res)


@router.get("/chips")
async def get_chip_strategy():
    effective_state = await manager_state_service.get_current_state()
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

    target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 6)

    # News adjustment path: identical across all surfaces
    horizon_proj = await analysis_service.projections(target_gw, 8, boot, fixtures)
    pool_df = horizon_proj.get(target_gw, pd.DataFrame())

    hist = None
    if effective_state.manager_id:
        try:
            hist, _ = await fpl_client.get_manager_history(effective_state.manager_id)
        except Exception as e:
            logger.warning("Manager history fetch warning: %s", e)

    squad_df, _, _ = await _get_effective_user_squad(pool_df, boot)

    res = await analysis_service.chip_strategy(
        current_gw=curr_gw or 5,
        current_squad_df=squad_df,
        horizon_projections=horizon_proj,
        fixtures=fixtures,
        bootstrap=boot,
        manager_history=hist,
    )
    joint = await analysis_service.joint_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=float(effective_state.bank_tenths),
        free_transfers=int(effective_state.free_transfers),
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        available_chips=effective_state.chips_remaining_set_1
        if target_gw <= 19
        else effective_state.chips_remaining_set_2,
        chips_by_set={1: effective_state.chips_remaining_set_1, 2: effective_state.chips_remaining_set_2},
        chips_already_used=[c["name"] for c in effective_state.chips_used if (c["event"] <= 19) == (target_gw <= 19)],
    )
    res = analysis_service.bind_chip_schedule(res, joint)
    res["recommend_chip_this_gw"] = joint["recommended_chip"] is not None
    res["recommended_chip"] = joint["recommended_chip"]
    res["chip_comparison_table"] = joint["chip_comparison_table"]
    res["stale"] = is_stale or effective_state.is_stale
    res["is_stale"] = res["stale"]
    res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(res)


@router.get("/league")
async def get_league_intel(league_id: int | None = None):
    effective_state = await manager_state_service.get_current_state()
    official_overall_rank = effective_state.overall_rank
    if (
        not isinstance(official_overall_rank, int)
        or isinstance(official_overall_rank, bool)
        or official_overall_rank <= 0
    ):
        official_overall_rank = None
    profile = data_store.get_profile()
    l_id = league_id or profile.target_league_id
    if not l_id:
        return safe_json_serialize(
            {
                "status": "unconfigured",
                "overall_rank": official_overall_rank,
                "current_league_rank": None,
                "league_rank_status": "league_unconfigured",
                "message": "No target mini-league ID configured. Enter a mini-league ID to see rival analysis.",
                "league_name": "None",
                "total_teams": 0,
                "standings": [],
                "template_players": [],
                "differential_players": [],
                "simulation": {"user_win_probability_pct": None, "expected_final_rank": None},
                "strategy": {
                    "mode_title": "Setup Required",
                    "rationale": "Set your mini-league ID in settings to activate rival analysis.",
                    "tactical_recommendations": [],
                },
                "stale": False,
                "is_stale": False,
                "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
            }
        )

    standings_data = await league_standings_manager.get_league_standings(l_id)
    from fpl_oracle.league.standings import current_manager_rank

    rank_fields = current_manager_rank(standings_data, effective_state.manager_id)
    boot, is_stale = await fpl_client.get_bootstrap_static()
    curr_gw, _ = await fpl_client.get_current_and_next_gw()

    if not standings_data.get("standings"):
        return safe_json_serialize(
            {
                "status": "empty",
                "overall_rank": official_overall_rank,
                "current_league_rank": None,
                "league_rank_status": "unavailable",
                "message": f"No standings found for mini-league {l_id}.",
                "league_name": standings_data.get("league_name", f"League #{l_id}"),
                "total_teams": 0,
                "standings": [],
                "template_players": [],
                "differential_players": [],
                "simulation": {"user_win_probability_pct": None, "expected_final_rank": None},
                "strategy": {
                    "mode_title": "No Standings",
                    "rationale": "No standings data returned from FPL API for this league.",
                    "tactical_recommendations": [],
                },
                "stale": is_stale,
                "is_stale": is_stale,
                "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
            }
        )

    rivals_res = await rival_analyzer.analyze_rivals(
        standings=standings_data["standings"],
        user_manager_id=effective_state.manager_id,
        current_gw=curr_gw or 5,
        bootstrap=boot,
    )

    from fpl_oracle.league.rivals import attach_squad_overlap

    standings_data["standings"] = attach_squad_overlap(
        standings_data["standings"],
        rivals_res["rival_squads"],
        effective_state.squad,
    )

    user_pts = effective_state.overall_points
    # Find user's rank in this mini-league if present, otherwise fallback to overall_rank or 1
    user_mini_rank = None
    if effective_state.manager_id and standings_data.get("standings"):
        for row in standings_data["standings"]:
            if row.get("entry") == effective_state.manager_id:
                user_mini_rank = row.get("rank")
                break
    user_rank: int = (
        int(user_mini_rank)
        if user_mini_rank is not None
        else (int(effective_state.overall_rank) if effective_state.overall_rank else 1)
    )

    # Monte Carlo simulation
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    sim_target_gw = (curr_gw + 1) if curr_gw else 2
    projections_horizon = await analysis_service.projections(sim_target_gw, 5, boot, fixtures)
    proj_df = projections_horizon.get(sim_target_gw)

    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) != 15:
        return {
            "status": "unavailable",
            "overall_rank": official_overall_rank,
            **rank_fields,
            "standings": standings_data["standings"],
            "simulation": None,
            "reason": "Configured squad unavailable",
        }

    # Same observed owned-squad/captain scenario as decision card, not the selected WC team.
    if proj_df is not None:
        user_squad_df["expected_points"] = user_squad_df["element"].map(proj_df.set_index("element")["expected_points"])
    mc_res = await asyncio.to_thread(
        monte_carlo_simulator.simulate_league,
        user_points=user_pts,
        user_squad_df=user_squad_df,
        rival_squads=rivals_res["rival_squads"],
        projections_df=proj_df,
        horizon_gws=5,
        seed=42,
        projections_by_gw=projections_horizon,
    )

    from fpl_oracle.league.strategy import automatic_strategy

    strategy = automatic_strategy(
        None
        if user_rank is None
        else dict(
            user_rank=user_rank,
            user_points=user_pts,
            standings=[dict(rank=r["rank"], points=r["total"]) for r in standings_data["standings"]],
        )
    )

    return safe_json_serialize(
        {
            "status": "success",
            "overall_rank": official_overall_rank,
            **rank_fields,
            "league_name": standings_data["league_name"],
            "total_teams": standings_data["total_teams"],
            "standings": standings_data["standings"],
            "template_players": rivals_res["template_players"],
            "differential_players": rivals_res["differential_players"],
            "simulation": mc_res,
            "strategy": strategy,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.get("/news")
async def get_news_signals(
    gw: int | None = None,
):
    """Return all reconciled news signals with verbatim quotes, source links, and mode badges (N7)."""
    boot, _ = await fpl_client.get_bootstrap_static()
    target_gw = gw or 6
    signals = await news_analyzer.get_player_news_signals(boot, target_gw=target_gw)
    return safe_json_serialize(
        {
            "target_gw": target_gw,
            "count": len(signals),
            "signals": signals,
        }
    )


@router.get("/decision-card")
async def get_decision_card_endpoint():
    """
    Returns the comprehensive per-gameweek unified decision card (D1).
    Reads from shared game state, projection snapshots, beam search transfers,
    chip calendar, and joint rival simulation.
    """
    card = await decision_card_generator.generate_decision_card()
    return safe_json_serialize(card)


@router.get("/decision-card/export", response_class=PlainTextResponse)
async def export_decision_card_endpoint():
    """
    Returns a printable plain text / markdown gameweek decision card (D2).
    """
    card = await decision_card_generator.generate_decision_card()
    markdown_text = format_decision_card_markdown(card)
    return PlainTextResponse(content=markdown_text, media_type="text/plain; charset=utf-8")


@router.get("/briefing")
async def get_briefing():
    return await weekly_briefing_generator.generate_briefing()


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    ans = await expert_agent.answer(user_message=req.message, session_id=req.session_id or "default")
    return safe_json_serialize({"response": ans, "data_as_of": fpl_client.get_data_as_of("bootstrap-static")})


@router.get("/price-changes")
async def get_price_changes():
    boot, is_stale = await fpl_client.get_bootstrap_static()
    preds = price_change_predictor.analyze_price_changes(boot)
    return safe_json_serialize(
        {
            "rises": [p for p in preds if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:10],
            "falls": [p for p in preds if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:10],
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


async def _get_effective_user_squad(target_df: pd.DataFrame, boot: Any) -> tuple[pd.DataFrame, float, int]:
    effective_state = await manager_state_service.get_current_state()
    user_squad_df = effective_state.to_squad_dataframe()
    if user_squad_df.empty or len(user_squad_df) != 15:
        raise HTTPException(status_code=409, detail="Configured squad unavailable; no personal advice generated")

    if not target_df.empty and "expected_points" in target_df.columns:
        xp_map = {int(r["element"]): float(r["expected_points"]) for _, r in target_df.iterrows()}
        p10_map = {int(r["element"]): float(r.get("p10", 0.0)) for _, r in target_df.iterrows()}
        p90_map = {int(r["element"]): float(r.get("p90", 0.0)) for _, r in target_df.iterrows()}
        user_squad_df["expected_points"] = user_squad_df["element"].map(xp_map).fillna(user_squad_df["expected_points"])
        user_squad_df["p10"] = user_squad_df["element"].map(p10_map).fillna(0.0)
        user_squad_df["p90"] = user_squad_df["element"].map(p90_map).fillna(0.0)

    return user_squad_df, float(effective_state.bank_tenths), int(effective_state.free_transfers)


@router.get("/contingency/plans")
async def get_contingency_plans(detailed: bool = True):
    """Returns precomputed Plan A, Plan B (injury pivot), and Plan C (differential/price pivot)."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)

    state = await manager_state_service.get_current_state()
    joint = await analysis_service.joint_plan(
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=free_transfers,
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 1,
        target_gw=target_gw,
        available_chips=state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2,
        chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
        chips_already_used=[c["name"] for c in state.chips_used if (c["event"] <= 19) == (target_gw <= 19)],
    )
    plans = await asyncio.to_thread(
        contingency_engine.generate_contingency_plans,
        current_squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=free_transfers,
        horizon_projections=horizon_proj,
        current_gw=curr_gw or 5,
        target_gw=target_gw,
        risk_preference="points",
        primary_plan=joint["recommended_plan"],
        compute_alternatives=detailed,
    )
    plans["stale"] = is_stale or state.is_stale
    plans["is_stale"] = plans["stale"]
    plans["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(plans)


async def _selected_advice_context(pool, projections, boot, current_gw, target_gw):
    owned, bank, ft = await _get_effective_user_squad(pool, boot)
    state = await manager_state_service.get_current_state()
    joint = await analysis_service.joint_plan(
        current_squad_df=owned,
        player_pool_df=pool,
        bank=bank,
        free_transfers=ft,
        horizon_projections=projections,
        current_gw=current_gw or 5,
        target_gw=target_gw,
        available_chips=state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2,
        chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
        chips_already_used=[c["name"] for c in state.chips_used if (c["event"] <= 19) == (target_gw <= 19)],
    )
    plan = joint["recommended_plan"]
    lineup = plan["lineup"]
    squad = pd.concat([lineup["starters"], lineup["bench"]], ignore_index=True)
    return (
        joint,
        squad,
        float(plan["remaining_bank"]) * 10,
        max(0, ft - (0 if joint["recommended_chip"] in ("wildcard", "freehit") else len(plan["transfers_in"]))),
    )


@router.get("/league/scenarios")
async def get_rival_scenarios():
    """On-demand diagnostic scenarios. Not calibrated odds or known hidden moves."""
    boot, stale = await fpl_client.get_bootstrap_static()
    fixtures, fixture_stale = await fpl_client.get_fixtures()
    if stale or fixture_stale:
        raise HTTPException(status_code=409, detail="Fresh sources required for rival stress")
    current, target = await fpl_client.get_current_and_next_gw()
    if not target:
        raise HTTPException(status_code=409, detail="Upcoming official gameweek unavailable")
    projections = await analysis_service.projections(target, 8, boot, fixtures)
    joint, _, _, _ = await _selected_advice_context(projections[target], projections, boot, current, target)
    return safe_json_serialize(await analysis_service.rival_scenarios(joint, projections, target))


@router.get("/contingency/matrix")
async def get_contingency_matrix():
    """Returns 'What if Player X is ruled out' matrix comparing auto-sub vs emergency transfer."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    joint, user_squad_df, bank, free_transfers = await _selected_advice_context(
        target_df, horizon_proj, boot, curr_gw, target_gw
    )

    matrix = await asyncio.to_thread(
        contingency_engine.compute_injury_matrix,
        squad_df=user_squad_df,
        player_pool_df=target_df,
        bank=bank,
        free_transfers=free_transfers,
        bootstrap=boot,
        lineup=joint["recommended_plan"]["lineup"],
    )
    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "scope": "selected_plan_not_submitted",
            "contingency_matrix": matrix,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.post("/contingency/panic")
async def post_contingency_panic(req: PanicRequest):
    """Emergency 1-click crisis solver for breaking team news (e.g. 'Saka ruled out 6 weeks')."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())
    state = await manager_state_service.get_current_state()
    if is_stale or state.is_stale or target_df.empty:
        raise HTTPException(status_code=409, detail="Fresh squad and predictions required. Update data and try again.")

    user_squad_df, bank, free_transfers = await _get_effective_user_squad(target_df, boot)

    import re
    import unicodedata

    def normalize(value):
        value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
        return re.sub(r"[^a-zA-Z0-9]+", " ", value).lower().strip()

    ids = set(req.ruled_out_ids or [])
    query = normalize(req.query or "")
    if not ids and query:
        ids = {
            int(row["element"])
            for row in user_squad_df.to_dict("records")
            if normalize(row["web_name"]) and f" {normalize(row['web_name'])} " in f" {query} "
        }
    if not ids or not ids.issubset(set(user_squad_df["element"])):
        return dict(status="needs_player", recommendation="Name an unavailable player in your squad.")
    if req.duration not in ("1", "2", "window"):
        return dict(
            status="needs_duration",
            recommendation="How long is he out? Choose this week, two weeks or until further notice.",
        )
    excluded_weeks = (
        sorted(horizon_proj) if req.duration == "window" else list(range(target_gw, target_gw + int(req.duration)))
    )
    constrained = {}
    for gw, frame in horizon_proj.items():
        frame = frame.copy()
        mask = frame["element"].isin(ids) & (gw in excluded_weeks)
        for column in ("expected_points", "p10", "p90"):
            if column in frame:
                frame.loc[mask, column] = 0.0
        frame["simulation_unavailable"] = mask
        constrained[gw] = frame
    try:
        joint = await analysis_service.joint_plan(
            current_squad_df=user_squad_df,
            player_pool_df=constrained[target_gw],
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=constrained,
            current_gw=curr_gw or target_gw - 1,
            target_gw=target_gw,
            available_chips=state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2,
            chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
            chips_already_used=[c["name"] for c in state.chips_used if (c["event"] <= 19) == (target_gw <= 19)],
            locked_out_ids=set(),
        )
    except ValueError:
        from fpl_oracle.optimise.squad import squad_optimizer
        from fpl_oracle.optimise.transfers import transfer_optimizer

        sells = {
            int(row["element"]): int(row.get("selling_price", row["value"])) for row in user_squad_df.to_dict("records")
        }
        try:
            repair = await asyncio.to_thread(
                squad_optimizer.solve_best_squad,
                constrained[target_gw],
                bank + sum(sells.values()),
                captain_mean_only=True,
                require_talisman=False,
                bench_weight=0,
                repair_elements=set(sells),
                repair_sell_prices=sells,
                repair_free_transfers=free_transfers,
                repair_min_transfers=True,
            )
            new_ids = set(int(e) for e in repair["squad"]["element"])
            ins, outs = new_ids - set(sells), set(sells) - new_ids
            cost = max(0, len(ins) - free_transfers) * transfer_optimizer.hit_penalty
            return safe_json_serialize(
                dict(
                    status="needs_decision",
                    recommendation=f"Not possible within the transfer/hit limit. Cheapest legal paid repair needs {len(ins)} transfers and costs {cost:g} points. This is above your limit, not a recommendation.",
                    above_cap_option=dict(transfers_in=sorted(ins), transfers_out=sorted(outs), hit_cost=cost),
                    chip_options=[],
                    excluded_gameweeks=excluded_weeks,
                )
            )
        except (ValueError, RuntimeError):
            raise HTTPException(
                status_code=409, detail="No legal plan found under budget and squad rules; no recommendation published."
            ) from None
    plan = joint["recommended_plan"]
    lineup = plan["lineup"]
    names = [row["web_name"] for row in user_squad_df.to_dict("records") if row["element"] in ids]
    moves = [
        f"{out['web_name']} → {incoming['web_name']}"
        for out, incoming in zip(plan["transfers_out"], plan["transfers_in"], strict=False)
    ]
    crisis_res = dict(
        status="recalculated",
        recommendation=("; ".join(moves) or "Keep the squad; use your bench.")
        + f" Captain: {lineup['captain']['web_name']}. Hit cost: {plan['hit_cost']} points.",
        affected_players=names,
        excluded_gameweeks=excluded_weeks,
        duration=req.duration,
        excluded_ids=sorted(ids),
        joint_plan=joint,
        chip_options=[
            dict(chip=c["chip_code"], hit_cost=c["plan"]["hit_cost"], net_points=c["plan"]["net_expected_points"])
            for c in joint.get("chip_comparison_table", [])
            if c.get("chip_code") in ("wildcard", "freehit")
        ],
        hit_policy=joint.get("hit_policy"),
        lineup_action=dict(
            new_formation=lineup["formation"],
            captain=lineup["captain"],
            vice_captain=lineup["vice_captain"],
            total_gameweek_expected_points=lineup["total_gameweek_expected_points"],
        ),
    )

    if joint.get("normal_plan_feasible") is False:
        crisis_res["recommendation"] = (
            "Not possible within the ordinary transfer/hit limit. "
            + str(plan.get("chip_applied", "Chip"))
            + " gives a legal XI. "
            + crisis_res["recommendation"]
        )
    crisis_res["stale"] = is_stale
    crisis_res["is_stale"] = is_stale
    crisis_res["data_as_of"] = fpl_client.get_data_as_of("bootstrap-static")
    return safe_json_serialize(crisis_res)


@router.get("/contingency/checklist")
async def get_pre_deadline_checklist():
    """Generates 5-point operational pre-deadline audit."""
    boot, is_stale = await fpl_client.get_bootstrap_static()
    fixtures, fixtures_stale = await fpl_client.get_fixtures()
    is_stale = is_stale or fixtures_stale
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
    target_gw = next_gw or 6

    horizon_proj = await analysis_service.projections(target_gw, 5, boot, fixtures)
    target_df = horizon_proj.get(target_gw, pd.DataFrame())

    joint, user_squad_df, bank, free_transfers = await _selected_advice_context(
        target_df, horizon_proj, boot, curr_gw, target_gw
    )
    game_state = await game_state_manager.get_game_state()

    profile = data_store.get_profile()
    history = None
    if profile.manager_id:
        try:
            history, _ = await fpl_client.get_manager_history(profile.manager_id)
        except Exception:
            pass

    chips_status = chip_planner.get_remaining_chips(history)
    checklist = await asyncio.to_thread(
        contingency_engine.generate_pre_deadline_checklist,
        squad_df=user_squad_df,
        bootstrap=boot,
        game_state_data=game_state.model_dump(),
        chips_status=chips_status,
        lineup=joint["recommended_plan"]["lineup"],
    )
    state = await manager_state_service.get_current_state()
    plan = joint["recommended_plan"]
    checklist.extend(
        [
            dict(
                item="Team",
                status="PASS" if state.team_confirmed else "WARNING",
                detail="Updated team"
                if state.team_confirmed
                else "Based on your last published team. Made transfers since then? Update your team.",
            ),
            dict(
                item="Bank",
                status="PASS" if plan["remaining_bank"] >= 0 else "WARNING",
                detail=f"£{plan['remaining_bank']:.1f}m left",
            ),
            dict(
                item="Transfers and hits",
                status="PASS" if plan["hit_cost"] <= 4 and len(plan["transfers_in"]) <= 2 else "WARNING",
                detail=f"{len(plan['transfers_in'])} transfers · {plan['hit_cost']} hit points",
            ),
            dict(
                item="Data",
                status="WARNING" if is_stale or state.is_stale else "PASS",
                detail="Refresh needed" if is_stale or state.is_stale else "Current",
            ),
        ]
    )
    return safe_json_serialize(
        {
            "gameweek": target_gw,
            "scope": "selected_plan_not_submitted",
            "seconds_to_deadline": game_state.seconds_to_deadline,
            "checklist": checklist,
            "stale": is_stale,
            "is_stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )


@router.get("/review")
async def get_gameweek_review(gameweek: int | None = None):
    """Produces post-gameweek review and performance diagnostics."""
    profile = data_store.get_profile()
    return await post_gameweek_reviewer.generate_gameweek_review(profile.manager_id, gameweek)


@router.get("/system/jobs")
async def get_system_jobs():
    """Returns execution status and telemetry for all background scheduler jobs."""
    status = get_jobs_status()
    return safe_json_serialize(status)


@router.post("/system/jobs/{job_id}/run")
async def trigger_job_run(job_id: str):
    """Manually triggers immediate execution of a scheduled background job."""
    res = await run_job_on_demand(job_id)
    if not res.get("success", False) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return safe_json_serialize(res)


@router.get("/system/models")
async def get_model_versions_endpoint():
    """Returns active model version, metric history, and rollback checkpoints."""
    active = model_registry.get_active_version()
    history = model_registry.get_version_history()
    return safe_json_serialize({"active_version": active, "history": history})


@router.post("/system/models/rollback")
async def rollback_model_endpoint(req: RollbackRequest):
    """Restores an archived model version checkpoint to active production."""
    res = model_registry.rollback_to_version(req.target_version)
    if not res.get("success", False):
        raise HTTPException(status_code=400, detail=res.get("error", "Rollback failed"))
    return safe_json_serialize(res)


@router.get("/system/status")
async def get_system_status():
    """Comprehensive system diagnostic telemetry."""
    game_state = await game_state_manager.get_game_state()
    active_model = model_registry.get_active_version()
    news_stat = news_ingestion.get_news_status()
    jobs_stat = get_jobs_status()
    db_runs = data_store.get_job_runs(limit=5)
    prices = data_store.get_latest_price_snapshots(limit=10)

    return safe_json_serialize(
        {
            "game_state": {
                "phase": game_state.phase.value,
                "seconds_to_deadline": round(game_state.seconds_to_deadline, 1)
                if game_state.seconds_to_deadline is not None
                else None,
                "current_gw": game_state.current_gw,
                "next_gw": game_state.next_gw,
            },
            "model": {
                "active_version": active_model.get("version"),
                "mae": active_model.get("ml_mae"),
                "spearman": active_model.get("ml_spearman"),
                "status": active_model.get("status"),
            },
            "scheduler": {"running": jobs_stat["scheduler_running"], "total_jobs": jobs_stat["total_jobs"]},
            "news": news_stat,
            "price_snapshots_recorded": len(prices),
            "recent_job_runs": db_runs,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }
    )
