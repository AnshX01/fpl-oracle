"""
Tool definitions and execution dispatcher for the FPL Expert Chat Agent.
Connects conversational queries directly to live ML projections, MILP optimizations,
chip strategies, news, and league simulations.
"""

import json
import logging
import time
from contextvars import ContextVar
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.data.store import data_store
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.llm.decision_record import SECTIONS, current_decision_record, published_result, section_of
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.server.analysis import analysis_service, target_gameweek

logger = logging.getLogger("fpl_oracle.llm.tools")

chat_tool_cache: ContextVar[dict[tuple[str, str], Any] | None] = ContextVar("chat_tool_cache", default=None)


MAX_TOOL_ROUNDS = 6

_MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "out": {"type": "integer", "description": "Player ID leaving the squad"},
        "in": {"type": "integer", "description": "Player ID joining the squad"},
    },
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_current_decision",
        "description": (
            "The advice the app has published for the next deadline: squad with xP/form/fixtures/availability, "
            "bank, free transfers, chips per set, recommended transfers and hits, XI/bench/captain/vice, chip "
            "decision with now-vs-later comparison, alternative plans with scores, league status, data freshness. "
            "Call it first for any question about what to do or why."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": list(SECTIONS),
                    "description": "Optional slice; omit for the whole record. 'raw' returns the published decision card.",
                }
            },
        },
    },
    {
        "name": "get_my_team",
        "description": "Fetch the configured squad (per-player form, minutes, ownership, next fixtures), bank, free transfers, chips used and remaining per set, overall points.",
        "parameters": {
            "type": "object",
            "properties": {"manager_id": {"type": "integer", "description": "Optional manager ID"}},
        },
    },
    {
        "name": "get_projections",
        "description": "Retrieve ML projected expected points (P10/P90, DefCon, ownership) ranked best first; horizon sums xP over upcoming gameweeks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Player name or position filter"},
                "horizon": {"type": "integer", "description": "Number of upcoming gameweeks (1-8)"},
                "limit": {"type": "integer", "description": "Players to return (default 15, max 800)"},
            },
        },
    },
    {
        "name": "optimise_transfers",
        "description": "Run mathematical MILP transfer optimizer evaluating candidate plans (roll transfer, 1 transfer, 2 transfers with hit evaluation).",
        "parameters": {
            "type": "object",
            "properties": {
                "locked_in": {"type": "array", "items": {"type": "integer"}, "description": "Player IDs to force keep"},
                "locked_out": {"type": "array", "items": {"type": "integer"}, "description": "Player IDs to never buy"},
            },
        },
    },
    {
        "name": "plan_chips",
        "description": "Compare chip options over the loaded upcoming weeks. No full-season best-date schedule is established.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "captain_options",
        "description": "Rank captaincy options for target gameweek with ceiling, floor, and safe vs differential picks.",
        "parameters": {
            "type": "object",
            "properties": {"gameweek": {"type": "integer", "description": "Target gameweek"}},
        },
    },
    {
        "name": "league_analysis",
        "description": "Analyze mini-league standings, rival squads, template vs differential ownership, and simulated outcomes.",
        "parameters": {
            "type": "object",
            "properties": {"league_id": {"type": "integer", "description": "Target mini-league ID"}},
        },
    },
    {
        "name": "price_change_watch",
        "description": "Check imminent price rises and falls tonight based on net transfers, hourly rate, and market momentum.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_players",
        "description": "Direct statistical and ML projection comparison between two or more players.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of player names to compare",
                }
            },
            "required": ["player_names"],
        },
    },
    {
        "name": "get_news",
        "description": "Fetch official injury, availability, and press conference signals for a specific player.",
        "parameters": {
            "type": "object",
            "properties": {"player_name": {"type": "string", "description": "Player name or web_name"}},
            "required": ["player_name"],
        },
    },
    {
        "name": "get_fixtures",
        "description": "Retrieve upcoming fixtures and difficulty ratings for a team or gameweek.",
        "parameters": {
            "type": "object",
            "properties": {
                "team_id": {"type": "integer", "description": "Team ID"},
                "gameweek": {"type": "integer", "description": "Gameweek number"},
            },
        },
    },
    {
        "name": "evaluate_plan",
        "description": "Scores a manager proposed plan (transfers, chip, captain) against the frontier and best alternatives.",
        "parameters": {
            "type": "object",
            "properties": {
                "proposed_transfers": {
                    "type": "array",
                    "items": _MOVE_SCHEMA,
                    "description": "List of transfers e.g. [{'out': 12, 'in': 34}]",
                },
                "proposed_chip": {
                    "type": "string",
                    "description": "Optional chip to play: wildcard, freehit, bboost or 3xc",
                },
                "proposed_captain": {"type": "integer", "description": "Optional captain player ID"},
            },
        },
    },
    {
        "name": "compare_options",
        "description": "Compares named alternative plans side by side showing net xP, hits, and risk.",
        "parameters": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "transfers": {"type": "array", "items": _MOVE_SCHEMA},
                            "chip": {"type": "string", "description": "wildcard, freehit, bboost or 3xc"},
                            "captain": {"type": "integer", "description": "Captain player ID"},
                        },
                    },
                    "description": "Options to compare side by side",
                }
            },
            "required": ["options"],
        },
    },
    {
        "name": "explain_why_not",
        "description": "Explains why the planner did not pick a hit or chip (wildcard, freehit, bboost, 3xc) this week, using the planner's own scored plans.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": "'hit', 'chip', or a chip: wildcard, freehit, bboost, 3xc",
                }
            },
            "required": ["action_type"],
        },
    },
    {
        "name": "affordability_path",
        "description": "Finds the cheapest legal paths to own a target player, including direct swaps and two-step moves.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_element": {"type": "integer", "description": "Target player ID"},
                "top_k": {"type": "integer", "description": "Number of alternative paths (default 3)"},
            },
            "required": ["target_element"],
        },
    },
]


def _next_fixtures(fixtures: list[Any], team_map: dict[int, str], team_id: int, from_gw: int, n: int = 3) -> list[str]:
    rows = sorted(
        (f for f in fixtures if f.event and f.event >= from_gw and team_id in (f.team_h, f.team_a)),
        key=lambda f: f.event,
    )[:n]
    out = []
    for f in rows:
        home = f.team_h == team_id
        opp = team_map.get(f.team_a if home else f.team_h, "?")
        out.append(
            f"GW{f.event} {opp}({'H' if home else 'A'},fdr{f.team_h_difficulty if home else f.team_a_difficulty})"
        )
    return out


class ToolExecutor:
    def __init__(self):
        pass

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        cache = chat_tool_cache.get()
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        if cache is not None and key in cache:
            import copy

            return copy.deepcopy(cache[key])
        began = time.monotonic()
        logger.info("Chat tool started: name=%s", tool_name)
        try:
            result = await self._execute(tool_name, arguments)
            if cache is not None and not result.get("error"):
                import copy

                cache[key] = copy.deepcopy(result)
            return result
        finally:
            logger.info("Chat tool finished: name=%s elapsed_seconds=%.1f", tool_name, time.monotonic() - began)

    async def _execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name == "get_current_decision":
                return section_of(await current_decision_record(), arguments.get("section"))
            elif tool_name == "get_my_team":
                return await self._tool_get_my_team(arguments)
            elif tool_name == "get_projections":
                return await self._tool_get_projections(arguments)
            elif tool_name == "optimise_transfers":
                return await self._tool_optimise_transfers(arguments)
            elif tool_name == "plan_chips":
                return await self._tool_plan_chips(arguments)
            elif tool_name == "captain_options":
                return await self._tool_captain_options(arguments)
            elif tool_name == "league_analysis":
                return await self._tool_league_analysis(arguments)
            elif tool_name == "price_change_watch":
                return await self._tool_price_change_watch(arguments)
            elif tool_name == "compare_players":
                return await self._tool_compare_players(arguments)
            elif tool_name == "get_news":
                return await self._tool_get_news(arguments)
            elif tool_name == "get_fixtures":
                return await self._tool_get_fixtures(arguments)
            elif tool_name == "evaluate_plan":
                return await self._tool_evaluate_plan(arguments)
            elif tool_name == "compare_options":
                return await self._tool_compare_options(arguments)
            elif tool_name == "explain_why_not":
                return await self._tool_explain_why_not(arguments)
            elif tool_name == "affordability_path":
                return await self._tool_affordability_path(arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e)}

    async def _tool_get_my_team(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.domain.manager_state import manager_state_service

        state = await manager_state_service.get_current_state()
        if args.get("manager_id") is not None and args["manager_id"] != state.manager_id:
            return {"status": "unavailable", "reason": "Requested manager differs from configured squad"}
        if len(state.squad) != 15:
            return {"status": "unavailable", "reason": state.error_message or "Configured squad unavailable"}
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        team_map = {t.id: t.short_name for t in boot.teams}
        elements = {int(e.id): e for e in boot.elements}
        squad = []
        for p in state.squad:
            row = p.model_dump(mode="json")
            e = elements.get(int(p.element))
            if e is not None:
                row.update(
                    form=e.form,
                    minutes=e.minutes,
                    season_points=e.total_points,
                    owned_pct=e.selected_by_percent,
                    news=e.news or "",
                    next_fixtures=_next_fixtures(fixtures, team_map, int(p.team), state.target_gw),
                )
            squad.append(row)
        return {
            "manager_name": state.manager_name,
            "team_name": state.team_name,
            "overall_points": state.overall_points,
            "overall_rank": state.overall_rank,
            "bank_millions": state.bank_millions,
            "free_transfers": state.free_transfers,
            "bank_source": state.bank_source,
            "ft_source": state.ft_source,
            "chips_used": state.chips_used,
            "chips_remaining_set_1": state.chips_remaining_set_1,
            "chips_remaining_set_2": state.chips_remaining_set_2,
            "active_chip": state.active_chip,
            "target_gameweek": state.target_gw,
            "deadline_utc": state.deadline_utc,
            "squad": squad,
            "is_stale": state.is_stale,
            "data_as_of": state.source_timestamp,
        }

    async def _tool_get_projections(self, args: dict[str, Any]) -> dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = target_gameweek(curr_gw, next_gw)

        query = (args.get("query") or "").strip().lower()
        horizon = min(8, max(1, int(args.get("horizon") or 3)))
        limit = min(800, max(1, int(args.get("limit") or 15)))

        horizon_proj = await analysis_service.projections(target_gw, horizon, boot, fixtures)
        df = horizon_proj.get(target_gw, pd.DataFrame())
        if df.empty:
            return {"gameweek": target_gw, "horizon": horizon, "players": []}
        weeks = [g for g in range(target_gw, target_gw + horizon) if g in horizon_proj]
        by_gw = {g: horizon_proj[g].set_index("element")["expected_points"] for g in weeks}
        df = df.assign(horizon_xp=df["element"].map(sum(by_gw.values())).fillna(0.0))
        if query:
            df = df[df["web_name"].str.lower().str.contains(query, regex=False) | (df["position"].str.lower() == query)]
        df = df.sort_values("horizon_xp" if len(weeks) > 1 else "expected_points", ascending=False)
        owned = {int(e.id): e.selected_by_percent for e in boot.elements}

        results = []
        for _, row in df.head(limit).iterrows():
            results.append(
                {
                    "element": int(row["element"]),
                    "web_name": row["web_name"],
                    "team": int(row["team"]),
                    "position": row["position"],
                    "cost": row["value"] / 10.0,
                    "expected_points_gw": round(float(row["expected_points"]), 2),
                    "expected_points_horizon": round(float(row["horizon_xp"]), 2),
                    "xp_by_gw": {g: round(float(by_gw[g].get(int(row["element"]), 0.0)), 2) for g in weeks},
                    "p10_floor": round(float(row["p10"]), 2),
                    "p90_ceiling": round(float(row["p90"]), 2),
                    "exp_defcon_points": round(float(row["exp_defcon_pts"]), 2),
                    "ownership_pct": float(owned.get(int(row["element"])) or 0.0),
                }
            )

        return {"gameweek": target_gw, "horizon": horizon, "gameweeks": weeks, "players": results}

    async def _tool_optimise_transfers(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.server.routes.api import OptimizeRequest, run_optimizer

        result = await run_optimizer(
            OptimizeRequest(locked_in=args.get("locked_in"), locked_out=args.get("locked_out"))
        )
        plan = result.get("recommended_plan", {})
        boot, _ = await fpl_client.get_bootstrap_static()
        elem_map = {int(e.id): e for e in boot.elements}

        def clean_player(p: Any) -> dict[str, Any]:
            if not isinstance(p, dict):
                return {}
            raw_val = p.get("value", p.get("cost", 50))
            val = float(raw_val) if raw_val is not None else 50.0
            cost = round(val / (10.0 if val > 20 else 1.0), 1)
            return {
                "element": p.get("element"),
                "web_name": p.get("web_name"),
                "position": p.get("position"),
                "cost": cost,
                "expected_points": round(float(p.get("expected_points", 0)), 2),
                "p10_floor": round(float(p.get("p10", 0)), 2),
                "p90_ceiling": round(float(p.get("p90", 0)), 2),
                "defcon_pts": round(float(p.get("exp_defcon_pts", 0)), 2),
                "opponent_difficulty": p.get("opponent_difficulty"),
                "is_home": bool(p.get("is_home", False)),
                "chance_of_playing": p.get("chance_of_playing", 100),
            }

        def format_moves(step: dict[str, Any]) -> list[dict[str, Any]]:
            moves = []
            t_ins = step.get("transfers_in", [])
            t_outs = step.get("transfers_out", [])
            max_len = max(len(t_ins), len(t_outs))
            for i in range(max_len):
                in_id = t_ins[i] if i < len(t_ins) else None
                out_id = t_outs[i] if i < len(t_outs) else None
                # MILP trajectories carry player dicts, the beam search carries ids.
                in_id = in_id.get("element") if isinstance(in_id, dict) else in_id
                out_id = out_id.get("element") if isinstance(out_id, dict) else out_id
                p_in = elem_map.get(in_id) if in_id else None
                p_out = elem_map.get(out_id) if out_id else None
                moves.append(
                    {
                        "transfer_in": p_in.web_name if p_in else (f"Player #{in_id}" if in_id else "None"),
                        "transfer_out": p_out.web_name if p_out else (f"Player #{out_id}" if out_id else "None"),
                        "cost_in": round(p_in.now_cost / 10.0, 1) if p_in else None,
                        "cost_out": round(p_out.now_cost / 10.0, 1) if p_out else None,
                    }
                )
            return moves

        first_step = plan.get("trajectory", [{}])[0] if plan.get("trajectory") else {}
        transfers_formatted = format_moves(first_step) if first_step else []

        candidate_plans_formatted = []
        for cp in result.get("candidate_plans", []):
            cp_step = cp.get("trajectory", [{}])[0] if cp.get("trajectory") else {}
            candidate_plans_formatted.append(
                {
                    "plan_type": cp.get("plan_type"),
                    "type": cp.get("plan_type"),
                    "summary": cp.get("recommendation_summary"),
                    "net_xp": round(float(cp.get("net_expected_points", 0)), 2),
                    "gain": round(float(cp.get("expected_gain", 0)), 2),
                    "gain_vs_roll": round(float(cp.get("expected_gain", 0)), 2),
                    "hit_cost": round(float(cp.get("hit_cost", 0)), 1),
                    "transfers": format_moves(cp_step) if cp_step else [],
                }
            )

        captain_clean = clean_player(plan.get("captain", {}))
        vice_captain_clean = clean_player(plan.get("vice_captain", {}))

        roadmap_clean = []
        for step in result.get("transfer_roadmap", []):
            roadmap_clean.append(
                {
                    "gameweek": step.get("gameweek"),
                    "action": step.get("action"),
                    "status": step.get("status"),
                    "banked_free_transfers": step.get("banked_free_transfers_projected", step.get("banked_ft")),
                    "banked_free_transfers_projected": step.get(
                        "banked_free_transfers_projected", step.get("banked_ft")
                    ),
                    "strategic_focus": step.get("strategic_focus"),
                    "key_targets": step.get("key_targets", []),
                    "captain": step.get("captain"),
                }
            )

        return {
            **result,
            "target_gameweek": result.get("target_gameweek"),
            "decision": plan.get("recommendation_summary", plan.get("plan_type", "Unavailable")),
            "recommended_plan": result["recommended_plan"],
            "plan_type": plan.get("plan_type"),
            "net_expected_points": round(float(plan.get("net_expected_points", 0)), 2),
            "expected_gain": round(float(plan.get("expected_gain", 0)), 2),
            "hit_cost": round(float(plan.get("hit_cost", 0)), 1),
            "bank_remaining": round(float(plan.get("bank_remaining", 0)), 2),
            "recommended_transfers": transfers_formatted,
            "hit_verdict": result.get("hit_verdict", "No hit recommended"),
            "captain": captain_clean,
            "vice_captain": vice_captain_clean,
            "formation": plan.get("formation"),
            "candidate_plans": candidate_plans_formatted,
            "roadmap": roadmap_clean,
            "transfer_roadmap": roadmap_clean,
            "chip_comparison_table": result.get("chip_comparison_table", []),
            "league_objective": result.get("league_objective"),
            "data_as_of": result.get("data_as_of"),
            "is_stale": result.get("is_stale", False),
        }

    async def _tool_plan_chips(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.server.routes.api import get_chip_strategy

        result = await get_chip_strategy()
        return {
            **result,
            "chip_table": result.get("chip_plan_table", []),
            "set_1_warning": result.get("set_1_deadline_warning"),
        }

    async def _tool_captain_options(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.server.routes.api import get_squad

        # The published advice is what the manager sees; only re-plan when nothing is published.
        result, _ = published_result()
        squad = result["squadData"] if result else await get_squad()
        if not squad.get("captain"):
            return {"status": "unavailable", "reason": squad.get("reason", "Configured lineup unavailable")}
        gw = squad["target_gameweek"]
        if args.get("gameweek") is not None and args["gameweek"] != gw:
            return {
                "status": "unavailable",
                "reason": "Requested gameweek differs from current configured recommendation",
                "target_gameweek": gw,
            }
        boot, _ = await fpl_client.get_bootstrap_static()
        owned = {int(e.id): float(e.selected_by_percent or 0.0) for e in boot.elements}
        captain, vice = squad["captain"], squad["vice_captain"]
        candidates = sorted(squad["starters"], key=lambda p: p.get("expected_points", 0), reverse=True)[:5]
        candidates_clean = [
            {
                "element": c.get("element"),
                "web_name": c.get("web_name"),
                "position": c.get("position"),
                "cost": round(float(c.get("now_cost", 0)), 1),
                "expected_points": round(float(c.get("expected_points", 0)), 2),
                "p10_floor": round(float(c.get("p10", 0)), 2),
                "p90_ceiling": round(float(c.get("p90", 0)), 2),
                "defcon_pts": round(float(c.get("exp_defcon_pts", 0)), 2),
                "opponent_difficulty": c.get("opponent_difficulty"),
                "is_home": c.get("is_home"),
                "chance_of_playing": c.get("chance_of_playing", 100),
                "ownership_pct": owned.get(int(c.get("element", 0)), 0.0),
            }
            for c in candidates
        ]
        return {
            "gameweek": gw,
            "risk_preference": "automatic",
            "scope": "selected_plan_not_submitted",
            "source": "published_decision" if result else "live_plan",
            "captain_name": captain["web_name"],
            "vice_captain_name": vice["web_name"],
            "captain": captain,
            "vice_captain": vice,
            "differential_option": next(
                (
                    c["web_name"]
                    for c in candidates_clean
                    if c["ownership_pct"] < 10.0 and c["element"] != captain["element"]
                ),
                None,
            ),
            "candidates": candidates_clean,
            "is_stale": squad.get("is_stale", False),
            "data_as_of": squad.get("data_as_of"),
        }

    async def _tool_league_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.domain.manager_state import manager_state_service

        profile = data_store.get_profile()
        league_id = args.get("league_id") or profile.target_league_id
        if not league_id:
            return {"message": "No target mini-league ID set in profile. Please enter your League ID."}

        standings = await league_standings_manager.get_league_standings(league_id)
        boot, _ = await fpl_client.get_bootstrap_static()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        effective_curr_gw = curr_gw or (next_gw - 1 if next_gw and next_gw > 1 else 1)

        rivals_res = await rival_analyzer.analyze_rivals(
            standings=standings["standings"],
            user_manager_id=profile.manager_id,
            current_gw=effective_curr_gw,
            bootstrap=boot,
            max_rivals_to_inspect=None,
            user_squad_df=(await manager_state_service.get_current_state()).to_squad_dataframe(),
        )

        return {
            "league_name": standings["league_name"],
            "total_teams": standings["total_teams"],
            "template_players": rivals_res["template_players"][:5],
            "differentials": rivals_res["differential_players"][:5],
            "top_rivals": [
                {
                    "rank": r["rank"],
                    "name": r["player_name"],
                    "total_points": r["total_points"],
                    "chips_used": r["chips_used"],
                }
                for r in rivals_res["rival_squads"][:5]
            ],
        }

    async def _tool_price_change_watch(self, args: dict[str, Any]) -> dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        predictions = price_change_predictor.analyze_price_changes(boot)
        rises = [p for p in predictions if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:5]
        falls = [p for p in predictions if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:5]
        return {"imminent_rises": rises, "imminent_falls": falls}

    async def _tool_compare_players(self, args: dict[str, Any]) -> dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = target_gameweek(curr_gw, next_gw)

        gw_df = projection_engine.predict_gameweek(
            target_gw, boot, fixtures, reconciled_inputs=news_analyzer.get_reconciled_inputs(boot, target_gw)
        )
        names = [n.strip().lower() for n in args.get("player_names", [])]

        matches = []
        for name in names:
            sub = gw_df[gw_df["web_name"].str.lower().str.contains(name)]
            if not sub.empty:
                r = sub.iloc[0]
                matches.append(
                    {
                        "element": int(r["element"]),
                        "web_name": r["web_name"],
                        "position": r["position"],
                        "cost": round(float(r["value"]) / 10.0, 1),
                        "expected_points": round(float(r["expected_points"]), 2),
                        "p10": round(float(r.get("p10", 0.0)), 2),
                        "p90": round(float(r.get("p90", 0.0)), 2),
                        "p10_floor": round(float(r.get("p10", 0.0)), 2),
                        "p90_ceiling": round(float(r.get("p90", 0.0)), 2),
                        "defcon_pts": round(float(r.get("exp_defcon_pts", 0.0)), 2),
                        "opponent_difficulty": r.get("opponent_difficulty"),
                        "is_home": bool(r.get("is_home", False)),
                        "chance_of_playing": r.get("chance_of_playing", 100),
                    }
                )

        return {"gameweek": target_gw, "comparisons": matches}

    async def _tool_get_news(self, args: dict[str, Any]) -> dict[str, Any]:
        player_query = (args.get("player_name") or args.get("query") or "").strip().lower()
        boot, stale = await fpl_client.get_bootstrap_static()
        if stale:
            return {"status": "stale"}
        team_map = {t.id: t.name for t in boot.teams}

        matching = [
            e
            for e in boot.elements
            if player_query in e.web_name.lower() or player_query in f"{e.first_name} {e.second_name}".lower()
        ]
        if not matching:
            return {"status": "not_found", "message": f"No player matching '{player_query}' found in database."}

        p = matching[0]
        is_fit = p.status == "a" and (p.chance_of_playing_next_round in [None, 100]) and not p.news
        return {
            "status": "found",
            "element": p.id,
            "web_name": p.web_name,
            "team": team_map.get(p.team, "Unknown"),
            "is_fit": is_fit,
            "availability_status": p.status,
            "chance_of_playing": p.chance_of_playing_next_round
            if p.chance_of_playing_next_round is not None
            else (100 if is_fit else None),
            "news": p.news or "No current injury or suspension news reported.",
            "source": "Official Premier League / FPL API",
        }

    async def _tool_get_fixtures(self, args: dict[str, Any]) -> dict[str, Any]:
        fixtures, _ = await fpl_client.get_fixtures()
        boot, _ = await fpl_client.get_bootstrap_static()
        team_map = {t.id: t.short_name for t in boot.teams}
        team_id = args.get("team_id")
        gw = args.get("gameweek")

        res = []
        for f in fixtures:
            if gw and f.event != gw:
                continue
            if team_id and f.team_h != team_id and f.team_a != team_id:
                continue
            res.append(
                {
                    "event": f.event,
                    "home_team": team_map.get(f.team_h, f"Team {f.team_h}"),
                    "away_team": team_map.get(f.team_a, f"Team {f.team_a}"),
                    "difficulty_home": f.team_h_difficulty,
                    "difficulty_away": f.team_a_difficulty,
                    "kickoff_time": f.kickoff_time,
                }
            )
        return {"fixtures": res[:10]}

    async def _critique_inputs(self, horizon: int = 8) -> dict[str, Any]:
        """Same state, units, chips and horizon the published plan is solved with."""
        from fpl_oracle.domain.manager_state import manager_state_service

        state = await manager_state_service.get_current_state()
        if len(state.squad) != 15:
            raise ValueError(state.error_message or "Configured squad unavailable")
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = target_gameweek(curr_gw, next_gw)
        projs = await analysis_service.projections(target_gw, horizon, boot, fixtures)
        last_gw = 19 if target_gw <= 19 else 38
        horizon_len = 0
        while target_gw + horizon_len in projs and target_gw + horizon_len <= last_gw:
            horizon_len += 1
        return {
            "state": state,
            "current_squad_df": state.to_squad_dataframe(),
            "player_pool_df": projs.get(target_gw, pd.DataFrame()),
            "bank": float(state.bank_tenths),
            "free_transfers": int(state.free_transfers),
            "horizon_projections": projs,
            "current_gw": curr_gw or target_gw - 1,
            "target_gw": target_gw,
            "horizon_len": max(1, horizon_len),
            "chips_available": list(state.chips_remaining_set_1 if target_gw <= 19 else state.chips_remaining_set_2),
        }

    async def _tool_evaluate_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.advisor.critique import plan_critique_service

        ctx = await self._critique_inputs()
        ctx.pop("state")
        res = plan_critique_service.evaluate_plan(
            proposed_transfers=args.get("proposed_transfers", []),
            proposed_chip=args.get("proposed_chip"),
            proposed_captain=args.get("proposed_captain"),
            **ctx,
        )
        return res.model_dump()

    async def _tool_compare_options(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.advisor.critique import plan_critique_service

        ctx = await self._critique_inputs()
        ctx.pop("state")
        res = plan_critique_service.compare_options(options=args.get("options", []), **ctx)
        return res.model_dump()

    async def _tool_explain_why_not(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.advisor.critique import plan_critique_service
        from fpl_oracle.domain.manager_state import manager_state_service

        state = await manager_state_service.get_current_state()
        # The planner's own scored plans: every candidate (hold, hits, chips) with its gain over rolling.
        opt = await self._tool_optimise_transfers({})
        plans = []
        for row in opt.get("chip_comparison_table", []):
            plan = row.get("plan") or {}
            chip = None if row.get("chip") == "HOLD" else row.get("chip")
            plans.append(
                {
                    "label": f"{plan.get('plan_type')} ({chip})" if chip else str(plan.get("plan_type")),
                    "chip": chip,
                    "hits": plan.get("hits", 0),
                    "gain_vs_roll": row.get("net_gain_vs_hold"),
                    "recommended": bool(row.get("is_recommended")),
                }
            )
        res = plan_critique_service.explain_why_not(
            action_type=args.get("action_type", "wildcard"), squad_df=state.to_squad_dataframe(), plans=plans
        )
        return res.model_dump()

    async def _tool_affordability_path(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.advisor.funding import AffordabilityPathFinder

        ctx = await self._critique_inputs(horizon=1)
        finder = AffordabilityPathFinder()
        paths = finder.find_cheapest_paths(
            current_squad_df=ctx["current_squad_df"],
            player_pool_df=ctx["player_pool_df"],
            bank_tenths=int(ctx["bank"]),
            target_element=int(args["target_element"]),
            free_transfers=ctx["free_transfers"],
            top_k=int(args.get("top_k", 3)),
        )
        return {
            "target_element": args["target_element"],
            "paths": [
                {
                    "target_element": p.target_element,
                    "path_type": p.path_type,
                    "transfers_count": p.transfers_count,
                    "hits": p.hits,
                    "hit_cost": p.hit_cost,
                    "initial_bank_tenths": p.initial_bank_tenths,
                    "final_bank_tenths": p.final_bank_tenths,
                    "is_legal": p.is_legal,
                    "arithmetic_explanation": p.arithmetic_explanation,
                }
                for p in paths
            ],
        }


tool_executor = ToolExecutor()
