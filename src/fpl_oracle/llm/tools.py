"""
Tool definitions and execution dispatcher for the FPL Expert Chat Agent.
Connects conversational queries directly to live ML projections, MILP optimizations,
chip strategies, news, and league simulations.
"""

import logging
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.data.store import data_store
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.server.analysis import analysis_service

logger = logging.getLogger("fpl_oracle.llm.tools")

TOOL_DEFINITIONS = [
    {
        "name": "get_my_team",
        "description": "Fetch current squad, bank, overall points, and chip history for user.",
        "parameters": {
            "type": "object",
            "properties": {"manager_id": {"type": "integer", "description": "Optional manager ID"}},
        },
    },
    {
        "name": "get_projections",
        "description": "Retrieve ML projected expected points and uncertainty (P10, P50, P90, DefCon) for upcoming gameweeks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Player name or position filter"},
                "horizon": {"type": "integer", "description": "Number of upcoming gameweeks (1-8)"},
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
        "description": "Run Joint Chip Strategy Planner evaluating optimal gameweeks for Wildcard, Free Hit, Triple Captain, and Bench Boost across both Set 1 (GW1-19) and Set 2 (GW20-38).",
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
        "description": "Analyze mini-league standings, rival squads, template vs differential ownership, and win probability.",
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
]


class ToolExecutor:
    def __init__(self):
        pass

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        logger.info(f"Executing tool {tool_name} with args {arguments}")
        try:
            if tool_name == "get_my_team":
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
            "squad": [p.model_dump(mode="json") for p in state.squad],
            "is_stale": state.is_stale,
            "data_as_of": state.source_timestamp,
        }

    async def _tool_get_projections(self, args: dict[str, Any]) -> dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 1)

        query = (args.get("query") or "").strip().lower()
        horizon = min(8, max(1, args.get("horizon", 3)))

        horizon_proj = await analysis_service.projections(target_gw, horizon, boot, fixtures)
        df_target = horizon_proj.get(target_gw, pd.DataFrame())

        if query:
            filtered = df_target[
                df_target["web_name"].str.lower().str.contains(query) | (df_target["position"].str.lower() == query)
            ]
        else:
            filtered = df_target.head(15)

        results = []
        for _, row in filtered.head(15).iterrows():
            results.append(
                {
                    "web_name": row["web_name"],
                    "team": int(row["team"]),
                    "position": row["position"],
                    "cost": row["value"] / 10.0,
                    "expected_points_gw": round(float(row["expected_points"]), 2),
                    "p10_floor": round(float(row["p10"]), 2),
                    "p90_ceiling": round(float(row["p90"]), 2),
                    "exp_defcon_points": round(float(row["exp_defcon_pts"]), 2),
                }
            )

        return {"gameweek": target_gw, "horizon": horizon, "players": results}

    async def _tool_optimise_transfers(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.server.routes.api import OptimizeRequest, run_optimizer

        return await run_optimizer(OptimizeRequest(locked_in=args.get("locked_in"), locked_out=args.get("locked_out")))

    async def _tool_plan_chips(self, args: dict[str, Any]) -> dict[str, Any]:
        from fpl_oracle.server.routes.api import get_chip_strategy

        result = await get_chip_strategy()
        return {
            **result,
            "chip_table": result.get("chip_plan_table", []),
            "set_1_warning": result.get("set_1_deadline_warning"),
        }

    async def _tool_captain_options(self, args: dict[str, Any]) -> dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        gw = args.get("gameweek") or next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 1)

        horizon = await analysis_service.projections(gw, 8, boot, fixtures)
        gw_df = horizon.get(gw, pd.DataFrame())
        top5 = gw_df.sort_values(by="expected_points", ascending=False).head(5)

        return {
            "gameweek": gw,
            "safe_captain": top5.iloc[0]["web_name"],
            "differential_captain": top5.iloc[1]["web_name"] if len(top5) > 1 else top5.iloc[0]["web_name"],
            "candidates": top5[["web_name", "position", "team", "expected_points", "p10", "p90"]].to_dict(
                orient="records"
            ),
        }

    async def _tool_league_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
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
        target_gw = next_gw or (curr_gw + 1 if curr_gw and curr_gw < 38 else 1)

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
                        "web_name": r["web_name"],
                        "position": r["position"],
                        "cost": r["value"] / 10.0,
                        "expected_points": round(float(r["expected_points"]), 2),
                        "p10": round(float(r["p10"]), 2),
                        "p90": round(float(r["p90"]), 2),
                        "defcon_pts": round(float(r["exp_defcon_pts"]), 2),
                    }
                )

        return {"gameweek": target_gw, "comparisons": matches}

    async def _tool_get_news(self, args: dict[str, Any]) -> dict[str, Any]:
        player_query = (args.get("player_name") or args.get("query") or "").strip().lower()
        boot, _ = await fpl_client.get_bootstrap_static()
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
            else (100 if is_fit else 0),
            "news": p.news or "No current injury or suspension news reported.",
            "source": "Official Premier League / FPL API",
            "confidence": 1.0,
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


tool_executor = ToolExecutor()
