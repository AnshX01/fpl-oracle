"""
Tool definitions and execution dispatcher for the FPL Expert Chat Agent.
Connects conversational queries directly to live ML projections, MILP optimizations,
chip strategies, news, and league simulations.
"""

from typing import Dict, Any, List, Optional
import json
import logging
import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.squad import squad_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.price_change import price_change_predictor
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.chips.calendar import fixture_calendar
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.news.analyse import news_analyzer
from fpl_oracle.data.store import data_store

logger = logging.getLogger("fpl_oracle.llm.tools")

TOOL_DEFINITIONS = [
    {
        "name": "get_my_team",
        "description": "Fetch current squad, bank, overall points, and chip history for user.",
        "parameters": {
            "type": "object",
            "properties": {
                "manager_id": {"type": "integer", "description": "Optional manager ID"}
            }
        }
    },
    {
        "name": "get_projections",
        "description": "Retrieve ML projected expected points and uncertainty (P10, P50, P90, DefCon) for upcoming gameweeks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Player name or position filter"},
                "horizon": {"type": "integer", "description": "Number of upcoming gameweeks (1-8)"}
            }
        }
    },
    {
        "name": "optimise_transfers",
        "description": "Run mathematical MILP transfer optimizer evaluating candidate plans (roll transfer, 1 transfer, 2 transfers with hit evaluation).",
        "parameters": {
            "type": "object",
            "properties": {
                "locked_in": {"type": "array", "items": {"type": "integer"}, "description": "Player IDs to force keep"},
                "locked_out": {"type": "array", "items": {"type": "integer"}, "description": "Player IDs to never buy"}
            }
        }
    },
    {
        "name": "plan_chips",
        "description": "Run Joint Chip Strategy Planner evaluating optimal gameweeks for Wildcard, Free Hit, Triple Captain, and Bench Boost across both Set 1 (GW1-19) and Set 2 (GW20-38).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "captain_options",
        "description": "Rank captaincy options for target gameweek with ceiling, floor, and safe vs differential picks.",
        "parameters": {
            "type": "object",
            "properties": {
                "gameweek": {"type": "integer", "description": "Target gameweek"}
            }
        }
    },
    {
        "name": "league_analysis",
        "description": "Analyze mini-league standings, rival squads, template vs differential ownership, and win probability.",
        "parameters": {
            "type": "object",
            "properties": {
                "league_id": {"type": "integer", "description": "Target mini-league ID"}
            }
        }
    },
    {
        "name": "price_change_watch",
        "description": "Check imminent price rises and falls tonight based on net transfers, hourly rate, and market momentum.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "compare_players",
        "description": "Direct statistical and ML projection comparison between two or more players.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_names": {"type": "array", "items": {"type": "string"}, "description": "List of player names to compare"}
            },
            "required": ["player_names"]
        }
    }
]

class ToolExecutor:
    def __init__(self):
        pass

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
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
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error in tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e)}

    async def _tool_get_my_team(self, args: Dict[str, Any]) -> Dict[str, Any]:
        profile = data_store.get_profile()
        m_id = args.get("manager_id") or profile.manager_id
        if not m_id:
            return {"status": "no_manager_id", "message": "No manager ID configured. Please set in profile or pass manager_id."}

        boot, _ = await fpl_client.get_bootstrap_static()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        gw_to_fetch = curr_gw or 1

        entry, _ = await fpl_client.get_manager_entry(m_id)
        picks, _ = await fpl_client.get_manager_picks(m_id, gw_to_fetch)
        history, _ = await fpl_client.get_manager_history(m_id)

        elem_map = {e.id: e for e in boot.elements}
        squad = []
        for p in picks.picks:
            elem = elem_map.get(p.element)
            squad.append({
                "element": p.element,
                "web_name": elem.web_name if elem else f"Player {p.element}",
                "team": elem.team if elem else 0,
                "cost": (elem.now_cost / 10.0) if elem else 0.0,
                "is_captain": p.is_captain,
                "is_vice_captain": p.is_vice_captain,
                "multiplier": p.multiplier
            })

        bank_m = (picks.entry_history.bank / 10.0) if picks.entry_history else 0.0
        return {
            "manager_name": f"{entry.player_first_name} {entry.player_last_name}",
            "team_name": entry.name,
            "overall_points": entry.summary_overall_points,
            "overall_rank": entry.summary_overall_rank,
            "bank_millions": bank_m,
            "chips_used": [c.name for c in history.chips],
            "squad": squad
        }

    async def _tool_get_projections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6

        query = (args.get("query") or "").strip().lower()
        horizon = min(8, max(1, args.get("horizon", 3)))

        horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, horizon, boot, fixtures)
        df_target = horizon_proj.get(target_gw, pd.DataFrame())

        if query:
            filtered = df_target[
                df_target["web_name"].str.lower().str.contains(query) |
                (df_target["position"].str.lower() == query)
            ]
        else:
            filtered = df_target.head(15)

        results = []
        for _, row in filtered.head(15).iterrows():
            results.append({
                "web_name": row["web_name"],
                "team": int(row["team"]),
                "position": row["position"],
                "cost": row["value"] / 10.0,
                "expected_points_gw": round(float(row["expected_points"]), 2),
                "p10_floor": round(float(row["p10"]), 2),
                "p90_ceiling": round(float(row["p90"]), 2),
                "exp_defcon_points": round(float(row["exp_defcon_pts"]), 2)
            })

        return {"gameweek": target_gw, "horizon": horizon, "players": results}

    async def _tool_optimise_transfers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        profile = data_store.get_profile()
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6

        horizon_proj = projection_engine.predict_multi_gameweeks(target_gw, 5, boot, fixtures)
        target_df = horizon_proj.get(target_gw, pd.DataFrame())

        # Load user squad or generate standard template if not loaded
        m_id = profile.manager_id
        current_squad_df = None
        bank = 5.0 # default £0.5m
        ft = profile.free_transfers or 1

        if m_id:
            try:
                picks, _ = await fpl_client.get_manager_picks(m_id, curr_gw or 5)
                picks_ids = [p.element for p in picks.picks]
                current_squad_df = target_df[target_df["element"].isin(picks_ids)].copy()
                if picks.entry_history:
                    bank = picks.entry_history.bank
            except Exception:
                pass

        if current_squad_df is None or len(current_squad_df) < 15:
            # Fallback to a valid 15-man squad within budget using MILP solver
            squad_res = squad_optimizer.solve_best_squad(player_pool_df=target_df, budget=1000.0)
            current_squad_df = squad_res["squad"].copy()

        opt_res = transfer_optimizer.evaluate_transfer_options(
            current_squad_df=current_squad_df,
            player_pool_df=target_df,
            bank=bank,
            free_transfers=ft,
            horizon_projections=horizon_proj,
            current_gw=curr_gw or 5,
            target_gw=target_gw,
            locked_in_ids=args.get("locked_in"),
            locked_out_ids=args.get("locked_out")
        )
        return {
            "decision": opt_res["recommended_plan"]["recommendation_summary"],
            "hit_verdict": opt_res["hit_verdict"],
            "candidate_plans": [
                {
                    "type": p["plan_type"],
                    "summary": p["recommendation_summary"],
                    "net_xp": p["net_expected_points"],
                    "gain": p["expected_gain"],
                    "hits": p["hits"]
                }
                for p in opt_res["candidate_plans"]
            ],
            "roadmap": opt_res["transfer_roadmap"]
        }

    async def _tool_plan_chips(self, args: Dict[str, Any]) -> Dict[str, Any]:
        profile = data_store.get_profile()
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

        horizon_proj = projection_engine.predict_multi_gameweeks(next_gw or 6, 8, boot, fixtures)
        pool_df = horizon_proj.get(next_gw or 6, pd.DataFrame())

        hist = None
        if profile.manager_id:
            try:
                hist, _ = await fpl_client.get_manager_history(profile.manager_id)
            except Exception:
                pass

        squad_df = squad_optimizer.solve_best_squad(player_pool_df=pool_df, budget=1000.0)["squad"]
        chip_res = chip_planner.generate_chip_strategy(
            current_gw=curr_gw or 5,
            current_squad_df=squad_df,
            horizon_projections=horizon_proj,
            fixtures=fixtures,
            bootstrap=boot,
            manager_history=hist
        )
        return {
            "set_1_warning": chip_res["set_1_deadline_warning"],
            "chip_table": chip_res["chip_plan_table"]
        }

    async def _tool_captain_options(self, args: Dict[str, Any]) -> Dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        gw = args.get("gameweek") or (await fpl_client.get_current_and_next_gw())[1] or 6

        gw_df = projection_engine.predict_gameweek(gw, boot, fixtures)
        top5 = gw_df.sort_values(by="expected_points", ascending=False).head(5)

        return {
            "gameweek": gw,
            "safe_captain": top5.iloc[0]["web_name"],
            "differential_captain": top5.iloc[1]["web_name"] if len(top5) > 1 else top5.iloc[0]["web_name"],
            "candidates": top5[["web_name", "position", "team", "expected_points", "p10", "p90"]].to_dict(orient="records")
        }

    async def _tool_league_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        profile = data_store.get_profile()
        league_id = args.get("league_id") or profile.target_league_id
        if not league_id:
            return {"message": "No target mini-league ID set in profile. Please enter your League ID."}

        standings = await league_standings_manager.get_league_standings(league_id, max_pages=1)
        boot, _ = await fpl_client.get_bootstrap_static()
        curr_gw, _ = await fpl_client.get_current_and_next_gw()

        rivals_res = await rival_analyzer.analyze_rivals(
            standings=standings["standings"],
            user_manager_id=profile.manager_id,
            current_gw=curr_gw or 5,
            bootstrap=boot,
            max_rivals_to_inspect=6
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
                    "chips_used": r["chips_used"]
                }
                for r in rivals_res["rival_squads"][:5]
            ]
        }

    async def _tool_price_change_watch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        predictions = price_change_predictor.analyze_price_changes(boot)
        rises = [p for p in predictions if p["direction"] in ["RISE_IMMINENT", "LIKELY_RISE"]][:5]
        falls = [p for p in predictions if p["direction"] in ["FALL_IMMINENT", "LIKELY_FALL"]][:5]
        return {
            "imminent_rises": rises,
            "imminent_falls": falls
        }

    async def _tool_compare_players(self, args: Dict[str, Any]) -> Dict[str, Any]:
        boot, _ = await fpl_client.get_bootstrap_static()
        fixtures, _ = await fpl_client.get_fixtures()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

        gw_df = projection_engine.predict_gameweek(next_gw or 6, boot, fixtures)
        names = [n.strip().lower() for n in args.get("player_names", [])]

        matches = []
        for name in names:
            sub = gw_df[gw_df["web_name"].str.lower().str.contains(name)]
            if not sub.empty:
                r = sub.iloc[0]
                matches.append({
                    "web_name": r["web_name"],
                    "position": r["position"],
                    "cost": r["value"] / 10.0,
                    "expected_points": round(float(r["expected_points"]), 2),
                    "p10": round(float(r["p10"]), 2),
                    "p90": round(float(r["p90"]), 2),
                    "defcon_pts": round(float(r["exp_defcon_pts"]), 2)
                })

        return {"gameweek": next_gw or 6, "comparisons": matches}

tool_executor = ToolExecutor()
