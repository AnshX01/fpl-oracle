"""
Per-Gameweek Unified Decision Card Engine (D1, D2, D3).
Generates the comprehensive operational decision card and printable markdown export
from the shared game state, projection snapshots, transfer engine, chip calendar,
and rival simulations.
"""

import asyncio
import logging
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.data.store import data_store
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
from fpl_oracle.league.standings import league_standings_manager
from fpl_oracle.league.strategy import league_strategy_advisor
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.server.analysis import analysis_service

logger = logging.getLogger("fpl_oracle.briefing.decision_card")


class DecisionCardGenerator:
    """Generates a per-gameweek decision card unifying all tactical facets."""

    async def generate_decision_card(self, include_league: bool = True) -> dict[str, Any]:
        game_state = await game_state_manager.get_game_state()
        effective_state = await manager_state_service.get_current_state()
        profile = data_store.get_profile()
        if len(effective_state.squad) != 15:
            return {
                "status": "unavailable",
                "transfers": None,
                "captain": None,
                "reason": "Configured squad unavailable",
                "is_stale": effective_state.is_stale,
            }

        boot, is_stale = await fpl_client.get_bootstrap_static()
        fixtures, fixtures_stale = await fpl_client.get_fixtures()
        is_stale = is_stale or fixtures_stale or effective_state.is_stale
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        effective_curr_gw = curr_gw or game_state.current_gw or 1
        target_gw = next_gw or game_state.next_gw or (effective_curr_gw + 1 if effective_curr_gw < 38 else 38)

        team_map = {t.id: t for t in boot.teams}

        # ----------------------------------------------------------------------
        # 1. Projections & Squad Resolution
        # ----------------------------------------------------------------------
        horizon_proj = await analysis_service.projections(target_gw, 8, boot, fixtures=fixtures)
        target_df = horizon_proj.get(target_gw, pd.DataFrame())

        user_squad_df = effective_state.to_squad_dataframe()
        if user_squad_df.empty or len(user_squad_df) != 15:
            return {
                "status": "unavailable",
                "transfers": None,
                "captain": None,
                "reason": "Configured squad unavailable",
                "is_stale": effective_state.is_stale,
            }

        if (
            target_df.empty
            or "element" not in target_df
            or not set(user_squad_df["element"]).issubset(set(target_df["element"]))
        ):
            return {
                "status": "unavailable",
                "captain": None,
                "transfers": None,
                "reason": "Projection snapshot missing configured players",
                "is_stale": True,
            }

        # Attach current projections to squad
        if not target_df.empty and "expected_points" in target_df.columns:
            xp_map = {int(r["element"]): float(r["expected_points"]) for _, r in target_df.iterrows()}
            p10_map = {int(r["element"]): float(r.get("p10", 0.0)) for _, r in target_df.iterrows()}
            p90_map = {int(r["element"]): float(r.get("p90", 0.0)) for _, r in target_df.iterrows()}
            user_squad_df["expected_points"] = (
                user_squad_df["element"].map(xp_map).fillna(user_squad_df["expected_points"])
            )
            user_squad_df["p10"] = user_squad_df["element"].map(p10_map).fillna(0.0)
            user_squad_df["p90"] = user_squad_df["element"].map(p90_map).fillna(0.0)

        # ----------------------------------------------------------------------
        # 3. Chip Strategy & Joint Optimization (G10)
        # ----------------------------------------------------------------------
        hist = None
        if effective_state.manager_id:
            try:
                hist, _ = await fpl_client.get_manager_history(effective_state.manager_id)
            except Exception as e:
                logger.warning("Decision card manager history fetch warning: %s", e)

        available_chips = (
            effective_state.chips_remaining_set_1 if target_gw <= 19 else effective_state.chips_remaining_set_2
        )
        chips_used = [c["name"] for c in effective_state.chips_used if (c["event"] <= 19) == (target_gw <= 19)]

        joint_res = await analysis_service.joint_plan(
            current_squad_df=user_squad_df,
            player_pool_df=target_df,
            bank=float(effective_state.bank_tenths),
            free_transfers=int(effective_state.free_transfers),
            horizon_projections=horizon_proj,
            current_gw=effective_curr_gw,
            target_gw=target_gw,
            available_chips=available_chips,
            chips_by_set={1: effective_state.chips_remaining_set_1, 2: effective_state.chips_remaining_set_2},
            chips_already_used=chips_used,
        )

        if include_league:
            chip_strat = await analysis_service.chip_strategy(
                current_gw=effective_curr_gw,
                current_squad_df=user_squad_df,
                horizon_projections=horizon_proj,
                fixtures=fixtures,
                bootstrap=boot,
                manager_history=hist,
            )

            chip_strat = analysis_service.bind_chip_schedule(chip_strat, joint_res)

        else:
            chip_strat = {
                "set_1_deadline_warning": "First-set chips expire at GW19; value beyond the loaded projection window remains unresolved."
            }

        # ----------------------------------------------------------------------
        # 4. Synchronized Transfer & Chip Decision (ONE Plan across surfaces)
        # ----------------------------------------------------------------------
        rec_plan = joint_res["recommended_plan"]
        # ----------------------------------------------------------------------
        # 2. Lineup & Captaincy
        # ----------------------------------------------------------------------
        lineup_res = rec_plan.get("lineup")
        if not isinstance(lineup_res, dict) or "starters" not in lineup_res:
            raise ValueError("Recommended plan missing verified post-action lineup")

        starters_list = []
        captain_elem = lineup_res["captain"]["element"]
        vice_elem = lineup_res["vice_captain"]["element"]

        for _, s in lineup_res["starters"].iterrows():
            eid = int(s["element"])
            tm = team_map.get(int(s["team"]))
            starters_list.append(
                {
                    "element": eid,
                    "web_name": s["web_name"],
                    "position": s["position"],
                    "team": int(s["team"]),
                    "team_short": tm.short_name if tm else "PL",
                    "expected_points": round(float(s["expected_points"]), 2),
                    "p10": round(float(s.get("p10", 0.0)), 2),
                    "p90": round(float(s.get("p90", 0.0)), 2),
                    "is_captain": eid == captain_elem,
                    "is_vice_captain": eid == vice_elem,
                }
            )

        bench_list = []
        for idx, (_, b) in enumerate(lineup_res["bench"].iterrows(), start=1):
            eid = int(b["element"])
            tm = team_map.get(int(b["team"]))
            bench_list.append(
                {
                    "element": eid,
                    "web_name": b["web_name"],
                    "position": b["position"],
                    "team": int(b["team"]),
                    "team_short": tm.short_name if tm else "PL",
                    "expected_points": round(float(b["expected_points"]), 2),
                    "bench_order": idx,
                }
            )

        cap_tm = team_map.get(int(lineup_res["captain"].get("team", 0)))
        vice_tm = team_map.get(int(lineup_res["vice_captain"].get("team", 0)))

        captain_dict = {
            "element": int(lineup_res["captain"]["element"]),
            "web_name": lineup_res["captain"]["web_name"],
            "team_short": cap_tm.short_name if cap_tm else "PL",
            "expected_points": round(float(lineup_res["captain"]["expected_points"]), 2),
            "p10": round(float(lineup_res["captain"].get("p10", 0.0)), 2),
            "p90": round(float(lineup_res["captain"].get("p90", 0.0)), 2),
        }

        vice_dict = {
            "element": int(lineup_res["vice_captain"]["element"]),
            "web_name": lineup_res["vice_captain"]["web_name"],
            "team_short": vice_tm.short_name if vice_tm else "PL",
            "expected_points": round(float(lineup_res["vice_captain"]["expected_points"]), 2),
            "p10": round(float(lineup_res["vice_captain"].get("p10", 0.0)), 2),
            "p90": round(float(lineup_res["vice_captain"].get("p90", 0.0)), 2),
        }

        t_in_raw = rec_plan.get("transfers_in", [])
        t_out_raw = rec_plan.get("transfers_out", [])

        transfers_in = []
        for p in t_in_raw:
            if isinstance(p, dict):
                transfers_in.append(
                    {
                        "element": int(p.get("element", 0)),
                        "web_name": p.get("web_name", ""),
                        "cost": round(float(p.get("cost", 0.0)), 1),
                        "expected_points": round(float(p.get("expected_points", 0.0)), 2),
                    }
                )

        transfers_out = []
        for p in t_out_raw:
            if isinstance(p, dict):
                transfers_out.append(
                    {
                        "element": int(p.get("element", 0)),
                        "web_name": p.get("web_name", ""),
                        "cost": round(float(p.get("sell_price", 0.0)), 1),
                        "expected_points": round(float(p.get("expected_points", 0.0)), 2),
                    }
                )

        is_roll = rec_plan.get("plan_type") == "ROLL_TRANSFER"
        action_name = "ROLL" if is_roll else ("SINGLE_TRANSFER" if len(transfers_in) == 1 else "MULTIPLE_TRANSFERS")

        transfers_summary = {
            "action": action_name,
            "is_roll": is_roll,
            "in": transfers_in,
            "out": transfers_out,
            "bank_after": round(float(rec_plan.get("remaining_bank", effective_state.bank_millions)), 2),
            "ft_used": 0
            if is_roll or joint_res.get("recommended_chip") in ("wildcard", "freehit")
            else min(effective_state.free_transfers, len(transfers_in)),
            "ft_remaining": max(
                0,
                int(effective_state.free_transfers)
                - (0 if joint_res.get("recommended_chip") in ("wildcard", "freehit") else len(transfers_in)),
            ),
            "ft_next_gw": int(rec_plan["next_banked_ft"]),
            "hit_cost": int(rec_plan.get("hits", 0) * 4),
            "hits_count": int(rec_plan.get("hits", 0)),
            "net_gain_vs_roll": round(float(rec_plan.get("horizon_gain_vs_roll", 0.0)), 2),
            "expected_gain_gw": round(float(rec_plan.get("expected_gain", 0.0)), 2),
            "horizon_pts": round(float(rec_plan.get("horizon_net_xp", 0.0)), 2),
            "no_regret_flag": bool(rec_plan.get("is_no_regret", False)),
            "action_summary": rec_plan.get("recommendation_summary", "Roll transfer for tactical flexibility."),
        }

        # Synchronize chip decision directly with joint planner
        chip_rec_now = bool(joint_res.get("recommended_chip") is not None)
        active_chip_code = joint_res.get("recommended_chip")
        from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

        best_cand = joint_res.get("best_candidate", {})
        active_chip_name = CHIP_DISPLAY_NAMES.get(active_chip_code, active_chip_code) if active_chip_code else None

        chip_decision = {
            "recommend": chip_rec_now,
            "chip_name": active_chip_code,
            "chip_display_name": active_chip_name,
            "reason": (
                best_cand.get("reason", "")
                if chip_rec_now
                else "No chip deployment recommended this gameweek. Save chips for confirmed DGWs."
            ),
            "next_best_window": next(
                (step["gameweek"] for step in rec_plan["trajectory"][1:] if step.get("chip")), None
            ),
            "gain_vs_hold": round(float(best_cand.get("gross_gain_vs_hold", 0.0)), 1) if chip_rec_now else 0.0,
            "set_1_deadline_warning": chip_strat.get("set_1_deadline_warning"),
        }

        # ----------------------------------------------------------------------
        # 5. League & Rival Analysis
        # ----------------------------------------------------------------------
        league_id = profile.target_league_id
        rivals_dict: dict[str, Any] = {
            "status": "unconfigured",
            "league_name": "None",
            "posture": "BALANCED_ATTACK",
            "posture_reason": "No target mini-league configured. Add your mini-league ID in settings to activate rival analysis.",
            "nearest_above_gap": None,
            "nearest_below_gap": None,
            "rival_count": 0,
            "exposure_players": [],
            "differential_players": [],
        }

        win_prob_dict: dict[str, Any] = {
            "status": "unconfigured",
            "p_first": None,
            "p_above_key_rivals": None,
            "expected_rank": None,
            "mc_se": None,
            "simulation_note": "No mini-league configured for Monte Carlo championship simulation.",
        }

        if league_id and not include_league:
            rivals_dict.update(
                status="deferred",
                posture_reason="Detailed league report is calculating separately; core plan already uses observed rival exposure.",
            )
            win_prob_dict.update(
                status="deferred",
                simulation_note="Detailed league simulation is separate from core advice. Probability not yet measured.",
            )

        if league_id and include_league:
            try:
                is_gw_known = (curr_gw is not None) or (game_state.current_gw is not None)
                standings_data = await league_standings_manager.get_league_standings(league_id)
                coverage_data = standings_data.get("coverage", {})

                if not is_gw_known:
                    rivals_dict = {
                        "status": "unavailable",
                        "league_name": standings_data.get("league_name", "Mini-League"),
                        "posture": "UNAVAILABLE",
                        "posture_reason": "Current gameweek is unknown.",
                        "nearest_above_gap": None,
                        "nearest_below_gap": None,
                        "rival_count": 0,
                        "exposure_players": [],
                        "differential_players": [],
                        "coverage": coverage_data,
                    }
                    win_prob_dict = {
                        "status": "unavailable",
                        "p_first": None,
                        "p_above_key_rivals": None,
                        "expected_rank": None,
                        "mc_se": None,
                        "simulation_note": "Current gameweek is unknown.",
                    }
                elif not standings_data.get("standings"):
                    rivals_dict = {
                        "status": "unavailable",
                        "league_name": standings_data.get("league_name", "Mini-League"),
                        "posture": "UNAVAILABLE",
                        "posture_reason": "Mini-league standings are empty or unavailable.",
                        "nearest_above_gap": None,
                        "nearest_below_gap": None,
                        "rival_count": 0,
                        "exposure_players": [],
                        "differential_players": [],
                        "coverage": coverage_data,
                    }
                    win_prob_dict = {
                        "status": "unavailable",
                        "p_first": None,
                        "p_above_key_rivals": None,
                        "expected_rank": None,
                        "mc_se": None,
                        "simulation_note": "Standings are empty or unavailable.",
                    }
                else:
                    user_rank = None
                    user_row = None
                    if effective_state.manager_id:
                        for row in standings_data["standings"]:
                            if row.get("entry") == effective_state.manager_id:
                                user_rank = row.get("rank")
                                user_row = row
                                break

                    if effective_state.manager_id is None or user_rank is None:
                        rivals_dict = {
                            "status": "unavailable",
                            "league_name": standings_data.get("league_name", "Mini-League"),
                            "posture": "UNAVAILABLE",
                            "posture_reason": "Manager not found in mini-league standings."
                            if effective_state.manager_id
                            else "Manager ID not configured in .env.",
                            "nearest_above_gap": None,
                            "nearest_below_gap": None,
                            "rival_count": 0,
                            "exposure_players": [],
                            "differential_players": [],
                            "coverage": coverage_data,
                        }
                        win_prob_dict = {
                            "status": "unavailable",
                            "p_first": None,
                            "p_above_key_rivals": None,
                            "expected_rank": None,
                            "mc_se": None,
                            "simulation_note": "Manager not found in mini-league standings."
                            if effective_state.manager_id
                            else "Manager ID not configured.",
                        }
                    else:
                        eval_gw = effective_curr_gw
                        rivals_res = await rival_analyzer.analyze_rivals(
                            standings=standings_data["standings"],
                            user_manager_id=effective_state.manager_id,
                            current_gw=eval_gw,
                            bootstrap=boot,
                        )

                        user_pts = (
                            int(user_row.get("total", effective_state.overall_points))
                            if user_row is not None
                            else int(effective_state.overall_points)
                        )
                        selection_mode = rivals_res.get("rival_selection_mode", "")

                        if selection_mode == "leader, no close chasers":
                            strategy = league_strategy_advisor.evaluate_strategy(
                                user_rank=user_rank,
                                user_total_points=user_pts,
                                rivals_analysis=rivals_res,
                                user_squad_df=user_squad_df,
                            )
                            rivals_dict = {
                                "status": "leader, no close chasers",
                                "league_name": standings_data.get("league_name", "Mini-League"),
                                "posture": strategy.get("posture", "CONSOLIDATE_LEAD"),
                                "posture_reason": "Leader with no close chasers within points window.",
                                "nearest_above_gap": 0,
                                "nearest_below_gap": strategy.get("gap_to_next"),
                                "rival_count": 0,
                                "exposure_players": [],
                                "differential_players": [],
                                "coverage": coverage_data,
                            }
                            win_prob_dict = {
                                "status": "leader, no close chasers",
                                "p_first": None,
                                "p_above_key_rivals": None,
                                "expected_rank": None,
                                "mc_se": None,
                                "simulation_note": "Leader with no rivals within points window.",
                            }
                        else:
                            strategy = league_strategy_advisor.evaluate_strategy(
                                user_rank=user_rank,
                                user_total_points=user_pts,
                                rivals_analysis=rivals_res,
                                user_squad_df=user_squad_df,
                            )

                            # Monte Carlo simulation across multi-gameweek horizon
                            mc_res = await asyncio.to_thread(
                                monte_carlo_simulator.simulate_league,
                                user_points=user_pts,
                                user_squad_df=user_squad_df,
                                rival_squads=rivals_res.get("rival_squads", []),
                                projections_df=target_df,
                                horizon_gws=5,
                                seed=42,
                                projections_by_gw=horizon_proj,
                            )

                            exposure_names = [
                                p.get("web_name", "") for p in rivals_res.get("differential_players", [])[:3]
                            ]
                            diff_names = [p.get("web_name", "") for p in rivals_res.get("template_players", [])[:3]]

                            rivals_dict = {
                                "status": "configured",
                                "league_name": standings_data.get("league_name", "Mini-League"),
                                "posture": strategy.get("posture", "BALANCED_ATTACK"),
                                "posture_reason": strategy.get(
                                    "rationale", "Maintain balanced upside with core template coverage."
                                ),
                                "nearest_above_gap": strategy.get("gap_to_leader"),
                                "nearest_below_gap": strategy.get("gap_to_next"),
                                "rival_count": len(rivals_res.get("rival_squads", [])),
                                "exposure_players": exposure_names,
                                "differential_players": diff_names,
                                "coverage": coverage_data,
                            }

                            if mc_res.get("status") == "NO_RIVALS_FOUND":
                                win_prob_dict = {
                                    "status": "no_rivals",
                                    "p_first": None,
                                    "p_above_key_rivals": None,
                                    "expected_rank": None,
                                    "mc_se": None,
                                    "simulation_note": mc_res.get(
                                        "message", "No rivals found in window for simulation."
                                    ),
                                }
                            elif mc_res.get("status") != "SIMULATION_SUCCESS":
                                win_prob_dict = {
                                    "status": "unavailable",
                                    "p_first": None,
                                    "p_above_key_rivals": None,
                                    "expected_rank": None,
                                    "mc_se": None,
                                    "simulation_note": mc_res.get("status", "Unknown simulation state"),
                                }
                            else:
                                win_prob_dict = {
                                    "status": "simulated",
                                    "p_first": round(float(mc_res.get("user_win_probability_pct", 0.0)), 1),
                                    "p_above_key_rivals": round(
                                        float(
                                            mc_res.get(
                                                "p_above_key_rivals", mc_res.get("user_win_probability_pct", 0.0)
                                            )
                                        ),
                                        1,
                                    ),
                                    "expected_rank": round(float(mc_res.get("expected_final_rank", 1.0)), 1),
                                    "mc_se": round(float(mc_res.get("win_prob_se", 0.5)), 2),
                                    "simulation_note": "Owned squad held with observed prior picks/captain; selected-rival horizon scenario, not selected transfer plan or season title odds.",
                                }
            except Exception as e:
                logger.error(f"[DecisionCard] Error during rival/Monte Carlo simulation: {e}")
                win_prob_dict = {
                    "status": "error",
                    "error_message": str(e),
                    "p_first": None,
                    "expected_rank": None,
                    "p_above_key_rivals": None,
                    "mc_se": None,
                    "simulation_note": f"Simulation failed: {e}",
                }

        # ----------------------------------------------------------------------
        # 6. Two-line Reasoning, Caveats & Freshness
        # ----------------------------------------------------------------------
        if is_roll:
            two_line = (
                f"Rolling the free transfer banks flexibility (reaching {transfers_summary['ft_next_gw']} FTs next GW) with starting XI projecting {lineup_res['starters_expected_points']:.1f} pts. "
                f"Captain {captain_dict['web_name']} leads output at {captain_dict['expected_points']:.1f} xP."
            )
        else:
            in_names = ", ".join(p["web_name"] for p in transfers_in)
            out_names = ", ".join(p["web_name"] for p in transfers_out)
            two_line = (
                f"Executing {out_names} -> {in_names} gains +{transfers_summary['expected_gain_gw']:.1f} xP this GW and +{transfers_summary['net_gain_vs_roll']:.1f} xP across the {len(joint_res['decision_scope']['horizon_gameweeks'])}-GW horizon. "
                f"Lineup totals {lineup_res['total_gameweek_expected_points']:.1f} xP led by captain {captain_dict['web_name']}."
            )

        caveats = [
            f"{len(joint_res['decision_scope']['horizon_gameweeks'])}-GW supported projection search. Value beyond this window is not estimated; chip decisions are conditional, not whole-season optima.",
            "Monitor Friday press conference updates for confirmed starter status.",
            "Verify lineup locking prior to the official deadline window.",
        ]
        if chip_decision["set_1_deadline_warning"]:
            caveats.append(str(chip_decision["set_1_deadline_warning"]))

        # News summary note
        news_count = 0
        try:
            news_signals = await analysis_service.news_signals(boot, target_gw)
            news_count = len(news_signals)
        except Exception:
            pass

        what_changed = (
            f"{news_count} team-news signals loaded; only gated, reconciled inputs may affect projections."
            if news_count > 0
            else "Projections refreshed against latest fixture and availability states."
        )

        deadline_seconds = game_state.seconds_to_deadline or 0.0
        active_ver = model_registry.get_active_version()

        return {
            "gameweek": target_gw,
            "deadline": {
                "deadline_time": game_state.deadline_time,
                "seconds_to_deadline": round(deadline_seconds, 1),
                "is_live": game_state.is_live,
                "label": f"GW{target_gw} Deadline: {game_state.deadline_time or 'TBD'}",
            },
            "data_freshness": {
                "projections_as_of": fpl_client.get_data_as_of("bootstrap-static"),
                "news_as_of": fpl_client.get_data_as_of("news") or fpl_client.get_data_as_of("bootstrap-static"),
                "rivals_as_of": fpl_client.get_data_as_of("league") or fpl_client.get_data_as_of("bootstrap-static"),
                "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
            },
            "model_version": active_ver.get("version", "v1.0.0"),
            "projection_snapshot_id": target_df.attrs.get("snapshot_id"),
            "decision_scope": joint_res.get("decision_scope"),
            "rival_scenarios": joint_res.get("rival_scenarios"),
            "resource_frontier": joint_res.get("resource_frontier"),
            "decision_readiness": joint_res.get("decision_readiness"),
            "ranked_options": [
                {
                    "rank": i + 1,
                    "chip": c["chip"],
                    "net_gain_vs_hold": c["net_gain_vs_hold"],
                    "reason": c["reason"],
                    "recommended": c.get("is_recommended", False),
                }
                for i, c in enumerate(joint_res.get("chip_comparison_table", []))
            ],
            "transfers": transfers_summary,
            "xi": starters_list,
            "formation": lineup_res.get("formation", "3-5-2"),
            "captain": captain_dict,
            "vice_captain": vice_dict,
            "bench": bench_list,
            "chip": chip_decision,
            "rivals": rivals_dict,
            "win_prob": win_prob_dict,
            "two_line_reasoning": two_line,
            "caveats": caveats,
            "what_changed": what_changed,
            "advice_disclaimer": "Advice only - nothing is submitted to FPL.",
            "is_stale": is_stale,
            "stale": is_stale,
            "data_as_of": fpl_client.get_data_as_of("bootstrap-static"),
        }


def format_decision_card_markdown(card: dict[str, Any]) -> str:
    """Formats decision card dictionary into a clean, printable markdown document."""
    gw = card.get("gameweek", 6)
    dl = card.get("deadline", {})
    t = card.get("transfers", {})
    cap = card.get("captain", {})
    vice = card.get("vice_captain", {})
    chip = card.get("chip", {})
    riv = card.get("rivals", {})
    wp = card.get("win_prob", {})

    lines = [
        f"# FPL Oracle — Gameweek {gw} Decision Card",
        f"**Generated:** {card.get('data_as_of', 'Recent')} | **Model:** {card.get('model_version', 'v1.0.0')}",
        f"**Deadline:** {dl.get('label', 'TBD')} ({dl.get('seconds_to_deadline', 0):.0f}s remaining)",
        f"**Disclaimer:** {card.get('advice_disclaimer', 'Advice only — nothing is submitted to FPL.')}",
        "",
        "---",
        "",
        "## 1. Chip Strategy",
        f"**Recommendation:** {'YES: ' + str(chip.get('chip_display_name')) if chip.get('recommend') else 'HOLD CHIP (Save for DGW/BGW)'}",
        f"- **Expected Gain vs Hold:** +{chip.get('gain_vs_hold', 0.0)} pts",
        f"- **Rationale:** {chip.get('reason', '')}",
        f"- **Next Best Window:** GW{chip.get('next_best_window', 'TBD')}",
    ]

    if chip.get("set_1_deadline_warning"):
        lines.append(f"- **Set 1 Expiry:** [Warning] {chip['set_1_deadline_warning']}")

    lines.extend(
        [
            "",
            "## 2. Transfer Plan",
            f"**Recommended Move:** {t.get('action_summary', '')}",
        ]
    )

    if t.get("is_roll"):
        lines.append(f"- **Strategy:** Roll transfer to accumulate {t.get('ft_remaining', 2)} free transfers.")
    else:
        in_names = ", ".join(f"{p['web_name']} (£{p['cost']}m)" for p in t.get("in", []))
        out_names = ", ".join(f"{p['web_name']} (£{p['cost']}m)" for p in t.get("out", []))
        lines.append(f"- **Transfers Out:** {out_names}")
        lines.append(f"- **Transfers In:** {in_names}")

    lines.extend(
        [
            f"- **GW Net Gain vs Roll:** +{t.get('expected_gain_gw', 0.0):.1f} pts",
            f"- **5-GW Horizon Net Gain:** +{t.get('net_gain_vs_roll', 0.0):.1f} pts",
            f"- **Hit Penalty:** -{t.get('hit_cost', 0)} pts ({t.get('hits_count', 0)} extra transfer(s))",
            f"- **Bank After:** £{t.get('bank_after', 0.0):.1f}m | **FTs Next Week:** {t.get('ft_remaining', 1)}",
            f"- **No-Regret Status:** {'Model stability flag, not verified future success' if t.get('no_regret_flag') else 'Conditional model candidate'}",
            "",
            "## 3. Starting XI & Captaincy",
            f"**Formation:** {card.get('formation', '3-5-2')}",
            f"- **Captain (C):** {cap.get('web_name', '')} ({cap.get('team_short', '')}) — {cap.get('expected_points', 0.0)} xP [P10: {cap.get('p10', 0.0)}, P90: {cap.get('p90', 0.0)}]",
            f"- **Vice-Captain (V):** {vice.get('web_name', '')} ({vice.get('team_short', '')}) — {vice.get('expected_points', 0.0)} xP",
            "",
            "### Starting Lineup",
            "| Pos | Player | Club | xP | P10-P90 | Role |",
            "|:---|:---|:---:|:---:|:---:|:---:|",
        ]
    )

    for p in card.get("xi", []):
        role = "Captain (C)" if p.get("is_captain") else ("Vice (V)" if p.get("is_vice_captain") else "")
        interval = f"[{p.get('p10', 0.0):.1f} - {p.get('p90', 0.0):.1f}]"
        lines.append(
            f"| {p.get('position')} | {p.get('web_name')} | {p.get('team_short')} | {p.get('expected_points', 0.0):.1f} | {interval} | {role} |"
        )

    lines.extend(
        [
            "",
            "### Bench Order",
        ]
    )
    for b in card.get("bench", []):
        lines.append(
            f"{b.get('bench_order')}. {b.get('web_name')} ({b.get('position')}, {b.get('team_short')}) — {b.get('expected_points', 0.0):.1f} xP"
        )

    lines.extend(
        [
            "",
            "## 4. Mini-League & Rivals",
            f"- **Mini-League:** {riv.get('league_name', 'None')} ({riv.get('status', 'unconfigured')})",
            f"- **Tactical Posture:** {riv.get('posture', 'BALANCED_ATTACK')}",
            f"- **Posture Rationale:** {riv.get('posture_reason', '')}",
            f"- **Championship Win Probability:** {wp.get('p_first', 0.0):.1f}% (Expected Finish: {wp.get('expected_rank', 1.0):.1f})",
            f"- **Key Exposure Risks:** {', '.join(riv.get('exposure_players', [])) or 'None'}",
            f"- **Key Differentials:** {', '.join(riv.get('differential_players', [])) or 'None'}",
            "",
            "## 5. Decision Summary & Pre-Deadline Caveats",
            f"**Rationale:** {card.get('two_line_reasoning', '')}",
            "",
            "**Caveats:**",
        ]
    )

    for c in card.get("caveats", []):
        lines.append(f"- {c}")

    lines.extend(
        [
            f"- {card.get('what_changed', '')}",
            "",
            "---",
            f"*{card.get('advice_disclaimer', '')}*",
        ]
    )

    return "\n".join(lines)


decision_card_generator = DecisionCardGenerator()
