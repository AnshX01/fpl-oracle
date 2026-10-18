"""
Per-Gameweek Unified Decision Card Engine (D1, D2, D3).
Generates the comprehensive operational decision card and printable markdown export
from the shared game state, projection snapshots, transfer engine, chip calendar,
and rival simulations.
"""

import asyncio
import json
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
from fpl_oracle.server.analysis import analysis_service, target_gameweek

logger = logging.getLogger("fpl_oracle.briefing.decision_card")


MIN_LATER_CHIP_GAIN = 1.0


def _millions(price: Any) -> float:
    """Plan prices arrive in tenths from the MILP trajectory and in millions from the beam search; no FPL
    player costs more than 20m, so anything above that is tenths."""
    value = float(price or 0.0)
    return value / 10.0 if value > 20.0 else value


def _better(value: Any) -> str:
    gain = float(value)
    return f"about {abs(gain):.1f} points {'better' if gain >= 0 else 'worse'}"


def _pair_moves(ins: list[dict[str, Any]], outs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match each player going out with one coming in, same position first, so a screen can show 'A -> B'."""
    free = list(ins)
    pairs = []
    for out in outs:
        pick = next((i for i in free if i.get("position") and i.get("position") == out.get("position")), None)
        pick = pick or (free[0] if free else None)
        if pick is not None:
            free.remove(pick)
        pairs.append({"out": out, "in": pick})
    pairs.extend({"out": None, "in": i} for i in free)
    return pairs


def _comparison_sentences(rec_plan: dict[str, Any], chip_display: str | None) -> tuple[str | None, str | None]:
    """(chip sentence, transfer sentence): the planner's own confidence check in plain words."""
    from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

    chip_text = move_text = None
    for check, rejected in ((rec_plan.get("decision_check"), False), (rec_plan.get("rejected_alternative"), True)):
        if not check:
            continue
        gain, versus = check.get("gain"), check.get("versus")
        if gain is None:
            continue
        if versus == "keeping the chip":
            name = CHIP_DISPLAY_NAMES.get(check.get("chip"), chip_display or "The chip")
            if rejected:
                chip_text = (
                    f"Keep your {name} for now: playing it this week is only {_better(gain)} "
                    "than waiting, too close to justify using it."
                )
            else:
                chip_text = f"Playing it now is {_better(gain)} than keeping your {name} for later."
        elif versus == "rolling":
            if rejected:
                move_text = (
                    f"The best move is only {_better(gain)} than saving the transfer, too close to justify making it."
                )
            else:
                move_text = f"These moves are {_better(gain)} than saving the transfer."
        elif versus == "best plan without a hit" and not rejected:
            move_text = f"The points hit pays off: {_better(gain)} than the best plan without one."
    return chip_text, move_text


def _later_chips(rec_plan: dict[str, Any], roadmap: list[dict[str, Any]], target_gw: int) -> list[dict[str, Any]]:
    """Chips the plan would play in later weeks. Provisional: the plan is re-decided at every deadline."""
    from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

    gains = {r.get("code"): r.get("expected_gain") for r in roadmap or []}
    later = []
    for act in rec_plan.get("trajectory") or []:
        chip, gw = act.get("chip"), int(act["gameweek"])
        if chip and gw > target_gw:
            gain = gains.get(chip)
            if gain is None or float(gain) < MIN_LATER_CHIP_GAIN:
                continue  # a gain this small is noise, and the plan is re-decided every week anyway
            later.append(
                {
                    "chip": chip,
                    "chip_display_name": CHIP_DISPLAY_NAMES.get(chip, chip),
                    "gameweek": gw,
                    "gain": round(float(gain), 1),
                    "provisional": True,
                    "label": "provisional - re-decided each week",
                }
            )
    return later


def _ordinal(n: Any) -> str:
    n = int(n or 0)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def track_record_summary() -> str | None:
    """One plain line from the scored pre-deadline forecasts (reports/holdout_log.json), or None before any."""
    from fpl_oracle.ml.holdout import HOLDOUT_LOG_PATH

    try:
        log = json.loads(HOLDOUT_LOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    scored = [r["comparison"] for r in log.get("evaluated_gameweeks", []) if r.get("comparison")]
    if not scored:
        return None

    def avg(source: str, key: str) -> float | None:
        vals = [c["sources"][source][key] for c in scored if source in c["sources"]]
        return round(sum(vals) / len(vals), 2) if vals else None

    ours, fpl = avg("FPL Oracle", "rmse"), avg("FPL ep_next", "rmse")
    top_ours, top_fpl = avg("FPL Oracle", "top20_actual_points"), avg("FPL ep_next", "top20_actual_points")
    text = f"Track record ({len(scored)} gameweek{'s' if len(scored) > 1 else ''}): forecast error {ours}"
    if fpl is not None:
        text += f" vs FPL's own {fpl}; our top-20 picks scored {top_ours} pts on average vs FPL's {top_fpl}"
    return text + " (lower error and higher points are better)."


def _your_choices(boot: Any) -> dict[str, list[dict[str, Any]]]:
    """The manager's 'keep' / "don't buy" calls the plan was built around, by name."""
    from fpl_oracle.data.store import data_store

    profile = data_store.get_profile()
    names = {e.id: e.web_name for e in boot.elements}
    return {
        kind: [{"element": i, "web_name": names.get(i, f"Player {i}")} for i in ids]
        for kind, ids in (("keep", profile.keep_ids), ("avoid", profile.avoid_ids))
    }


def standing_summary(standings_data: dict[str, Any], manager_id: int | None) -> dict[str, Any] | None:
    """Where the manager stands in the mini-league: position, gap to the leader and to the next rival."""
    rows = sorted(standings_data.get("standings") or [], key=lambda r: r.get("rank") or 10**9)
    me = next((r for r in rows if manager_id and r.get("entry") == manager_id), None)
    if me is None:
        return None
    idx = rows.index(me)

    def person(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {"manager": row.get("player_name"), "team": row.get("entry_name"), "points": row.get("total")}

    points = int(me.get("total") or 0)
    above, below = (rows[idx - 1] if idx else None), (rows[idx + 1] if idx + 1 < len(rows) else None)
    table = [
        {
            "rank": r.get("rank"),
            "manager": r.get("player_name"),
            "team": r.get("entry_name"),
            "points": r.get("total"),
            "last_gw_points": r.get("event_total"),
            "is_you": r is me,
        }
        for i, r in enumerate(rows)
        if i < 10 or r is me
    ]
    league = standings_data.get("league_name") or "your mini-league"
    if idx == 0:
        lead = max(0, points - int(below.get("total") or 0)) if below else None
        text = f"You lead {league}" + (
            f", {lead} points clear of {below.get('player_name')}." if below is not None else "."
        )
    else:
        assert above is not None  # idx > 0
        gap_leader = max(0, int(rows[0].get("total") or 0) - points)
        gap_next = max(0, int(above.get("total") or 0) - points)
        text = (
            f"You are {_ordinal(me.get('rank'))} in {league}: {gap_leader} points behind the leader "
            f"({rows[0].get('player_name')})"
            + (f", {gap_next} behind {above.get('player_name')} just above you." if idx > 1 else ".")
        )
    return {
        "text": text,
        "position": me.get("rank"),
        "league_size": (standings_data.get("coverage") or {}).get("total_entries") or len(rows),
        "points": points,
        "is_leader": idx == 0,
        "leader": person(rows[0]),
        "gap_to_leader": max(0, int(rows[0].get("total") or 0) - points),
        "next_above": person(above),
        "gap_to_next_above": max(0, int(above.get("total") or 0) - points) if above else None,
        "next_below": person(below),
        "lead_over_next_below": max(0, points - int(below.get("total") or 0)) if below else None,
        "table": table,
    }


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
        target_gw = target_gameweek(effective_curr_gw, next_gw or game_state.next_gw)

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
                "set_1_deadline_warning": (
                    "First-set chips expire at GW19; value beyond the loaded projection window remains unresolved."
                    if target_gw <= 19 and effective_state.chips_remaining_set_1
                    else None
                )
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

        pool_lookup = (
            target_df.drop_duplicates("element").set_index("element")[["position", "team"]].to_dict("index")
            if {"position", "team"} <= set(target_df.columns)
            else {}
        )

        def move_rows(raw: list[Any], price_key: str) -> list[dict[str, Any]]:
            rows = []
            for p in raw:
                if not isinstance(p, dict):
                    continue
                eid = int(p.get("element", 0))
                known = pool_lookup.get(eid, {})
                tm = team_map.get(int(p.get("team") or known.get("team") or 0))
                rows.append(
                    {
                        "element": eid,
                        "web_name": p.get("web_name", ""),
                        "position": p.get("position") or known.get("position"),
                        "team_short": tm.short_name if tm else None,
                        "cost": round(_millions(p.get(price_key)), 1),
                        "expected_points": round(float(p.get("expected_points", 0.0)), 2),
                    }
                )
            return rows

        transfers_in = move_rows(t_in_raw, "cost")
        transfers_out = move_rows(t_out_raw, "sell_price")

        is_roll = rec_plan.get("plan_type") == "ROLL_TRANSFER"
        action_name = (
            "ROLL"
            if is_roll
            else "CHIP_ONLY"
            if not transfers_in
            else "SINGLE_TRANSFER"
            if len(transfers_in) == 1
            else "MULTIPLE_TRANSFERS"
        )

        # The planner's trajectory is the horizon that horizon_pts / net_gain_vs_roll are summed over.
        horizon_gws = [int(a["gameweek"]) for a in rec_plan.get("trajectory") or []] or [target_gw]
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
            "horizon_label": f"GW{horizon_gws[0]}-{horizon_gws[-1]}" if len(horizon_gws) > 1 else f"GW{horizon_gws[0]}",
            "horizon_gameweeks": horizon_gws,
            "no_regret_flag": bool(rec_plan.get("is_no_regret", False)),
            "pairs": _pair_moves(transfers_in, transfers_out),
            "temporary": rec_plan.get("chip") == "freehit",
        }

        # Synchronize chip decision directly with joint planner
        chip_rec_now = bool(joint_res.get("recommended_chip") is not None)
        active_chip_code = joint_res.get("recommended_chip")
        from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

        best_cand = joint_res.get("best_candidate", {})
        active_chip_name = CHIP_DISPLAY_NAMES.get(active_chip_code, active_chip_code) if active_chip_code else None

        chip_decision_gain = best_cand.get("net_gain_vs_hold")
        chip_text, move_text = _comparison_sentences(rec_plan, active_chip_name)
        if chip_rec_now and not chip_text:
            if not joint_res.get("normal_plan_feasible", True):
                chip_text = f"{active_chip_name} is needed: no legal team is possible this week without it."
            elif chip_decision_gain is not None:
                chip_text = (
                    f"Playing it now is {_better(chip_decision_gain)} than keeping your {active_chip_name} for later."
                )
        transfers_summary["comparison"] = move_text
        chip_decision = {
            "recommend": chip_rec_now,
            "chip_name": active_chip_code,
            "chip_display_name": active_chip_name,
            "reason": chip_text or ("Save your chips this week." if not chip_rec_now else None),
            "comparison": chip_text,
            "check": rec_plan.get("decision_check") or rec_plan.get("rejected_alternative"),
            "later": _later_chips(rec_plan, joint_res.get("measured_chip_roadmap") or [], target_gw),
            "next_best_window": None,
            "gain_vs_hold": (
                round(float(best_cand.get("net_gain_vs_hold", 0.0)), 1)
                if joint_res.get("normal_plan_feasible", True)
                else None
            )
            if chip_rec_now
            else 0.0,
            "baseline_available": joint_res.get("normal_plan_feasible", True),
            "set_1_deadline_warning": chip_strat.get("set_1_deadline_warning"),
        }

        # ----------------------------------------------------------------------
        # 5. League & Rival Analysis
        # ----------------------------------------------------------------------
        league_id = profile.target_league_id
        rivals_dict: dict[str, Any] = {
            "status": "unconfigured",
            "league_name": None,
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
            "expected_rank": None,
            "mc_se": None,
            "simulation_note": "No mini-league configured for Monte Carlo championship simulation.",
        }

        standings_data: dict[str, Any] | None = None
        if league_id and not include_league:
            rivals_dict.update(
                league_name=f"League #{league_id}",
                status="unavailable",
                posture=None,
                posture_reason="Mini-league standings could not be loaded.",
            )
            win_prob_dict.update(
                status="not_simulated", simulation_note="Win chances are not part of the weekly advice."
            )
            try:
                standings_data = await league_standings_manager.get_league_standings(league_id)
                rivals_dict["league_name"] = standings_data.get("league_name") or rivals_dict["league_name"]
            except Exception as e:
                logger.warning("Decision card league standings warning: %s", e)

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
                            user_squad_df=user_squad_df,
                        )

                        user_pts = (
                            int(user_row.get("total", effective_state.overall_points))
                            if user_row is not None
                            else int(effective_state.overall_points)
                        )
                        selection_mode = rivals_res.get("rival_selection_mode", "")

                        if selection_mode == "leader, no close chasers":
                            others = [
                                float(r.get("total", 0))
                                for r in standings_data["standings"]
                                if r.get("entry") != effective_state.manager_id
                            ]
                            rivals_dict = {
                                "status": "leader, no close chasers",
                                "league_name": standings_data.get("league_name", "Mini-League"),
                                "posture": "CONSOLIDATE_LEAD",
                                "posture_reason": "Leader with no close chasers within points window.",
                                "nearest_above_gap": 0,
                                "nearest_below_gap": int(user_pts - max(others)) if others else None,
                                "rival_count": 0,
                                "exposure_players": [],
                                "differential_players": [],
                                "coverage": coverage_data,
                            }
                            win_prob_dict = {
                                "status": "leader, no close chasers",
                                "p_first": None,
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
                            # A failed simulation must not hide the rival analysis that already succeeded.
                            try:
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
                            except Exception as e:
                                logger.error(f"[DecisionCard] Error during rival/Monte Carlo simulation: {e}")
                                mc_res = {"status": "SIMULATION_ERROR", "error": str(e)}

                            exposure_names = [p.get("web_name", "") for p in rivals_res.get("template_players", [])[:3]]
                            diff_names = [p.get("web_name", "") for p in rivals_res.get("differential_players", [])[:3]]

                            rivals_dict = {
                                "status": "configured",
                                "league_name": standings_data.get("league_name", "Mini-League"),
                                "posture": strategy.get("strategy_mode", "BALANCED_ATTACK"),
                                "posture_reason": strategy.get(
                                    "rationale", "Maintain balanced upside with core template coverage."
                                ),
                                "nearest_above_gap": strategy.get("nearest_above_gap"),
                                "nearest_below_gap": strategy.get("nearest_below_gap"),
                                "rival_count": len(rivals_res.get("rival_squads", [])),
                                "exposure_players": exposure_names,
                                "differential_players": diff_names,
                                "coverage": coverage_data,
                            }

                            if mc_res.get("status") == "NO_RIVALS_FOUND":
                                win_prob_dict = {
                                    "status": "no_rivals",
                                    "p_first": None,
                                    "expected_rank": None,
                                    "mc_se": None,
                                    "simulation_note": mc_res.get(
                                        "message", "No rivals found in window for simulation."
                                    ),
                                }
                            elif mc_res.get("status") == "SIMULATION_ERROR":
                                win_prob_dict = {
                                    "status": "error",
                                    "error_message": mc_res["error"],
                                    "p_first": None,
                                    "expected_rank": None,
                                    "mc_se": None,
                                    "simulation_note": f"Simulation failed: {mc_res['error']}",
                                }
                            elif mc_res.get("status") != "SIMULATION_SUCCESS":
                                win_prob_dict = {
                                    "status": "unavailable",
                                    "p_first": None,
                                    "expected_rank": None,
                                    "mc_se": None,
                                    "simulation_note": mc_res.get("status", "Unknown simulation state"),
                                }
                            else:
                                win_prob_dict = {
                                    "status": "simulated",
                                    "p_first": round(float(mc_res.get("user_win_probability_pct", 0.0)), 1),
                                    "expected_rank": round(float(mc_res.get("expected_final_rank", 1.0)), 1),
                                    "mc_se": round(float(mc_res.get("win_prob_se", 0.5)), 2),
                                    "simulation_note": (
                                        "Owned squad held with observed prior picks/captain; selected-rival horizon scenario, "
                                        "not selected transfer plan or season title odds. "
                                        f"{mc_res.get('rivals_simulated')} of {mc_res.get('rivals_requested')} rivals simulated."
                                    ),
                                }
            except Exception as e:
                logger.error(f"[DecisionCard] Error during rival/Monte Carlo simulation: {e}")
                win_prob_dict = {
                    "status": "error",
                    "error_message": str(e),
                    "p_first": None,
                    "expected_rank": None,
                    "mc_se": None,
                    "simulation_note": f"Simulation failed: {e}",
                }

        summary = standing_summary(standings_data, effective_state.manager_id) if standings_data else None
        league_odds = rec_plan.get("league") if summary else None
        if summary and league_odds:
            # From the planner's win-probability model (docs/decision_model.md section 4), same plan as the card
            summary["win_chance"] = league_odds["p_win"]
            summary["text"] += f" Chance of finishing first with this plan: about {league_odds['p_win']:.0%}."
        if summary:
            finished = [e.id for e in boot.events if e.finished]
            rivals_dict.update(
                standing=summary,
                summary=summary["text"],
                standings_after_gameweek=max(finished) if finished else None,
            )
            if rivals_dict["status"] == "unavailable":
                rivals_dict.update(status="configured", posture_reason=None)
            rivals_dict["nearest_above_gap"] = summary["gap_to_next_above"]
            rivals_dict["nearest_below_gap"] = summary["lead_over_next_below"]
        elif league_id and rivals_dict["status"] == "unavailable" and standings_data:
            rivals_dict["posture_reason"] = "You are not listed in this mini-league."

        # ----------------------------------------------------------------------
        # 6. Two-line Reasoning, Caveats & Freshness
        # ----------------------------------------------------------------------
        n_moves = len(transfers_in)
        hit_note = f" (-{transfers_summary['hit_cost']} point hit)" if transfers_summary["hit_cost"] else ""
        if active_chip_code in ("wildcard", "freehit"):
            if active_chip_code == "freehit":
                headline = f"Play your Free Hit this week: {n_moves} players change for this gameweek only"
            else:
                headline = f"Play your Wildcard now: {n_moves} players change"
        elif active_chip_name:
            extra = f", plus {n_moves} transfer{'s' if n_moves != 1 else ''}{hit_note}" if n_moves else ""
            headline = f"Play your {active_chip_name} this week{extra}"
        elif is_roll:
            headline = "Save your free transfer"
        else:
            moves = [
                f"{pair['out']['web_name']} for {pair['in']['web_name']}"
                for pair in transfers_summary["pairs"]
                if pair["out"] and pair["in"]
            ]
            headline = (
                f"Transfer {moves[0]}{hit_note}"
                if len(moves) == 1
                else f"Make {n_moves} transfers{hit_note}: {', '.join(moves)}"
            )
        transfers_summary["action_summary"] = headline
        team_pts = lineup_res["total_gameweek_expected_points"]
        if is_roll:
            two_line = (
                f"Saving it gives you {transfers_summary['ft_next_gw']} free transfers next week. Your team is "
                f"predicted to score {team_pts:.1f} points this week with {captain_dict['web_name']} as captain."
            )
        elif not transfers_in:
            two_line = (
                f"Your team is predicted to score {team_pts:.1f} points this week with "
                f"{captain_dict['web_name']} as captain."
            )
        else:
            two_line = (
                f"Compared with saving your transfer, this gains about {transfers_summary['expected_gain_gw']:+.1f} "
                f"points this week and {transfers_summary['net_gain_vs_roll']:+.1f} over {transfers_summary['horizon_label']}. "
                f"Your team is predicted to score {team_pts:.1f} points this week with "
                f"{captain_dict['web_name']} as captain."
            )

        caveats = [
            "Uses current fixtures and forecasts through chip expiry.",
            "Check the latest team news.",
            "Confirm your team in FPL before the deadline.",
        ]
        if chip_decision["set_1_deadline_warning"]:
            caveats.append(str(chip_decision["set_1_deadline_warning"]))
        if effective_state.error_message:
            caveats.insert(0, effective_state.error_message)
        # One short line for the screen; the longer list above stays for the chat context.
        caution = (
            effective_state.error_message
            or ("Chips left from the first set expire after GW19." if chip_decision["set_1_deadline_warning"] else None)
            or "Check team news before the deadline."
        )

        # News summary note
        news_count = 0
        press_notes: list[dict[str, Any]] = []
        try:
            news_signals = await analysis_service.news_signals(boot, target_gw)
            news_count = len(news_signals)
            from fpl_oracle.news.press import applied_press_notes

            ids = {p.element for p in effective_state.squad} | {p["element"] for p in transfers_in + transfers_out}
            press_notes = applied_press_notes(news_signals, ids)
        except Exception:
            pass

        what_changed = (
            f"{news_count} team-news items checked against the official injury and availability flags."
            if news_count > 0
            else "Predictions use the latest fixtures and official availability flags."
        )
        if press_notes:
            what_changed += " " + " ".join(n["text"] for n in press_notes)

        deadline_seconds = game_state.seconds_to_deadline or 0.0
        active_ver = model_registry.get_active_version()

        return {
            "confirmation_recommendation": dict(
                manager_id=effective_state.manager_id,
                gameweek=target_gw,
                baseline=[p.element for p in effective_state.squad],
                baseline_bank=effective_state.bank_tenths,
                baseline_ft=effective_state.free_transfers,
                buy_prices={p["element"]: int(round(p["cost"] * 10)) for p in transfers_in},
                ins=[p["element"] for p in transfers_in],
                outs=[p["element"] for p in transfers_out],
                hit_cost=int(rec_plan.get("hits", 0) * 4),
                expected_gain=rec_plan["trajectory"][0].get("expected_player_gain"),
                remaining_bank=int(round(rec_plan["remaining_bank"] * 10)),
                captain=captain_elem,
                vice_captain=vice_elem,
                bench=[p["element"] for p in bench_list],
                chip=active_chip_code,
                baseline_purchase={p.element: p.purchase_price for p in effective_state.squad},
                baseline_selling={p.element: p.selling_price for p in effective_state.squad},
            ),
            "gameweek": target_gw,
            "deadline": {
                "deadline_time": game_state.deadline_time,
                "seconds_to_deadline": round(deadline_seconds, 1),
                "is_live": game_state.is_live,
                "label": f"GW{target_gw} Deadline: {game_state.deadline_time or 'TBD'}",
            },
            "data_freshness": {
                "projections_as_of": fpl_client.get_data_as_of("bootstrap-static"),
                "news_as_of": fpl_client.get_data_as_of("news"),
                "rivals_as_of": fpl_client.get_data_as_of("league"),
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
                    "chip": c.get("chip", "HOLD"),
                    "net_gain_vs_hold": c.get("net_gain_vs_hold", c.get("horizon_gain_vs_roll", 0.0)),
                    "reason": c.get("reason", "Projection horizon trajectory."),
                    "recommended": c.get("is_recommended", False),
                }
                for i, c in enumerate(joint_res.get("chip_comparison_table", []))
            ],
            "hit_policy": joint_res.get("hit_policy"),
            "transfers": transfers_summary,
            "xi": starters_list,
            "formation": lineup_res.get("formation", "3-5-2"),
            "captain": captain_dict,
            "vice_captain": vice_dict,
            "bench": bench_list,
            "chip": chip_decision,
            "rivals": rivals_dict,
            "win_prob": win_prob_dict,
            "headline": headline,
            "team_points": round(float(team_pts), 1),
            "two_line_reasoning": two_line,
            "caveats": caveats,
            "caution": caution,
            "your_choices": _your_choices(boot),
            "track_record": track_record_summary(),
            "what_changed": what_changed,
            "press_signals": press_notes,
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
        f"# FPL Oracle - Gameweek {gw}",
        f"**Deadline:** {dl.get('deadline_time') or dl.get('label') or 'Unknown'}",
        "",
        "## Transfers",
    ]
    if t.get("is_roll"):
        lines.append("Save your free transfer.")
    else:
        for out, incoming in zip(t.get("out", []), t.get("in", []), strict=False):
            lines.append(f"- {out['web_name']} -> {incoming['web_name']}")
    lines += [
        f"Hit: {t.get('hit_cost', 0)} pts · Bank: £{t.get('bank_after', 0):.1f}m · Free transfers left: {t.get('ft_remaining', 0)}",
        "",
        "## Captain and bench",
        f"Captain: {cap.get('web_name', 'Unknown')}",
        f"Vice captain: {vice.get('web_name', 'Unknown')}",
        f"Formation: {card.get('formation', 'Unknown')}",
        "",
        "| Player | Position | Predicted points |",
        "|---|---|---|",
    ]
    for player in card.get("xi", []):
        lines.append(
            f"| {player.get('web_name')} | {player.get('position')} | {player.get('expected_points', 0):.1f} |"
        )
    lines += ["", "### Bench"]
    for player in card.get("bench", []):
        lines.append(f"{player.get('bench_order')}. {player.get('web_name')}")
    lines += ["", "## Chips", str(chip.get("chip_display_name")) if chip.get("recommend") else "Save chips"]
    if chip.get("recommend") and chip.get("gain_vs_hold") is not None:
        lines.append(chip.get("comparison") or f"+{chip['gain_vs_hold']} predicted points")
    if chip.get("set_1_deadline_warning"):
        lines.append(chip["set_1_deadline_warning"])
    if riv.get("league_name") and riv.get("status") not in ("unconfigured", "unavailable"):
        lines += ["", "## League", str(riv["league_name"])]
    if wp.get("p_first") is not None:
        lines.append(f"Chance of finishing first: {wp['p_first']:.0f}%")
    if riv.get("summary"):
        lines.append(str(riv["summary"]))
    return "\n".join(lines)


decision_card_generator = DecisionCardGenerator()
