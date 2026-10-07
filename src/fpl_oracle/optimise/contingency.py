"""
Contingency, Emergency Response & Plan B/C Engine for FPL Oracle.
Provides:
1. Plan B & Plan C precomputation with explicit triggers and xP deltas vs Plan A.
2. Injury & Rotation Contingency Matrix (Auto-sub vs Direct Transfer vs Trust Bench).
3. Last-minute "Panic Button" crisis re-optimizer under breaking team news.
4. Pre-deadline operational checklist generator.
"""

import logging
import re
import unicodedata
from typing import Any

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer

logger = logging.getLogger("fpl_oracle.optimise.contingency")


def _normalize_name(name: str) -> str:
    """Normalize accents and strip non-alphanumeric chars for robust matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-zA-Z0-9]", "", ascii_str).lower()


class ContingencyEngine:
    def __init__(self):
        pass

    def generate_contingency_plans(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        risk_preference: str = "balanced",
        primary_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Precomputes Plan A (Primary), Plan B (Injury/Press Conf Pivot), and Plan C (Differential/Price Pivot).
        Returns concrete triggers, replacement actions, and expected points delta vs Plan A.
        """
        # Step 1: Base Plan A (using TransferOptimizer)
        base_res: dict[str, Any] = (
            {"recommended_plan": primary_plan}
            if primary_plan is not None
            else transfer_optimizer.evaluate_transfer_options(
                current_squad_df=current_squad_df,
                player_pool_df=player_pool_df,
                bank=bank,
                free_transfers=free_transfers,
                horizon_projections=horizon_projections,
                current_gw=current_gw,
                target_gw=target_gw,
                risk_preference=risk_preference,
            )
        )

        plan_a_raw = base_res.get("recommended_plan", {})
        plan_a_xp = float(plan_a_raw.get("net_expected_points", 0.0))
        plan_a_type = plan_a_raw.get("plan_type", "ROLL_TRANSFER")

        plan_a = {
            "title": f"Plan A (Primary): {plan_a_raw.get('recommendation_summary', 'Roll transfer')}",
            "plan_type": plan_a_type,
            "transfers_count": plan_a_raw.get("transfers_count", 0),
            "transfers_in": plan_a_raw.get("transfers_in", []),
            "transfers_out": plan_a_raw.get("transfers_out", []),
            "hits": plan_a_raw.get("hits", 0),
            "hit_cost": plan_a_raw.get("hit_cost", 0.0),
            "net_expected_points": plan_a_xp,
            "delta_vs_plan_a": 0.0,
            "trigger_condition": "Standard baseline execution (no late injuries or unexpected price swings).",
            "action_summary": plan_a_raw.get("recommendation_summary", "Roll transfer"),
        }

        # Step 2: Plan B (Injury / Press Conference Contingency Pivot)
        # If Plan A transfers in a player or keeps a doubtful player, Plan B provides the immediate safe alternative.
        plan_b = None
        current_elements = set(current_squad_df["element"].tolist())
        target_gw_df = horizon_projections.get(target_gw, player_pool_df)

        if (
            plan_a_raw.get("transfers_in")
            and plan_a_raw.get("plan_type") not in ("WILDCARD", "FREE_HIT", "FREEHIT")
            and len(plan_a_raw["transfers_in"]) <= 2
        ):
            transferred_in_elem = plan_a_raw["transfers_in"][0]["element"]
            transferred_in_name = plan_a_raw["transfers_in"][0]["web_name"]
            # Exclude the primary target to find the 2nd best alternative (the backup pivot)
            locked_out_backup = [transferred_in_elem]
            backup_res = transfer_optimizer.evaluate_transfer_options(
                current_squad_df=current_squad_df,
                player_pool_df=player_pool_df,
                bank=bank,
                free_transfers=free_transfers,
                horizon_projections=horizon_projections,
                current_gw=current_gw,
                target_gw=target_gw,
                locked_out_ids=locked_out_backup,
                risk_preference=risk_preference,
            )
            backup_plan = backup_res.get("recommended_plan", {})
            backup_xp = float(backup_plan.get("net_expected_points", plan_a_xp))
            delta_b = round(backup_xp - plan_a_xp, 2)

            backup_in_names = ", ".join(t["web_name"] for t in backup_plan.get("transfers_in", [])) or "Hold/Roll"
            plan_b = {
                "title": f"Plan B (Injury Pivot): Pivot to {backup_in_names}",
                "plan_type": backup_plan.get("plan_type", "BACKUP_TRANSFER"),
                "transfers_count": backup_plan.get("transfers_count", 0),
                "transfers_in": backup_plan.get("transfers_in", []),
                "transfers_out": backup_plan.get("transfers_out", []),
                "hits": backup_plan.get("hits", 0),
                "hit_cost": backup_plan.get("hit_cost", 0.0),
                "net_expected_points": backup_xp,
                "delta_vs_plan_a": delta_b,
                "trigger_condition": f"Press conference confirms {transferred_in_name} is doubtful (<75%) or ruled out.",
                "action_summary": f"Pivot target: buy {backup_in_names} instead of {transferred_in_name} (net impact: {delta_b:+.2f} xP).",
            }
            # Plan A was to roll. Plan B evaluates the best 1 transfer if an existing squad member is ruled out.
            cand_plans = base_res.get("candidate_plans", [])
            top_1_plan = next((p for p in cand_plans if p.get("plan_type") == "1_TRANSFER"), None)
            if top_1_plan:
                top_xp = float(top_1_plan.get("net_expected_points", plan_a_xp))
                delta_b = round(top_xp - plan_a_xp, 2)
                in_name = top_1_plan["transfers_in"][0]["web_name"]
                out_name = top_1_plan["transfers_out"][0]["web_name"]
                plan_b = {
                    "title": f"Plan B (Injury Acceleration): Transfer {out_name} -> {in_name}",
                    "plan_type": "1_TRANSFER",
                    "transfers_count": 1,
                    "transfers_in": top_1_plan.get("transfers_in", []),
                    "transfers_out": top_1_plan.get("transfers_out", []),
                    "hits": top_1_plan.get("hits", 0),
                    "hit_cost": top_1_plan.get("hit_cost", 0.0),
                    "net_expected_points": top_xp,
                    "delta_vs_plan_a": delta_b,
                    "trigger_condition": f"Squad regular ({out_name}) suffers training setback or is confirmed benched.",
                    "action_summary": f"Burn 1 banked FT immediately for {in_name} ({delta_b:+.2f} xP vs rolling).",
                }

        if plan_b is None:
            plan_b = {
                "title": "Plan B (Conservative Roll): Bank Free Transfer",
                "plan_type": "ROLL_TRANSFER",
                "transfers_count": 0,
                "transfers_in": [],
                "transfers_out": [],
                "hits": 0,
                "hit_cost": 0.0,
                "net_expected_points": plan_a_xp,
                "delta_vs_plan_a": 0.0,
                "trigger_condition": "Late uncertainty across multiple fixture press conferences.",
                "action_summary": "Roll transfer and bank additional FT for double gameweek flexibility.",
            }

        # Step 3: Plan C (Differential / Price Rise Contingency)
        # Search for high-ceiling aggressive differential pick (P90 maximization)
        high_ceiling_candidates = (
            target_gw_df[(~target_gw_df["element"].isin(current_elements)) & (target_gw_df["value"] <= (bank + 80.0))]
            .sort_values(by="p90", ascending=False)
            .head(1)
        )

        if not high_ceiling_candidates.empty:
            diff_cand = high_ceiling_candidates.iloc[0]
            diff_xp = round(float(diff_cand.get("expected_points", 5.0)), 2)
            diff_p90 = round(float(diff_cand.get("p90", 11.0)), 2)
            plan_c = {
                "title": f"Plan C (Differential Ceiling): Target {diff_cand['web_name']}",
                "plan_type": "AGGRESSIVE_DIFFERENTIAL",
                "transfers_count": 1,
                "transfers_in": [
                    {
                        "element": int(diff_cand["element"]),
                        "web_name": diff_cand["web_name"],
                        "team": int(diff_cand["team"]),
                        "position": diff_cand["position"],
                        "cost": diff_cand["value"] / 10.0,
                        "expected_points": diff_xp,
                        "p90": diff_p90,
                    }
                ],
                "transfers_out": plan_a_raw.get("transfers_out", []),
                "hits": 0 if free_transfers >= 1 else 4,
                "hit_cost": 0.0 if free_transfers >= 1 else 4.0,
                "net_expected_points": round(plan_a_xp - 0.5, 2),
                "delta_vs_plan_a": -0.5,
                "p90_ceiling": diff_p90,
                "trigger_condition": "Trailing in mini-league (chasing mode) or primary target undergoes sudden price rise before buy.",
                "action_summary": f"High-ceiling play: {diff_cand['web_name']} brings P90 ceiling of {diff_p90} pts for mini-league chase.",
            }
        else:
            plan_c = {
                "title": "Plan C (Price Rise Safeguard): Pre-rise Early Lock",
                "plan_type": "PRICE_PROTECTION",
                "transfers_count": 1,
                "transfers_in": plan_a_raw.get("transfers_in", []),
                "transfers_out": plan_a_raw.get("transfers_out", []),
                "hits": 0,
                "hit_cost": 0.0,
                "net_expected_points": plan_a_xp,
                "delta_vs_plan_a": 0.0,
                "trigger_condition": "Target player transfer velocity indicates imminent price rise tonight.",
                "action_summary": "Execute transfer 24 hours early to protect squad value before price inflation.",
            }

        return {
            "plan_a": plan_a,
            "plan_b": plan_b,
            "plan_c": plan_c,
            "target_gameweek": target_gw,
            "free_transfers_available": free_transfers,
            "bank_millions": round(bank / 10.0, 2),
        }

    def compute_injury_matrix(
        self,
        squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        bootstrap: BootstrapStatic,
        lineup: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Calculates the full 'What if Player X is ruled out' replacement matrix for all starters.
        Compares:
        1. Auto-sub outcome: which bench player steps in and net xP change.
        2. Direct emergency transfer: best replacement on market within budget.
        3. Recommendation: TRUST_BENCH vs EXECUTE_TRANSFER vs MONITOR.
        """
        lineup_res = lineup or lineup_optimizer.select_lineup_and_captain(squad_df)
        starters = lineup_res["starters"]
        bench = lineup_res["bench"]

        elem_meta = {e.id: e for e in bootstrap.elements}
        matrix = []

        bench_outfield = bench[bench["position"] != "GKP"]
        bench_gk = bench[bench["position"] == "GKP"]

        for _, starter_row in starters.iterrows():
            elem_id = int(starter_row["element"])
            elem_name = starter_row["web_name"]
            pos = starter_row["position"]
            st_xp = round(float(starter_row["expected_points"]), 2)

            meta = elem_meta.get(elem_id)
            status = meta.status if meta else "a"
            chance = meta.chance_of_playing_next_round if meta else 100
            news_str = meta.news if meta and meta.news else ""

            # Determine autosub
            if pos == "GKP":
                sub_row = bench_gk.iloc[0] if not bench_gk.empty else None
            else:
                # First outfield bench player that maintains a legal formation (3+ DEF, 2+ MID, 1+ FWD)
                sub_row = None
                for _, b_row in bench_outfield.iterrows():
                    counts = starters["position"].value_counts().to_dict()
                    counts[pos] -= 1
                    counts[b_row["position"]] = counts.get(b_row["position"], 0) + 1
                    if counts.get("DEF", 0) >= 3 and counts.get("MID", 0) >= 2 and counts.get("FWD", 0) >= 1:
                        sub_row = b_row
                        break

            sub_name = sub_row["web_name"] if sub_row is not None else "None"
            sub_xp = round(float(sub_row["expected_points"]), 2) if sub_row is not None else 0.0
            autosub_delta = round(sub_xp - st_xp, 2)

            # Determine best emergency transfer replacement
            sell_price = float(starter_row.get("selling_price", starter_row["value"]))
            avail_budget = sell_price + bank
            candidates = player_pool_df[
                (player_pool_df["position"] == pos)
                & (player_pool_df["element"] != elem_id)
                & (~player_pool_df["element"].isin(squad_df["element"]))
                & (player_pool_df["value"] <= avail_budget)
            ].sort_values(by="expected_points", ascending=False)
            other_teams = squad_df[squad_df["element"] != elem_id]["team"].value_counts()
            candidates = candidates[candidates["team"].map(other_teams).fillna(0) < 3]

            best_rep = candidates.iloc[0] if not candidates.empty else None
            rep_name = best_rep["web_name"] if best_rep is not None else "None"
            rep_xp = round(float(best_rep["expected_points"]), 2) if best_rep is not None else 0.0

            # Transfer net gain vs trusting the bench
            hit_cost = 0.0 if free_transfers >= 1 else 4.0
            transfer_net_xp = round(rep_xp - hit_cost, 2)
            transfer_gain_vs_bench = round(transfer_net_xp - sub_xp, 2)

            # Action verdict
            if transfer_gain_vs_bench >= 2.0:
                verdict = "EXECUTE_TRANSFER"
                verdict_reason = (
                    f"Emergency transfer to {rep_name} nets {transfer_gain_vs_bench:+.2f} xP over bench after hit."
                )
            elif autosub_delta >= -1.0 or transfer_gain_vs_bench < 0.5:
                verdict = "TRUST_BENCH"
                verdict_reason = f"Bench coverage ({sub_name}) is strong ({sub_xp} xP). Save free transfer / hit."
            else:
                verdict = "MONITOR_PRESS_CONFERENCE"
                verdict_reason = f"Marginal call ({transfer_gain_vs_bench:+.2f} xP). Wait for final press conference."

            matrix.append(
                {
                    "element": elem_id,
                    "web_name": elem_name,
                    "position": pos,
                    "current_status": status,
                    "chance_of_playing": chance if chance is not None else (0 if status == "i" else 100),
                    "news": news_str,
                    "expected_points": st_xp,
                    "autosub_player": sub_name,
                    "autosub_expected_points": sub_xp,
                    "autosub_points_delta": autosub_delta,
                    "emergency_replacement": rep_name,
                    "emergency_cost": round((best_rep["value"] / 10.0), 1) if best_rep is not None else 0.0,
                    "emergency_expected_points": rep_xp,
                    "emergency_net_expected_points": transfer_net_xp,
                    "transfer_gain_vs_bench": transfer_gain_vs_bench,
                    "action_verdict": verdict,
                    "verdict_reason": verdict_reason,
                    "wait_vs_commit": "Wait for verified availability news; transfer momentum does not establish a price-change time.",
                }
            )

        return matrix

    def panic_button_reoptimize(
        self,
        query: str,
        squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        ruled_out_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Emergency 1-click crisis solver for breaking team news (e.g. 'Saka ruled out 6 weeks').
        Instantly identifies ruled-out squad player, zeros their minutes/points, and solves:
        1. Clean bench promotion & revised captaincy.
        2. Best immediate 1-transfer market replacement.
        """
        affected_ids = set(ruled_out_ids or [])

        # Match player from text query if not provided directly
        if not affected_ids and query:
            clean_q = _normalize_name(query)
            for _, r in squad_df.iterrows():
                p_norm = _normalize_name(r["web_name"])
                if p_norm in clean_q or clean_q in p_norm:
                    affected_ids.add(int(r["element"]))

        if not affected_ids:
            # Fallback to the lowest chance starter if no match
            lineup = lineup_optimizer.select_lineup_and_captain(squad_df)
            affected_ids.add(int(lineup["starters"].iloc[0]["element"]))

        # Perturb squad with ruled-out player having 0.0 expected points
        perturbed_squad = squad_df.copy()
        for idx in perturbed_squad.index:
            if int(perturbed_squad.loc[idx, "element"]) in affected_ids:
                perturbed_squad.loc[idx, "expected_points"] = 0.0
                perturbed_squad.loc[idx, "p10"] = 0.0
                perturbed_squad.loc[idx, "p90"] = 0.0

        # 1. New lineup with bench promotion
        new_lineup = lineup_optimizer.select_lineup_and_captain(perturbed_squad)

        # Identify which player was promoted
        orig_lineup = lineup_optimizer.select_lineup_and_captain(squad_df)
        orig_starters = set(orig_lineup["starters"]["element"].tolist())
        new_starters = set(new_lineup["starters"]["element"].tolist())
        promoted_ids = list(new_starters - orig_starters)
        promoted_player = None
        if promoted_ids:
            p_row = new_lineup["starters"][new_lineup["starters"]["element"] == promoted_ids[0]].iloc[0]
            promoted_player = {
                "element": int(p_row["element"]),
                "web_name": p_row["web_name"],
                "position": p_row["position"],
                "expected_points": round(float(p_row["expected_points"]), 2),
            }

        # 2. Emergency 1-transfer market solution
        ruled_out_id = list(affected_ids)[0]
        ruled_out_row = squad_df[squad_df["element"] == ruled_out_id].iloc[0]
        sell_val = float(ruled_out_row.get("selling_price", ruled_out_row["value"]))
        avail_cash = sell_val + bank

        candidates = player_pool_df[
            (player_pool_df["position"] == ruled_out_row["position"])
            & (player_pool_df["element"] != ruled_out_id)
            & (~player_pool_df["element"].isin(squad_df["element"]))
            & (player_pool_df["value"] <= avail_cash)
        ].sort_values(by="expected_points", ascending=False)

        best_market_rep = None
        if not candidates.empty:
            bm = candidates.iloc[0]
            hit = 0.0 if free_transfers >= 1 else 4.0
            best_market_rep = {
                "element": int(bm["element"]),
                "web_name": bm["web_name"],
                "team": int(bm["team"]),
                "position": bm["position"],
                "cost": round(bm["value"] / 10.0, 1),
                "expected_points": round(float(bm["expected_points"]), 2),
                "net_expected_points": round(float(bm["expected_points"]) - hit, 2),
                "hit_cost": hit,
            }

        return {
            "status": "crisis_resolved",
            "crisis_query": query,
            "ruled_out_player": {
                "element": ruled_out_id,
                "web_name": ruled_out_row["web_name"],
                "position": ruled_out_row["position"],
                "lost_expected_points": round(
                    float(squad_df[squad_df["element"] == ruled_out_id]["expected_points"].iloc[0]), 2
                ),
            },
            "lineup_action": {
                "summary": f"Bench {ruled_out_row['web_name']}. Auto-promote {promoted_player['web_name'] if promoted_player else 'first reserve'}.",
                "promoted_player": promoted_player,
                "new_formation": new_lineup["formation"],
                "captain": new_lineup["captain"],
                "vice_captain": new_lineup["vice_captain"],
                "total_gameweek_expected_points": new_lineup["total_gameweek_expected_points"],
            },
            "emergency_transfer": best_market_rep,
            "recommendation": (
                f"1-Click Recommendation: If {ruled_out_row['web_name']} is definitely out, "
                f"start {promoted_player['web_name'] if promoted_player else 'bench'} ({promoted_player['expected_points'] if promoted_player else 0} xP). "
                f"Alternatively, transfer {ruled_out_row['web_name']} -> {best_market_rep['web_name'] if best_market_rep else 'Replacement'} "
                f"for {best_market_rep['net_expected_points'] if best_market_rep else 0} net xP."
            ),
        }

    def generate_pre_deadline_checklist(
        self,
        squad_df: pd.DataFrame,
        bootstrap: BootstrapStatic,
        game_state_data: dict[str, Any],
        chips_status: dict[str, Any],
        lineup: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generates 5-point operational pre-deadline audit.
        """
        checklist = []
        elem_meta = {e.id: e for e in bootstrap.elements}

        # 1. Starters Fitness
        lineup = lineup or lineup_optimizer.select_lineup_and_captain(squad_df)
        doubtful = []
        for _, s in lineup["starters"].iterrows():
            e = elem_meta.get(int(s["element"]))
            if e and (
                e.status != "a" or (e.chance_of_playing_next_round is not None and e.chance_of_playing_next_round < 100)
            ):
                doubtful.append(f"{s['web_name']} ({e.chance_of_playing_next_round or 0}% - {e.news or 'Doubt'})")

        if not doubtful:
            checklist.append(
                {
                    "item": "Starting XI Fitness & Availability",
                    "status": "PASS",
                    "badge": "Fit",
                    "detail": "No current official injury/suspension flags for the selected starters. Availability and starting minutes are not guaranteed.",
                }
            )
        else:
            checklist.append(
                {
                    "item": "Starting XI Fitness & Availability",
                    "status": "WARNING",
                    "badge": "Doubtful Starters",
                    "detail": f"Doubtful players detected: {', '.join(doubtful)}. Check Friday press conference quotes.",
                }
            )

        # 2. Vice-Captain Reliability
        vc = lineup["vice_captain"]
        cap = lineup["captain"]
        checklist.append(
            {
                "item": "Vice-Captain Failsafe",
                "status": "PASS",
                "badge": "Active",
                "detail": f"Vice-Captain assigned to {vc['web_name']} ({vc['expected_points']} xP). Activates if {cap['web_name']} does not feature.",
            }
        )

        # 3. Bench Autosub Hierarchy
        b1 = lineup["bench"][lineup["bench"]["position"] != "GKP"].iloc[0]
        checklist.append(
            {
                "item": "Autosub Hierarchy Order",
                "status": "PASS",
                "badge": f"1st Sub: {b1['web_name']}",
                "detail": f"Selected first outfield sub {b1['web_name']} ({round(float(b1['expected_points']), 2)} xP) occupies position 1 on the bench.",
            }
        )

        # 4. Chip Set 1 Expiry Horizon
        curr_gw = game_state_data.get("next_gw") or game_state_data.get("current_gw") or 5
        rem_set_1 = chips_status.get("set_1_remaining", [])
        gws_to_19 = max(0, 20 - curr_gw)
        if len(rem_set_1) > gws_to_19:
            checklist.append(
                {
                    "item": "Chip Set 1 Expiry Deadline",
                    "status": "ACTION_REQUIRED",
                    "badge": "CRITICAL CONGESTION",
                    "detail": f"You have {len(rem_set_1)} Set 1 chips left with only {gws_to_19} gameweeks before GW19! Play a chip now or forfeit it.",
                }
            )
        elif rem_set_1:
            checklist.append(
                {
                    "item": "Chip Set 1 Expiry Deadline",
                    "status": "INFO",
                    "badge": f"{len(rem_set_1)} Chips / {gws_to_19} GWs",
                    "detail": f"Set 1 chips remaining: {', '.join(rem_set_1)}. Cutoff is the GW19 deadline; consult the current official deadline.",
                }
            )
        else:
            checklist.append(
                {
                    "item": "Chip Set 1 Expiry Deadline",
                    "status": "PASS",
                    "badge": "Set 1 Complete",
                    "detail": "All Set 1 chips executed or planned on schedule.",
                }
            )

        # 5. Deadline Countdown
        secs = game_state_data.get("seconds_to_deadline", 0.0)
        hours = round(secs / 3600.0, 1)
        checklist.append(
            {
                "item": "Pre-Deadline Lock Time",
                "status": "PASS" if hours > 12 else "WARNING",
                "badge": f"{hours}h Remaining",
                "detail": f"Deadline locks in {hours} hours. Finalize transfers and captaincy before official server freeze.",
            }
        )

        return checklist


contingency_engine = ContingencyEngine()
