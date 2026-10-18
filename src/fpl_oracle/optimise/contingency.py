"""
Contingency, Emergency Response & Plan B/C Engine for FPL Oracle.
Provides:
1. Plan B & Plan C precomputation with explicit triggers and xP deltas vs Plan A.
2. Injury & Rotation Contingency Matrix (Auto-sub vs Direct Transfer vs Trust Bench).
3. Last-minute "Panic Button" crisis re-optimizer under breaking team news.
4. Pre-deadline operational checklist generator.
"""

import logging
from typing import Any

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.optimise.lineup import lineup_optimizer, play_probability
from fpl_oracle.optimise.transfers import transfer_optimizer

logger = logging.getLogger("fpl_oracle.optimise.contingency")


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
        compute_alternatives: bool = True,
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

        if not compute_alternatives:
            pending = dict(
                title="Conditional alternative not calculated",
                plan_type="UNAVAILABLE",
                transfers_count=0,
                transfers_in=[],
                transfers_out=[],
                hits=0,
                hit_cost=0,
                net_expected_points=None,
                delta_vs_plan_a=None,
                trigger_condition="Run the detailed contingency analysis if conditions change.",
                action_summary="No alternative recommendation published. Primary plan remains below.",
            )
            return dict(
                plan_a=plan_a,
                plan_b=dict(pending),
                plan_c=dict(pending),
                target_gameweek=target_gw,
                free_transfers_available=free_transfers,
                bank_millions=bank / 10,
            )

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

        if plan_b is None:
            owned_line = lineup_optimizer.select_lineup_and_captain(current_squad_df, risk_preference=risk_preference)
            owned_xp = float(owned_line["total_gameweek_expected_points"])
            plan_b = dict(
                title="Plan B: Hold owned squad while evidence is unresolved",
                plan_type="ROLL_TRANSFER",
                transfers_count=0,
                transfers_in=[],
                transfers_out=[],
                hits=0,
                hit_cost=0,
                net_expected_points=round(owned_xp, 2),
                delta_vs_plan_a=round(owned_xp - plan_a_xp, 2),
                trigger_condition="Primary restructure or target cannot be confirmed before deadline.",
                action_summary="No transfer or chip. Reassess current owned XI on verified news; no invented alternate gain.",
            )

        # Plan C is a legal full-squad single-transfer alternative, not a cheap
        # high-P90 player stapled onto unrelated Plan A transfer-outs.
        pmap = {int(r["element"]): r for r in target_gw_df.to_dict("records")}
        purchase = {
            int(r["element"]): int(r.get("purchase_price", r["value"])) for r in current_squad_df.to_dict("records")
        }
        moves = transfer_optimizer._get_candidate_1_transfers(
            current_elements, int(bank), purchase, target_gw_df, pmap, set(), set(), set(), max_per_pos=3
        )
        choices = []
        for move in moves:
            ids = (current_elements - set(move["transfers_out"])) | set(move["transfers_in"])
            frame = pd.DataFrame([pmap[e] for e in sorted(ids)])
            line = lineup_optimizer.select_lineup_and_captain(
                frame, risk_preference="aggressive", simulate_captaincy=False
            )
            hits = max(0, len(move["transfers_in"]) - free_transfers)
            net = float(line["total_gameweek_expected_points"]) - 4 * hits
            # P90 of the lineup total: independent normals, sigma from each player's p10-p90 spread.
            spread = (line["starters"].get("p90", line["starters"]["expected_points"] * 1.8)) - (
                line["starters"].get("p10", line["starters"]["expected_points"] * 0.4)
            )
            var = float(((spread / 2.563) ** 2).sum())
            cap = line["captain"]
            var += ((float(cap.get("p90", 0.0)) - float(cap.get("p10", 0.0))) / 2.563) ** 2 * (
                cap["multiplier"] ** 2 - 1
            )
            ceiling = net + 1.2816 * var**0.5
            choices.append((ceiling, net, move))
        if choices:
            ceiling, net, move = max(choices, key=lambda choice: (choice[0], choice[1]))
            ins = [
                dict(
                    element=e,
                    web_name=pmap[e]["web_name"],
                    position=pmap[e]["position"],
                    team=pmap[e]["team"],
                    cost=pmap[e]["value"] / 10,
                    expected_points=pmap[e]["expected_points"],
                )
                for e in move["transfers_in"]
            ]
            outs = [
                dict(
                    element=e,
                    web_name=pmap[e]["web_name"],
                    position=pmap[e]["position"],
                    team=pmap[e]["team"],
                    sell_price=move["sell_prices"][e] / 10,
                )
                for e in move["transfers_out"]
            ]
            hits = max(0, len(ins) - free_transfers)
            plan_c = dict(
                title="Plan C: Legal higher-ceiling single-transfer alternative",
                plan_type="1_TRANSFER",
                transfers_count=len(ins),
                transfers_in=ins,
                transfers_out=outs,
                hits=hits,
                hit_cost=hits * 4,
                net_expected_points=round(net, 2),
                delta_vs_plan_a=round(net - plan_a_xp, 2),
                p90_ceiling=round(ceiling, 2),
                remaining_bank=round((bank + move["bank_delta"]) / 10, 2),
                trigger_condition="User chooses higher-ceiling exposure after reviewing the points/hit trade-off.",
                action_summary="Model ceiling comparison only; not an imminent-price claim or automatic instruction.",
            )
        else:
            plan_c = dict(plan_b, title="Plan C: No legal differential candidate found")

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
        3. One-week comparison, not an instruction to transfer.
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
            sub_xp = (
                round(float(sub_row["expected_points"]) * float(play_probability(sub_row.to_frame().T).iloc[0]), 2)
                if sub_row is not None
                else 0.0
            )
            autosub_delta = round(sub_xp - st_xp, 2)

            # Determine best emergency transfer replacement
            sell_price = float(starter_row.get("selling_price", starter_row["value"]))
            avail_budget = sell_price + bank
            candidates = player_pool_df[
                (player_pool_df["position"] == pos)
                & (player_pool_df["element"] != elem_id)
                & (~player_pool_df["element"].isin(squad_df["element"]))
                & (player_pool_df["value"] <= avail_budget)
            ]
            other_teams = squad_df[squad_df["element"] != elem_id]["team"].value_counts()
            candidates = candidates[candidates["team"].map(other_teams).fillna(0) < 3]
            # Only fit-enough replacements, ranked by availability-adjusted xP.
            candidates = candidates.assign(_avail_xp=candidates["expected_points"] * play_probability(candidates))
            candidates = candidates[play_probability(candidates) >= 0.75].sort_values("_avail_xp", ascending=False)

            best_rep = candidates.iloc[0] if not candidates.empty else None
            rep_name = best_rep["web_name"] if best_rep is not None else "None"
            rep_xp = round(float(best_rep["_avail_xp"]), 2) if best_rep is not None else 0.0

            # Transfer net gain vs trusting the bench
            hit_cost = 0.0 if free_transfers >= 1 else 4.0
            transfer_net_xp = round(rep_xp - hit_cost, 2)
            transfer_gain_vs_bench = round(transfer_net_xp - sub_xp, 2)

            # Action verdict
            if transfer_gain_vs_bench >= 2.0:
                verdict = "CHECK_TRANSFER"
                verdict_reason = (
                    f"{rep_name}: {transfer_gain_vs_bench:+.2f} est. points vs bench. Run the emergency check."
                )
            elif autosub_delta >= -1.0 or transfer_gain_vs_bench < 0.5:
                verdict = "TRUST_BENCH"
                verdict_reason = (
                    f"Bench coverage ({sub_name}) is strong ({sub_xp} predicted pts). Save free transfer / hit."
                )
            else:
                verdict = "MONITOR_PRESS_CONFERENCE"
                verdict_reason = (
                    f"Marginal call ({transfer_gain_vs_bench:+.2f} predicted pts). Wait for final press conference."
                )

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
                    "detail": "No injury flags.",
                }
            )
        else:
            checklist.append(
                {
                    "item": "Starting XI Fitness & Availability",
                    "status": "WARNING",
                    "badge": "Doubtful Starters",
                    "detail": f"Check {', '.join(doubtful)}.",
                }
            )

        # 2. Vice-Captain Reliability
        vc = lineup["vice_captain"]
        cap = lineup["captain"]
        checklist.append(
            {
                "item": "Vice-Captain Failsafe",
                "status": "PASS"
                if vc["element"] != cap["element"]
                and elem_meta.get(int(vc["element"]))
                and elem_meta[int(vc["element"])].status == "a"
                else "WARNING",
                "badge": "Assigned",
                "detail": f"{vc['web_name']}",
            }
        )

        # 3. Bench Autosub Hierarchy
        b1 = lineup["bench"][lineup["bench"]["position"] != "GKP"].iloc[0]
        checklist.append(
            {
                "item": "Autosub Hierarchy Order",
                "status": "PASS"
                if elem_meta.get(int(b1["element"])) and elem_meta[int(b1["element"])].status == "a"
                else "WARNING",
                "badge": f"1st Sub: {b1['web_name']}",
                "detail": f"{b1['web_name']}",
            }
        )

        # 4. Chip set expiry horizon (set 1 expires after GW19, set 2 after GW38)
        curr_gw = game_state_data.get("next_gw") or game_state_data.get("current_gw") or 5
        set_no, last_gw = (1, 19) if curr_gw <= 19 else (2, 38)
        rem_chips = chips_status.get(f"set_{set_no}_remaining", [])
        gws_left = last_gw + 1 - curr_gw
        item = f"Chip Set {set_no} Expiry Deadline"
        if len(rem_chips) > max(0, gws_left):
            checklist.append(
                {
                    "item": item,
                    "status": "ACTION_REQUIRED",
                    "badge": "CRITICAL CONGESTION",
                    "detail": f"You have {len(rem_chips)} Set {set_no} chips left with only {max(0, gws_left)} gameweeks up to GW{last_gw}! Play a chip now or forfeit it.",
                }
            )
        elif rem_chips:
            checklist.append(
                {
                    "item": item,
                    "status": "INFO",
                    "badge": f"{len(rem_chips)} Chips / {gws_left} GWs",
                    "detail": f"{len(rem_chips)} chips expire at GW{last_gw}.",
                }
            )
        else:
            checklist.append(
                {
                    "item": item,
                    "status": "PASS",
                    "badge": f"Set {set_no} Complete",
                    "detail": f"No set {set_no} chips left.",
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
                "detail": f"{hours} hours left.",
            }
        )

        return checklist


contingency_engine = ContingencyEngine()
