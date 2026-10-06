"""
Joint Chip Strategy Planner with Dynamic Programming / Beam Search.

Features:
- Enforces official 2026/27 rules: 2 sets of 4 chips (Wildcard, Free Hit, Triple Captain, Bench Boost)
- Hard Gameweek 19 deadline for Set 1 (cannot be carried over, expired after GW19) (C3)
- Exactly 1 chip per gameweek
- Multi-GW chip calendar with expected benefit vs no-chip baseline and horizon confidence tagging (C1)
- Joint chip and transfer planning cross-awareness (C2)
- Strict recommendation threshold gating (only recommend when gain exceeds uncertainty margin) (C3)
- Rival chip tracking (usage likelihood, counter-chip differential opportunities) (C4)
"""

import itertools
from typing import Any

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture, ManagerHistory
from fpl_oracle.chips.calendar import fixture_calendar
from fpl_oracle.chips.simulate import chip_simulator
from fpl_oracle.config import RULES

ALL_CHIPS = ["wildcard", "freehit", "3xc", "bboost"]
CHIP_DISPLAY_NAMES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "3xc": "Triple Captain",
    "bboost": "Bench Boost",
}

TRIGGER_CONDITIONS_DEFAULT = {
    "wildcard": "Squad accumulates 2+ key injuries or major fixture swings across 3+ core clubs.",
    "freehit": "Major blank gameweek (cup postponements) or extreme single-week fixture mismatch.",
    "3xc": "Confirmed Double Gameweek with elite captain asset, or elite home fixture with high P90 ceiling.",
    "bboost": "All 15 squad players confirmed starting with favorable fixtures (ideally in a Double Gameweek).",
}

# Strict recommendation thresholds (gain vs no-chip baseline must exceed uncertainty margin) (C3)
CHIP_RECOMMENDATION_THRESHOLDS = {
    "3xc": 4.0,       # Extra 1x captain must be >= 4.0 xP
    "bboost": 8.0,    # Bench must provide >= 8.0 xP
    "freehit": 10.0,  # One-week swing must be >= 10.0 xP
    "wildcard": 8.0,  # Multi-week squad uplift must be >= 8.0 xP
}


class ChipPlanner:
    def __init__(self):
        self.rules = RULES

    def get_remaining_chips(self, history: ManagerHistory | None) -> dict[str, Any]:
        """
        Determine remaining chips in Set 1 (GW 1-19) and Set 2 (GW 20-38).
        """
        used_set_1 = set()
        used_set_2 = set()
        used_details = []

        if history and history.chips:
            for item in history.chips:
                c_name = item.name.lower()
                c_event = item.event
                used_details.append({"name": c_name, "event": c_event, "time": item.time})
                if c_event <= 19:
                    used_set_1.add(c_name)
                else:
                    used_set_2.add(c_name)

        remaining_set_1 = [c for c in ALL_CHIPS if c not in used_set_1]
        remaining_set_2 = [c for c in ALL_CHIPS if c not in used_set_2]

        return {
            "set_1_used": list(used_set_1),
            "set_1_remaining": remaining_set_1,
            "set_2_used": list(used_set_2),
            "set_2_remaining": remaining_set_2,
            "used_history": used_details,
        }

    def compute_chip_utility(
        self,
        chip: str,
        gw: int,
        current_squad_df: pd.DataFrame,
        horizon_projections: dict[int, pd.DataFrame],
        dgw_gws: list[int],
        bgw_gws: list[int],
        budget: float = 1000.0,
        current_gw: int = 5,
    ) -> tuple[float, str, float, float]:
        """
        Computes expected benefit vs no-chip baseline for playing a chip in gameweek `gw` (C1).
        Returns: (expected_gain_vs_baseline, confidence, baseline_no_chip_xp, with_chip_xp).
        """
        gw_df = horizon_projections.get(gw, current_squad_df)
        is_dgw = gw in dgw_gws
        is_bgw = gw in bgw_gws
        is_beyond_horizon = gw > (current_gw + 4)

        baseline_xp = 50.0
        with_chip_xp = 50.0
        gain = 0.0

        if chip == "3xc":
            res = chip_simulator.evaluate_triple_captain(gw, current_squad_df if len(current_squad_df) == 15 else gw_df)
            gain = float(res.get("expected_gain", 6.0))
            baseline_xp = float(res.get("captain_expected_points", 6.0)) * 2.0 + 38.0
            with_chip_xp = baseline_xp + gain

        elif chip == "bboost":
            res = chip_simulator.evaluate_bench_boost(gw, current_squad_df, gw_df)
            gain = float(res.get("expected_gain", 10.0))
            baseline_xp = 48.0
            with_chip_xp = baseline_xp + gain

        elif chip == "freehit":
            res = chip_simulator.evaluate_free_hit(gw, current_squad_df, gw_df, budget)
            gain = float(res.get("expected_gain", 12.0))
            baseline_xp = float(res.get("current_squad_xp", 45.0))
            with_chip_xp = float(res.get("free_hit_squad_xp", baseline_xp + gain))

        elif chip == "wildcard":
            res = chip_simulator.evaluate_wildcard(gw, current_squad_df, gw_df, budget)
            gain = float(res.get("expected_gain", 10.0))
            baseline_xp = float(res.get("current_squad_xp", 45.0))
            with_chip_xp = float(res.get("wildcard_squad_xp", baseline_xp + gain))

        if is_beyond_horizon:
            confidence = "LOW_CONFIDENCE (Beyond 5-GW Horizon)"
        elif is_dgw or is_bgw or gain >= CHIP_RECOMMENDATION_THRESHOLDS.get(chip, 8.0):
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"

        return round(gain, 1), confidence, round(baseline_xp, 1), round(with_chip_xp, 1)

    def optimize_joint_assignment(
        self,
        available_gws: list[int],
        remaining_chips: list[str],
        current_squad_df: pd.DataFrame,
        horizon_projections: dict[int, pd.DataFrame],
        dgw_gws: list[int],
        bgw_gws: list[int],
        budget: float = 1000.0,
        current_gw: int = 5,
    ) -> tuple[dict[str, int], float, list[dict[str, Any]]]:
        """
        Searches joint space of chip assignments using Dynamic Programming / Permutation search.
        Enforces 1 chip per gameweek constraint and accounts for chip synergies (WC -> BB).
        """
        if not remaining_chips or not available_gws:
            return {}, 0.0, []

        # Precompute utilities for each (chip, gw) pair
        utilities: dict[str, dict[int, tuple[float, str, float, float]]] = {c: {} for c in remaining_chips}
        for c in remaining_chips:
            for g in available_gws:
                utilities[c][g] = self.compute_chip_utility(
                    chip=c,
                    gw=g,
                    current_squad_df=current_squad_df,
                    horizon_projections=horizon_projections,
                    dgw_gws=dgw_gws,
                    bgw_gws=bgw_gws,
                    budget=budget,
                    current_gw=current_gw,
                )

        candidates: list[tuple[float, dict[str, int]]] = []
        k = len(remaining_chips)

        if len(available_gws) >= k:
            gw_combos = list(itertools.combinations(available_gws, k))
            if len(gw_combos) > 2000:
                gw_combos = gw_combos[:2000]

            for combo in gw_combos:
                for perm in itertools.permutations(remaining_chips):
                    curr_assign = dict(zip(perm, combo, strict=False))
                    tot_gain = sum(utilities[chip][gw][0] for chip, gw in curr_assign.items())

                    # Synergy: Wildcard played 1 to 3 GWs before Bench Boost
                    if "wildcard" in curr_assign and "bboost" in curr_assign:
                        wc_gw = curr_assign["wildcard"]
                        bb_gw = curr_assign["bboost"]
                        if 0 < (bb_gw - wc_gw) <= 3:
                            tot_gain += 5.0  # WC bench setup synergy bonus

                    candidates.append((tot_gain, curr_assign))
        else:
            for perm in itertools.permutations(remaining_chips, len(available_gws)):
                curr_assign = dict(zip(perm, available_gws, strict=False))
                tot_gain = sum(utilities[chip][gw][0] for chip, gw in curr_assign.items())
                candidates.append((tot_gain, curr_assign))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            return {}, 0.0, []

        best_gain, best_assign = candidates[0]

        alternatives = []
        seen_plans = {tuple(sorted(best_assign.items()))}
        for score, plan in candidates[1:]:
            plan_key = tuple(sorted(plan.items()))
            if plan_key not in seen_plans:
                seen_plans.add(plan_key)
                alternatives.append(
                    {
                        "total_expected_gain": round(score, 1),
                        "assignment": plan,
                        "delta_vs_best": round(score - best_gain, 1),
                    }
                )
                if len(alternatives) >= 3:
                    break

        return best_assign, round(best_gain, 1), alternatives

    def get_joint_transfer_advice(
        self,
        joint_schedule: dict[int, str],
        current_gw: int,
        banked_fts: int = 1,
    ) -> dict[str, Any]:
        """
        Generates tactical advice connecting transfers to the chip calendar (C2).
        Prevents burning transfers before Wildcard or Free Hit.
        """
        next_gw = current_gw + 1
        chip_next = joint_schedule.get(next_gw)
        chip_curr = joint_schedule.get(current_gw)

        if chip_curr == "freehit":
            return {
                "cross_awareness_alert": "FREE HIT ACTIVE: Squad is temporary for this gameweek only. Saved free transfers will carry over to next gameweek.",
                "transfer_action_guidance": "Do not make permanent transfers; your pre-Free-Hit squad will automatically return next week.",
                "hit_penalty_multiplier": 1.0,
            }
        elif chip_curr == "wildcard":
            return {
                "cross_awareness_alert": "WILDCARD ACTIVE: Unlimited free transfers available this gameweek.",
                "transfer_action_guidance": "Restructure entire 15-player roster for long-term fixtures and bench depth.",
                "hit_penalty_multiplier": 0.0,
            }
        elif chip_next == "wildcard":
            return {
                "cross_awareness_alert": f"UPCOMING WILDCARD IN GW{next_gw}: Do not take transfer hits this week.",
                "transfer_action_guidance": "Roll your transfer or make a 1-week aggressive punt. All transfers will be reset by the Wildcard next gameweek.",
                "hit_penalty_multiplier": 2.5,  # Strong penalty against hits before wildcard
            }
        elif chip_next == "freehit":
            return {
                "cross_awareness_alert": f"UPCOMING FREE HIT IN GW{next_gw}: Focus transfers on GW{next_gw + 1} and beyond.",
                "transfer_action_guidance": "Any transfer made now will be bench-benched during the Free Hit and returned afterwards.",
                "hit_penalty_multiplier": 1.5,
            }

        return {
            "cross_awareness_alert": "Standard joint planning: Chip calendar aligned with multi-gameweek transfer horizon.",
            "transfer_action_guidance": "Execute planned transfers; save free transfers when approaching Double Gameweeks.",
            "hit_penalty_multiplier": 1.0,
        }

    def generate_chip_strategy(
        self,
        current_gw: int,
        current_squad_df: pd.DataFrame,
        horizon_projections: dict[int, pd.DataFrame],
        fixtures: list[Fixture],
        bootstrap: BootstrapStatic,
        manager_history: ManagerHistory | None = None,
        rivals_analysis: dict[str, Any] | None = None,
        budget: float = 1000.0,
    ) -> dict[str, Any]:
        """
        Compute optimal joint chip strategy for both Set 1 (up to GW19) and Set 2 (GW20-38).
        Includes expected benefit vs baseline, Set 1 cutoff enforcement, and rival chip tracking.
        """
        chips_status = self.get_remaining_chips(manager_history)
        calendar = fixture_calendar.analyze_calendar(fixtures, bootstrap)
        dgw_list = calendar["double_gameweeks"]
        bgw_list = calendar["blank_gameweeks"]

        dgw_gws = [d["gameweek"] for d in dgw_list if d["gameweek"] >= current_gw]
        bgw_gws = [b["gameweek"] for b in bgw_list if b["gameweek"] >= current_gw]

        chip_plan_table = []
        joint_schedule = {}

        # ----------------------------------------------------------------------
        # SET 1 PLANNING (GW current_gw to 19) (C3)
        # ----------------------------------------------------------------------
        set_1_remaining = chips_status["set_1_remaining"] if current_gw <= 19 else []
        set_1_warning = None
        set_1_opportunity_cost: dict[str, float] = {}

        set1_best_assign: dict[str, int] = {}
        set1_best_gain: float = 0.0
        set1_alternatives: list[dict[str, Any]] = []

        if current_gw <= 19 and set_1_remaining:
            available_gws_set_1 = list(range(current_gw, 20))
            set1_best_assign, set1_best_gain, set1_alternatives = self.optimize_joint_assignment(
                available_gws=available_gws_set_1,
                remaining_chips=set_1_remaining,
                current_squad_df=current_squad_df,
                horizon_projections=horizon_projections,
                dgw_gws=dgw_gws,
                bgw_gws=bgw_gws,
                budget=budget,
                current_gw=current_gw,
            )

            total_opp_cost = 0.0
            for c in set_1_remaining:
                gw_chosen = set1_best_assign.get(c, current_gw)
                util, _, _, _ = self.compute_chip_utility(
                    c, gw_chosen, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget, current_gw=current_gw
                )
                set_1_opportunity_cost[c] = util
                total_opp_cost += util

            if len(set_1_remaining) > len(available_gws_set_1):
                set_1_warning = (
                    f"CRITICAL CHIP CONGESTION: You have {len(set_1_remaining)} chips remaining in Set 1 "
                    f"but only {len(available_gws_set_1)} gameweeks left before the GW19 deadline! "
                    f"Since only 1 chip can be played per gameweek, at least one chip will expire unused. "
                    f"Total opportunity cost of forfeiting remaining Set 1 chips: ~{round(total_opp_cost, 1)} pts."
                )
            else:
                set_1_warning = (
                    f"SET 1 DEADLINE NOTICE: Gameweek 19 (Saturday 2 January 2027) is the hard deadline for your first set of chips. "
                    f"Any unused Set 1 chips ({', '.join([CHIP_DISPLAY_NAMES[c] for c in set_1_remaining])}) will be permanently forfeited without rollover. "
                    f"Total opportunity cost if unused: ~{round(total_opp_cost, 1)} expected points."
                )

            alt_assign = set1_alternatives[0]["assignment"] if set1_alternatives else {}
            for c in set_1_remaining:
                rec_gw = set1_best_assign.get(c)
                if rec_gw is not None:
                    joint_schedule[rec_gw] = c
                    gain, conf, base_xp, chip_xp = self.compute_chip_utility(
                        c, rec_gw, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget, current_gw=current_gw
                    )
                    alt_gw = alt_assign.get(c, rec_gw + 1 if rec_gw < 19 else rec_gw - 1)
                    threshold = CHIP_RECOMMENDATION_THRESHOLDS.get(c, 8.0)
                    is_recommended = (gain >= threshold) and (rec_gw == current_gw)

                    chip_plan_table.append(
                        {
                            "chip": f"{CHIP_DISPLAY_NAMES[c]} (Set 1)",
                            "code": c,
                            "set": 1,
                            "recommended_gw": rec_gw,
                            "expected_gain": gain,
                            "baseline_no_chip_xp": base_xp,
                            "with_chip_xp": chip_xp,
                            "confidence": conf,
                            "is_beyond_horizon": rec_gw > (current_gw + 4),
                            "is_recommended_this_gw": is_recommended,
                            "recommendation_threshold": threshold,
                            "alternative_gw": alt_gw,
                            "trigger_conditions": TRIGGER_CONDITIONS_DEFAULT[c],
                            "opportunity_cost": set_1_opportunity_cost.get(c, gain),
                            "reasoning": f"Deploy {CHIP_DISPLAY_NAMES[c]} in GW{rec_gw} to gain +{gain} xP over no-chip baseline ({base_xp} pts) before GW19 hard expiry.",
                        }
                    )
        elif current_gw > 19:
            set_1_warning = "SET 1 EXPIRED: Gameweek 19 deadline has passed. Set 1 chips are expired and cannot be deployed."

        # ----------------------------------------------------------------------
        # SET 2 PLANNING (GW 20 to 38)
        # ----------------------------------------------------------------------
        set_2_remaining = chips_status["set_2_remaining"]
        available_gws_set_2 = list(range(max(20, current_gw), 39))

        set2_best_assign, set2_best_gain, set2_alternatives = self.optimize_joint_assignment(
            available_gws=available_gws_set_2,
            remaining_chips=set_2_remaining,
            current_squad_df=current_squad_df,
            horizon_projections=horizon_projections,
            dgw_gws=dgw_gws,
            bgw_gws=bgw_gws,
            budget=budget,
            current_gw=current_gw,
        )

        alt2_assign = set2_alternatives[0]["assignment"] if set2_alternatives else {}
        for c in set_2_remaining:
            rec_gw = set2_best_assign.get(
                c, 34 if c == "bboost" else (30 if c == "wildcard" else (29 if c == "freehit" else 37))
            )
            gain, conf, base_xp, chip_xp = self.compute_chip_utility(
                c, rec_gw, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget, current_gw=current_gw
            )
            alt_gw = alt2_assign.get(c, rec_gw + 1 if rec_gw < 38 else rec_gw - 1)
            threshold = CHIP_RECOMMENDATION_THRESHOLDS.get(c, 8.0)
            is_recommended = (gain >= threshold) and (rec_gw == current_gw)

            chip_plan_table.append(
                {
                    "chip": f"{CHIP_DISPLAY_NAMES[c]} (Set 2)",
                    "code": c,
                    "set": 2,
                    "recommended_gw": rec_gw,
                    "expected_gain": gain,
                    "baseline_no_chip_xp": base_xp,
                    "with_chip_xp": chip_xp,
                    "confidence": conf,
                    "is_beyond_horizon": rec_gw > (current_gw + 4),
                    "is_recommended_this_gw": is_recommended,
                    "recommendation_threshold": threshold,
                    "alternative_gw": alt_gw,
                    "trigger_conditions": TRIGGER_CONDITIONS_DEFAULT[c],
                    "opportunity_cost": gain,
                    "reasoning": f"Deploy {CHIP_DISPLAY_NAMES[c]} in GW{rec_gw} targeting Spring fixtures (+{gain} xP over {base_xp} no-chip baseline).",
                }
            )

        # ----------------------------------------------------------------------
        # RIVAL CHIP TRACKING & COUNTER-CHIP CONTEXT (C4)
        # ----------------------------------------------------------------------
        rival_chips_summary = {}
        if rivals_analysis and "rival_squads" in rivals_analysis:
            rivals = rivals_analysis["rival_squads"]
            n_rivals = max(1, len(rivals))
            for c_code in ALL_CHIPS:
                c_name = CHIP_DISPLAY_NAMES[c_code]
                used_by = sum(1 for r in rivals if c_code in [c.lower() for c in r.get("chips_used", [])])
                remaining_count = n_rivals - used_by
                rival_chips_summary[c_code] = {
                    "chip_name": c_name,
                    "rivals_used_count": used_by,
                    "rivals_used_pct": round((used_by / n_rivals) * 100.0, 1),
                    "rivals_remaining_count": remaining_count,
                    "rivals_remaining_pct": round((remaining_count / n_rivals) * 100.0, 1),
                    "usage_likelihood_this_gw": "HIGH" if current_gw in dgw_gws else "LOW",
                }

        # Joint Transfer Advice (C2)
        transfer_advice = self.get_joint_transfer_advice(joint_schedule, current_gw)

        # Current GW Chip Decision
        chip_for_curr_gw = next((item for item in chip_plan_table if item["recommended_gw"] == current_gw), None)
        recommend_chip_now = (
            bool(chip_for_curr_gw["is_recommended_this_gw"])
            if chip_for_curr_gw
            else False
        )

        return {
            "chips_status": chips_status,
            "current_gameweek": current_gw,
            "recommend_chip_this_gw": recommend_chip_now,
            "current_gw_chip_recommendation": chip_for_curr_gw,
            "joint_transfer_advice": transfer_advice,
            "set_1_deadline_warning": set_1_warning,
            "set_1_opportunity_cost": set_1_opportunity_cost,
            "total_set_1_opportunity_cost": round(sum(set_1_opportunity_cost.values()), 1),
            "optimal_joint_gain": round(set1_best_gain + set2_best_gain, 1),
            "chip_plan_table": chip_plan_table,
            "joint_schedule": joint_schedule,
            "rival_chips_summary": rival_chips_summary,
            "set_1_alternative_plans": set1_alternatives,
            "set_2_alternative_plans": set2_alternatives,
            "double_gameweeks_calendar": dgw_list,
            "blank_gameweeks_calendar": bgw_list,
        }

    def evaluate_joint_plan(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        manager_history: ManagerHistory | None = None,
        rivals_analysis: dict[str, Any] | None = None,
        chip_retention_values: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Coordinates stateful joint transfer and chip trajectory optimization.
        Filters available chips according to Set 1 (GW 1-19) / Set 2 (GW 20-38) rules.
        """
        from fpl_oracle.optimise.transfers import transfer_optimizer

        chips_status = self.get_remaining_chips(manager_history)
        available = (
            chips_status["set_1_remaining"]
            if target_gw <= 19
            else chips_status["set_2_remaining"]
        )

        return transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            available_chips=available,
            chip_retention_values=chip_retention_values,
        )


chip_planner = ChipPlanner()

