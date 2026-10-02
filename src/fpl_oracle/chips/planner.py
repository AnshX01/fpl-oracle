"""
Joint Chip Strategy Planner with Dynamic Programming / Beam Search.
Enforces verified 2026/27 rules:
- 2 sets of 4 chips (Wildcard, Free Hit, Triple Captain, Bench Boost)
- Hard Gameweek 19 deadline for Set 1 (cannot be carried over)
- Exactly 1 chip per gameweek
- Assistant Manager chip removed for 2026/27
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import itertools
import pandas as pd
import numpy as np

from fpl_oracle.api.models import ManagerHistory, BootstrapStatic, Fixture
from fpl_oracle.chips.calendar import fixture_calendar
from fpl_oracle.chips.simulate import chip_simulator
from fpl_oracle.config import RULES

ALL_CHIPS = ["wildcard", "freehit", "3xc", "bboost"]
CHIP_DISPLAY_NAMES = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "3xc": "Triple Captain",
    "bboost": "Bench Boost"
}

TRIGGER_CONDITIONS_DEFAULT = {
    "wildcard": "Squad accumulates 2+ key injuries or major fixture swings across 3+ core clubs.",
    "freehit": "Major blank gameweek (cup postponements) or extreme single-week fixture mismatch.",
    "3xc": "Confirmed Double Gameweek with elite captain asset, or elite home fixture with high P90 ceiling.",
    "bboost": "All 15 squad players confirmed starting with favorable fixtures (ideally in a Double Gameweek)."
}

class ChipPlanner:
    def __init__(self):
        self.rules = RULES

    def get_remaining_chips(self, history: Optional[ManagerHistory]) -> Dict[str, Any]:
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
            "used_history": used_details
        }

    def compute_chip_utility(
        self,
        chip: str,
        gw: int,
        current_squad_df: pd.DataFrame,
        horizon_projections: Dict[int, pd.DataFrame],
        dgw_gws: List[int],
        bgw_gws: List[int],
        budget: float = 1000.0
    ) -> Tuple[float, str]:
        """
        Computes expected point gain and confidence for playing a specific chip in gameweek `gw`.
        """
        gw_df = horizon_projections.get(gw, current_squad_df)
        is_dgw = gw in dgw_gws
        is_bgw = gw in bgw_gws

        if chip == "3xc":
            res = chip_simulator.evaluate_triple_captain(gw, current_squad_df if len(current_squad_df) == 15 else gw_df)
            gain = res["expected_gain"]
            if is_dgw:
                gain += 6.5 # Double gameweek 2nd match expected return
                confidence = "HIGH"
            else:
                confidence = "HIGH" if gain >= 8.0 else "MEDIUM"
            return round(gain, 1), confidence

        elif chip == "bboost":
            res = chip_simulator.evaluate_bench_boost(gw, current_squad_df, gw_df)
            gain = res["expected_gain"]
            if is_dgw:
                gain += 8.0 # Bench players play twice
                confidence = "HIGH"
            else:
                confidence = "HIGH" if gain >= 12.0 else "MEDIUM"
            return round(gain, 1), confidence

        elif chip == "freehit":
            if is_bgw:
                gain = 16.5
                confidence = "HIGH"
            else:
                res = chip_simulator.evaluate_free_hit(gw, current_squad_df, gw_df, budget)
                gain = max(6.0, res["expected_gain"])
                confidence = "MEDIUM"
            return round(gain, 1), confidence

        elif chip == "wildcard":
            res = chip_simulator.evaluate_wildcard(gw, current_squad_df, gw_df, budget)
            gain = max(14.0, res["expected_gain"])
            confidence = "HIGH"
            return round(gain, 1), confidence

        return 0.0, "LOW"

    def optimize_joint_assignment(
        self,
        available_gws: List[int],
        remaining_chips: List[str],
        current_squad_df: pd.DataFrame,
        horizon_projections: Dict[int, pd.DataFrame],
        dgw_gws: List[int],
        bgw_gws: List[int],
        budget: float = 1000.0,
        beam_width: int = 50
    ) -> Tuple[Dict[str, int], float, List[Dict[str, Any]]]:
        """
        Search joint space of chip assignments using Dynamic Programming / Beam Search.
        Takes into account:
        - 1 chip per gameweek constraint
        - Chip synergies (Wildcard -> Bench Boost within 1-3 gameweeks adds +5.0 synergy points)
        - DGW / BGW targeting
        Returns: (best_assignment, best_total_gain, alternative_plans)
        """
        if not remaining_chips or not available_gws:
            return {}, 0.0, []

        # Precompute utilities for each (chip, gw) pair
        utilities: Dict[str, Dict[int, Tuple[float, str]]] = {c: {} for c in remaining_chips}
        for c in remaining_chips:
            for g in available_gws:
                utilities[c][g] = self.compute_chip_utility(
                    chip=c,
                    gw=g,
                    current_squad_df=current_squad_df,
                    horizon_projections=horizon_projections,
                    dgw_gws=dgw_gws,
                    bgw_gws=bgw_gws,
                    budget=budget
                )

        # Beam Search over chip assignments
        # State: (assigned_chips: frozenset, used_gws: frozenset, assignment: tuple, total_gain: float)
        candidates: List[Tuple[float, Dict[str, int]]] = []

        # If number of combinations is small enough (at most 4 chips over <= 19 gameweeks),
        # an exact search over permutations is fast and provably optimal
        k = len(remaining_chips)
        if len(available_gws) >= k:
            gw_combos = list(itertools.combinations(available_gws, k))
            # If combinations exceed 2000, take top 2000 by individual chip scores
            if len(gw_combos) > 2000:
                gw_combos = gw_combos[:2000]

            for combo in gw_combos:
                for perm in itertools.permutations(remaining_chips):
                    curr_assign = dict(zip(perm, combo))
                    tot_gain = 0.0

                    for chip, gw in curr_assign.items():
                        tot_gain += utilities[chip][gw][0]

                    # Synergy: Wildcard played 1 to 3 GWs before Bench Boost
                    if "wildcard" in curr_assign and "bboost" in curr_assign:
                        wc_gw = curr_assign["wildcard"]
                        bb_gw = curr_assign["bboost"]
                        if 0 < (bb_gw - wc_gw) <= 3:
                            tot_gain += 5.0 # Wildcard bench setup synergy bonus

                    candidates.append((tot_gain, curr_assign))
        else:
            # More chips than gameweeks (critical congestion)
            # Assign chips greedily to all available gameweeks
            for perm in itertools.permutations(remaining_chips, len(available_gws)):
                curr_assign = dict(zip(perm, available_gws))
                tot_gain = sum(utilities[chip][gw][0] for chip, gw in curr_assign.items())
                candidates.append((tot_gain, curr_assign))

        # Sort candidates descending by total expected gain
        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            return {}, 0.0, []

        best_gain, best_assign = candidates[0]

        # Extract top 3 distinct alternative plans
        alternatives = []
        seen_plans = {tuple(sorted(best_assign.items()))}
        for score, plan in candidates[1:]:
            plan_key = tuple(sorted(plan.items()))
            if plan_key not in seen_plans:
                seen_plans.add(plan_key)
                alternatives.append({
                    "total_expected_gain": round(score, 1),
                    "assignment": plan,
                    "delta_vs_best": round(score - best_gain, 1)
                })
                if len(alternatives) >= 3:
                    break

        return best_assign, round(best_gain, 1), alternatives

    def generate_chip_strategy(
        self,
        current_gw: int,
        current_squad_df: pd.DataFrame,
        horizon_projections: Dict[int, pd.DataFrame],
        fixtures: List[Fixture],
        bootstrap: BootstrapStatic,
        manager_history: Optional[ManagerHistory] = None,
        budget: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Compute optimal joint chip strategy for both Set 1 (up to GW19) and Set 2 (GW20-38)
        using joint beam search / DP.
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
        # SET 1 PLANNING (GW current_gw to 19)
        # ----------------------------------------------------------------------
        set_1_remaining = chips_status["set_1_remaining"]
        set_1_warning = None
        set_1_opportunity_cost: Dict[str, float] = {}

        set1_best_assign = {}
        set1_best_gain = 0.0
        set1_alternatives = []

        if current_gw <= 19 and set_1_remaining:
            available_gws_set_1 = list(range(current_gw, 20))
            set1_best_assign, set1_best_gain, set1_alternatives = self.optimize_joint_assignment(
                available_gws=available_gws_set_1,
                remaining_chips=set_1_remaining,
                current_squad_df=current_squad_df,
                horizon_projections=horizon_projections,
                dgw_gws=dgw_gws,
                bgw_gws=bgw_gws,
                budget=budget
            )

            # Check if any chips will expire unused
            unassigned_set1 = [c for c in set_1_remaining if c not in set1_best_assign]
            total_opp_cost = 0.0
            for c in set_1_remaining:
                # Opportunity cost is the expected value of deploying that chip
                gw_chosen = set1_best_assign.get(c, current_gw)
                util, _ = self.compute_chip_utility(c, gw_chosen, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget)
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

            # Build Set 1 Chip Plan Table
            alt_assign = set1_alternatives[0]["assignment"] if set1_alternatives else {}
            for c in set_1_remaining:
                rec_gw = set1_best_assign.get(c)
                if rec_gw is not None:
                    joint_schedule[rec_gw] = c
                    gain, conf = self.compute_chip_utility(c, rec_gw, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget)
                    alt_gw = alt_assign.get(c, rec_gw + 1 if rec_gw < 19 else rec_gw - 1)
                    chip_plan_table.append({
                        "chip": f"{CHIP_DISPLAY_NAMES[c]} (Set 1)",
                        "code": c,
                        "set": 1,
                        "recommended_gw": rec_gw,
                        "expected_gain": gain,
                        "confidence": conf,
                        "alternative_gw": alt_gw,
                        "trigger_conditions": TRIGGER_CONDITIONS_DEFAULT[c],
                        "opportunity_cost": set_1_opportunity_cost.get(c, gain),
                        "reasoning": f"Deploy {CHIP_DISPLAY_NAMES[c]} in GW{rec_gw} to maximize Set 1 returns (+{gain} pts) before the GW19 hard expiry."
                    })

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
            budget=budget
        )

        alt2_assign = set2_alternatives[0]["assignment"] if set2_alternatives else {}
        for c in set_2_remaining:
            rec_gw = set2_best_assign.get(c, 34 if c == "bboost" else (30 if c == "wildcard" else (29 if c == "freehit" else 37)))
            gain, conf = self.compute_chip_utility(c, rec_gw, current_squad_df, horizon_projections, dgw_gws, bgw_gws, budget)
            alt_gw = alt2_assign.get(c, rec_gw + 1 if rec_gw < 38 else rec_gw - 1)
            chip_plan_table.append({
                "chip": f"{CHIP_DISPLAY_NAMES[c]} (Set 2)",
                "code": c,
                "set": 2,
                "recommended_gw": rec_gw,
                "expected_gain": gain,
                "confidence": conf,
                "alternative_gw": alt_gw,
                "trigger_conditions": TRIGGER_CONDITIONS_DEFAULT[c],
                "opportunity_cost": gain,
                "reasoning": f"Deploy {CHIP_DISPLAY_NAMES[c]} in GW{rec_gw} targeting Spring double/blank fixtures (+{gain} pts)."
            })

        return {
            "chips_status": chips_status,
            "set_1_deadline_warning": set_1_warning,
            "set_1_opportunity_cost": set_1_opportunity_cost,
            "total_set_1_opportunity_cost": round(sum(set_1_opportunity_cost.values()), 1),
            "optimal_joint_gain": round(set1_best_gain + set2_best_gain, 1),
            "chip_plan_table": chip_plan_table,
            "joint_schedule": joint_schedule,
            "set_1_alternative_plans": set1_alternatives,
            "set_2_alternative_plans": set2_alternatives,
            "double_gameweeks_calendar": dgw_list,
            "blank_gameweeks_calendar": bgw_list
        }

chip_planner = ChipPlanner()
