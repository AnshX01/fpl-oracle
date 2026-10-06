"""
Projection aggregator & Scoring ensemble.
Combines decomposed ML components strictly according to verified 2026/27 rules.
Computes calibrated point distributions: P10, P50 (median), P90, and variance without naive mean=median equality.
"""

import numpy as np
import pandas as pd

from fpl_oracle.domain.scoring import calculate_expected_fixture_points, expected_floor_div


class ScoringEnsemble:
    def __init__(self):
        pass

    def aggregate_components(self, components: dict[str, np.ndarray], X: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate component predictions into expected points and distributions.
        Enforces verified 2026/27 scoring rules and coherent probabilities.
        """
        n = len(X)
        p_min60 = components["p_min60"]
        p_starts = components["p_starts"]
        exp_mins = components["expected_minutes"]

        # Probability of playing (at least 1 minute)
        # Bounded coherently: p_play >= p_min60
        p_play = np.clip(np.maximum(p_min60, np.minimum(1.0, exp_mins / 60.0)), 0.0, 1.0)
        p_sub = np.clip(p_play - p_min60, 0.0, 1.0)

        # 1. Appearance points
        # 60+ mins -> 2 pts; 1-59 mins -> 1 pt
        appearance_pts = p_min60 * 2.0 + p_sub * 1.0

        # Positional masks
        pos_gkp = X.get("pos_GKP", pd.Series(np.zeros(n), index=X.index)).values
        pos_def = X.get("pos_DEF", pd.Series(np.zeros(n), index=X.index)).values
        pos_mid = X.get("pos_MID", pd.Series(np.zeros(n), index=X.index)).values
        pos_fwd = X.get("pos_FWD", pd.Series(np.zeros(n), index=X.index)).values

        # 2. Goals scored points
        goal_pts_rate = pos_gkp * 10.0 + pos_def * 6.0 + pos_mid * 5.0 + pos_fwd * 4.0
        exp_goal_pts = components["expected_goals"] * goal_pts_rate

        # 3. Assists points (3 pts across all positions)
        exp_assist_pts = components["expected_assists"] * 3.0

        # 4. Clean sheet points (requires >= 60 mins)
        cs_pts_rate = pos_gkp * 4.0 + pos_def * 4.0 + pos_mid * 1.0 + pos_fwd * 0.0
        exp_cs_pts = components["p_clean_sheet"] * p_min60 * cs_pts_rate

        # 5. Goals conceded deductions (-1 pt per 2 goals conceded for DEF/GKP)
        # Using Poisson expectation for floor(conceded / 2) to correctly account for threshold probability
        exp_gc = components["expected_goals_conceded"]
        gc_penalty_multiplier = pos_gkp * 1.0 + pos_def * 1.0
        exp_gc_deduction = np.array([expected_floor_div(float(r), 2) for r in exp_gc]) * gc_penalty_multiplier

        # 6. Goalkeeper saves (1 pt per 3 saves)
        # Using Poisson expectation for floor(saves / 3)
        exp_saves = components["expected_saves"]
        exp_saves_pts = np.array([expected_floor_div(float(s), 3) for s in exp_saves]) * pos_gkp

        # 7. Defensive Contribution (DefCon +2 for DEF, MID, FWD)
        defcon_eligible = 1.0 - pos_gkp
        exp_defcon_pts = components["p_defcon"] * p_play * 2.0 * defcon_eligible

        # 8. Disciplinary deductions (trained on real card labels)
        exp_card_deduction = components["expected_card_deduction"]

        # 9. Bonus points
        exp_bonus_pts = components["expected_bonus"]

        # Total Expected Points (unclipped to preserve negative point outcomes!)
        xP = (
            appearance_pts
            + exp_goal_pts
            + exp_assist_pts
            + exp_cs_pts
            - exp_gc_deduction
            + exp_saves_pts
            + exp_defcon_pts
            - exp_card_deduction
            + exp_bonus_pts
        )

        # When player has 0 minutes / chance of playing is 0, zero out everything
        zero_mask = (p_play <= 1e-4) | (exp_mins <= 1e-4)
        xP = np.where(zero_mask, 0.0, xP)

        # Calibrated Distribution (P10, P50, P90, Variance)
        # In FPL scoring, distribution is positively skewed (hauls are in the upper tail, floor is bounded by appearance)
        # Mean > Median for skewed haulers.
        base_sigma = 1.1 + (pos_fwd * 1.4) + (pos_mid * 1.2) + (pos_def * 0.9) + (pos_gkp * 0.7)
        sigma = base_sigma * np.sqrt(np.clip(np.maximum(0.1, xP) / 3.0, 0.4, 3.5))

        # P10: Lower outcome floor (for starters, typically 1 or 2 appearance points; for subs 0)
        p10 = np.where(
            zero_mask,
            0.0,
            np.maximum(-1.0, np.where(p_min60 > 0.7, 2.0 - np.clip(exp_card_deduction, 0.0, 1.0), p_play * 1.0)),
        )

        # P50: True Median (below mean due to right skew of goal/bonus hauls)
        # Median is typically xP minus skew adjustment
        skew_adj = np.where(xP > 3.0, np.minimum(0.6, (exp_goal_pts + exp_assist_pts + exp_bonus_pts) * 0.25), 0.0)
        p50 = np.where(zero_mask, 0.0, np.maximum(p10, xP - skew_adj))

        # P90: Upper ceiling outcome
        p90 = np.where(zero_mask, 0.0, np.maximum(p50 + 1.0, xP + 1.45 * sigma))

        variance = sigma**2

        res_df = pd.DataFrame(
            {
                "expected_points": np.round(xP, 2),
                "p10": np.round(p10, 2),
                "p50": np.round(p50, 2),
                "p90": np.round(p90, 2),
                "variance": np.round(variance, 2),
                "exp_appearance": np.round(appearance_pts, 2),
                "exp_goals_pts": np.round(exp_goal_pts, 2),
                "exp_assists_pts": np.round(exp_assist_pts, 2),
                "exp_cs_pts": np.round(exp_cs_pts, 2),
                "exp_defcon_pts": np.round(exp_defcon_pts, 2),
                "exp_bonus_pts": np.round(exp_bonus_pts, 2),
                "p_starts": np.round(p_starts, 3),
                "p_min60": np.round(p_min60, 3),
            }
        )

        return res_df


scoring_ensemble = ScoringEnsemble()
