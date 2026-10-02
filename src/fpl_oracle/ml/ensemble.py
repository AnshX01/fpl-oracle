"""
Projection aggregator & Scoring ensemble.
Combines decomposed ML components strictly according to verified 2026/27 rules.
Computes point distributions: P10, P50 (median), P90, and variance.
"""

import numpy as np
import pandas as pd

from fpl_oracle.config import SCORING


class ScoringEnsemble:
    def __init__(self):
        self.scoring = SCORING

    def aggregate_components(self, components: dict[str, np.ndarray], X: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate component predictions into expected points and distributions.
        """
        n = len(X)
        p_min60 = components["p_min60"]
        p_starts = components["p_starts"]
        exp_mins = components["expected_minutes"]

        # 1. Appearance points
        # If min >= 60 -> 2 pts; if 0 < min < 60 -> 1 pt
        p_play = np.clip(exp_mins / 70.0, 0.0, 1.0)
        p_sub = np.clip(p_play - p_min60, 0.0, 1.0)
        appearance_pts = p_min60 * self.scoring["minutes"]["long_play"] + p_sub * self.scoring["minutes"]["short_play"]

        # 2. Position weights
        pos_gkp = X.get("pos_GKP", np.zeros(n)).values
        pos_def = X.get("pos_DEF", np.zeros(n)).values
        pos_mid = X.get("pos_MID", np.zeros(n)).values
        pos_fwd = X.get("pos_FWD", np.zeros(n)).values

        # Goals scored points
        goal_pts_multiplier = (
            pos_gkp * self.scoring["goals_scored"]["GKP"] +
            pos_def * self.scoring["goals_scored"]["DEF"] +
            pos_mid * self.scoring["goals_scored"]["MID"] +
            pos_fwd * self.scoring["goals_scored"]["FWD"]
        )
        exp_goal_pts = components["expected_goals"] * goal_pts_multiplier

        # Assists points (3 pts across all positions)
        exp_assist_pts = components["expected_assists"] * self.scoring["assists"]["MID"]

        # Clean sheet points (requires >= 60 mins)
        cs_pts_multiplier = (
            pos_gkp * self.scoring["clean_sheets"]["GKP"] +
            pos_def * self.scoring["clean_sheets"]["DEF"] +
            pos_mid * self.scoring["clean_sheets"]["MID"] +
            pos_fwd * self.scoring["clean_sheets"]["FWD"]
        )
        exp_cs_pts = components["p_clean_sheet"] * p_min60 * cs_pts_multiplier

        # Goals conceded deductions (-1 pt per 2 goals conceded for DEF/GKP)
        gc_penalty_multiplier = (
            pos_gkp * abs(self.scoring["goals_conceded"]["GKP"]) +
            pos_def * abs(self.scoring["goals_conceded"]["DEF"])
        )
        exp_gc_deduction = (components["expected_goals_conceded"] / 2.0) * gc_penalty_multiplier

        # Goalkeeper saves (1 pt per 3 saves)
        exp_saves_pts = (components["expected_saves"] / 3.0) * self.scoring["saves"]["points"]

        # Defensive Contribution (DefCon) - verified 2026/27 rule (+2 pts for DEF/MID/FWD)
        defcon_eligible = 1.0 - pos_gkp
        defcon_pts_rate = self.scoring["defensive_contribution"]["DEF"]
        exp_defcon_pts = components["p_defcon"] * defcon_eligible * defcon_pts_rate

        # Disciplinary deductions
        exp_card_deduction = components["expected_card_deduction"]

        # Bonus points
        exp_bonus_pts = components["expected_bonus"]

        # Total Expected Points
        xP = (
            appearance_pts +
            exp_goal_pts +
            exp_assist_pts +
            exp_cs_pts -
            exp_gc_deduction +
            exp_saves_pts +
            exp_defcon_pts -
            exp_card_deduction +
            exp_bonus_pts
        )
        xP = np.clip(xP, 0.0, 25.0)

        # Uncertainty modeling: position-specific variance
        # Strikers/attacking mids have higher variance (higher ceiling/floor spread)
        base_sigma = 1.2 + (pos_fwd * 1.5) + (pos_mid * 1.3) + (pos_def * 1.0) + (pos_gkp * 0.8)
        sigma = base_sigma * np.sqrt(np.clip(xP / 3.5, 0.5, 3.0))

        p10 = np.clip(xP - 1.28 * sigma, 0.0, None)
        p50 = xP
        p90 = xP + 1.28 * sigma
        variance = sigma ** 2

        res_df = pd.DataFrame({
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
            "p_min60": np.round(p_min60, 3)
        })

        return res_df

scoring_ensemble = ScoringEnsemble()
