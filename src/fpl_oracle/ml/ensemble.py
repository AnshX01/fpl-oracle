"""
Projection aggregator & Scoring ensemble.
Combines decomposed ML components strictly according to verified 2026/27 rules.
Computes calibrated point distributions: P10, P50 (median), P90, and variance without naive mean=median equality.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.config import MODELS_DIR
from fpl_oracle.domain.scoring import expected_floor_div


class ScoringEnsemble:
    def __init__(
        self,
        z10: float | None = None,
        z90: float | None = None,
        blend_weights: tuple[float, float, float] | None = None,
    ):
        self.calibrated_z10 = -0.806 if z10 is None else z10
        self.calibrated_z90 = 1.009 if z90 is None else z90
        # Learned validation stacking weights: (w_ml, w_recent_form, w_season_avg)
        self.blend_weights = blend_weights or (0.72, 0.04, 0.24)
        self.bucket_quantiles: dict[str, dict[str, float]] = {}
        if z10 is None and z90 is None and blend_weights is None:
            self._load_default_calibration()

    def _load_default_calibration(self):
        cal_path = MODELS_DIR / "calibration.json"
        if cal_path.exists():
            self.load_calibration(cal_path)

    def load_calibration(self, path: Path) -> bool:
        """Load persisted calibration parameters from JSON artifact."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.calibrated_z10 = float(data.get("z10", self.calibrated_z10))
            self.calibrated_z90 = float(data.get("z90", self.calibrated_z90))
            if "blend_weights" in data:
                bw = data["blend_weights"]
                self.blend_weights = (float(bw[0]), float(bw[1]), float(bw[2]))
            if "bucket_quantiles" in data:
                self.bucket_quantiles = data["bucket_quantiles"]
            return True
        except Exception:
            return False

    def save_calibration(self, path: Path | None = None) -> Path:
        """Persist calibration quantiles and blend weights to disk."""
        target_path = path or (MODELS_DIR / "calibration.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "z10": round(self.calibrated_z10, 4),
            "z90": round(self.calibrated_z90, 4),
            "blend_weights": [round(w, 4) for w in self.blend_weights],
            "bucket_quantiles": self.bucket_quantiles,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return target_path

    def calibrate(
        self,
        components_cal: dict[str, np.ndarray],
        X_cal: pd.DataFrame,
        Y_cal: pd.DataFrame,
    ) -> tuple[float, float]:
        """
        Fit empirical standardized residual quantiles and bucketed quantiles on a disjoint calibration set.
        """
        cal_preds = self.aggregate_components(components_cal, X_cal)
        xp = cal_preds["expected_points"].values
        actual = Y_cal["target_points"].values
        sigma = np.sqrt(cal_preds["variance"].values)

        z = (actual - xp) / np.maximum(0.5, sigma)
        self.calibrated_z10 = float(np.percentile(z, 10))
        self.calibrated_z90 = float(np.percentile(z, 90))

        # Position and minutes bucket calibration
        exp_mins = components_cal.get("expected_minutes", np.zeros(len(X_cal)))
        buckets = {}
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_mask = (X_cal.get(f"pos_{pos}", pd.Series(0, index=X_cal.index)).values > 0)
            for mins_cat, mins_mask in [
                ("low", exp_mins < 30),
                ("mid", (exp_mins >= 30) & (exp_mins < 65)),
                ("high", exp_mins >= 65),
            ]:
                combined_mask = pos_mask & mins_mask
                if combined_mask.sum() >= 20:
                    r = actual[combined_mask] - xp[combined_mask]
                    buckets[f"{pos}_{mins_cat}"] = {
                        "q10": float(round(np.percentile(r, 10), 3)),
                        "q90": float(round(np.percentile(r, 90), 3)),
                        "n": int(combined_mask.sum()),
                    }
        self.bucket_quantiles = buckets
        self.save_calibration()
        return self.calibrated_z10, self.calibrated_z90

    def aggregate_components(self, components: dict[str, np.ndarray], X: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate component predictions into expected points and distributions.
        Enforces verified 2026/27 scoring rules and coherent probabilities.
        """
        n = len(X)
        p_min60 = components["p_min60"]
        p_starts = components["p_starts"]
        exp_mins = components["expected_minutes"]

        cop_scale = (
            np.clip(X["chance_of_playing"].values / 100.0, 0.0, 1.0)
            if "chance_of_playing" in X.columns
            else 1.0
        )
        p_min60 = p_min60 * cop_scale
        p_starts = p_starts * cop_scale
        exp_mins = exp_mins * cop_scale

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
        exp_goal_pts = components["expected_goals"] * goal_pts_rate * cop_scale

        # 3. Assists points (3 pts across all positions)
        exp_assist_pts = components["expected_assists"] * 3.0 * cop_scale

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

        # 10. Goalkeeper penalty saves (+5 pts)
        exp_pen_saves = components.get("expected_penalties_saved", np.zeros(n))
        exp_pen_saves_pts = exp_pen_saves * 5.0 * pos_gkp

        # Total Expected Points (unclipped to preserve negative point outcomes!)
        xP = (
            appearance_pts
            + exp_goal_pts
            + exp_assist_pts
            + exp_cs_pts
            - exp_gc_deduction
            + exp_saves_pts
            + exp_pen_saves_pts
            + exp_defcon_pts
            - exp_card_deduction
            + exp_bonus_pts
        )

        # When player has 0 minutes / chance of playing is 0, zero out everything
        zero_mask = (p_play <= 1e-4) | (exp_mins <= 1e-4)
        xP = np.where(zero_mask, 0.0, xP)

        # Apply learned validation blend with heuristic baselines if configured and baseline features present
        if self.blend_weights is not None and any(w > 1e-4 for w in self.blend_weights[1:]):
            if "roll_points_5" in X.columns and "roll_starts_ratio_5" in X.columns and "opponent_difficulty" in X.columns:
                from fpl_oracle.ml.eval import model_evaluator
                base_recent = model_evaluator.compute_baseline_projections(X)
                base_season = model_evaluator.compute_season_avg_baseline(X)
                w_ml, w_rec, w_sea = self.blend_weights
                xP = w_ml * xP + w_rec * base_recent + w_sea * base_season
                xP = np.where(zero_mask, 0.0, xP)

        # Calibrated Distribution (P10, P50, P90, Variance)
        # In FPL scoring, distribution is positively skewed (hauls are in the upper tail, floor is bounded by appearance)
        base_sigma = 1.1 + (pos_fwd * 1.4) + (pos_mid * 1.2) + (pos_def * 0.9) + (pos_gkp * 0.7)
        sigma = base_sigma * np.sqrt(np.clip(np.maximum(0.1, xP) / 3.0, 0.4, 3.5))

        # P50: True Median (below mean due to right skew of goal/bonus hauls)
        skew_adj = np.where(xP > 3.0, np.minimum(0.6, (exp_goal_pts + exp_assist_pts + exp_bonus_pts) * 0.25), 0.0)
        p50 = np.where(zero_mask, 0.0, np.maximum(0.0, xP - skew_adj))

        # Check if bucketed quantile parameters exist
        if self.bucket_quantiles:
            p10_vals = np.zeros(n)
            p90_vals = np.zeros(n)
            for i in range(n):
                if zero_mask[i]:
                    p10_vals[i] = 0.0
                    p90_vals[i] = 0.0
                    continue
                pos = "GKP" if pos_gkp[i] > 0 else ("DEF" if pos_def[i] > 0 else ("MID" if pos_mid[i] > 0 else "FWD"))
                m_val = exp_mins[i]
                m_cat = "low" if m_val < 30 else ("mid" if m_val < 65 else "high")
                b_key = f"{pos}_{m_cat}"
                if b_key in self.bucket_quantiles:
                    q10 = self.bucket_quantiles[b_key]["q10"]
                    q90 = self.bucket_quantiles[b_key]["q90"]
                    p10_vals[i] = max(0.0, xP[i] + q10)
                    p90_vals[i] = max(p50[i] + 1.0, xP[i] + q90)
                else:
                    p10_vals[i] = max(0.0, xP[i] + self.calibrated_z10 * max(0.5, sigma[i]))
                    p90_vals[i] = max(p50[i] + 1.0, xP[i] + self.calibrated_z90 * max(0.5, sigma[i]))
            p10 = p10_vals
            p90 = p90_vals
        else:
            # Fallback to standard calibrated empirical residual factors
            p10 = np.where(
                zero_mask,
                0.0,
                np.maximum(0.0, xP + self.calibrated_z10 * np.maximum(0.5, sigma)),
            )
            p90 = np.where(
                zero_mask,
                0.0,
                np.maximum(p50 + 1.0, xP + self.calibrated_z90 * np.maximum(0.5, sigma)),
            )

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
