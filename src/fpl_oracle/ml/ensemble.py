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
from sklearn.metrics import mean_absolute_error

from fpl_oracle.config import MODELS_DIR
from fpl_oracle.domain.scoring import expected_floor_div
from fpl_oracle.rules.scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    GOAL_POINTS,
    OWN_GOAL_POINTS,
    PENALTY_MISSED_POINTS,
    PENALTY_SAVE_POINTS,
)


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
        self.blend_weights: tuple[float, float, float] | None = blend_weights or (0.72, 0.04, 0.24)
        self.positional_blend_weights: dict[str, tuple[float, float, float]] | None = None
        self.bucket_quantiles: dict[str, dict[str, float]] = {}
        self.cqr: Any | None = None
        if z10 is None and z90 is None and blend_weights is None:
            self._load_default_calibration()

    def _load_default_calibration(self):
        cal_path = MODELS_DIR / "calibration.json"
        if cal_path.exists():
            self.load_calibration(cal_path)
        cqr_path = MODELS_DIR / "cqr"
        if cqr_path.exists():
            from fpl_oracle.ml.quantiles import ConformalQuantileRegressor

            cqr = ConformalQuantileRegressor()
            if cqr.load(cqr_path):
                self.cqr = cqr

    def set_cqr(self, cqr: Any) -> None:
        """Set active ConformalQuantileRegressor instance."""
        self.cqr = cqr

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
            if "positional_blend_weights" in data:
                self.positional_blend_weights = {
                    pos: (float(w[0]), float(w[1]), float(w[2]))
                    for pos, w in data["positional_blend_weights"].items()
                }
            if "bucket_quantiles" in data:
                self.bucket_quantiles = data["bucket_quantiles"]
            return True
        except Exception:
            return False

    def get_calibration_dict(self) -> dict[str, Any]:
        """Get current calibration quantiles and blend weights as dictionary."""
        res: dict[str, Any] = {
            "z10": round(self.calibrated_z10, 4),
            "z90": round(self.calibrated_z90, 4),
            "blend_weights": (
                [round(w, 4) for w in self.blend_weights] if self.blend_weights is not None else [0.72, 0.04, 0.24]
            ),
            "bucket_quantiles": self.bucket_quantiles,
        }
        if self.positional_blend_weights is not None:
            res["positional_blend_weights"] = {
                pos: [round(w, 4) for w in weights]
                for pos, weights in self.positional_blend_weights.items()
            }
        return res

    def save_calibration(self, path: Path | None = None) -> Path:
        """Persist calibration quantiles and blend weights to disk. Does NOT default to production path."""
        if path is None:
            target_path = Path("scratch") / "calibration.json"
        else:
            target_path = path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.get_calibration_dict()
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
            pos_mask = X_cal.get(f"pos_{pos}", pd.Series(0, index=X_cal.index)).values > 0
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
        # In-memory fitting only; disk persistence is handled exclusively by promotion routine
        return self.calibrated_z10, self.calibrated_z90

    def fit_stacking_weights(
        self,
        components_val: dict[str, np.ndarray],
        X_val: pd.DataFrame,
        Y_val: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Fit optimal blending weights w = (w_ml, w_recent, w_season) on validation set.
        Updates self.blend_weights in-place.
        """
        from scipy.optimize import minimize

        from fpl_oracle.ml.eval import model_evaluator

        old_weights = self.blend_weights
        self.blend_weights = None  # Extract unblended ML xP
        raw_preds = self.aggregate_components(components_val, X_val)["expected_points"].values
        self.blend_weights = old_weights

        base_rec = model_evaluator.compute_baseline_projections(X_val)
        base_sea = model_evaluator.compute_season_avg_baseline(X_val)
        actual = Y_val["target_points"].values

        def _mae_loss(w: np.ndarray) -> float:
            pred = w[0] * raw_preds + w[1] * base_rec + w[2] * base_sea
            return float(mean_absolute_error(actual, pred))

        res = minimize(
            _mae_loss,
            [0.72, 0.04, 0.24],
            bounds=[(0, 1), (0, 1), (0, 1)],
            constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
        )
        self.blend_weights = (float(round(res.x[0], 4)), float(round(res.x[1], 4)), float(round(res.x[2], 4)))
        return self.blend_weights

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
            np.clip(X["chance_of_playing"].values / 100.0, 0.0, 1.0) if "chance_of_playing" in X.columns else 1.0
        )
        p_min60 = p_min60 * cop_scale
        p_starts = p_starts * cop_scale
        exp_mins = exp_mins * cop_scale
        if "news_start_probability" in X:
            start = X["news_start_probability"].to_numpy(dtype=float)
            valid = np.isfinite(start)
            p_starts = np.where(valid, np.minimum(p_starts, start * cop_scale), p_starts)
            p_min60 = np.minimum(p_min60, p_starts)
        if "news_minutes_limit" in X:
            limit = X["news_minutes_limit"].to_numpy(dtype=float)
            valid = np.isfinite(limit)
            ratio = np.where(valid, np.minimum(1.0, limit / np.maximum(exp_mins, 1e-9)), 1.0)
            exp_mins = exp_mins * ratio
            p_min60 = np.where(valid & (limit < 60), 0.0, p_min60)
            # Scale unconditional point components once for explicit minutes restriction.
            components = dict(components)
            for key in (
                "expected_goals",
                "expected_assists",
                "expected_goals_conceded",
                "expected_saves",
                "expected_bonus",
                "expected_card_deduction",
                "expected_penalties_saved",
                "expected_own_goals",
                "expected_penalties_missed",
            ):
                if key in components:
                    components[key] = components[key] * ratio

        # Probability of playing (at least 1 minute)
        # Bounded coherently: p_play >= p_min60
        if "p_play" in components:
            p_play = np.clip(components["p_play"] * cop_scale, 0.0, 1.0)
        else:
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
        goal_pts_rate = (
            pos_gkp * float(GOAL_POINTS["GKP"])
            + pos_def * float(GOAL_POINTS["DEF"])
            + pos_mid * float(GOAL_POINTS["MID"])
            + pos_fwd * float(GOAL_POINTS["FWD"])
        )
        exp_goal_pts = components["expected_goals"] * goal_pts_rate * cop_scale

        # 3. Assists points
        exp_assist_pts = components["expected_assists"] * float(ASSIST_POINTS) * cop_scale

        # 4. Clean sheet points (requires >= 60 mins)
        cs_pts_rate = (
            pos_gkp * float(CLEAN_SHEET_POINTS["GKP"])
            + pos_def * float(CLEAN_SHEET_POINTS["DEF"])
            + pos_mid * float(CLEAN_SHEET_POINTS["MID"])
            + pos_fwd * float(CLEAN_SHEET_POINTS["FWD"])
        )
        exp_cs_pts = components["p_clean_sheet"] * p_min60 * cs_pts_rate

        # 5. Goals conceded deductions (-1 pt per 2 goals conceded for DEF/GKP)
        # Using Poisson expectation for floor(conceded / 2) to correctly account for threshold probability
        exp_gc = components["expected_goals_conceded"]
        gc_penalty_multiplier = pos_gkp * 1.0 + pos_def * 1.0
        exp_gc_deduction = (
            np.array([expected_floor_div(float(r), 2) for r in exp_gc]) * gc_penalty_multiplier * cop_scale
        )

        # 6. Goalkeeper saves (1 pt per 3 saves)
        # Using Poisson expectation for floor(saves / 3)
        exp_saves = components["expected_saves"]
        exp_saves_pts = np.array([expected_floor_div(float(s), 3) for s in exp_saves]) * pos_gkp * cop_scale

        # 7. Defensive Contribution (DefCon +2 for DEF, MID, FWD)
        defcon_eligible = 1.0 - pos_gkp
        exp_defcon_pts = components["p_defcon"] * p_play * float(DEFCON_POINTS) * defcon_eligible

        # 8. Disciplinary deductions (trained on real card labels)
        exp_card_deduction = components["expected_card_deduction"] * cop_scale

        # 9. Bonus points
        exp_bonus_pts = components["expected_bonus"] * cop_scale

        # 10. Goalkeeper penalty saves (+5 pts)
        exp_pen_saves = components.get("expected_penalties_saved", np.zeros(n))
        exp_pen_saves_pts = exp_pen_saves * float(PENALTY_SAVE_POINTS) * pos_gkp * cop_scale

        # 11. Own goals (-2 pts per own goal, WI-1.2)
        if "expected_own_goals" in components:
            exp_og = np.asarray(components["expected_own_goals"], dtype=float)
        elif "roll_own_goals_5" in X.columns:
            og_rate = np.clip(X["roll_own_goals_5"].to_numpy(dtype=float) / 5.0, 0.0, 1.0)
            pos_og_prior = pos_def * 0.008 + pos_gkp * 0.003 + pos_mid * 0.002 + pos_fwd * 0.001
            exp_og = 0.5 * og_rate + 0.5 * pos_og_prior
        else:
            exp_og = pos_def * 0.008 + pos_gkp * 0.003 + pos_mid * 0.002 + pos_fwd * 0.001
        exp_og_deduction = exp_og * float(abs(OWN_GOAL_POINTS)) * p_play * cop_scale

        # 12. Penalties missed (-2 pts per penalty missed, WI-1.2)
        # Expectation = P(takes penalty) * P(penalty awarded) * miss_rate, shrunk to positional prior
        if "expected_penalties_missed" in components:
            exp_pm = np.asarray(components["expected_penalties_missed"], dtype=float)
        elif "roll_penalties_missed_5" in X.columns:
            pm_rate = np.clip(X["roll_penalties_missed_5"].to_numpy(dtype=float) / 5.0, 0.0, 1.0)
            taker_prior = (pos_fwd * 0.015 + pos_mid * 0.008) * np.clip(
                components.get("expected_goals", np.zeros(n)) / 0.4, 0.0, 1.0
            )
            exp_pm = np.where(pm_rate > 0, 0.6 * pm_rate + 0.4 * taker_prior, taker_prior * 0.5)
        else:
            exp_pm = (pos_fwd * 0.005 + pos_mid * 0.003) * p_play
        exp_pen_miss_deduction = exp_pm * float(abs(PENALTY_MISSED_POINTS)) * p_play * cop_scale

        # Zero out components when player does not participate
        zero_mask = (p_play <= 1e-4) | (exp_mins <= 1e-4)
        appearance_pts = np.where(zero_mask, 0.0, appearance_pts)
        exp_goal_pts = np.where(zero_mask, 0.0, exp_goal_pts)
        exp_assist_pts = np.where(zero_mask, 0.0, exp_assist_pts)
        exp_cs_pts = np.where(zero_mask, 0.0, exp_cs_pts)
        exp_gc_deduction = np.where(zero_mask, 0.0, exp_gc_deduction)
        exp_saves_pts = np.where(zero_mask, 0.0, exp_saves_pts)
        exp_pen_saves_pts = np.where(zero_mask, 0.0, exp_pen_saves_pts)
        exp_defcon_pts = np.where(zero_mask, 0.0, exp_defcon_pts)
        exp_card_deduction = np.where(zero_mask, 0.0, exp_card_deduction)
        exp_bonus_pts = np.where(zero_mask, 0.0, exp_bonus_pts)
        exp_og_deduction = np.where(zero_mask, 0.0, exp_og_deduction)
        exp_pen_miss_deduction = np.where(zero_mask, 0.0, exp_pen_miss_deduction)

        # Total Expected Points (unclipped to preserve negative point outcomes!)
        raw_xP = (
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
            - exp_og_deduction
            - exp_pen_miss_deduction
        )
        xP = raw_xP

        exp_blend_adj = np.zeros(n)
        # Apply learned validation blend with heuristic baselines (WI-3.10)
        has_baseline_features = (
            "roll_points_5" in X.columns
            and "roll_starts_ratio_5" in X.columns
            and "opponent_difficulty" in X.columns
        )
        if has_baseline_features and (
            (self.positional_blend_weights is not None)
            or (self.blend_weights is not None and any(w > 1e-4 for w in self.blend_weights[1:]))
        ):
            from fpl_oracle.ml.eval import model_evaluator

            base_recent = np.asarray(model_evaluator.compute_baseline_projections(X))
            base_season = np.asarray(model_evaluator.compute_season_avg_baseline(X))

            if self.positional_blend_weights is not None:
                blended_xP = np.copy(raw_xP)
                for pos, (w_ml, w_rec, w_sea) in self.positional_blend_weights.items():
                    pos_mask = (
                        (pos_gkp if pos == "GKP" else (pos_def if pos == "DEF" else (pos_mid if pos == "MID" else pos_fwd)))
                        > 0.5
                    )
                    if np.any(pos_mask):
                        blended_xP[pos_mask] = (
                            w_ml * raw_xP[pos_mask]
                            + w_rec * base_recent[pos_mask]
                            + w_sea * base_season[pos_mask]
                        )
            else:
                assert self.blend_weights is not None
                w_ml, w_rec, w_sea = self.blend_weights
                blended_xP = w_ml * raw_xP + w_rec * base_recent + w_sea * base_season

            blended_xP = np.where(zero_mask, 0.0, blended_xP)
            exp_blend_adj = blended_xP - raw_xP
            xP = blended_xP

        # Calibrated Distribution (P10, P50, P90, Variance)
        # In FPL scoring, distribution is positively skewed (hauls are in the upper tail, floor is bounded by appearance)
        base_sigma = 1.1 + (pos_fwd * 1.4) + (pos_mid * 1.2) + (pos_def * 0.9) + (pos_gkp * 0.7)
        sigma = base_sigma * np.sqrt(np.clip(np.maximum(0.1, xP) / 3.0, 0.4, 3.5))

        # P50: True Median (below mean due to right skew of goal/bonus hauls)
        skew_adj = np.where(xP > 3.0, np.minimum(0.6, (exp_goal_pts + exp_assist_pts + exp_bonus_pts) * 0.25), 0.0)
        p50 = np.where(zero_mask, 0.0, np.maximum(0.0, xP - skew_adj))

        # Check if caller explicitly injected custom bucketed quantiles (e.g. in synthetic tests)
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
        # Otherwise use calibrated CQR model
        elif self.cqr is not None and getattr(self.cqr, "is_calibrated", False):
            pos_labels = np.where(
                pos_gkp > 0, "GKP", np.where(pos_def > 0, "DEF", np.where(pos_mid > 0, "MID", "FWD"))
            )
            p10, p90, _ = self.cqr.predict_quantiles(
                X=X,
                pos_labels=pos_labels,
                xp=xP,
                p_play=p_play,
            )
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

        r_app = np.round(appearance_pts, 2)
        r_goals = np.round(exp_goal_pts, 2)
        r_assists = np.round(exp_assist_pts, 2)
        r_cs = np.round(exp_cs_pts, 2)
        r_defcon = np.round(exp_defcon_pts, 2)
        r_bonus = np.round(exp_bonus_pts, 2)
        r_gc = np.round(exp_gc_deduction, 2)
        r_saves = np.round(exp_saves_pts, 2)
        r_pen_saves = np.round(exp_pen_saves_pts, 2)
        r_card = np.round(exp_card_deduction, 2)
        r_og = np.round(exp_og_deduction, 2)
        r_pm = np.round(exp_pen_miss_deduction, 2)
        r_blend = np.round(exp_blend_adj, 2)

        # Decomposed raw expectation sums exactly to the penny
        decomp_xp = np.round(
            r_app
            + r_goals
            + r_assists
            + r_cs
            + r_defcon
            + r_bonus
            + r_saves
            + r_pen_saves
            - r_gc
            - r_card
            - r_og
            - r_pm,
            2,
        )
        final_xp = np.round(decomp_xp + r_blend, 2)

        res_df = pd.DataFrame(
            {
                "expected_points": final_xp,
                "raw_expected_points": decomp_xp,
                "p10": np.round(p10, 2),
                "p50": np.round(p50, 2),
                "p90": np.round(p90, 2),
                "variance": np.round(variance, 2),
                "exp_appearance": r_app,
                "exp_goals_pts": r_goals,
                "exp_assists_pts": r_assists,
                "exp_cs_pts": r_cs,
                "exp_defcon_pts": r_defcon,
                "exp_bonus_pts": r_bonus,
                "p_starts": np.round(p_starts, 3),
                "p_min60": np.round(p_min60, 3),
                "p_clean_sheet": np.round(components.get("p_clean_sheet", np.full(n, 0.30)), 4),
                "p_play": np.round(p_play, 3),
                "exp_gc_deduction": r_gc,
                "exp_saves_pts": r_saves,
                "exp_pen_saves_pts": r_pen_saves,
                "exp_card_deduction": r_card,
                "exp_og_deduction": r_og,
                "exp_pen_miss_deduction": r_pm,
                "exp_blend_adj": r_blend,
            }
        )

        return res_df

    @staticmethod
    def verify_decomposition(df: pd.DataFrame) -> pd.Series:
        """
        Reconstructs unblended expected points from constituent additive and deductive components.
        Matches raw_expected_points exactly (and expected_points when unblended).
        """
        return np.round(
            df.get("exp_appearance", 0.0)
            + df.get("exp_goals_pts", 0.0)
            + df.get("exp_assists_pts", 0.0)
            + df.get("exp_cs_pts", 0.0)
            + df.get("exp_defcon_pts", 0.0)
            + df.get("exp_bonus_pts", 0.0)
            + df.get("exp_saves_pts", 0.0)
            + df.get("exp_pen_saves_pts", 0.0)
            - df.get("exp_gc_deduction", 0.0)
            - df.get("exp_card_deduction", 0.0)
            - df.get("exp_og_deduction", 0.0)
            - df.get("exp_pen_miss_deduction", 0.0),
            2,
        )

    def fit_positional_stacking_weights(
        self,
        raw_xp: np.ndarray,
        base_recent: np.ndarray,
        base_season: np.ndarray,
        targets: np.ndarray,
        positions: np.ndarray,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Fits non-negative least squares stacking weights per position (WI-3.10)
        constrained to sum to 1 on out-of-fold predictions.
        """
        from scipy.optimize import nnls

        weights: dict[str, tuple[float, float, float]] = {}
        for pos in ("GKP", "DEF", "MID", "FWD"):
            mask = positions == pos
            if not np.any(mask) or len(targets[mask]) < 10:
                weights[pos] = (0.72, 0.04, 0.24)
                continue
            A = np.column_stack([raw_xp[mask], base_recent[mask], base_season[mask]])
            b = targets[mask]
            w, _ = nnls(A, b)
            s = float(np.sum(w))
            if s > 0.0:
                w_norm = w / s
                weights[pos] = (float(round(w_norm[0], 4)), float(round(w_norm[1], 4)), float(round(w_norm[2], 4)))
            else:
                weights[pos] = (0.72, 0.04, 0.24)

        self.positional_blend_weights = weights
        return weights


scoring_ensemble = ScoringEnsemble()
