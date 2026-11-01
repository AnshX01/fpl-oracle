"""
Bonus Points component model and zero-sum fixture allocator.
The 2026/27 BPS rules era (tackled penalty removed, CBI 1 BPS per 3 actions, GK 2 BPS per save) is learned
from 2026-27 rows through the is_2026_27 feature, which the final refit up-weights; there are no hand-set
positional multipliers.
- Strictly enforces fixture-level zero-sum constraint: sum(E[bonus]) <= 6.0 per fixture.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS
from fpl_oracle.ml.components.base import BaseComponent

logger = logging.getLogger("fpl_oracle.ml.components.bonus")


def project_to_box_simplex(
    b_hat: np.ndarray,
    target_sum: float = 6.0,
    lower: float = 0.0,
    upper: float = 3.0,
) -> np.ndarray:
    """
    Exact L2 Euclidean projection onto the bounded box simplex:
        min_b  0.5 * ||b - b_hat||_2^2
        s.t.   sum_{i=1}^K b_i = target_sum,  lower <= b_i <= upper for all i.

    Optimal solution has KKT form: b_i*(lambda) = clip(b_hat_i - lambda, lower, upper).
    Computes exact lambda via piecewise-linear breakpoint analysis with bisection fallback.
    """
    b_arr = np.asarray(b_hat, dtype=np.float64)
    k = len(b_arr)
    if k == 0:
        return np.array([], dtype=np.float64)

    # Infeasible extreme cases: clamp to bounds
    if target_sum <= k * lower:
        return np.full(k, lower, dtype=np.float64)
    if target_sum >= k * upper:
        return np.full(k, upper, dtype=np.float64)

    # Breakpoints in lambda where coordinates hit lower or upper bounds:
    # b_hat_i - lambda = upper  =>  lambda = b_hat_i - upper
    # b_hat_i - lambda = lower  =>  lambda = b_hat_i - lower
    bp_upper = b_arr - upper
    bp_lower = b_arr - lower
    breakpoints = np.unique(np.concatenate([bp_upper, bp_lower]))
    breakpoints.sort()

    def g(lam: float) -> float:
        return float(np.sum(np.clip(b_arr - lam, lower, upper)))

    # Find the interval [lam_lo, lam_hi] containing the root g(lam) = target_sum
    # Note: g(lam) is monotonically non-increasing in lam.
    found = False
    opt_lambda = 0.0

    # Check boundaries of breakpoints
    if g(breakpoints[0]) < target_sum:
        # Root is to the left of all breakpoints: all coords at upper bound
        opt_lambda = breakpoints[0]
        found = True
    elif g(breakpoints[-1]) > target_sum:
        # Root is to the right of all breakpoints: all coords at lower bound
        opt_lambda = breakpoints[-1]
        found = True
    else:
        for j in range(len(breakpoints) - 1):
            lam_a = breakpoints[j]
            lam_b = breakpoints[j + 1]
            g_a = g(lam_a)
            g_b = g(lam_b)

            if abs(g_a - target_sum) < 1e-12:
                opt_lambda = lam_a
                found = True
                break
            if abs(g_b - target_sum) < 1e-12:
                opt_lambda = lam_b
                found = True
                break

            if g_a > target_sum > g_b:
                # Root lies strictly in (lam_a, lam_b).
                # On this interval, coordinate classification (upper, lower, active) is constant!
                lam_mid = 0.5 * (lam_a + lam_b)
                active_mask = (b_arr - lam_mid > lower) & (b_arr - lam_mid < upper)
                n_active = int(np.sum(active_mask))
                if n_active > 0:
                    upper_sum = np.sum(b_arr - lam_mid >= upper) * upper
                    lower_sum = np.sum(b_arr - lam_mid <= lower) * lower
                    active_sum = np.sum(b_arr[active_mask])
                    opt_lambda = float((upper_sum + lower_sum + active_sum - target_sum) / n_active)
                    found = True
                    break

    # Robust high-precision bisection fallback if numerical edge cases occur
    if not found:
        lo = float(np.min(b_arr) - upper)
        hi = float(np.max(b_arr) - lower)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            val = g(mid)
            if val > target_sum:
                lo = mid
            else:
                hi = mid
        opt_lambda = 0.5 * (lo + hi)

    return np.clip(b_arr - opt_lambda, lower, upper)


class BonusModel(BaseComponent):
    """
    Predicts expected bonus points with 2026/27 rules-era rebalancing and
    post-hoc fixture-level zero-sum constraint.
    """

    def __init__(self):
        super().__init__("bonus_model")
        self.reg = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        feat_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_train = X[feat_cols].copy()
        for c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)

        y = Y["target_bonus"].values
        weights = np.ones(len(X_train))
        if "is_2026_27" in X_train.columns:
            weights = np.where(X_train["is_2026_27"] > 0, 4.0, 1.0)

        self.reg.fit(X_train, y, sample_weight=weights)
        self.is_fitted = True

    def constrain_fixture_bonus(
        self,
        exp_bonus: np.ndarray,
        X: pd.DataFrame,
        fixture_ids: pd.Series | np.ndarray | None = None,
        max_fixture_bonus: float = 6.0,
    ) -> np.ndarray:
        """
        Enforce post-hoc fixture-level zero-sum constraint.
        Total bonus points awarded across all players in a match cannot exceed 6.0. With real fixture ids
        (fixture_ids / X["fixture_id"]) the cap is exactly 6.0 per fixture; groupings by gameweek (training
        corpora) scale it by the number of matches they contain.
        """
        n = len(exp_bonus)
        if n == 0:
            return np.array([])

        constrained = exp_bonus.copy()

        meta_df = X.attrs.get("meta") if hasattr(X, "attrs") else None
        if isinstance(meta_df, pd.DataFrame):
            if (
                len(meta_df) != n
                and X.index.is_unique
                and meta_df.index.is_unique
                and set(X.index).issubset(meta_df.index)
            ):
                meta_df = meta_df.loc[X.index]
            elif len(meta_df) != n:
                meta_df = None

        # Resolve fixture identifiers
        exact_fixtures = fixture_ids is not None or "fixture_id" in X.columns
        if fixture_ids is not None:
            fixtures = np.asarray(fixture_ids)
        elif "fixture_id" in X.columns:
            fixtures = X["fixture_id"].values
        elif meta_df is not None and "fixture_id" in meta_df.columns:
            fixtures = meta_df["fixture_id"].values
        elif meta_df is not None and "round" in meta_df.columns:
            fixtures = meta_df["round"].values
        else:
            surrogate_cols = [c for c in ["implied_team_xG", "implied_opp_xG", "days_rest"] if c in X.columns]
            if surrogate_cols and len(X) >= 20:
                fixtures = pd.factorize(X[surrogate_cols].apply(tuple, axis=1))[0]
            else:
                return constrained

        s_fixtures = pd.Series(fixtures, index=np.arange(n))
        for f_id, idx_list in s_fixtures.groupby(s_fixtures).groups.items():
            idx = idx_list.values if hasattr(idx_list, "values") else np.array(idx_list)
            target_cap = (
                max_fixture_bonus if exact_fixtures or len(idx) <= 35 else float(max_fixture_bonus * (len(idx) / 30.0))
            )
            f_sum = float(np.sum(constrained[idx]))
            if f_sum > target_cap and target_cap > 0:
                constrained[idx] = project_to_box_simplex(
                    constrained[idx],
                    target_sum=target_cap,
                    lower=0.0,
                    upper=3.0,
                )
                logger.debug(
                    "L2 simplex projected fixture %s bonus from %.3f to %.3f (cap=%.2f)",
                    f_id,
                    f_sum,
                    float(np.sum(constrained[idx])),
                    target_cap,
                )

        return constrained

    @staticmethod
    def project_to_box_simplex(
        b_hat: np.ndarray,
        target_sum: float = 6.0,
        lower: float = 0.0,
        upper: float = 3.0,
    ) -> np.ndarray:
        return project_to_box_simplex(b_hat, target_sum=target_sum, lower=lower, upper=upper)

    def predict(
        self,
        X: pd.DataFrame,
        fixture_ids: pd.Series | np.ndarray | None = None,
        apply_fixture_constraint: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Predict expected bonus points per player, constraining fixture sums <= 6.0.
        """
        if not self.is_fitted:
            raise RuntimeError("BonusModel is not fitted.")

        if X.empty:
            return {"expected_bonus": np.array([])}

        feat_cols = getattr(self.reg, "feature_name_", None)
        if feat_cols is None:
            feat_cols = list(FEATURE_COLUMNS)
        X_feat = X.reindex(columns=feat_cols, fill_value=0.0).copy()
        for c in X_feat.columns:
            X_feat[c] = pd.to_numeric(X_feat[c], errors="coerce").fillna(0.0)

        exp_bonus = np.clip(self.reg.predict(X_feat), 0.0, 3.0)

        # Post-hoc fixture-level zero-sum constraint
        if apply_fixture_constraint:
            exp_bonus = self.constrain_fixture_bonus(exp_bonus, X, fixture_ids=fixture_ids)

        exp_bonus = np.clip(exp_bonus, 0.0, 3.0)
        return {"expected_bonus": exp_bonus}
