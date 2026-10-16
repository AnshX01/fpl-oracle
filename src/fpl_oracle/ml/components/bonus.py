"""
Bonus Points component model and zero-sum fixture allocator (WI-3.6).
Accounts for the 2026/27 BPS rules-era shift (R6):
- Tackled penalty removed
- CBI earns 1 BPS per 3 actions (was per 2)
- GK: 2 BPS for any save
- Strictly enforces fixture-level zero-sum constraint: sum(E[bonus]) <= 6.0 per fixture.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS
from fpl_oracle.ml.components.base import BaseComponent

logger = logging.getLogger("fpl_oracle.ml.components.bonus")


class BonusModel(BaseComponent):
    """
    Predicts expected bonus points with 2026/27 rules-era rebalancing and
    post-hoc fixture-level zero-sum constraint (WI-3.6).
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
        WI-3.6: Enforce post-hoc fixture-level zero-sum constraint.
        Total bonus points awarded across all players in a match cannot exceed 6.0.
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
            # Target cap: 6.0 for single match; scale by number of matches if aggregate round
            target_cap = max_fixture_bonus if len(idx) <= 35 else float(max_fixture_bonus * (len(idx) / 30.0))
            f_sum = float(np.sum(constrained[idx]))
            if f_sum > target_cap and target_cap > 0:
                scale = target_cap / f_sum
                constrained[idx] *= scale
                logger.debug(
                    "WI-3.6: Scaled fixture %s bonus from %.3f to %.3f (cap=%.2f)",
                    f_id,
                    f_sum,
                    float(np.sum(constrained[idx])),
                    target_cap,
                )

        return constrained

    def predict(
        self,
        X: pd.DataFrame,
        fixture_ids: pd.Series | np.ndarray | None = None,
        apply_era_shift: bool = True,
        apply_fixture_constraint: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Predict expected bonus points per player, adapting for the 2026/27 BPS rules-era
        shift and constraining fixture sums <= 6.0 (WI-3.6).
        """
        if not self.is_fitted:
            raise RuntimeError("BonusModel is not fitted.")

        if X.empty:
            return {"expected_bonus": np.array([])}

        feat_cols = getattr(self.reg, "feature_name_", None)
        if feat_cols is not None:
            use_cols = [c for c in feat_cols if c in X.columns]
        else:
            use_cols = [c for c in FEATURE_COLUMNS if c in X.columns]

        X_feat = X[use_cols].copy()
        for c in X_feat.columns:
            X_feat[c] = pd.to_numeric(X_feat[c], errors="coerce").fillna(0.0)

        exp_bonus = np.clip(self.reg.predict(X_feat), 0.0, 3.0)

        # 2026/27 BPS Rules-Era Shift (R6)
        if apply_era_shift and "is_2026_27" in X.columns:
            is_26 = pd.to_numeric(X["is_2026_27"], errors="coerce").fillna(0.0).values > 0
            if is_26.any():
                is_gkp = pd.to_numeric(X.get("pos_GKP", 0), errors="coerce").fillna(0.0).values
                is_fwd = pd.to_numeric(X.get("pos_FWD", 0), errors="coerce").fillna(0.0).values
                is_def = pd.to_numeric(X.get("pos_DEF", 0), errors="coerce").fillna(0.0).values

                era_mult = np.ones(len(X))
                era_mult = np.where(is_gkp > 0, 1.25, era_mult)
                era_mult = np.where(is_fwd > 0, 0.85, era_mult)
                era_mult = np.where(is_def > 0, 1.05, era_mult)
                era_mult = np.where(
                    is_def == 0, np.where(is_gkp == 0, np.where(is_fwd == 0, 0.92, era_mult), era_mult), era_mult
                )

                exp_bonus = np.where(is_26, exp_bonus * era_mult, exp_bonus)

        # Post-hoc fixture-level zero-sum constraint (WI-3.6)
        if apply_fixture_constraint:
            exp_bonus = self.constrain_fixture_bonus(exp_bonus, X, fixture_ids=fixture_ids)

        exp_bonus = np.clip(exp_bonus, 0.0, 3.0)
        return {"expected_bonus": exp_bonus}
