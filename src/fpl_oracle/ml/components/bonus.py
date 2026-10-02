"""
Bonus Points component model.
Accounts for 2026/27 BPS rebalancing by weighting recent gameweeks
and utilizing the is_2026_27 indicator feature.
"""


import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.components.base import BaseComponent


class BonusModel(BaseComponent):
    def __init__(self):
        super().__init__("bonus_model")
        self.reg = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y = Y["target_bonus"].values
        # Weight recent seasons and 2026/27 data more heavily
        weights = np.ones(len(X))
        if "is_2026_27" in X.columns:
            weights = np.where(X["is_2026_27"] > 0, 4.0, 1.0)

        self.reg.fit(X, y, sample_weight=weights)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("BonusModel is not fitted.")

        exp_bonus = np.clip(self.reg.predict(X), 0.0, 3.0)
        return {
            "expected_bonus": exp_bonus
        }
