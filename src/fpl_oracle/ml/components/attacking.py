"""
Attacking component model.
Predicts expected goals (xG) and expected assists (xA).
"""


import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.components.base import BaseComponent


class AttackingModel(BaseComponent):
    def __init__(self):
        super().__init__("attacking_model")
        self.reg_goals = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.reg_assists = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.reg_goals.fit(X, Y["target_goals"].values)
        self.reg_assists.fit(X, Y["target_assists"].values)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("AttackingModel is not fitted.")

        exp_goals = np.clip(self.reg_goals.predict(X), 0.0, 3.5)
        exp_assists = np.clip(self.reg_assists.predict(X), 0.0, 3.0)

        # Suppress attacking expectations for GKP
        if "pos_GKP" in X.columns:
            is_gkp = X["pos_GKP"].values
            exp_goals = exp_goals * (1.0 - is_gkp * 0.98)
            exp_assists = exp_assists * (1.0 - is_gkp * 0.95)

        return {
            "expected_goals": exp_goals,
            "expected_assists": exp_assists
        }
