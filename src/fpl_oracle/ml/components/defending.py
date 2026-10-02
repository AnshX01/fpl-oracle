"""
Defending component model.
Predicts clean sheet probability P(CS) and expected goals conceded.
"""


import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent


class DefendingModel(BaseComponent):
    def __init__(self):
        super().__init__("defending_model")
        self.clf_cs = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.reg_gc = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.calibrator_cs = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_cs = Y["target_clean_sheet"].values
        y_gc = Y["target_goals_conceded"].values

        self.clf_cs.fit(X, y_cs)
        self.reg_gc.fit(X, y_gc)

        prob_cs_raw = self.clf_cs.predict_proba(X)[:, 1]
        self.calibrator_cs.fit(prob_cs_raw, y_cs)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("DefendingModel is not fitted.")

        p_cs_raw = self.clf_cs.predict_proba(X)[:, 1]
        p_cs = self.calibrator_cs.calibrate(p_cs_raw)
        exp_gc = np.clip(self.reg_gc.predict(X), 0.0, 5.0)

        return {
            "p_clean_sheet": p_cs,
            "expected_goals_conceded": exp_gc
        }
