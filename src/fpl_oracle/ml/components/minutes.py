"""
Minutes and Starting probability component model.
Predicts P(starts), P(min60+), and expected minutes.
"""


import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent


class MinutesModel(BaseComponent):
    def __init__(self):
        super().__init__("minutes_model")
        self.clf_starts = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.clf_min60 = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.reg_minutes = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.calibrator_starts = Calibrator("isotonic")
        self.calibrator_min60 = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_starts = Y["target_starts"].values
        y_min60 = Y["target_min60"].values
        y_mins = Y["target_minutes"].values

        self.clf_starts.fit(X, y_starts)
        self.clf_min60.fit(X, y_min60)
        self.reg_minutes.fit(X, y_mins)

        # Calibrate
        prob_starts_raw = self.clf_starts.predict_proba(X)[:, 1]
        self.calibrator_starts.fit(prob_starts_raw, y_starts)

        prob_min60_raw = self.clf_min60.predict_proba(X)[:, 1]
        self.calibrator_min60.fit(prob_min60_raw, y_min60)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("MinutesModel is not fitted.")

        p_starts_raw = self.clf_starts.predict_proba(X)[:, 1]
        p_starts = self.calibrator_starts.calibrate(p_starts_raw)

        p_min60_raw = self.clf_min60.predict_proba(X)[:, 1]
        p_min60 = self.calibrator_min60.calibrate(p_min60_raw)

        exp_mins = np.clip(self.reg_minutes.predict(X), 0.0, 95.0)

        # Apply chance_of_playing adjustment
        if "chance_of_playing" in X.columns:
            cop = X["chance_of_playing"].values / 100.0
            p_starts = p_starts * cop
            p_min60 = p_min60 * cop
            exp_mins = exp_mins * cop

        return {
            "p_starts": p_starts,
            "p_min60": p_min60,
            "expected_minutes": exp_mins
        }
