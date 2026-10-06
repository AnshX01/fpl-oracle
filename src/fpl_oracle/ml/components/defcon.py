"""
Defensive Contribution (DefCon) component model for 2026/27 rules.
Predicts the probability that an outfield player achieves the 2-point DefCon reward.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent


class DefConModel(BaseComponent):
    def __init__(self):
        super().__init__("defcon_model")
        self.clf = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1
        )
        self.calibrator = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # We give higher sample weight to 2026/27 data where DefCon is actively scored
        weights = np.ones(len(X))
        if "is_2026_27" in X.columns:
            weights = np.where(X["is_2026_27"] > 0, 5.0, 1.0)

        y = Y["target_defcon"].values
        self.clf.fit(X, y, sample_weight=weights)

        probs_raw = self.clf.predict_proba(X)[:, 1]
        self.calibrator.fit(probs_raw, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("DefConModel is not fitted.")

        probs_raw = self.clf.predict_proba(X)[:, 1]
        p_defcon = self.calibrator.calibrate(probs_raw)

        # Goalkeepers do not receive DefCon per 2026/27 rules
        if "pos_GKP" in X.columns:
            p_defcon = p_defcon * (1.0 - X["pos_GKP"].values)

        return {"p_defcon": p_defcon}
