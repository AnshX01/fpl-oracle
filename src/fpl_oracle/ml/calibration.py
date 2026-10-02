"""
Probability calibration using Isotonic Regression and Platt scaling.
Ensures predicted probabilities (e.g. P(starts), P(clean_sheet)) are well-calibrated.
"""

from typing import Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(y_prob, y_true)
        else: # platt scaling
            self.model = LogisticRegression()
            self.model.fit(y_prob.reshape(-1, 1), y_true)

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.clip(y_prob, 0.0, 1.0)
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(y_prob), 0.0, 1.0)
        else:
            return self.model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
