"""
Probability calibration using Isotonic Regression and Platt scaling.
Ensures predicted probabilities (e.g. P(starts), P(clean_sheet)) are well-calibrated.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p)).reshape(-1, 1)


class Calibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(y_prob, y_true)
            self.model = iso
        else:  # Platt scaling on the logit: smooth and monotone, no plateau on thin top bins
            lr = LogisticRegression()
            lr.fit(_logit(y_prob), y_true)
            self.model = lr

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.clip(y_prob, 0.0, 1.0)
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(y_prob), 0.0, 1.0)
        else:
            return self.model.predict_proba(_logit(y_prob))[:, 1]
