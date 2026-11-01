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
            lr = LogisticRegression(C=1e5, solver="lbfgs")
            lr.fit(_logit(y_prob), y_true)
            # Enforce monotonicity: Platt scaling requires a non-negative slope (a >= 0)
            if lr.coef_[0, 0] < 0:
                lr.coef_[0, 0] = 0.0
                base_rate = float(np.clip(np.mean(y_true), 1e-6, 1.0 - 1e-6))
                lr.intercept_[0] = float(np.log(base_rate / (1.0 - base_rate)))
            self.model = lr

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.clip(y_prob, 0.0, 1.0)
        y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.model.predict(y_prob), 0.0, 1.0)
        else:
            return self.model.predict_proba(_logit(y_prob))[:, 1]

    def brier_score(self, y_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes Mean Squared Error (Brier score) between predicted probabilities and binary outcomes:
        Brier = mean((y_prob - y_true)^2)
        Lower is better (0.0 is perfect calibration and sharpness).
        """
        p = np.asarray(y_prob, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        if len(p) == 0:
            return 0.0
        return float(np.mean((p - y) ** 2))

    def log_loss(self, y_prob: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
        """
        Computes negative log-likelihood (binary cross-entropy) with numerical clipping:
        Log-Loss = -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
        """
        p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
        y = np.asarray(y_true, dtype=np.float64)
        if len(p) == 0:
            return 0.0
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def evaluate(self, y_prob: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """
        Evaluates calibration performance:
        Returns {'brier_score': float, 'log_loss': float}.
        """
        return {
            "brier_score": self.brier_score(y_prob, y_true),
            "log_loss": self.log_loss(y_prob, y_true),
        }
