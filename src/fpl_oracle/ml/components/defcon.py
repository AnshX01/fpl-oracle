"""
Defensive Contribution (DefCon) component model for 2026/27 rules.
Predicts the probability that an outfield player achieves the 2-point DefCon reward.
Implements out-of-fold probability calibration to prevent in-sample overfitting.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent
from fpl_oracle.ml.temporal import temporal_folds


class DefConModel(BaseComponent):
    def __init__(self):
        super().__init__("defcon_model")
        self.clf = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.calibrator = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        weights = np.ones(len(X))
        if "is_2026_27" in X.columns:
            weights = np.where(X["is_2026_27"] > 0, 5.0, 1.0)

        y = Y["target_defcon"].values

        # Independent out-of-fold probability estimation for calibration
        folds = list(temporal_folds(X))
        oof_probs = np.zeros(len(X))

        for train_idx, val_idx in folds:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y[train_idx]
            w_tr = weights[train_idx]

            fold_clf = lgb.LGBMClassifier(
                n_estimators=80, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
            )
            fold_clf.fit(X_tr, y_tr, sample_weight=w_tr)
            oof_probs[val_idx] = np.asarray(fold_clf.predict_proba(X_val))[:, 1]

        # Fit calibrator on truly out-of-fold predictions
        valid = np.concatenate([test for _, test in folds])
        self.calibrator.fit(oof_probs[valid], y[valid])

        # Fit main classifier on full dataset
        self.clf.fit(X, y, sample_weight=weights)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("DefConModel is not fitted.")

        probs_raw = self.clf.predict_proba(X)[:, 1]
        p_defcon = self.calibrator.calibrate(probs_raw)

        # Goalkeepers do not receive DefCon per official 2026/27 rules
        if "pos_GKP" in X.columns:
            p_defcon = p_defcon * (1.0 - X["pos_GKP"].values)

        return {"p_defcon": p_defcon}
