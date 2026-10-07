"""
Minutes and Starting probability component model.
Predicts P(starts), P(min60+), and expected minutes.
Implements out-of-fold probability calibration to prevent in-sample overfitting.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent
from fpl_oracle.ml.temporal import temporal_folds


class MinutesModel(BaseComponent):
    def __init__(self):
        super().__init__("minutes_model")
        self.clf_starts = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.clf_min60 = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.reg_minutes = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.calibrator_starts = Calibrator("isotonic")
        self.calibrator_min60 = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_starts = Y["target_starts"].values
        y_min60 = Y["target_min60"].values
        y_mins = Y["target_minutes"].values

        # 1. Independent out-of-fold probability estimation for calibration
        folds = list(temporal_folds(X))
        oof_starts = np.zeros(len(X))
        oof_min60 = np.zeros(len(X))

        for train_idx, val_idx in folds:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_s_tr = y_starts[train_idx]
            y_m_tr = y_min60[train_idx]

            fold_clf_s = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
            )
            fold_clf_m = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
            )

            fold_clf_s.fit(X_tr, y_s_tr)
            fold_clf_m.fit(X_tr, y_m_tr)

            oof_starts[val_idx] = np.asarray(fold_clf_s.predict_proba(X_val))[:, 1]
            oof_min60[val_idx] = np.asarray(fold_clf_m.predict_proba(X_val))[:, 1]

        # Fit calibrators on truly out-of-fold predictions
        valid = np.concatenate([test for _, test in folds])
        self.calibrator_starts.fit(oof_starts[valid], y_starts[valid])
        self.calibrator_min60.fit(oof_min60[valid], y_min60[valid])

        # 2. Fit main models on full dataset
        self.clf_starts.fit(X, y_starts)
        self.clf_min60.fit(X, y_min60)
        self.reg_minutes.fit(X, y_mins)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("MinutesModel is not fitted.")

        p_starts_raw = self.clf_starts.predict_proba(X)[:, 1]
        p_starts = self.calibrator_starts.calibrate(p_starts_raw)

        p_min60_raw = self.clf_min60.predict_proba(X)[:, 1]
        p_min60 = self.calibrator_min60.calibrate(p_min60_raw)

        exp_mins = np.clip(self.reg_minutes.predict(X), 0.0, 95.0)

        # Enforce probabilistic coherence
        # A player playing 60+ minutes implies expected minutes >= 60 * p_min60
        exp_mins = np.maximum(exp_mins, p_min60 * 60.0)

        return {"p_starts": p_starts, "p_min60": p_min60, "expected_minutes": exp_mins}
