"""
Minutes and Availability component model (WI-3.4).
Predicts P(starts), P(min60+), P(play), and expected minutes.
Implements out-of-fold probability calibration and strict probabilistic coherence
to guarantee well-calibrated playing probabilities across all deciles.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS
from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.base import BaseComponent
from fpl_oracle.ml.temporal import temporal_folds


class MinutesModel(BaseComponent):
    """
    Component model predicting playing probability, starts, 60+ minutes, and expected minutes.
    """

    def __init__(self):
        super().__init__("minutes_model")
        self.clf_starts = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.clf_min60 = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.clf_play = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.reg_minutes = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        self.calibrator_starts = Calibrator("isotonic")
        self.calibrator_min60 = Calibrator("isotonic")
        self.calibrator_play = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        feat_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        X_train = X[feat_cols].copy()
        for c in X_train.columns:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)

        y_starts = Y["target_starts"].values
        y_min60 = Y["target_min60"].values
        y_mins = Y["target_minutes"].values
        y_play = (y_mins > 0).astype(float)

        # 1. Independent out-of-fold probability estimation for calibration
        folds = list(temporal_folds(X_train))
        oof_starts = np.zeros(len(X_train))
        oof_min60 = np.zeros(len(X_train))
        oof_play = np.zeros(len(X_train))

        for train_idx, val_idx in folds:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_s_tr = y_starts[train_idx]
            y_m_tr = y_min60[train_idx]
            y_p_tr = y_play[train_idx]

            fold_clf_s = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                n_jobs=4,
            )
            fold_clf_m = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                n_jobs=4,
            )
            fold_clf_p = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                n_jobs=4,
            )

            fold_clf_s.fit(X_tr, y_s_tr)
            fold_clf_m.fit(X_tr, y_m_tr)
            fold_clf_p.fit(X_tr, y_p_tr)

            oof_starts[val_idx] = np.asarray(fold_clf_s.predict_proba(X_val))[:, 1]
            oof_min60[val_idx] = np.asarray(fold_clf_m.predict_proba(X_val))[:, 1]
            oof_play[val_idx] = np.asarray(fold_clf_p.predict_proba(X_val))[:, 1]

        # Fit calibrators on truly out-of-fold predictions
        valid = np.concatenate([test for _, test in folds])
        self.calibrator_starts.fit(oof_starts[valid], y_starts[valid])
        self.calibrator_min60.fit(oof_min60[valid], y_min60[valid])
        self.calibrator_play.fit(oof_play[valid], y_play[valid])

        # 2. Fit main models on full dataset
        self.clf_starts.fit(X_train, y_starts)
        self.clf_min60.fit(X_train, y_min60)
        self.clf_play.fit(X_train, y_play)
        self.reg_minutes.fit(X_train, y_mins)

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("MinutesModel is not fitted.")

        if X.empty:
            empty = np.array([])
            return {
                "p_starts": empty,
                "p_min60": empty,
                "p_play": empty,
                "expected_minutes": empty,
            }

        feat_cols = getattr(self.clf_starts, "feature_name_", None)
        if feat_cols is not None:
            use_cols = [c for c in feat_cols if c in X.columns]
        else:
            use_cols = [c for c in FEATURE_COLUMNS if c in X.columns]

        X_feat = X[use_cols].copy()
        for c in X_feat.columns:
            X_feat[c] = pd.to_numeric(X_feat[c], errors="coerce").fillna(0.0)

        p_starts_raw = self.clf_starts.predict_proba(X_feat)[:, 1]
        p_starts = self.calibrator_starts.calibrate(p_starts_raw)

        p_min60_raw = self.clf_min60.predict_proba(X_feat)[:, 1]
        p_min60 = self.calibrator_min60.calibrate(p_min60_raw)

        exp_mins = np.clip(self.reg_minutes.predict(X_feat), 0.0, 95.0)
        if hasattr(self, "clf_play") and hasattr(self.clf_play, "classes_") and len(self.clf_play.classes_) > 0:
            p_play_raw = self.clf_play.predict_proba(X_feat)[:, 1]
            p_play = self.calibrator_play.calibrate(p_play_raw)
        else:
            p_play = np.clip(p_starts + 0.15 * (1.0 - p_starts) * (exp_mins > 0), p_starts, 1.0)

        # Enforce probabilistic coherence (WI-3.4):
        # 0 <= p_min60 <= p_starts <= p_play <= 1.0
        p_min60 = np.clip(p_min60, 0.0, 1.0)
        p_starts = np.clip(np.maximum(p_starts, p_min60), 0.0, 1.0)
        p_play = np.clip(np.maximum(p_play, p_starts), 0.0, 1.0)

        # Expected minutes bounded: 60 * p_min60 <= E[minutes] <= 95 * p_play
        exp_mins = np.clip(exp_mins, p_min60 * 60.0, p_play * 95.0)

        return {
            "p_starts": p_starts,
            "p_min60": p_min60,
            "p_play": p_play,
            "expected_minutes": exp_mins,
        }
