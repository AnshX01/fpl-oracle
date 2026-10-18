"""
Defending component model.
Predicts clean sheet probability P(CS | player plays 60+ minutes) and expected goals conceded.
The ensemble multiplies P(CS) by P(60+ minutes), so the label must be conditional on playing 60+ minutes
(clean sheets only score for 60+ minute players): training on all rows would discount minutes twice.
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
            n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.reg_gc = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.calibrator_cs = Calibrator("isotonic")

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_gc = Y["target_goals_conceded"].values
        self.reg_gc.fit(X, y_gc)

        # Clean-sheet classifier + calibrator: only rows where the player completed 60+ minutes
        played_60 = Y["target_min60"].to_numpy() > 0 if "target_min60" in Y.columns else np.ones(len(X), dtype=bool)
        X_cs = X[played_60]
        y_cs = Y["target_clean_sheet"].to_numpy()[played_60]
        self.clf_cs.fit(X_cs, y_cs)

        if len(X_cs) >= 6:
            from fpl_oracle.ml.temporal import temporal_folds

            folds = list(temporal_folds(X_cs))
            prob_oof = np.zeros(len(X_cs))
            for train_idx, val_idx in folds:
                clf_fold = lgb.LGBMClassifier(
                    n_estimators=self.clf_cs.n_estimators,
                    learning_rate=self.clf_cs.learning_rate,
                    num_leaves=self.clf_cs.num_leaves,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=4,
                )
                clf_fold.fit(X_cs.iloc[train_idx], y_cs[train_idx])
                prob_oof[val_idx] = np.asarray(clf_fold.predict_proba(X_cs.iloc[val_idx]))[:, 1]
            valid = np.concatenate([test for _, test in folds])
            self.calibrator_cs.fit(prob_oof[valid], y_cs[valid])
        else:
            raise ValueError("Insufficient temporal rows for clean-sheet calibration")

        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("DefendingModel is not fitted.")

        p_cs_raw = self.clf_cs.predict_proba(X)[:, 1]
        p_cs = self.calibrator_cs.calibrate(p_cs_raw)
        exp_gc = np.clip(self.reg_gc.predict(X), 0.0, 5.0)

        return {"p_clean_sheet": p_cs, "expected_goals_conceded": exp_gc}
