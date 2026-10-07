"""
Cards and Saves component model.
Predicts goalkeeper saves, disciplinary card/own-goal/penalty deductions,
and goalkeeper penalty saves from actual historical event labels.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.components.base import BaseComponent


class CardsSavesModel(BaseComponent):
    def __init__(self):
        super().__init__("cards_saves_model")
        self.reg_saves = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.reg_cards = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        self.reg_pen_saves = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.03, num_leaves=15, random_state=42, verbosity=-1, n_jobs=4
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_saves = Y["target_saves"].values

        if "target_card_deduction" in Y.columns:
            y_cards = Y["target_card_deduction"].values
        else:
            y_cards = (
                Y.get("yellow_cards", pd.Series(0, index=Y.index)).values * 1.0
                + Y.get("red_cards", pd.Series(0, index=Y.index)).values * 3.0
            )

        if "target_penalties_saved" in Y.columns:
            y_pen_saves = Y["target_penalties_saved"].values
        else:
            y_pen_saves = np.zeros(len(X))

        self.reg_saves.fit(X, y_saves)
        self.reg_cards.fit(X, y_cards)
        self.reg_pen_saves.fit(X, y_pen_saves)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("CardsSavesModel is not fitted.")

        exp_saves = np.clip(self.reg_saves.predict(X), 0.0, 15.0)
        exp_pen_saves = np.clip(self.reg_pen_saves.predict(X), 0.0, 1.0)

        if "pos_GKP" in X.columns:
            gkp_mask = X["pos_GKP"].values
            exp_saves = exp_saves * gkp_mask
            exp_pen_saves = exp_pen_saves * gkp_mask

        # Expected disciplinary/negative deduction in points (typically 0.05 to 0.40 pts per match)
        exp_cards = np.clip(self.reg_cards.predict(X), 0.0, 3.5)

        return {
            "expected_saves": exp_saves,
            "expected_card_deduction": exp_cards,
            "expected_penalties_saved": exp_pen_saves,
        }
