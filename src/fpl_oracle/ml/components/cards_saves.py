"""
Cards and Saves component model.
Predicts goalkeeper saves and disciplinary card deductions from actual ground truth disciplinary labels.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from fpl_oracle.ml.components.base import BaseComponent


class CardsSavesModel(BaseComponent):
    def __init__(self):
        super().__init__("cards_saves_model")
        self.reg_saves = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1
        )
        self.reg_cards = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_saves = Y["target_saves"].values

        # Correct ground truth disciplinary target: yellow cards (-1) + red cards (-3)
        if "target_card_deduction" in Y.columns:
            y_cards = Y["target_card_deduction"].values
        else:
            y_cards = (
                Y.get("yellow_cards", pd.Series(0, index=Y.index)).values * 1.0
                + Y.get("red_cards", pd.Series(0, index=Y.index)).values * 3.0
            )

        self.reg_saves.fit(X, y_saves)
        self.reg_cards.fit(X, y_cards)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("CardsSavesModel is not fitted.")

        exp_saves = np.clip(self.reg_saves.predict(X), 0.0, 15.0)
        if "pos_GKP" in X.columns:
            # Outfield players do not make goalkeeper saves
            exp_saves = exp_saves * X["pos_GKP"].values

        # Expected disciplinary deduction in points (typically 0.05 to 0.40 pts per match)
        exp_cards = np.clip(self.reg_cards.predict(X), 0.0, 3.5)

        return {"expected_saves": exp_saves, "expected_card_deduction": exp_cards}
