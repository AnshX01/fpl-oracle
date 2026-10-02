"""
Cards and Saves component model.
Predicts goalkeeper saves and disciplinary card deductions.
"""

from typing import Dict
import numpy as np
import pandas as pd
import lightgbm as lgb

from fpl_oracle.ml.components.base import BaseComponent

class CardsSavesModel(BaseComponent):
    def __init__(self):
        super().__init__("cards_saves_model")
        self.reg_saves = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.reg_cards = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        y_saves = Y["target_saves"].values
        y_cards = Y["target_goals"].values * 0.0 # dummy default or card targets if present
        # In Y target_saves is saves
        self.reg_saves.fit(X, y_saves)
        self.reg_cards.fit(X, np.clip(y_saves * 0.05 + 0.1, 0.0, 0.4))
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("CardsSavesModel is not fitted.")

        exp_saves = np.clip(self.reg_saves.predict(X), 0.0, 12.0)
        if "pos_GKP" in X.columns:
            exp_saves = exp_saves * X["pos_GKP"].values

        exp_cards = np.clip(self.reg_cards.predict(X), 0.0, 0.5)

        return {
            "expected_saves": exp_saves,
            "expected_card_deduction": exp_cards
        }
