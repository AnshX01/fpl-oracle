"""
Unit tests for ML Component Models:
- DefendingModel
- MinutesModel
- BonusModel
- CardsSavesModel
- AttackingModel
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel


@pytest.fixture
def synthetic_component_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "pos_DEF": np.random.choice([0.0, 1.0], size=n),
            "pos_MID": np.random.choice([0.0, 1.0], size=n),
            "pos_FWD": np.random.choice([0.0, 1.0], size=n),
            "pos_GKP": np.random.choice([0.0, 1.0], size=n),
            "roll_minutes_3": np.random.uniform(0, 90, size=n),
            "roll_minutes_5": np.random.uniform(0, 90, size=n),
            "roll_minutes_8": np.random.uniform(0, 90, size=n),
            "roll_starts_ratio_5": np.random.uniform(0, 1, size=n),
            "roll_min60_ratio_5": np.random.uniform(0, 1, size=n),
            "std_minutes_per_gw": np.random.uniform(0, 30, size=n),
            "chance_of_playing": np.full(n, 100.0),
            "roll_points_3": np.random.uniform(0, 10, size=n),
            "roll_points_5": np.random.uniform(0, 10, size=n),
            "roll_points_8": np.random.uniform(0, 10, size=n),
            "roll_xG_3": np.random.uniform(0, 1, size=n),
            "roll_xG_5": np.random.uniform(0, 1, size=n),
            "roll_xG_8": np.random.uniform(0, 1, size=n),
            "roll_xA_3": np.random.uniform(0, 1, size=n),
            "roll_xA_5": np.random.uniform(0, 1, size=n),
            "roll_xA_8": np.random.uniform(0, 1, size=n),
            "roll_xGI_5": np.random.uniform(0, 2, size=n),
            "roll_goals_5": np.random.uniform(0, 2, size=n),
            "roll_assists_5": np.random.uniform(0, 2, size=n),
            "roll_xGC_5": np.random.uniform(0.5, 2.5, size=n),
            "roll_clean_sheets_5": np.random.uniform(0, 0.6, size=n),
            "roll_saves_5": np.random.uniform(0, 5, size=n),
            "roll_goals_conceded_5": np.random.uniform(0.5, 2.5, size=n),
            "roll_defcon_5": np.random.uniform(0, 0.5, size=n),
            "roll_ict_5": np.random.uniform(0, 20, size=n),
            "roll_bps_5": np.random.uniform(5, 35, size=n),
            "roll_cards_5": np.random.uniform(0, 0.4, size=n),
            "implied_team_xG": np.random.uniform(0.8, 2.5, size=n),
            "implied_team_cs_prob": np.random.uniform(0.1, 0.5, size=n),
            "implied_opp_xG": np.random.uniform(0.8, 2.5, size=n),
            "implied_opp_cs_prob": np.random.uniform(0.1, 0.5, size=n),
            "was_home": np.random.choice([0.0, 1.0], size=n),
            "value": np.random.uniform(4.0, 14.0, size=n),
            "is_2026_27": np.zeros(n),
        }
    )

    # Attach meta for temporal folds
    meta = pd.DataFrame(
        {
            "season": ["2025-26"] * n,
            "round": [(i % 20) + 1 for i in range(n)],
            "position": np.random.choice(["DEF", "MID", "FWD", "GKP"], size=n),
        }
    )
    X.attrs["meta"] = meta

    Y = pd.DataFrame(
        {
            "target_minutes": np.random.uniform(0, 90, size=n),
            "target_starts": np.random.choice([0.0, 1.0], size=n),
            "target_min60": np.random.choice([0.0, 1.0], size=n),
            "target_goals": np.random.poisson(0.3, size=n),
            "target_assists": np.random.poisson(0.2, size=n),
            "target_clean_sheet": np.random.binomial(1, 0.3, size=n),
            "target_goals_conceded": np.random.poisson(1.2, size=n),
            "target_saves": np.random.poisson(2.5, size=n),
            "target_card_deduction": np.random.choice([0.0, 1.0, 3.0], size=n, p=[0.85, 0.12, 0.03]),
            "target_bonus": np.random.choice([0.0, 1.0, 2.0, 3.0], size=n, p=[0.7, 0.15, 0.1, 0.05]),
            "target_defcon": np.random.binomial(1, 0.15, size=n),
            "target_defcon_actions": np.random.poisson(5.0, size=n),
        }
    )
    return X, Y


def test_defending_model_fit_predict(synthetic_component_data):
    X, Y = synthetic_component_data
    model = DefendingModel()
    with pytest.raises(RuntimeError):
        model.predict(X)

    model.fit(X, Y)
    preds = model.predict(X)
    assert "p_clean_sheet" in preds
    assert "expected_goals_conceded" in preds
    assert len(preds["p_clean_sheet"]) == len(X)
    assert np.all(preds["p_clean_sheet"] >= 0.0)
    assert np.all(preds["p_clean_sheet"] <= 1.0)


def test_minutes_model_fit_predict(synthetic_component_data):
    X, Y = synthetic_component_data
    model = MinutesModel()
    with pytest.raises(RuntimeError):
        model.predict(X)

    model.fit(X, Y)
    preds = model.predict(X)
    assert "expected_minutes" in preds
    assert "p_min60" in preds
    assert "p_starts" in preds
    assert len(preds["expected_minutes"]) == len(X)


def test_bonus_model_fit_predict(synthetic_component_data):
    X, Y = synthetic_component_data
    model = BonusModel()
    with pytest.raises(RuntimeError):
        model.predict(X)

    model.fit(X, Y)
    preds = model.predict(X)
    assert "expected_bonus" in preds
    assert len(preds["expected_bonus"]) == len(X)


def test_cards_saves_model_fit_predict(synthetic_component_data):
    X, Y = synthetic_component_data
    model = CardsSavesModel()
    with pytest.raises(RuntimeError):
        model.predict(X)

    model.fit(X, Y)
    preds = model.predict(X)
    assert "expected_saves" in preds
    assert "expected_card_deduction" in preds
    assert len(preds["expected_saves"]) == len(X)


def test_attacking_model_fit_predict(synthetic_component_data):
    X, Y = synthetic_component_data
    model = AttackingModel()
    with pytest.raises(RuntimeError):
        model.predict(X)

    model.fit(X, Y)
    preds = model.predict(X)
    assert "expected_goals" in preds
    assert "expected_assists" in preds
    assert len(preds["expected_goals"]) == len(X)
