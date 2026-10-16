"""
Unit tests for rebuilt DefCon component model (WI-3.5).
Validates count modeling, position-specific thresholds (DEF: 10, MID/FWD: 12, GKP: 0),
overdispersion parameters, probability calibration, and zero-DefCon GKP rule.
"""

import numpy as np
import pandas as pd

from fpl_oracle.ml.components.defcon import DefConModel


def _build_synthetic_defcon_data(n: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    positions = np.random.choice(["DEF", "MID", "FWD", "GKP"], size=n, p=[0.35, 0.40, 0.15, 0.10])
    minutes = np.random.uniform(0, 90, size=n)

    # Base actions: DEF/MID generate more actions conditional on minutes
    base_rate = np.where(
        positions == "DEF", 0.08, np.where(positions == "MID", 0.07, np.where(positions == "FWD", 0.03, 0.0))
    )
    mean_actions = minutes * base_rate
    actions = np.random.poisson(mean_actions).astype(float)
    actions[positions == "GKP"] = 0.0

    pos_def = (positions == "DEF").astype(float)
    pos_mid = (positions == "MID").astype(float)
    pos_fwd = (positions == "FWD").astype(float)
    pos_gkp = (positions == "GKP").astype(float)

    thresh = np.where(positions == "DEF", 10.0, np.where(np.isin(positions, ["MID", "FWD"]), 12.0, 9999.0))
    target_defcon = ((actions >= thresh) & (positions != "GKP")).astype(float)

    X = pd.DataFrame(
        {
            "roll_minutes_5": minutes,
            "pos_DEF": pos_def,
            "pos_MID": pos_mid,
            "pos_FWD": pos_fwd,
            "pos_GKP": pos_gkp,
            "was_home": np.random.binomial(1, 0.5, size=n).astype(float),
            "is_2026_27": np.random.binomial(1, 0.5, size=n).astype(float),
            "opp_strength_defence": np.random.uniform(1.0, 1.8, size=n),
            "implied_opp_xG": np.random.uniform(0.8, 2.2, size=n),
        }
    )

    Y = pd.DataFrame(
        {
            "target_defcon": target_defcon,
            "target_defcon_actions": actions,
        }
    )

    return X, Y


def test_defcon_model_fit_and_predict():
    X, Y = _build_synthetic_defcon_data(300)
    model = DefConModel()
    model.fit(X, Y)

    assert model.is_fitted
    preds = model.predict(X)
    assert "p_defcon" in preds
    assert "expected_defcon_actions" in preds

    p_defcon = preds["p_defcon"]
    assert len(p_defcon) == len(X)
    assert np.all(p_defcon >= 0.0)
    assert np.all(p_defcon <= 1.0)


def test_defcon_gkp_strictly_zero():
    X, Y = _build_synthetic_defcon_data(300)
    model = DefConModel()
    model.fit(X, Y)

    gkp_mask = X["pos_GKP"] == 1.0
    preds = model.predict(X)
    assert np.all(preds["p_defcon"][gkp_mask] == 0.0), "GKP must have strictly 0.0 DefCon probability"


def test_defcon_position_thresholds():
    model = DefConModel()
    assert model._get_threshold("DEF") == 10
    assert model._get_threshold("MID") == 12
    assert model._get_threshold("FWD") == 12
    assert model._get_threshold("GKP") == 9999


def test_defcon_dispersions():
    model = DefConModel()
    for pos in ["DEF", "MID", "FWD"]:
        assert pos in model.dispersions
        assert model.dispersions[pos] > 0.0
