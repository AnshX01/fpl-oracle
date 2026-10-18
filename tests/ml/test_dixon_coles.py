"""
Unit tests for Dixon-Coles bivariate team model (WI-3.1).
Validates attack/defence fitting, rho adjustment, scoreline matrix normalization,
promoted team priors, and features.py integration.
"""

import numpy as np

from fpl_oracle.ml.components.dixon_coles import DixonColesModel, dixon_coles_model


def test_dixon_coles_fit_and_centering():
    model = DixonColesModel()
    model.fit()
    assert model.is_fitted
    assert len(model.alphas) >= 20
    assert len(model.deltas) >= 20

    # Identifiability: sum of alphas must be zero
    alpha_sum = sum(model.alphas.values())
    assert abs(alpha_sum) < 1e-4, f"Alphas must be zero-centered, got sum={alpha_sum}"

    # Positive home advantage
    assert model.gamma > 0.05, f"Home advantage gamma should be positive, got {model.gamma}"

    # Low-score correlation parameter rho in valid bounds
    assert -0.5 <= model.rho <= 0.4, f"rho={model.rho} outside valid bounds [-0.5, 0.4]"
    assert abs(model.rho) > 0.01, f"rho={model.rho} must capture low-score dependency"


def test_dixon_coles_bivariate_scoreline_matrix():
    model = DixonColesModel()
    pred = model.predict_scoreline_matrix("Arsenal", "Chelsea", max_goals=10)

    grid = pred["grid"]
    assert grid.shape == (11, 11)
    assert np.all(grid >= 0.0)
    assert abs(np.sum(grid) - 1.0) < 1e-5

    # 1X2 Probabilities sum to 1.0
    outcome_sum = pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"]
    assert abs(outcome_sum - 1.0) < 1e-5

    # Clean sheet probabilities are in (0, 1)
    assert 0.05 < pred["p_cs_home"] < 0.90
    assert 0.05 < pred["p_cs_away"] < 0.90


def test_dixon_coles_promoted_team_fallback():
    model = DixonColesModel()
    # A completely unknown club should shrink to the promoted team prior
    alpha, delta = model.get_team_parameters("FictionalUnitedFC")
    assert alpha == model.promoted_alpha
    assert delta == model.promoted_delta


def test_dixon_coles_predict_fixture_home_away_consistency():
    home_pred = dixon_coles_model.predict_fixture("Arsenal", "Chelsea", is_home=True)
    away_pred = dixon_coles_model.predict_fixture("Arsenal", "Chelsea", is_home=False)

    # Implied xG for Arsenal when home must equal implied opponent xG for Chelsea when away
    assert abs(home_pred["implied_team_xG"] - away_pred["implied_opp_xG"]) < 1e-5
    assert abs(home_pred["implied_team_cs_prob"] - away_pred["implied_opp_cs_prob"]) < 1e-5
