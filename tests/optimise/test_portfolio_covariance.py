"""
Unit tests for Markowitz Squad Portfolio Covariance & Correlation Engine.
Verifies mathematical identities, defensive double-up variance amplification,
and analytical portfolio volatility bounds.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.optimise.covariance import SquadPortfolioOptimizer, ledoit_wolf_constant_correlation
from fpl_oracle.sim.match_sim import MatchSimulator


@pytest.fixture
def sample_squad_df() -> pd.DataFrame:
    """Creates a realistic 15-player squad with Arsenal defensive double-up and City/Liverpool attack."""
    rows = [
        # Starters (11)
        {"element": 1, "web_name": "GKP_1", "team": 1, "position": "GKP", "is_starter": True, "expected_points": 4.5},
        {"element": 2, "web_name": "DEF_1A", "team": 1, "position": "DEF", "is_starter": True, "expected_points": 4.2},
        {"element": 3, "web_name": "DEF_1B", "team": 1, "position": "DEF", "is_starter": True, "expected_points": 4.0},
        {"element": 4, "web_name": "DEF_2", "team": 2, "position": "DEF", "is_starter": True, "expected_points": 3.8},
        {"element": 5, "web_name": "DEF_3", "team": 3, "position": "DEF", "is_starter": True, "expected_points": 3.5},
        {"element": 6, "web_name": "MID_2A", "team": 2, "position": "MID", "is_starter": True, "expected_points": 6.5},
        {"element": 7, "web_name": "MID_2B", "team": 2, "position": "MID", "is_starter": True, "expected_points": 5.8},
        {"element": 8, "web_name": "MID_4", "team": 4, "position": "MID", "is_starter": True, "expected_points": 5.2},
        {"element": 9, "web_name": "MID_5", "team": 5, "position": "MID", "is_starter": True, "expected_points": 4.8},
        {
            "element": 10,
            "web_name": "FWD_2",
            "team": 2,
            "position": "FWD",
            "is_starter": True,
            "expected_points": 7.5,
            "is_captain": True,
        },
        {"element": 11, "web_name": "FWD_6", "team": 6, "position": "FWD", "is_starter": True, "expected_points": 5.0},
        # Bench (4)
        {"element": 12, "web_name": "GKP_B", "team": 7, "position": "GKP", "is_starter": False, "expected_points": 2.5},
        {"element": 13, "web_name": "DEF_B", "team": 8, "position": "DEF", "is_starter": False, "expected_points": 2.0},
        {"element": 14, "web_name": "MID_B", "team": 9, "position": "MID", "is_starter": False, "expected_points": 2.2},
        {
            "element": 15,
            "web_name": "FWD_B",
            "team": 10,
            "position": "FWD",
            "is_starter": False,
            "expected_points": 1.8,
        },
    ]
    return pd.DataFrame(rows)


def test_squad_covariance_shape_and_diagonal_invariants(sample_squad_df):
    optimizer = SquadPortfolioOptimizer(MatchSimulator(n_sims=1000, seed=42))
    res = optimizer.analyze_squad_covariance(sample_squad_df)

    # 1. 15x15 Matrix dimensions
    assert len(res.covariance_matrix) == 15
    assert len(res.correlation_matrix) == 15
    assert all(len(row) == 15 for row in res.covariance_matrix)
    assert all(len(row) == 15 for row in res.correlation_matrix)

    # 2. Correlation diagonal must be strictly 1.0
    for i in range(15):
        assert abs(res.correlation_matrix[i][i] - 1.0) < 1e-5

    # 3. Symmetry
    for i in range(15):
        for j in range(15):
            assert abs(res.covariance_matrix[i][j] - res.covariance_matrix[j][i]) < 1e-5
            assert abs(res.correlation_matrix[i][j] - res.correlation_matrix[j][i]) < 1e-5


def test_markowitz_variance_identity(sample_squad_df):
    """Verifies that w^T Sigma w equals sum(w_i^2 sigma_i^2) + 2*sum(w_i w_j cov_ij)."""
    optimizer = SquadPortfolioOptimizer(MatchSimulator(n_sims=1000, seed=123))
    res = optimizer.analyze_squad_covariance(sample_squad_df)

    w = np.array([m.weight for m in res.player_metrics])
    sigma = np.array(res.covariance_matrix)

    computed_var = float(w @ sigma @ w)
    reported_var = res.portfolio_variance

    # Portfolio variance matches within rounding precision
    assert abs(computed_var - reported_var) < 0.5

    # Covariance delta matches
    ind_var = res.independent_volatility**2
    assert abs((reported_var - ind_var) - res.covariance_delta_variance) < 0.5


def test_defensive_double_up_positive_covariance(sample_squad_df):
    """Teammates in defence (Team 1 GKP, DEF_1A, DEF_1B) must show positive covariance."""
    optimizer = SquadPortfolioOptimizer(MatchSimulator(n_sims=1500, seed=999))
    res = optimizer.analyze_squad_covariance(sample_squad_df)

    # Elements 1, 2, 3 are all Team 1 backline
    cov_gkp_def = res.covariance_matrix[0][1]
    corr_gkp_def = res.correlation_matrix[0][1]
    cov_def_def = res.covariance_matrix[1][2]
    corr_def_def = res.correlation_matrix[1][2]

    assert cov_gkp_def >= 0.0, f"Expected non-negative covariance for GKP+DEF, got {cov_gkp_def}"
    assert corr_gkp_def >= 0.0, f"Expected non-negative correlation for GKP+DEF, got {corr_gkp_def}"
    assert cov_def_def >= 0.0, f"Expected non-negative covariance for DEF+DEF, got {cov_def_def}"
    assert corr_def_def >= 0.0, f"Expected non-negative correlation for DEF+DEF, got {corr_def_def}"

    # Defensive stack risk is positive
    assert res.defensive_stack_risk >= 0.0


def test_captain_multipliers_propagate_to_weights(sample_squad_df):
    optimizer = SquadPortfolioOptimizer(MatchSimulator(n_sims=500, seed=7))

    # Standard captain (2x)
    res_reg = optimizer.analyze_squad_covariance(sample_squad_df, captain_id=10, is_triple_captain=False)
    cap_metric = next(m for m in res_reg.player_metrics if m.element == 10)
    assert cap_metric.weight == 2.0
    assert cap_metric.is_captain is True

    # Triple captain (3x)
    res_tc = optimizer.analyze_squad_covariance(sample_squad_df, captain_id=10, is_triple_captain=True)
    tc_metric = next(m for m in res_tc.player_metrics if m.element == 10)
    assert tc_metric.weight == 3.0
    assert tc_metric.is_captain is True
    assert res_tc.portfolio_volatility > res_reg.portfolio_volatility


def test_serialization_and_to_dict(sample_squad_df):
    optimizer = SquadPortfolioOptimizer(MatchSimulator(n_sims=500, seed=1))
    res = optimizer.analyze_squad_covariance(sample_squad_df)
    d = res.to_dict()

    assert "player_metrics" in d
    assert "covariance_matrix" in d
    assert "correlation_matrix" in d
    assert "portfolio_expected_points" in d
    assert "portfolio_volatility" in d
    assert "defensive_stack_risk" in d
    assert "sharpe_ratio" in d
    assert isinstance(d["summary"], str)
    assert len(d["player_metrics"]) == 15
    assert "shrinkage_intensity" in d


def test_ledoit_wolf_shrinkage_guarantees_positive_definite():
    """Verifies that Ledoit-Wolf shrinkage produces strictly positive-definite covariance matrices."""
    # Create collinear Monte Carlo draws
    rng = np.random.default_rng(42)
    base = rng.normal(size=(1, 500))
    collinear_draws = np.repeat(base, 10, axis=0) + 1e-4 * rng.normal(size=(10, 500))

    cov, shrink = ledoit_wolf_constant_correlation(collinear_draws)

    # 1. Shrinkage intensity is bounded in [0, 1]
    assert 0.0 <= shrink <= 1.0

    # 2. Strict positive definiteness: all eigenvalues > 0
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0), f"Expected strictly positive eigenvalues, got min {eigvals.min()}"

    # 3. Symmetry
    assert np.allclose(cov, cov.T, atol=1e-8)


def test_opponent_hedge_detection_gating():
    """
    Opponent hedge detection must be strictly gated on clubs facing each other in the gameweek,
    preventing Monte Carlo sampling noise from creating false hedges between unrelated clubs.
    """
    rows = [
        {"element": 1, "web_name": "DEF_T1", "team": 1, "position": "DEF", "is_starter": True, "expected_points": 4.0},
        {"element": 2, "web_name": "FWD_T2", "team": 2, "position": "FWD", "is_starter": True, "expected_points": 6.0},
    ]
    df = pd.DataFrame(rows)
    opt = SquadPortfolioOptimizer()

    class MockNegativeCorrelatedSim:
        n_sims = 100
        player_ids = [1, 2]
        # Anti-correlated scores
        scores = np.array(
            [
                [0.0, 6.0] * 50,
                [8.0, 0.0] * 50,
            ]
        )

    mock_sim = MockNegativeCorrelatedSim()

    # Case A: Teams do NOT play each other (e.g. Team 1 vs 3, Team 2 vs 4) -> No hedge
    res_no_fixture = opt.analyze_squad_covariance(
        df,
        sim_result=mock_sim,
        opp_team={1: 3, 2: 4},
    )
    assert res_no_fixture.opponent_hedge_benefit == 0.0
    assert not any(p.relationship == "OPPONENT_HEDGE" for p in res_no_fixture.top_hedged_pairs)

    # Case B: Teams DO play each other (Team 1 vs Team 2) -> Legitimate opponent hedge detected
    res_fixture = opt.analyze_squad_covariance(
        df,
        sim_result=mock_sim,
        opp_team={1: 2, 2: 1},
    )
    assert res_fixture.opponent_hedge_benefit > 0.0
    assert any(p.relationship == "OPPONENT_HEDGE" for p in res_fixture.top_hedged_pairs)
