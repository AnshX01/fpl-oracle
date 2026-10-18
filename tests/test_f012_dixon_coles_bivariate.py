"""
Tests for F-012 / WI-3.1: Bivariate Dixon-Coles (1997) Model.
Verifies:
1. When rho == 0, the bivariate scoreline probability matrix equals independent Poisson products exactly.
2. Parameter recovery on simulated match scorelines recovers ground-truth parameters (alpha, delta, gamma, rho).
"""

import math

import numpy as np
import pandas as pd

from fpl_oracle.ml.components.dixon_coles import DixonColesModel


def test_dixon_coles_rho_zero_equals_independent_poisson():
    """
    WI-3.1 / Item C.(6):
    Assert that when rho = 0, Dixon-Coles bivariate distribution strictly equals
    independent Poisson products across all scorelines (0..10 x 0..10).
    """
    model = DixonColesModel(default_rho=0.0)
    model.rho = 0.0
    model.alphas = {"TeamA": 0.20, "TeamB": -0.10}
    model.deltas = {"TeamA": 0.15, "TeamB": -0.05}
    model.gamma = 0.18
    model.teams = ["TeamA", "TeamB"]
    model.is_fitted = True

    # Entire scoreline matrix check
    pred = model.predict_scoreline_matrix("TeamA", "TeamB", max_goals=10)
    grid = pred["grid"]
    lam = pred["lambda"]
    mu = pred["mu"]

    # Independent Poisson grid
    pois_grid = np.zeros((11, 11), dtype=float)
    for x in range(11):
        px = math.exp(-lam) * (lam**x) / math.factorial(x)
        for y in range(11):
            py = math.exp(-mu) * (mu**y) / math.factorial(y)
            pois_grid[x, y] = px * py
    pois_grid /= np.sum(pois_grid)

    assert np.allclose(grid, pois_grid, atol=1e-5), (
        "When rho=0, Dixon-Coles grid must match independent Poisson grid across all (x, y) cells"
    )


def test_dixon_coles_parameter_recovery_simulated():
    """
    WI-3.1 / Item C.(6):
    Assert parameter recovery on synthetic bivariate data with known true parameters.
    Simulates matches from a 4-team league and verifies convergence within sampling tolerances.
    """
    np.random.seed(42)
    teams = ["Team1", "Team2", "Team3", "Team4"]

    # True parameter definitions with sufficient separation above sampling noise
    true_alphas = {"Team1": 0.40, "Team2": 0.15, "Team3": -0.15, "Team4": -0.40}
    true_deltas = {"Team1": 0.30, "Team2": 0.10, "Team3": -0.10, "Team4": -0.30}
    true_gamma = 0.22
    true_rho = -0.12

    # Simulate 800 matches
    sim_rows = []
    base_date = pd.Timestamp("2025-08-01")

    for m_idx in range(800):
        h_team, a_team = np.random.choice(teams, size=2, replace=False)
        lam = math.exp(true_alphas[h_team] - true_deltas[a_team] + true_gamma)
        mu = math.exp(true_alphas[a_team] - true_deltas[h_team])

        # Bivariate Dixon-Coles probabilities for 0..6
        probs = np.zeros((7, 7))
        for x in range(7):
            px = math.exp(-lam) * (lam**x) / math.factorial(x)
            for y in range(7):
                py = math.exp(-mu) * (mu**y) / math.factorial(y)
                tau = 1.0
                if x == 0 and y == 0:
                    tau = max(1e-6, 1.0 - lam * mu * true_rho)
                elif x == 0 and y == 1:
                    tau = max(1e-6, 1.0 + lam * true_rho)
                elif x == 1 and y == 0:
                    tau = max(1e-6, 1.0 + mu * true_rho)
                elif x == 1 and y == 1:
                    tau = max(1e-6, 1.0 - true_rho)
                probs[x, y] = max(0.0, tau * px * py)

        probs /= np.sum(probs)
        flat_idx = np.random.choice(49, p=probs.flatten())
        hg = flat_idx // 7
        ag = flat_idx % 7

        sim_rows.append(
            {
                "season": "2025-26",
                "round": (m_idx % 38) + 1,
                "home_team": h_team,
                "away_team": a_team,
                "home_goals": hg,
                "away_goals": ag,
                "kickoff_time": (base_date + pd.Timedelta(days=m_idx % 250)).isoformat(),
            }
        )

    sim_df = pd.DataFrame(sim_rows)
    model = DixonColesModel(xi=0.0001, ridge_lambda=0.01)
    model.fit(matches_df=sim_df)

    # Assert home advantage recovery within sampling tolerance
    assert abs(model.gamma - true_gamma) < 0.12, (
        f"Recovered gamma {model.gamma:.4f} should be close to true gamma {true_gamma:.4f}"
    )

    # Assert low-score correlation rho recovery within sampling tolerance
    assert abs(model.rho - true_rho) < 0.15, f"Recovered rho {model.rho:.4f} should be close to true rho {true_rho:.4f}"

    # Assert relative attack strength ordering is preserved: Team1 > Team2 > Team3 > Team4
    assert model.alphas["Team1"] > model.alphas["Team2"], "Team1 alpha must exceed Team2"
    assert model.alphas["Team2"] > model.alphas["Team3"], "Team2 alpha must exceed Team3"
    assert model.alphas["Team3"] > model.alphas["Team4"], "Team3 alpha must exceed Team4"

    # Assert relative defensive strength ordering is preserved: Team1 > Team2 > Team3 > Team4
    assert model.deltas["Team1"] > model.deltas["Team2"], "Team1 delta must exceed Team2"
    assert model.deltas["Team2"] > model.deltas["Team3"], "Team2 delta must exceed Team3"
    assert model.deltas["Team3"] > model.deltas["Team4"], "Team3 delta must exceed Team4"
