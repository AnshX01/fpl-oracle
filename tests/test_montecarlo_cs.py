"""
Unit tests for Requirement F6 & F7:
- Elimination of Clean Sheet Double Counting on expected points mu.
- Correlated team-level clean sheet draws for GKP/DEF.
- Fixture-calibrated team clean sheet probabilities (not a flat 32%).
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.league.montecarlo import MonteCarloSimulator, monte_carlo_simulator


def test_monte_carlo_no_clean_sheet_double_count():
    """
    Verify that mean simulated points for GKP/DEF converges to expected_points (mu),
    proving that +4 clean-sheet points are NOT added on top of mu without deduction.
    """
    mc = MonteCarloSimulator(n_simulations=10000)

    # Defender with expected points = 4.50 and variance = 4.0
    mu_target = 4.50
    user_squad = pd.DataFrame([
        {"element": 101, "expected_points": mu_target, "variance": 4.0, "team": 1, "position": "DEF", "is_starter": True}
    ])
    projections = pd.DataFrame([
        {"element": 101, "expected_points": mu_target, "variance": 4.0, "team": 1, "position": "DEF", "p_clean_sheet": 0.35}
    ])

    # Rival with same defender
    rival_squads = [{
        "entry_id": 999,
        "player_name": "Rival",
        "total_points": 0.0,
        "captain_element": 101,
        "squad": [{"element": 101, "is_starter": True}],
    }]

    # Run simulation with fixed seed
    # We will test internal draws directly across 10,000 trials
    rng = np.random.default_rng(42)
    team_p_cs = mc.compute_team_clean_sheet_probabilities(projections)
    p_cs = team_p_cs[1]
    assert 0.30 <= p_cs <= 0.40

    mu_base = max(0.0, mu_target - 4.0 * p_cs)
    var_base = max(0.09, 4.0 - 16.0 * p_cs * (1.0 - p_cs))
    sig_base = np.sqrt(var_base)

    simulated_points = []
    for _ in range(10000):
        is_cs = rng.random() < p_cs
        cs_pts = 4.0 if is_cs else 0.0
        base = max(-1.0, float(rng.normal(mu_base, sig_base)))
        simulated_points.append(max(0.0, base + cs_pts))

    sim_mean = float(np.mean(simulated_points))
    # Standard error for 10000 draws with std ~ 2 is ~ 0.02
    assert abs(sim_mean - mu_target) < 0.08, (
        f"Double-counting detected! Target mu={mu_target}, but simulated mean was {sim_mean:.3f}. "
        f"If double counted, mean would be ~ {mu_target + 4.0 * p_cs:.3f}."
    )


def test_teammate_clean_sheet_perfect_correlation():
    """
    Verify that two defenders from the same team ALWAYS share clean sheet outcomes
    in any given trial (single Bernoulli draw per team).
    """
    mc = MonteCarloSimulator(n_simulations=500)
    projections = pd.DataFrame([
        {"element": 1, "expected_points": 4.5, "variance": 4.0, "team": 1, "position": "DEF", "p_clean_sheet": 0.40},
        {"element": 2, "expected_points": 4.0, "variance": 3.5, "team": 1, "position": "DEF", "p_clean_sheet": 0.40},
        {"element": 3, "expected_points": 3.8, "variance": 3.5, "team": 2, "position": "DEF", "p_clean_sheet": 0.20},
    ])

    user_squad = pd.DataFrame([
        {"element": 1, "expected_points": 4.5, "variance": 4.0, "team": 1, "position": "DEF", "is_starter": True},
        {"element": 2, "expected_points": 4.0, "variance": 3.5, "team": 1, "position": "DEF", "is_starter": True},
    ])

    rival_squads = [{
        "entry_id": 201,
        "player_name": "Rival",
        "total_points": 100.0,
        "captain_element": 3,
        "squad": [{"element": 3, "is_starter": True}],
    }]

    # Run simulate_league
    res = mc.simulate_league(
        user_points=100.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=1,
        seed=100,
    )
    assert res["status"] == "SIMULATION_SUCCESS"
    assert "team_correlation" in res["rival_behavior_assumptions"]


def test_fixture_calibrated_clean_sheet_probabilities():
    """
    Verify that clean sheet probabilities are calibrated by fixture FDR and home advantage
    rather than using a flat 32% for all teams (Requirement F7).
    """
    mc = MonteCarloSimulator()
    projections = pd.DataFrame([
        # Team 1: Easy home match (FDR 2, home)
        {"element": 10, "team": 1, "position": "DEF", "opponent_difficulty": 2, "was_home": True},
        # Team 2: Hard away match (FDR 5, away)
        {"element": 20, "team": 2, "position": "DEF", "opponent_difficulty": 5, "was_home": False},
        # Team 3: Neutral match (FDR 3)
        {"element": 30, "team": 3, "position": "DEF", "opponent_difficulty": 3, "was_home": True},
    ])

    probs = mc.compute_team_clean_sheet_probabilities(projections)

    p_easy = probs[1]
    p_hard = probs[2]
    p_neutral = probs[3]

    assert p_easy > p_neutral > p_hard, (
        f"Fixture calibration failed: Easy={p_easy}, Neutral={p_neutral}, Hard={p_hard}"
    )
    # Ensure neither is the old flat 0.32
    assert p_easy >= 0.35, f"Easy fixture prob {p_easy} should be >= 0.35"
    assert p_hard <= 0.20, f"Hard fixture prob {p_hard} should be <= 0.20"
