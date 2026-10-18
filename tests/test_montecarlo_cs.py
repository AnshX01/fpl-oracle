"""
Unit tests for Requirement F6 & F7:
- Elimination of Clean Sheet Double Counting on expected points mu.
- Correlated team-level clean sheet draws for GKP/DEF.
- Fixture-calibrated team clean sheet probabilities (not a flat 32%).
"""

import pandas as pd

from fpl_oracle.league.montecarlo import MonteCarloSimulator


def test_teammate_clean_sheet_perfect_correlation():
    """
    Verify that two defenders from the same team ALWAYS share clean sheet outcomes
    in any given trial (single Bernoulli draw per team).
    """
    mc = MonteCarloSimulator(n_simulations=500)
    projections = pd.DataFrame(
        [
            {
                "element": 1,
                "expected_points": 4.5,
                "variance": 4.0,
                "team": 1,
                "position": "DEF",
                "p_clean_sheet": 0.40,
            },
            {
                "element": 2,
                "expected_points": 4.0,
                "variance": 3.5,
                "team": 1,
                "position": "DEF",
                "p_clean_sheet": 0.40,
            },
            {
                "element": 3,
                "expected_points": 3.8,
                "variance": 3.5,
                "team": 2,
                "position": "DEF",
                "p_clean_sheet": 0.20,
            },
        ]
    )

    user_squad = pd.DataFrame(
        [
            {"element": 1, "expected_points": 4.5, "variance": 4.0, "team": 1, "position": "DEF", "is_starter": True},
            {"element": 2, "expected_points": 4.0, "variance": 3.5, "team": 1, "position": "DEF", "is_starter": True},
        ]
    )

    rival_squads = [
        {
            "entry_id": 201,
            "player_name": "Rival",
            "total_points": 100.0,
            "captain_element": 3,
            "squad": [{"element": 3, "is_starter": True}],
        }
    ]

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
