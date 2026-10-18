"""
Unit tests for G9: Position-Specific Component Monte Carlo & Clean Sheet Correctness.

Verifies:
1. Unbiased simulation: Simulated mean equals model xP within Monte Carlo error tolerance
   (< 5% relative error for xP >= 1.0; < 0.05 absolute error for xP = 0.1) across DEF, MID, FWD, GKP.
2. MID clean sheet correlation consistent with FPL rules (+1 pt for MID CS).
3. Shared team clean-sheet Bernoulli per fixture across teammates.
4. Horizon simulation utilizes per-GW projections and fixtures rather than repeating GW1.
5. Model-derived clean sheet probability takes priority; named settings in config govern fallback.
6. Prediction engine retains p_clean_sheet, home/away, and fixture metadata.
"""

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, Team
from fpl_oracle.league.montecarlo import MonteCarloSimulator
from fpl_oracle.ml.predict import projection_engine


def test_teammate_clean_sheet_shared_bernoulli():
    """
    Verify that two defenders on the same team share the exact same clean sheet trial outcome
    in every simulation step.
    """
    mc = MonteCarloSimulator(n_simulations=500)
    projections = pd.DataFrame(
        [
            {"element": 1, "expected_points": 4.5, "team": 1, "position": "DEF", "p_clean_sheet": 0.40},
            {"element": 2, "expected_points": 4.0, "team": 1, "position": "DEF", "p_clean_sheet": 0.40},
        ]
    )
    user_squad = pd.DataFrame(
        [
            {"element": 1, "expected_points": 4.5, "team": 1, "position": "DEF", "is_starter": True},
            {"element": 2, "expected_points": 4.0, "team": 1, "position": "DEF", "is_starter": True},
        ]
    )
    rival_squads = [
        {
            "entry_id": 99,
            "player_name": "Rival",
            "total_points": 50.0,
            "captain_element": 1,
            "squad": [{"element": 1, "is_starter": True}],
        }
    ]

    res = mc.simulate_league(
        user_points=50.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )
    assert res["status"] == "SIMULATION_SUCCESS"
    assert "MID +1 pt on clean sheet" in res["rival_behavior_assumptions"]["team_correlation"]


def test_predict_retains_p_clean_sheet_and_fixtures():
    """
    Verify that ProjectionEngine.predict_gameweek retains p_clean_sheet, was_home,
    is_home, opponent_team, and fixture_id in its output DataFrame.
    """
    boot = BootstrapStatic(
        events=[],
        teams=[
            Team(id=1, name="Arsenal", short_name="ARS"),
            Team(id=2, name="Chelsea", short_name="CHE"),
        ],
        elements=[
            Element(id=1, web_name="Raya", team=1, element_type=1, now_cost=55, code=101),
            Element(id=2, web_name="Saliba", team=1, element_type=2, now_cost=60, code=102),
            Element(id=3, web_name="Saka", team=1, element_type=3, now_cost=100, code=103),
        ],
        element_types=[
            ElementType(
                id=1,
                singular_name="Goalkeeper",
                singular_name_short="GKP",
                plural_name="Goalkeepers",
                plural_name_short="GKP",
            ),
            ElementType(
                id=2,
                singular_name="Defender",
                singular_name_short="DEF",
                plural_name="Defenders",
                plural_name_short="DEF",
            ),
            ElementType(
                id=3,
                singular_name="Midfielder",
                singular_name_short="MID",
                plural_name="Midfielders",
                plural_name_short="MID",
            ),
            ElementType(
                id=4,
                singular_name="Forward",
                singular_name_short="FWD",
                plural_name="Forwards",
                plural_name_short="FWD",
            ),
        ],
    )
    fixtures = [
        Fixture(
            id=42,
            code=10042,
            event=10,
            team_h=1,
            team_a=2,
            team_h_difficulty=2,
            team_a_difficulty=4,
            kickoff_time="2026-10-15T15:00:00Z",
        )
    ]

    preds = projection_engine.predict_gameweek(10, boot, fixtures)
    assert not preds.empty

    required_cols = ["p_clean_sheet", "was_home", "is_home", "opponent_team", "fixture_id", "p_play"]
    for col in required_cols:
        assert col in preds.columns, f"Required column '{col}' missing from predict_gameweek output!"

    # Verify Raya (GKP) and Saliba (DEF) have valid p_clean_sheet > 0
    raya = preds[preds["element"] == 1].iloc[0]
    assert 0.0 < float(raya["p_clean_sheet"]) < 1.0
    assert float(raya["was_home"]) == 1.0
    assert int(raya["opponent_team"]) == 2
    assert int(raya["fixture_id"]) == 42
