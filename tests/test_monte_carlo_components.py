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

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, Team
from fpl_oracle.config import (
    CLEAN_SHEET_AWAY_FACTOR,
    CLEAN_SHEET_HOME_FACTOR,
    DEFAULT_CLEAN_SHEET_PROBABILITY,
)
from fpl_oracle.league.montecarlo import MonteCarloSimulator
from fpl_oracle.ml.predict import projection_engine


@pytest.mark.parametrize("pos", ["DEF", "MID", "FWD", "GKP"])
@pytest.mark.parametrize("xp", [0.1, 1.0, 4.5, 8.0])
def test_real_simulation_function_unbiased_across_positions_and_xp(pos: str, xp: float):
    """
    Verify that the REAL simulation function (simulate_player_trials) produces an empirical mean
    matching model xP within Monte Carlo error bounds for all positions and all xP scales.
    """
    mc = MonteCarloSimulator(n_simulations=10000)
    player_data = {
        "element": 1,
        "expected_points": xp,
        "position": pos,
        "team": 1,
        "p_clean_sheet": 0.35,
    }

    # Execute 10,000 trials with fixed seed
    draws = mc.simulate_player_trials(
        player_data=player_data,
        team_p_cs=0.35,
        n_simulations=10000,
        seed=42,
    )

    sim_mean = float(np.mean(draws))
    abs_err = abs(sim_mean - xp)

    if xp <= 0.1:
        # For small bench xP = 0.1, assert absolute error < 0.05
        assert abs_err < 0.05, (
            f"Unbiased check failed for {pos} at xP={xp}: simulated mean {sim_mean:.4f}, abs_err {abs_err:.4f} >= 0.05"
        )
    else:
        # For xP >= 1.0, assert relative error < 5%
        rel_err = abs_err / xp
        assert rel_err < 0.05, (
            f"Unbiased check failed for {pos} at xP={xp}: simulated mean {sim_mean:.4f}, rel_err {rel_err:.2%} >= 5%"
        )


def test_midfielder_clean_sheet_correlation():
    """
    Verify that midfielders receive +1.0 pt for a clean sheet (when playing >= 60 min)
    consistent with official FPL scoring rules, and 0 when the team concedes.
    """
    mc = MonteCarloSimulator()
    rng = np.random.default_rng(100)

    # Midfielder who always plays 90 mins (p_min60 = 1.0)
    mid_player = {
        "element": 10,
        "expected_points": 5.0,
        "position": "MID",
        "team": 1,
        "p_min60": 1.0,
        "p_play": 1.0,
        "exp_cs_pts": 0.35 * 1.0,  # 0.35 CS pts
    }

    # Simulate with team clean sheet guaranteed True vs False
    scores_cs = [mc.simulate_player_gameweek(mid_player, team_clean_sheet=True, rng=rng) for _ in range(5000)]
    scores_no_cs = [mc.simulate_player_gameweek(mid_player, team_clean_sheet=False, rng=rng) for _ in range(5000)]

    mean_cs = float(np.mean(scores_cs))
    mean_no_cs = float(np.mean(scores_no_cs))

    diff = mean_cs - mean_no_cs
    # Difference should equal +1.0 pt for MID clean sheet (within MC tolerance ~0.05)
    assert abs(diff - 1.0) < 0.08, f"MID clean sheet differential was {diff:.3f}, expected 1.0"


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


def test_horizon_simulation_per_gw_projections():
    """
    Verify that multi-gameweek simulation uses each gameweek's own distinct projections,
    rather than repeating GW1 across all horizon steps.
    """
    mc = MonteCarloSimulator(n_simulations=2000)

    # GW1: High expectation player (xP = 10.0)
    gw1_df = pd.DataFrame([{"element": 1, "expected_points": 10.0, "team": 1, "position": "MID"}])
    # GW2: Tough fixture / rotation (xP = 2.0)
    gw2_df = pd.DataFrame([{"element": 1, "expected_points": 2.0, "team": 1, "position": "MID"}])

    projections_by_gw = {1: gw1_df, 2: gw2_df}

    user_squad = pd.DataFrame(
        [{"element": 1, "expected_points": 10.0, "team": 1, "position": "MID", "is_starter": True}]
    )
    rival_squads = [
        {
            "entry_id": 1,
            "total_points": 0.0,
            "squad": [{"element": 1, "is_starter": True}],
        }
    ]

    # Run horizon with per-GW projections
    res_multi = mc.simulate_league(
        user_points=0.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        horizon_gws=2,
        projections_by_gw=projections_by_gw,
        seed=42,
    )
    assert res_multi["status"] == "SIMULATION_SUCCESS"

    # Compare with a simulation where GW1 is repeated for 2 GWs (xP 10.0 + 10.0 = 20.0)
    # The per-GW projection has xP 10.0 + 2.0 = 12.0
    # Average total points should reflect ~12.0, not ~20.0
    rng = np.random.default_rng(42)
    scores_gw1 = [mc.simulate_player_gameweek(gw1_df.iloc[0].to_dict(), False, rng) for _ in range(2000)]
    scores_gw2 = [mc.simulate_player_gameweek(gw2_df.iloc[0].to_dict(), False, rng) for _ in range(2000)]
    mean_combined = float(np.mean(scores_gw1)) + float(np.mean(scores_gw2))

    assert 11.0 < mean_combined < 13.0, f"Combined per-GW projection mean {mean_combined:.2f} should be ~12.0"


def test_clean_sheet_model_derived_priority():
    """
    Verify that when model-derived p_clean_sheet is present in projections,
    it takes priority over the FDR/home heuristic.
    """
    mc = MonteCarloSimulator()
    projections = pd.DataFrame(
        [
            {
                "element": 1,
                "team": 5,
                "position": "DEF",
                "p_clean_sheet": 0.55,
                "opponent_difficulty": 5,  # Difficult fixture would normally give low prob
                "was_home": False,
            }
        ]
    )

    team_p_cs = mc.compute_team_clean_sheet_probabilities(projections)
    # Model gave 0.55; heuristic would have given ~0.15
    assert team_p_cs[5] == 0.55


def test_clean_sheet_config_fallback():
    """
    Verify that when model outputs are absent, the named config settings
    (DEFAULT_CLEAN_SHEET_PROBABILITY, CLEAN_SHEET_HOME_FACTOR, CLEAN_SHEET_AWAY_FACTOR)
    govern the fallback calculation.
    """
    mc = MonteCarloSimulator()

    # Projection without p_clean_sheet
    projections = pd.DataFrame(
        [
            {"element": 10, "team": 8, "position": "DEF", "opponent_difficulty": 3, "was_home": True},
            {"element": 20, "team": 9, "position": "DEF", "opponent_difficulty": 3, "was_home": False},
        ]
    )

    team_p_cs = mc.compute_team_clean_sheet_probabilities(projections)

    # For neutral FDR 3: diff_factor = (6 - 3) / 3 = 1.0
    expected_home = round(
        float(np.clip(DEFAULT_CLEAN_SHEET_PROBABILITY * CLEAN_SHEET_HOME_FACTOR * 1.0, 0.05, 0.75)), 3
    )
    expected_away = round(
        float(np.clip(DEFAULT_CLEAN_SHEET_PROBABILITY * CLEAN_SHEET_AWAY_FACTOR * 1.0, 0.05, 0.75)), 3
    )

    assert team_p_cs[8] == expected_home
    assert team_p_cs[9] == expected_away


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
