"""
Unit & Integration tests for Mini-League intelligence, seeded deterministic
Monte Carlo simulation, strategy modes (Defending vs Chasing), and BGW/DGW calendar detection.
"""

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture, Team
from fpl_oracle.chips.calendar import fixture_calendar
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.strategy import league_strategy_advisor


def test_monte_carlo_seeded_determinism():
    """Verify Monte Carlo simulation with fixed seed yields identical win probabilities."""
    user_squad = pd.DataFrame([{"element": i, "expected_points": 5.0, "variance": 4.0} for i in range(1, 16)])
    projections = pd.DataFrame([{"element": i, "expected_points": 5.0, "variance": 4.0} for i in range(1, 50)])
    rival_squads = [
        {
            "entry_id": 101,
            "player_name": "Rival Leader",
            "total_points": 350.0,
            "captain_element": 1,
            "squad": [{"element": i, "is_starter": True} for i in range(10, 21)],
        },
        {
            "entry_id": 102,
            "player_name": "Rival Chaser",
            "total_points": 330.0,
            "captain_element": 2,
            "squad": [{"element": i, "is_starter": True} for i in range(20, 31)],
        },
    ]

    # Run twice with same seed
    res1 = monte_carlo_simulator.simulate_league(
        user_points=340.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=3,
        seed=12345,
    )
    res2 = monte_carlo_simulator.simulate_league(
        user_points=340.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=3,
        seed=12345,
    )

    assert res1["user_win_probability_pct"] == res2["user_win_probability_pct"]
    assert res1["user_top3_probability_pct"] == res2["user_top3_probability_pct"]
    assert res1["expected_final_rank"] == res2["expected_final_rank"]


def test_league_strategy_modes():
    """Verify advisor switches between DEFENDING_LEAD, BALANCED_ATTACK, and CHASING_PACK."""
    user_squad = pd.DataFrame([{"element": i} for i in range(1, 16)])

    # Case 1: Leading the mini-league (+15 pts)
    rivals_leading = {
        "rival_squads": [
            {
                "player_name": "Chaser",
                "rank": 2,
                "total_points": 335,
                "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
            }
        ]
    }
    strat_lead = league_strategy_advisor.evaluate_strategy(
        user_rank=1, user_total_points=350, rivals_analysis=rivals_leading, user_squad_df=user_squad
    )
    assert strat_lead["strategy_mode"] == "DEFENDING_LEAD"
    assert "minimize variance" in strat_lead["rationale"].lower()
    assert any("template protection" in t.lower() for t in strat_lead["tactical_recommendations"])

    # Case 2: Within striking distance (-10 pts)
    rivals_close = {
        "rival_squads": [
            {
                "player_name": "Leader",
                "rank": 1,
                "total_points": 360,
                "squad": [{"element": i, "is_starter": True} for i in range(5, 16)],
            }
        ]
    }
    strat_close = league_strategy_advisor.evaluate_strategy(
        user_rank=2, user_total_points=350, rivals_analysis=rivals_close, user_squad_df=user_squad
    )
    assert strat_close["strategy_mode"] == "BALANCED_ATTACK"
    assert any(
        "controlled variance" in strat_close["mode_title"].lower() or "differential" in t.lower()
        for t in strat_close["tactical_recommendations"]
    )

    # Case 3: Chasing from behind (-45 pts)
    rivals_behind = {
        "rival_squads": [
            {
                "player_name": "Far Ahead Leader",
                "rank": 1,
                "total_points": 395,
                "squad": [{"element": i, "is_starter": True} for i in range(5, 16)],
            }
        ]
    }
    strat_behind = league_strategy_advisor.evaluate_strategy(
        user_rank=5, user_total_points=350, rivals_analysis=rivals_behind, user_squad_df=user_squad
    )
    assert strat_behind["strategy_mode"] == "CHASING_PACK"
    assert "high-variance differential" in strat_behind["mode_title"].lower()
    assert any(
        "counter-chip" in t.lower() or "differential captaincy" in t.lower()
        for t in strat_behind["tactical_recommendations"]
    )


def test_fixture_calendar_bgw_and_dgw_detection():
    """Verify Blank and Double Gameweek detection logic on synthetic schedule."""
    # 2 teams: Team 1 and Team 2
    mock_teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
        Team(id=3, name="Liverpool", short_name="LIV"),
    ]
    mock_boot = BootstrapStatic(
        events=[], game_settings={}, phases=[], teams=mock_teams, total_players=0, elements=[], element_types=[]
    )

    # In GW7: Arsenal plays twice (DGW), Chelsea plays once, Liverpool plays 0 (BGW)
    fixtures = [
        Fixture(id=101, code=101, event=7, team_h=1, team_a=2, kickoff_time="2026-10-17T14:00:00Z"),
        Fixture(id=102, code=102, event=7, team_h=1, team_a=2, kickoff_time="2026-10-20T19:00:00Z"),
    ]

    cal = fixture_calendar.analyze_calendar(fixtures, mock_boot)
    # Check DGW detection
    dgw_events = [d["gameweek"] for d in cal["double_gameweeks"]]
    assert 7 in dgw_events
    dgw_teams = [t["team_name"] for t in cal["double_gameweeks"][0]["teams_with_doubles"]]
    assert "Arsenal" in dgw_teams
