import pandas as pd
import pytest

from fpl_oracle.league.montecarlo import monte_carlo_simulator


@pytest.fixture
def base_squad_and_projections():
    user_squad = pd.DataFrame([
        {"element": i, "position": "MID", "team": 1, "expected_points": 5.0, "variance": 4.0, "is_starter": True}
        for i in range(1, 16)
    ])
    projections = pd.DataFrame([
        {"element": i, "position": "MID", "team": 1, "expected_points": 5.0, "variance": 4.0}
        for i in range(1, 30)
    ])
    return user_squad, projections


def test_identical_two_squads_yields_50_percent(base_squad_and_projections):
    """
    F-002: When user and 1 rival have identical squads and identical points,
    win probability must be ~50.0%, not 100.0%.
    """
    user_squad, projections = base_squad_and_projections
    rivals = [
        {
            "entry_id": 101,
            "player_name": "Rival Symmetric",
            "total_points": 300.0,
            "captain_element": 1,
            "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
        }
    ]

    res = monte_carlo_simulator.simulate_league(
        user_points=300.0,
        user_squad_df=user_squad,
        rival_squads=rivals,
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )

    assert res["status"] == "SIMULATION_SUCCESS"
    win_prob = res["user_win_probability_pct"]
    assert 49.0 <= win_prob <= 51.0, f"Expected ~50% win prob, got {win_prob}%"
    assert 1.45 <= res["expected_final_rank"] <= 1.55, f"Expected ~1.5 rank, got {res['expected_final_rank']}"


def test_three_way_identical_tie_yields_33_percent(base_squad_and_projections):
    """
    F-002: In a 3-way identical squad tie, each participant should win ~33.3%, not 100.0%.
    """
    user_squad, projections = base_squad_and_projections
    rivals = [
        {
            "entry_id": 101,
            "player_name": "Rival 1",
            "total_points": 300.0,
            "captain_element": 1,
            "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
        },
        {
            "entry_id": 102,
            "player_name": "Rival 2",
            "total_points": 300.0,
            "captain_element": 1,
            "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
        },
    ]

    res = monte_carlo_simulator.simulate_league(
        user_points=300.0,
        user_squad_df=user_squad,
        rival_squads=rivals,
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )

    assert res["status"] == "SIMULATION_SUCCESS"
    win_prob = res["user_win_probability_pct"]
    assert 32.0 <= win_prob <= 34.5, f"Expected ~33.3% win prob, got {win_prob}%"
    assert 1.95 <= res["expected_final_rank"] <= 2.05, f"Expected ~2.0 rank, got {res['expected_final_rank']}"


def test_symmetric_swap_of_user_and_rival(base_squad_and_projections):
    """
    F-002: Swapping user and rival should give symmetric results.
    """
    user_squad, projections = base_squad_and_projections
    rival_squad = [
        {"element": i, "is_starter": True} for i in range(1, 12)
    ]

    # Sim 1: User A vs Rival B
    res1 = monte_carlo_simulator.simulate_league(
        user_points=300.0,
        user_squad_df=user_squad,
        rival_squads=[{"entry_id": 101, "total_points": 300.0, "captain_element": 1, "squad": rival_squad}],
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )

    # Sim 2: User B vs Rival A
    res2 = monte_carlo_simulator.simulate_league(
        user_points=300.0,
        user_squad_df=user_squad,
        rival_squads=[{"entry_id": 102, "total_points": 300.0, "captain_element": 1, "squad": rival_squad}],
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )

    assert res1["user_win_probability_pct"] == res2["user_win_probability_pct"]
    assert res1["expected_final_rank"] == res2["expected_final_rank"]


def test_win_probabilities_sum_to_100_percent(base_squad_and_projections):
    """
    F-002: Total win probabilities across all competitors must sum to 100%.
    """
    _, projections = base_squad_and_projections

    # 3 distinct squads
    squad_a = pd.DataFrame([
        {"element": i, "position": "MID", "team": 1, "expected_points": 6.0, "variance": 3.0, "is_starter": True}
        for i in range(1, 12)
    ])
    squad_b = pd.DataFrame([
        {"element": i + 10, "position": "MID", "team": 1, "expected_points": 5.5, "variance": 4.0, "is_starter": True}
        for i in range(1, 12)
    ])
    squad_c = pd.DataFrame([
        {"element": i + 20, "position": "MID", "team": 1, "expected_points": 5.0, "variance": 5.0, "is_starter": True}
        for i in range(1, 12)
    ])

    proj_full = pd.DataFrame([
        {"element": i, "position": "MID", "team": 1, "expected_points": 5.5, "variance": 4.0}
        for i in range(1, 40)
    ])

    # Participant A as user
    res_a = monte_carlo_simulator.simulate_league(
        user_points=100.0,
        user_squad_df=squad_a,
        rival_squads=[
            {"entry_id": 2, "total_points": 100.0, "captain_element": 11, "squad": squad_b.to_dict(orient="records")},
            {"entry_id": 3, "total_points": 100.0, "captain_element": 21, "squad": squad_c.to_dict(orient="records")},
        ],
        projections_df=proj_full,
        horizon_gws=1,
        seed=100,
    )

    # Participant B as user
    res_b = monte_carlo_simulator.simulate_league(
        user_points=100.0,
        user_squad_df=squad_b,
        rival_squads=[
            {"entry_id": 1, "total_points": 100.0, "captain_element": 1, "squad": squad_a.to_dict(orient="records")},
            {"entry_id": 3, "total_points": 100.0, "captain_element": 21, "squad": squad_c.to_dict(orient="records")},
        ],
        projections_df=proj_full,
        horizon_gws=1,
        seed=100,
    )

    # Participant C as user
    res_c = monte_carlo_simulator.simulate_league(
        user_points=100.0,
        user_squad_df=squad_c,
        rival_squads=[
            {"entry_id": 1, "total_points": 100.0, "captain_element": 1, "squad": squad_a.to_dict(orient="records")},
            {"entry_id": 2, "total_points": 100.0, "captain_element": 11, "squad": squad_b.to_dict(orient="records")},
        ],
        projections_df=proj_full,
        horizon_gws=1,
        seed=100,
    )

    total_prob = (
        res_a["user_win_probability_pct"]
        + res_b["user_win_probability_pct"]
        + res_c["user_win_probability_pct"]
    )
    assert 98.0 <= total_prob <= 102.0, f"Expected total win probability ~100%, got {total_prob}%"

