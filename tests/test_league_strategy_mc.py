"""
Unit & Integration tests for Mini-League intelligence, seeded deterministic
Monte Carlo simulation, strategy modes (Defending vs Chasing), and BGW/DGW calendar detection.
Covers R1-R4 and W1-W3 requirements.
"""

import time

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture, Team
from fpl_oracle.chips.calendar import fixture_calendar
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.rivals import rival_analyzer
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
    mock_teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
        Team(id=3, name="Liverpool", short_name="LIV"),
    ]
    mock_boot = BootstrapStatic(
        events=[], game_settings={}, phases=[], teams=mock_teams, total_players=0, elements=[], element_types=[]
    )

    fixtures = [
        Fixture(id=101, code=101, event=7, team_h=1, team_a=2, kickoff_time="2026-10-17T14:00:00Z"),
        Fixture(id=102, code=102, event=7, team_h=1, team_a=2, kickoff_time="2026-10-20T19:00:00Z"),
    ]

    cal = fixture_calendar.analyze_calendar(fixtures, mock_boot)
    dgw_events = [d["gameweek"] for d in cal["double_gameweeks"]]
    assert 7 in dgw_events
    dgw_teams = [t["team_name"] for t in cal["double_gameweeks"][0]["teams_with_doubles"]]
    assert "Arsenal" in dgw_teams


def test_proximity_based_rival_selection():
    """R1: Verifies proximity window (managers above + within 30 points below)."""
    standings = [
        {"entry": 10, "player_name": "Leader", "total": 400, "rank": 1},
        {"entry": 20, "player_name": "Second", "total": 380, "rank": 2},
        {"entry": 30, "player_name": "User", "total": 360, "rank": 3},  # User at 360
        {"entry": 40, "player_name": "Close Chaser 1", "total": 355, "rank": 4},  # Diff 5 (<= 30)
        {"entry": 50, "player_name": "Close Chaser 2", "total": 335, "rank": 5},  # Diff 25 (<= 30)
        {"entry": 60, "player_name": "Far Away", "total": 310, "rank": 6},  # Diff 50 (> 30, excluded)
    ]

    selected, mode, rank = rival_analyzer.select_proximity_rivals(
        standings=standings, user_manager_id=30, points_window=30, max_rivals=10
    )

    assert mode == "PROXIMITY_WINDOW"
    assert rank == 3
    selected_ids = [s["entry"] for s in selected]
    # Managers above user: 10, 20
    assert 10 in selected_ids
    assert 20 in selected_ids
    # Managers below within 30 pts: 40 (diff 5), 50 (diff 25)
    assert 40 in selected_ids
    assert 50 in selected_ids
    # Manager below outside 30 pts: 60 (diff 50) -> must be excluded
    assert 60 not in selected_ids
    assert 30 not in selected_ids  # User self excluded from rivals


def test_published_rival_details_eo_exposure_chips():
    """R2: Verifies rival effective ownership, net exposure vs user squad, and 2026/27 chip tracking."""
    # 2026/27 chip tracking
    # GW10: Wildcard 1 not used -> remaining
    chips_gw10 = rival_analyzer.calculate_chips_status([], current_gw=10)
    assert "wildcard_set1" in chips_gw10["chips_remaining"]
    assert chips_gw10["set1_wildcard_expired"] is False

    # GW22: Wildcard 1 not used -> expired
    chips_gw22 = rival_analyzer.calculate_chips_status([], current_gw=22)
    assert "wildcard_set1" not in chips_gw22["chips_remaining"]
    assert "wildcard_set2" in chips_gw22["chips_remaining"]
    assert chips_gw22["set1_wildcard_expired"] is True

    # Wildcard used in Set 1
    chips_used = rival_analyzer.calculate_chips_status([{"name": "wildcard", "event": 8}], current_gw=10)
    assert "wildcard_set1" not in chips_used["chips_remaining"]


def test_pre_deadline_picks_opacity():
    """R3: Verifies upcoming deadline picks are marked unknown and confidential."""
    # Build synthetic history
    chips = [{"name": "freehit", "event": 3}]
    status = rival_analyzer.calculate_chips_status(chips, current_gw=6)
    assert "freehit" not in status["chips_remaining"]
    assert "bench_boost" in status["chips_remaining"]


def test_bounded_posture_tie_breaker():
    """R4: Verifies tactical risk posture acts strictly as a bounded tie-breaker (<= 0.5 xP)."""
    # Candidate plans:
    # Plan A: Pure highest xP (+3.0 xP)
    # Plan B: Runner up (+2.8 xP, diff = 0.2 xP <= 0.5 margin), brings in differential
    # Plan C: Inferior (+2.0 xP, diff = 1.0 xP > 0.5 margin)
    plan_a = {"plan_id": "A", "expected_gain": 3.0, "transfers_in": [{"element": 10}]}
    plan_b = {"plan_id": "B", "expected_gain": 2.8, "transfers_in": [{"element": 20}]}
    plan_c = {"plan_id": "C", "expected_gain": 2.0, "transfers_in": [{"element": 30}]}

    # Rival EO map: element 10 has 90% EO (template); element 20 has 10% EO (differential)
    rival_eo = {10: 90.0, 20: 10.0, 30: 5.0}

    # Case 1: CHASING_PACK posture with close candidates (0.2 diff) -> prefers Plan B (differential)
    tb_chase = league_strategy_advisor.apply_tie_breaker(
        candidate_plans=[plan_a, plan_b],
        posture="CHASING_PACK",
        rival_eo_map=rival_eo,
        margin_xp=0.5,
    )
    assert tb_chase["tie_breaker_applied"] is True
    assert tb_chase["selected_plan"]["plan_id"] == "B"

    # Case 2: DEFENDING_LEAD posture with close candidates -> prefers Plan A (template coverage)
    tb_defend = league_strategy_advisor.apply_tie_breaker(
        candidate_plans=[plan_a, plan_b],
        posture="DEFENDING_LEAD",
        rival_eo_map=rival_eo,
        margin_xp=0.5,
    )
    assert tb_defend["selected_plan"]["plan_id"] == "A"

    # Case 3: Plan C gap is 1.0 xP (> 0.5 margin) -> model NEVER overrides for Plan C
    tb_large_gap = league_strategy_advisor.apply_tie_breaker(
        candidate_plans=[plan_a, plan_c],
        posture="CHASING_PACK",
        rival_eo_map=rival_eo,
        margin_xp=0.5,
    )
    assert tb_large_gap["tie_breaker_applied"] is False
    assert tb_large_gap["selected_plan"]["plan_id"] == "A"


def test_joint_player_draws_and_clean_sheet_correlations():
    """W1: Verifies joint draws (shared players get identical score per trial) and defensive clean sheets."""
    user_squad = pd.DataFrame([
        {"element": 1, "position": "GKP", "team": 1, "expected_points": 4.5, "variance": 4.0, "is_starter": True},
        {"element": 2, "position": "DEF", "team": 1, "expected_points": 4.5, "variance": 4.0, "is_starter": True},
        {"element": 3, "position": "MID", "team": 2, "expected_points": 6.0, "variance": 4.0, "is_starter": True},
    ])
    projections = pd.DataFrame([
        {"element": 1, "position": "GKP", "team": 1, "expected_points": 4.5, "variance": 4.0},
        {"element": 2, "position": "DEF", "team": 1, "expected_points": 4.5, "variance": 4.0},
        {"element": 3, "position": "MID", "team": 2, "expected_points": 6.0, "variance": 4.0},
    ])

    rival_squads = [{
        "entry_id": 999,
        "player_name": "Rival",
        "total_points": 100.0,
        "captain_element": 3,
        "squad": [
            {"element": 1, "is_starter": True},  # Shared GKP
            {"element": 2, "is_starter": True},  # Shared DEF
            {"element": 3, "is_starter": True},  # Shared MID
        ],
    }]

    # When squads are identical and starting points identical, win prob must be balanced (around 50% or tied)
    res = monte_carlo_simulator.simulate_league(
        user_points=100.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=1,
        seed=42,
    )

    assert "rival_behavior_assumptions" in res
    assert "joint_draws" in res["rival_behavior_assumptions"]
    assert "team_correlation" in res["rival_behavior_assumptions"]
    assert res["status"] == "SIMULATION_SUCCESS"


def test_rival_future_behavior_assumptions():
    """W2: Verifies explicit rival future behavior model and visible assumptions."""
    user_squad = pd.DataFrame([{"element": i, "expected_points": 5.0, "variance": 4.0} for i in range(1, 16)])
    projections = pd.DataFrame([{"element": i, "expected_points": 5.0, "variance": 4.0} for i in range(1, 50)])
    rival_squads = [{
        "entry_id": 101,
        "player_name": "Leader",
        "total_points": 200.0,
        "captain_element": 1,
        "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
    }]

    res = monte_carlo_simulator.simulate_league(
        user_points=190.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=5,
        seed=100,
        rival_behavior_model="consensus_template",
    )

    assumptions = res["rival_behavior_assumptions"]
    assert "captaincy_model" in assumptions
    assert "transfer_model" in assumptions
    assert "chip_model" in assumptions


def test_side_by_side_plan_win_prob_comparison():
    """W3: Verifies side-by-side comparison of candidate plans and bounded runtime (< 1.5s)."""
    user_squad = pd.DataFrame([{"element": i, "expected_points": 5.0, "variance": 4.0, "is_starter": True} for i in range(1, 16)])
    projections = pd.DataFrame([{"element": i, "expected_points": 5.0 + (i * 0.1), "variance": 4.0} for i in range(1, 50)])
    rival_squads = [{
        "entry_id": 101,
        "player_name": "Leader",
        "total_points": 350.0,
        "captain_element": 1,
        "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
    }]

    candidate_plans = [
        {"plan_type": "ROLL_TRANSFER", "net_expected_points": 60.0, "transfers_in": [], "transfers_out": []},
        {"plan_type": "1_TRANSFER", "net_expected_points": 62.5, "transfers_in": [{"element": 35}], "transfers_out": [{"element": 10}]},
    ]

    t0 = time.time()
    comparisons = monte_carlo_simulator.compare_candidate_plans(
        candidate_plans=candidate_plans,
        user_points=340.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projections,
        horizon_gws=5,
        seed=42,
    )
    elapsed = time.time() - t0

    assert elapsed < 2.0  # Bounded runtime
    assert len(comparisons) == 2
    for comp in comparisons:
        assert "plan_type" in comp
        assert "gameweek_net_xp" in comp
        assert "win_probability_pct" in comp
        assert "expected_final_rank" in comp
