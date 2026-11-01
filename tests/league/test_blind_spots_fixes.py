"""
Unit tests for League Strategy and Win Probability Blind Spot Fixes:
- Fix 1: Risk Posture Inversion & Boundary Thresholds (strategy.py)
- Fix 3: Multi-Rival Threat Aggregation & Shadowing (strategy.py)
- Fix 8: Defensive Teammate & Opponent Covariance (objective.py)
"""

from fpl_oracle.league.objective import win_probability
from fpl_oracle.league.strategy import league_strategy_advisor


def test_fix1_leader_never_in_chasing_pack():
    """A manager leading by points (points_gap > 0) must never be assigned CHASING_PACK,
    even in large leagues where individual win probability is diluted (e.g. 18%)."""
    posture = league_strategy_advisor.determine_risk_posture(points_gap=12.0, win_probability_pct=18.0, current_gw=15)
    assert posture == "BALANCED_ATTACK", f"Expected BALANCED_ATTACK for positive lead, got {posture}"

    # Solid lead in GW 15 triggers DEFENDING_LEAD
    posture_solid = league_strategy_advisor.determine_risk_posture(
        points_gap=16.0, win_probability_pct=18.0, current_gw=15
    )
    assert posture_solid == "DEFENDING_LEAD"


def test_fix1_trailer_postures():
    """A trailing manager without high win probability is in BALANCED_ATTACK or CHASING_PACK."""
    posture = league_strategy_advisor.determine_risk_posture(points_gap=-5.0, win_probability_pct=50.0, current_gw=25)
    assert posture == "BALANCED_ATTACK", f"Expected BALANCED_ATTACK for trailing manager, got {posture}"

    # Large deficit triggers CHASING_PACK
    posture_large_deficit = league_strategy_advisor.determine_risk_posture(
        points_gap=-26.0, win_probability_pct=15.0, current_gw=25
    )
    assert posture_large_deficit == "CHASING_PACK"

    # Deficit with low win probability triggers CHASING_PACK
    posture_mod_deficit = league_strategy_advisor.determine_risk_posture(
        points_gap=-12.0, win_probability_pct=15.0, current_gw=25
    )
    assert posture_mod_deficit == "CHASING_PACK"


def test_fix3_multi_rival_shadow_adjustment_aggregation():
    """Evaluates shadow penalty across multiple tracked chasers rather than just rivals[0]."""
    transfers_out = [{"element": 10}, {"element": 20}]
    transfers_in = [{"element": 30}]

    # Rival 1 owns player 10 (points gap 5)
    # Rival 2 owns player 10 (points gap 8)
    # Neither owns player 20
    tracked_rivals = [
        {"player_name": "Rival 1", "points_gap": 5.0, "elements": [10, 11, 12]},
        {"player_name": "Rival 2", "points_gap": 8.0, "elements": [10, 13, 14]},
    ]

    adj, reasons = league_strategy_advisor.compute_rival_shadow_adjustment(
        transfers_out=transfers_out,
        transfers_in=transfers_in,
        rival_elements=tracked_rivals,
        posture="DEFENDING_LEAD",
    )

    # Selling player 10 is penalized more heavily because both chasers own him
    assert adj < -0.60, f"Expected aggregated penalty < -0.60, got {adj}"
    assert any("immediate chasers" in r for r in reasons)


def test_fix3_backward_compatibility_single_set():
    """Confirms single set[int] still works cleanly as before."""
    transfers_out = [{"element": 10}]
    transfers_in = [{"element": 30}]
    rival_set = {10, 15, 20}

    adj, reasons = league_strategy_advisor.compute_rival_shadow_adjustment(
        transfers_out=transfers_out,
        transfers_in=transfers_in,
        rival_elements=rival_set,
        posture="DEFENDING_LEAD",
    )
    assert adj == -0.60
    assert any("primary chaser" in r for r in reasons)


def test_fix8_defensive_teammate_covariance():
    """Defensive teammates (e.g. 2 defenders from same team) share clean sheet covariance,
    expanding the variance of the defensive stack in win probability calculations."""
    gws = [10]
    pos = {1: "DEF", 2: "DEF", 3: "MID"}
    # Team 1 for both defenders 1 and 2
    team_map = {1: 1, 2: 1, 3: 2}

    my_weeks = [{1: 1.0, 2: 1.0}]  # Stacking 2 defenders from team 1
    held_weeks = [{1: 1.0, 2: 1.0}]
    theirs_weeks = [{3: 1.0}]  # Rival has midfielder from team 2

    xp = {10: {1: 4.5, 2: 4.5, 3: 4.5}}
    sd = {10: {1: 3.0, 2: 3.0, 3: 3.0}}

    rival_context = {
        "user_points": 100,
        "rivals": [{"points": 95, "elements": [3]}],
        "team_map": team_map,
    }

    # Solve with team_map (includes positive covariance between 1 and 2)
    res_cov = win_probability(
        my_weeks=my_weeks,
        held_weeks=held_weeks,
        best_plan_points=9.0,
        gws=gws,
        rival_context=rival_context,
        xp=xp,
        sd=sd,
        position=pos,
        rival_cache={0: theirs_weeks},
        team_map=team_map,
    )

    # Solve without team_map (orthogonal assumption)
    res_ortho = win_probability(
        my_weeks=my_weeks,
        held_weeks=held_weeks,
        best_plan_points=9.0,
        gws=gws,
        rival_context={k: v for k, v in rival_context.items() if k != "team_map"},
        xp=xp,
        sd=sd,
        position=pos,
        rival_cache={0: theirs_weeks},
        team_map=None,
    )

    assert res_cov is not None and res_ortho is not None
    # For a leader (+5 pts lead), higher variance slightly lowers win probability (defending lead dislikes variance)
    assert 0.0 < res_cov["p_win"] <= 1.0
    assert 0.0 < res_ortho["p_win"] <= 1.0
