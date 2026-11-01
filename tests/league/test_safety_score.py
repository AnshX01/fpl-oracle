"""
Tests for Live Matchday Safety Score & Arrow Threshold Engine (LiveFPL / Ragabolly Parity).
"""

from fpl_oracle.league.live_matchday import (
    LiveManagerSquad,
    LivePlayerScore,
    live_matchday_engine,
)


def _make_dummy_player(element_id: int, web_name: str, points: int) -> LivePlayerScore:
    return LivePlayerScore(
        element_id=element_id,
        web_name=web_name,
        position="MID",
        team_id=1,
        minutes=90,
        goals=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        bps=10,
        official_bonus=0,
        provisional_bonus=0,
        live_points=points,
        status="FINISHED",
    )


def test_calculate_safety_score_green_arrow():
    # 3 managers in pool:
    # Manager 1 (User): 60 pts
    # Manager 2: 40 pts
    # Manager 3: 50 pts
    # Field average / Safety Score = (60 + 40 + 50) / 3 = 50.0 pts
    # User gap = +10.0 pts -> GREEN_ARROW

    p1 = _make_dummy_player(101, "Talisman A", 10)
    p2 = _make_dummy_player(102, "Talisman B", 8)

    squads = {
        1: LiveManagerSquad(
            manager_id=1,
            manager_name="User",
            team_name="Team User",
            active_chip=None,
            transfer_cost=0,
            live_gw_gross=60,
            live_gw_net=60,
            live_total_points=500,
            starters=[p1, p2],
            bench=[],
            autosubs=[],
            effective_captain_id=101,
            players_played=11,
            players_playing=0,
            players_remaining=0,
        ),
        2: LiveManagerSquad(
            manager_id=2,
            manager_name="Rival 1",
            team_name="Team 1",
            active_chip=None,
            transfer_cost=4,
            live_gw_gross=44,
            live_gw_net=40,
            live_total_points=480,
            starters=[p1],
            bench=[],
            autosubs=[],
            effective_captain_id=101,
            players_played=11,
            players_playing=0,
            players_remaining=0,
        ),
        3: LiveManagerSquad(
            manager_id=3,
            manager_name="Rival 2",
            team_name="Team 2",
            active_chip=None,
            transfer_cost=0,
            live_gw_gross=50,
            live_gw_net=50,
            live_total_points=490,
            starters=[p2],
            bench=[],
            autosubs=[],
            effective_captain_id=102,
            players_played=11,
            players_playing=0,
            players_remaining=0,
        ),
    }

    safety = live_matchday_engine.calculate_safety_score(squads, user_manager_id=1)
    assert safety is not None
    assert safety.safety_score == 50.0
    assert safety.user_live_net_points == 60
    assert safety.safety_gap == 10.0
    assert safety.projected_arrow == "GREEN_ARROW"
    assert "Talisman A" in safety.field_effective_ownership
    # Talisman A was captained by M1 (2x) and M2 (2x) = 4 / 3 = 133.3% EO
    assert safety.field_effective_ownership["Talisman A"] == 133.3


def test_calculate_safety_score_red_arrow():
    # User at 35 pts when safety threshold is 48.0 pts -> RED_ARROW
    p1 = _make_dummy_player(1, "Player 1", 5)
    squads = {
        10: LiveManagerSquad(
            manager_id=10,
            manager_name="Trailing User",
            team_name="User",
            active_chip=None,
            transfer_cost=4,
            live_gw_gross=39,
            live_gw_net=35,
            live_total_points=400,
            starters=[p1],
            bench=[],
            autosubs=[],
            effective_captain_id=1,
            players_played=11,
            players_playing=0,
            players_remaining=0,
        ),
        20: LiveManagerSquad(
            manager_id=20,
            manager_name="Leader Rival",
            team_name="Rival",
            active_chip=None,
            transfer_cost=0,
            live_gw_gross=61,
            live_gw_net=61,
            live_total_points=450,
            starters=[p1],
            bench=[],
            autosubs=[],
            effective_captain_id=1,
            players_played=11,
            players_playing=0,
            players_remaining=0,
        ),
    }

    safety = live_matchday_engine.calculate_safety_score(squads, user_manager_id=10)
    assert safety is not None
    assert safety.safety_score == 48.0
    assert safety.user_live_net_points == 35
    assert safety.safety_gap == -13.0
    assert safety.projected_arrow == "RED_ARROW"
    assert "Deficit of 13.0 pts" in safety.summary_text
