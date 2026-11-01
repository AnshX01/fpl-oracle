"""
Unit tests for WhatIfEngine.
Verifies positional goal/assist/CS scoring, captaincy doubling,
mini-league rank movement recalculation, and narrative generation.
"""

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Team
from fpl_oracle.league.live_matchday import (
    LiveLeagueRow,
    LiveManagerSquad,
    LiveMatchdaySnapshot,
    LivePlayerScore,
)
from fpl_oracle.league.what_if import (
    HypotheticalEvent,
    calculate_points_from_events,
    what_if_engine,
)


def make_mock_bootstrap():
    elements = [
        Element(id=1, web_name="PlayerA", element_type=4, team=1, now_cost=100),  # FWD
        Element(id=2, web_name="PlayerB", element_type=3, team=1, now_cost=80),  # MID
        Element(id=3, web_name="PlayerC", element_type=2, team=2, now_cost=50),  # DEF
    ]
    element_types = [
        ElementType(
            id=2, singular_name_short="DEF", singular_name="Defender", plural_name="Defenders", plural_name_short="DEF"
        ),
        ElementType(
            id=3,
            singular_name_short="MID",
            singular_name="Midfielder",
            plural_name="Midfielders",
            plural_name_short="MID",
        ),
        ElementType(
            id=4, singular_name_short="FWD", singular_name="Forward", plural_name="Forwards", plural_name_short="FWD"
        ),
    ]
    teams = [
        Team(id=1, name="Team1", short_name="T1"),
        Team(id=2, name="Team2", short_name="T2"),
    ]
    return BootstrapStatic(elements=elements, element_types=element_types, teams=teams, events=[])


def test_calculate_points_from_events_positional():
    # FWD: 1 goal (4), 1 assist (3), 3 bonus -> 10 pts
    ev_fwd = HypotheticalEvent(element_id=1, goals=1, assists=1, bonus=3)
    assert calculate_points_from_events("FWD", ev_fwd) == 10

    # MID: 1 goal (5), 1 CS (1), 2 bonus -> 8 pts
    ev_mid = HypotheticalEvent(element_id=2, goals=1, clean_sheets=1, bonus=2)
    assert calculate_points_from_events("MID", ev_mid) == 8

    # DEF: 1 goal (6), 1 CS (4), 3 bonus -> 13 pts
    ev_def = HypotheticalEvent(element_id=3, goals=1, clean_sheets=1, bonus=3)
    assert calculate_points_from_events("DEF", ev_def) == 13


def test_what_if_rank_overtake():
    boot = make_mock_bootstrap()

    # User is in 2nd place (526 pts), rival is in 1st place (529 pts)
    # User owns PlayerA (FWD) as captain!
    user_squad = LiveManagerSquad(
        manager_id=100,
        manager_name="User",
        team_name="My Team",
        active_chip=None,
        transfer_cost=0,
        live_gw_gross=50,
        live_gw_net=50,
        live_total_points=526,
        starters=[
            LivePlayerScore(
                element_id=1,
                web_name="PlayerA",
                position="FWD",
                team_id=1,
                minutes=90,
                goals=0,
                assists=0,
                clean_sheets=0,
                goals_conceded=0,
                bps=10,
                official_bonus=0,
                provisional_bonus=0,
                live_points=2,
                status="FINISHED",
                multiplier=2,
            )
        ],
        bench=[],
        autosubs=[],
        effective_captain_id=1,
        players_played=1,
        players_playing=0,
        players_remaining=0,
    )

    standings = [
        LiveLeagueRow(
            live_rank=1,
            previous_rank=1,
            rank_movement=0,
            manager_id=200,
            manager_name="Rival Leader",
            team_name="Rival Team",
            live_gw_points=40,
            live_total_points=529,
            gap_to_leader=0,
            players_remaining=0,
            captain_name="PlayerB",
            active_chip=None,
        ),
        LiveLeagueRow(
            live_rank=2,
            previous_rank=2,
            rank_movement=0,
            manager_id=100,
            manager_name="User",
            team_name="My Team",
            live_gw_points=50,
            live_total_points=526,
            gap_to_leader=3,
            players_remaining=0,
            captain_name="PlayerA",
            active_chip=None,
        ),
    ]

    snapshot = LiveMatchdaySnapshot(
        gameweek=9,
        league_id=435956,
        league_name="IceMakers League",
        user_manager_id=100,
        user_live_rank=2,
        user_live_total=526,
        is_live=True,
        fixtures_summary=[],
        standings=standings,
        user_squad=user_squad,
    )

    # Hypothesis: PlayerA (User's Captain) scores 1 goal (4 pts) + 3 bonus -> 7 base pts * 2 = 14 pts!
    events = [HypotheticalEvent(element_id=1, goals=1, bonus=3)]
    res = what_if_engine.simulate_events(snapshot, events, boot)

    assert res.user_rank_before == 2
    assert res.user_rank_after == 1  # User moved to 1st!
    assert res.user_rank_delta == 1
    assert res.user_points_delta == 14
    assert res.gap_to_leader_after == 0
    assert "elevating your rank from #2 to #1" in res.summary_narrative
