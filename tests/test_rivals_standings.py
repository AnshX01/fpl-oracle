"""
Tests for G8: Rival logic, Standings pagination, and UI/DecisionCard honesty.
Verifies:
1. Leader case: user is 1st with no close chasers -> returns empty list and 'leader, no close chasers', never forced 10 chasers.
2. Leader case with close chasers: strictly respects points window below user.
3. Missing manager: returns explicit 'unavailable' status without falling back to top managers or defaulting rank to 1.
4. Mocked 120-page league: paginates past old 50-page cap without truncation.
5. Partial standings: marks partial coverage when safety limit is reached or page retries fail.
"""

from unittest.mock import AsyncMock, patch

import pytest

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import ClassicLeagueResponse, ClassicStandingResult, StandingsPage
from fpl_oracle.league.rivals import get_rival_set
from fpl_oracle.league.standings import league_standings_manager


@pytest.fixture(autouse=True)
def clean_client():
    """Ensure fpl_client is cleanly terminated to avoid Windows transport exit errors."""
    yield
    try:
        if getattr(fpl_client, "_client", None) is not None and not fpl_client._client.is_closed:
            import asyncio

            asyncio.run(fpl_client.aclose())
    except Exception:
        pass


def test_leader_case_no_close_chasers():
    """User is 1st; all chasers are beyond the points window (20 pts)."""
    user_id = 12345
    standings = [
        {"entry": user_id, "player_name": "Leader", "rank": 1, "total": 500},
        {"entry": 2001, "player_name": "Chaser 1", "rank": 2, "total": 470},  # 30 pts behind
        {"entry": 2002, "player_name": "Chaser 2", "rank": 3, "total": 465},
        {"entry": 2003, "player_name": "Chaser 3", "rank": 4, "total": 460},
        {"entry": 2004, "player_name": "Chaser 4", "rank": 5, "total": 450},
    ]

    rivals, mode, user_rank = get_rival_set(standings, user_manager_id=user_id, points_window=20)
    assert user_rank == 1
    assert rivals == [], "Must not force 10 chasers when user is leader"
    assert mode == "leader, no close chasers"


def test_leader_case_with_close_chasers():
    """User is 1st; only chasers within the points window are included."""
    user_id = 12345
    standings = [
        {"entry": user_id, "player_name": "Leader", "rank": 1, "total": 500},
        {"entry": 2001, "player_name": "Chaser 1", "rank": 2, "total": 492},  # 8 pts behind -> in window
        {"entry": 2002, "player_name": "Chaser 2", "rank": 3, "total": 481},  # 19 pts behind -> in window
        {"entry": 2003, "player_name": "Chaser 3", "rank": 4, "total": 479},  # 21 pts behind -> out of window
        {"entry": 2004, "player_name": "Chaser 4", "rank": 5, "total": 450},
    ]

    rivals, mode, user_rank = get_rival_set(standings, user_manager_id=user_id, points_window=20)
    assert user_rank == 1
    assert mode == "PROXIMITY_WINDOW"
    assert len(rivals) == 2
    assert [r["entry"] for r in rivals] == [2001, 2002]


def test_missing_manager_returns_unavailable():
    """User manager ID is missing from standings -> returns explicit unavailable state."""
    standings = [
        {"entry": 1001, "player_name": "Manager 1", "rank": 1, "total": 500},
        {"entry": 1002, "player_name": "Manager 2", "rank": 2, "total": 480},
    ]

    # When user_manager_id is not found
    rivals, mode, user_rank = get_rival_set(standings, user_manager_id=99999, points_window=20)
    assert rivals == []
    assert mode == "unavailable"
    assert user_rank is None

    # When user_manager_id is None
    rivals_none, mode_none, user_rank_none = get_rival_set(standings, user_manager_id=None, points_window=20)
    assert rivals_none == []
    assert mode_none == "unavailable"
    assert user_rank_none is None


@pytest.mark.anyio
async def test_mocked_120_page_league():
    """Verify LeagueStandingsManager paginates past the old 50-page cap through 120 pages."""

    def make_page_resp(page_num: int):
        has_next = page_num < 120
        results = [
            ClassicStandingResult(
                id=page_num * 1000 + i,
                event_total=50,
                player_name=f"Player P{page_num}_{i}",
                rank=(page_num - 1) * 10 + i,
                last_rank=(page_num - 1) * 10 + i,
                rank_sort=(page_num - 1) * 10 + i,
                total=1000 - ((page_num - 1) * 10 + i),
                entry=page_num * 1000 + i,
                entry_name=f"Team P{page_num}_{i}",
            )
            for i in range(1, 11)
        ]
        return ClassicLeagueResponse(
            league={"id": 777, "name": "Massive 120-Page League", "total_entries": 1200},
            standings=StandingsPage(
                has_next=has_next,
                page=page_num,
                results=results,
            ),
        ), False

    mock_client = AsyncMock()
    mock_client.get_classic_league_standings.side_effect = lambda league_id, page=1: make_page_resp(page)

    with patch("fpl_oracle.league.standings.fpl_client", mock_client):
        res = await league_standings_manager.get_league_standings(league_id=777, max_pages=150)

    assert res["total_teams"] == 1200
    assert res["coverage"]["pages_fetched"] == 120
    assert res["coverage"]["partial"] is False
    assert res["coverage"]["reason"] is None
    assert len(res["standings"]) == 1200


@pytest.mark.anyio
async def test_partial_standings_hit_safety_limit():
    """Verify safety limit halts pagination and sets coverage.partial = True."""

    def make_infinite_page(page_num: int):
        results = [
            ClassicStandingResult(
                id=page_num * 100 + i,
                event_total=50,
                player_name=f"Player P{page_num}_{i}",
                rank=(page_num - 1) * 5 + i,
                last_rank=(page_num - 1) * 5 + i,
                rank_sort=(page_num - 1) * 5 + i,
                total=500 - ((page_num - 1) * 5 + i),
                entry=page_num * 100 + i,
                entry_name=f"Team P{page_num}_{i}",
            )
            for i in range(1, 6)
        ]
        return ClassicLeagueResponse(
            league={"id": 888, "name": "Infinite League", "total_entries": 10000},
            standings=StandingsPage(
                has_next=True,
                page=page_num,
                results=results,
            ),
        ), False

    mock_client = AsyncMock()
    mock_client.get_classic_league_standings.side_effect = lambda league_id, page=1: make_infinite_page(page)

    with patch("fpl_oracle.league.standings.fpl_client", mock_client):
        res = await league_standings_manager.get_league_standings(league_id=888, max_pages=5)

    assert res["coverage"]["pages_fetched"] == 5
    assert res["coverage"]["partial"] is True
    assert "safety limit" in res["coverage"]["reason"].lower()
    assert res["total_teams"] == 25


@pytest.mark.anyio
async def test_partial_standings_failed_page_with_retry():
    """Verify failed page retries 3 times then sets coverage.partial = True."""
    attempts = 0

    def make_page_with_fail(page_num: int):
        nonlocal attempts
        if page_num == 1:
            return ClassicLeagueResponse(
                league={"id": 999, "name": "Faulty League", "total_entries": 100},
                standings=StandingsPage(
                    has_next=True,
                    page=1,
                    results=[
                        ClassicStandingResult(
                            id=101,
                            event_total=50,
                            player_name="P1",
                            rank=1,
                            last_rank=1,
                            rank_sort=1,
                            total=500,
                            entry=101,
                            entry_name="T1",
                        )
                    ],
                ),
            ), False
        else:
            attempts += 1
            raise ConnectionError(f"HTTP 503 Gateway Timeout attempt {attempts}")

    mock_client = AsyncMock()
    mock_client.get_classic_league_standings.side_effect = lambda league_id, page=1: make_page_with_fail(page)

    with patch("fpl_oracle.league.standings.fpl_client", mock_client):
        res = await league_standings_manager.get_league_standings(league_id=999, max_pages=10)

    assert res["coverage"]["pages_fetched"] == 1
    assert res["coverage"]["partial"] is True
    assert "failed to fetch page 2" in res["coverage"]["reason"].lower()
    assert attempts == 3, "Must have retried page 2 three times"
    assert res["total_teams"] == 1


def _make_dummy_projections():
    import pandas as pd

    rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    for i, pos in enumerate(positions, 1):
        rows.append(
            {
                "element": i,
                "web_name": f"Player_{i}",
                "team": (i % 20) + 1,
                "position": pos,
                "expected_points": 5.0,
                "variance": 4.0,
                "value": 60,
                "p10": 2.0,
                "p90": 8.0,
                "exp_defcon_pts": 1.0,
                "is_starter": i <= 11,
                "is_captain": i == 1,
                "is_vice_captain": i == 2,
                "multiplier": 2 if i == 1 else (1 if i <= 11 else 0),
            }
        )
    df = pd.DataFrame(rows)
    return {gw: df.copy() for gw in range(1, 40)}, df


@pytest.mark.anyio
async def test_decision_card_missing_manager_shows_unavailable(configured_advisor):
    """Verify decision card reflects explicit unavailable state when manager is missing from standings."""

    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import ProfileData
    from fpl_oracle.domain.manager_state import manager_state_service
    from fpl_oracle.ml.predict import projection_engine

    multi_proj, single_proj = _make_dummy_projections()
    mock_profile = ProfileData(
        manager_id=99999,
        target_league_id=555,
    )
    mock_state = configured_advisor[0].model_copy(update={"manager_id":99999,"overall_points":500})

    standings_payload = {
        "league_id": 555,
        "league_name": "Test Mini-League",
        "total_teams": 2,
        "coverage": {"pages_fetched": 1, "total_entries": 2, "partial": False, "reason": None},
        "standings": [
            {"entry": 101, "player_name": "Rival 1", "rank": 1, "total": 500},
            {"entry": 102, "player_name": "Rival 2", "rank": 2, "total": 490},
        ],
    }

    with (
        patch("fpl_oracle.briefing.decision_card.data_store.get_profile", return_value=mock_profile),
        patch.object(manager_state_service, "get_current_state", AsyncMock(return_value=mock_state)),
        patch.object(league_standings_manager, "get_league_standings", AsyncMock(return_value=standings_payload)),
        patch.object(projection_engine, "predict_multi_gameweeks", return_value=multi_proj),
        patch.object(projection_engine, "predict_gameweek", return_value=single_proj),
    ):
        card = await decision_card_generator.generate_decision_card()

    assert card["rivals"]["status"] == "unavailable"
    assert "not found in mini-league standings" in card["rivals"]["posture_reason"].lower()
    assert card["rivals"]["coverage"]["pages_fetched"] == 1
    assert card["win_prob"]["status"] == "unavailable"


@pytest.mark.anyio
async def test_decision_card_leader_no_close_chasers(configured_advisor):
    """Verify decision card reflects 'leader, no close chasers' state without forcing 10 rivals."""

    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import ProfileData
    from fpl_oracle.domain.manager_state import manager_state_service
    from fpl_oracle.ml.predict import projection_engine

    multi_proj, single_proj = _make_dummy_projections()
    mock_profile = ProfileData(
        manager_id=101,
        target_league_id=555,
    )
    mock_state = configured_advisor[0].model_copy(update={"manager_id":101,"overall_points":600})

    standings_payload = {
        "league_id": 555,
        "league_name": "Test Mini-League",
        "total_teams": 3,
        "coverage": {"pages_fetched": 1, "total_entries": 3, "partial": False, "reason": None},
        "standings": [
            {"entry": 101, "player_name": "Leader", "rank": 1, "total": 600},
            {"entry": 102, "player_name": "Rival 1", "rank": 2, "total": 550},  # 50 pts behind
            {"entry": 103, "player_name": "Rival 2", "rank": 3, "total": 540},  # 60 pts behind
        ],
    }

    with (
        patch("fpl_oracle.briefing.decision_card.data_store.get_profile", return_value=mock_profile),
        patch.object(manager_state_service, "get_current_state", AsyncMock(return_value=mock_state)),
        patch.object(league_standings_manager, "get_league_standings", AsyncMock(return_value=standings_payload)),
        patch.object(projection_engine, "predict_multi_gameweeks", return_value=multi_proj),
        patch.object(projection_engine, "predict_gameweek", return_value=single_proj),
    ):
        card = await decision_card_generator.generate_decision_card()

    assert card["rivals"]["status"] == "leader, no close chasers"
    assert card["rivals"]["rival_count"] == 0
    assert card["win_prob"]["status"] == "leader, no close chasers"
