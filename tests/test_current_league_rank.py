from fpl_oracle.league.standings import current_manager_rank


def test_current_rank_uses_owner_entry_not_overall_or_simulation():
    data = {"standings": [{"entry": 11, "rank": 2}, {"entry": 22, "rank": 7}]}
    assert current_manager_rank(data, 22) == {"current_league_rank": 7, "league_rank_status": "available"}
    assert current_manager_rank(data, 99) == {"current_league_rank": None, "league_rank_status": "not_listed"}
    assert current_manager_rank(data, None)["league_rank_status"] == "manager_unconfigured"
    assert current_manager_rank({**data, "coverage": {"partial": True}}, 99)["league_rank_status"] == "unavailable"
    assert current_manager_rank({"standings": [{"entry": 22, "rank": None}]}, 22)["current_league_rank"] is None


async def _endpoint_case(monkeypatch, manager_id):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import pandas as pd

    from fpl_oracle.server.routes import api

    state = SimpleNamespace(
        manager_id=manager_id,
        overall_points=338,
        overall_rank=123456,
        squad=[],
        to_squad_dataframe=lambda: pd.DataFrame({"element": range(1, 16)}),
    )
    monkeypatch.setattr(api.manager_state_service, "get_current_state", AsyncMock(return_value=state))
    monkeypatch.setattr(api.data_store, "get_profile", lambda: SimpleNamespace(target_league_id=55))
    monkeypatch.setattr(
        api.league_standings_manager,
        "get_league_standings",
        AsyncMock(
            return_value={
                "league_name": "Test League",
                "total_teams": 2,
                "standings": [{"entry": 11, "rank": 1, "total": 400}, {"entry": 22, "rank": 7, "total": 338}],
            }
        ),
    )
    monkeypatch.setattr(api.fpl_client, "get_bootstrap_static", AsyncMock(return_value=(None, False)))
    monkeypatch.setattr(api.fpl_client, "get_fixtures", AsyncMock(return_value=([], False)))
    monkeypatch.setattr(api.fpl_client, "get_current_and_next_gw", AsyncMock(return_value=(6, 7)))
    monkeypatch.setattr(api.fpl_client, "get_data_as_of", lambda _: None)
    monkeypatch.setattr(
        api.rival_analyzer,
        "analyze_rivals",
        AsyncMock(return_value={"rival_squads": [], "template_players": [], "differential_players": []}),
    )
    monkeypatch.setattr(api.analysis_service, "projections", AsyncMock(return_value={}))
    monkeypatch.setattr(
        api.monte_carlo_simulator,
        "simulate_league",
        lambda **kwargs: {"status": "SIMULATION_SUCCESS", "user_win_probability_pct": 5.0, "expected_final_rank": 6.2},
    )
    return await api.get_league_intel()


def test_endpoint_exposes_current_rank_outside_real_simulation_shape(monkeypatch):
    import asyncio

    response = asyncio.run(_endpoint_case(monkeypatch, 22))
    assert response["overall_rank"] == 123456
    assert response["current_league_rank"] == 7
    assert response["league_rank_status"] == "available"
    assert "user_rank" not in response["simulation"]
    assert response["simulation"]["user_win_probability_pct"] == 5.0


def test_missing_owner_never_exposes_overall_rank_as_league_rank(monkeypatch):
    import asyncio

    response = asyncio.run(_endpoint_case(monkeypatch, 99))
    assert response["overall_rank"] == 123456
    assert response["current_league_rank"] is None
    assert response["league_rank_status"] == "not_listed"
