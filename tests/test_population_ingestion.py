import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import PlayerMatchHistory
from fpl_oracle.data.historical import HistoricalDataManager


def test_zero_minutes_player_is_included_in_training_population(monkeypatch):
    never_played = SimpleNamespace(
        id=1,
        minutes=0,
        total_points=0,
        first_name="Unused",
        second_name="Player",
        web_name="Unused",
        team=1,
        element_type=1,
    )
    boot = SimpleNamespace(
        elements=[never_played],
        teams=[SimpleNamespace(id=1, name="Club")],
        element_types=[SimpleNamespace(id=1, singular_name_short="GKP")],
    )
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    summary = SimpleNamespace(history=[PlayerMatchHistory(element=1, fixture=1, opponent_team=2, round=1)])
    fetch = AsyncMock(return_value=(summary, False))
    monkeypatch.setattr(fpl_client, "get_element_summary", fetch)
    result = asyncio.run(HistoricalDataManager()._fetch_live_season_async())
    assert len(result) == 1 and result.iloc[0]["minutes"] == 0 and result.iloc[0]["total_points"] == 0
    fetch.assert_awaited_once_with(1, force_refresh=True)


def test_refresh_preserves_prior_season_double_gameweek_rows(tmp_path, monkeypatch):
    import pandas as pd

    manager = HistoricalDataManager()
    manager.output_file = tmp_path / "history.csv"
    # Old corpus has two legitimate matches for one player in the same GW,
    # without fixture IDs. Both must survive a current-season refresh.
    old = pd.DataFrame(
        [
            dict(season="2025-26", element=1, round=1, total_points=2),
            dict(season="2025-26", element=1, round=1, total_points=8),
        ]
    )
    old.to_csv(manager.output_file, index=False)
    boot = SimpleNamespace(events=[SimpleNamespace(id=1, finished=True, data_checked=True)])
    fixture = SimpleNamespace(id=2, event=1, finished=True)
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([fixture], False)))
    fresh = pd.DataFrame([dict(season="2026-27", element=1, round=1, fixture=2, total_points=3)])
    monkeypatch.setattr(manager, "_fetch_live_season_async", AsyncMock(return_value=fresh))
    result = asyncio.run(manager.refresh_current_season())
    assert result["complete"]
    saved = pd.read_csv(manager.output_file)
    assert len(saved[saved.season == "2025-26"]) == 2
