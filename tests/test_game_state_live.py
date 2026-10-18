import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fpl_oracle.api import fpl_client as fc
from fpl_oracle.api.game_state import GameStateManager


def test_cached_fixtures_refreshed_when_a_match_may_be_live(monkeypatch):
    kick = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    cached = [SimpleNamespace(event=7, finished=False, started=False, kickoff_time=kick, team_h=1, team_a=2)]
    fresh = [SimpleNamespace(event=7, finished=False, started=True, kickoff_time=kick, team_h=1, team_a=2)]
    calls = []

    async def boot(force_refresh=False):
        return SimpleNamespace(
            events=[
                SimpleNamespace(id=7, is_current=True, is_next=False, deadline_time=None),
                SimpleNamespace(id=8, is_current=False, is_next=True, deadline_time="2099-01-01T00:00:00Z"),
            ]
        ), False

    async def fixtures(force_refresh=False):
        calls.append(force_refresh)
        return (fresh if force_refresh else cached), False

    async def status():
        return {"status": [], "leagues": "Unknown"}, False

    monkeypatch.setattr(fc.fpl_client, "get_bootstrap_static", boot)
    monkeypatch.setattr(fc.fpl_client, "get_fixtures", fixtures)
    monkeypatch.setattr(fc.fpl_client, "get_event_status", status)
    st = asyncio.run(GameStateManager().get_game_state())
    assert st.is_live and st.phase.value == "LIVE"
    assert calls == [False, True]
