import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.league.rivals import RivalAnalyzer


def test_current_and_target_callers_use_same_latest_released_picks(monkeypatch):
    boot = SimpleNamespace(
        elements=[],
        events=[
            SimpleNamespace(id=4, finished=True, deadline_time="2026-09-12T12:30:00Z", is_next=False),
            SimpleNamespace(id=5, finished=True, deadline_time="2026-09-18T17:30:00Z", is_next=False),
            SimpleNamespace(id=6, finished=False, deadline_time="2099-10-10T10:00:00Z", is_next=True),
        ],
    )
    picks = AsyncMock(return_value=(SimpleNamespace(picks=[]), False))
    monkeypatch.setattr(fpl_client, "get_manager_picks", picks)
    monkeypatch.setattr(fpl_client, "get_manager_history", AsyncMock(return_value=(SimpleNamespace(chips=[]), False)))
    rows = [dict(entry=2, total=110, rank=1), dict(entry=1, total=100, rank=2)]
    for arg in [5, 6]:
        asyncio.run(RivalAnalyzer().analyze_rivals(rows, 1, arg, boot))
    assert [c.args[1] for c in picks.await_args_list] == [5, 5]
