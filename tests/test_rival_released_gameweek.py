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


def test_tied_leaders_stay_in_rival_set():
    from fpl_oracle.league.rivals import get_rival_set

    rows = [dict(entry=2, total=100, rank=1), dict(entry=1, total=100, rank=1), dict(entry=3, total=95, rank=3)]
    selected, mode, rank = get_rival_set(rows, 1, points_window=20)
    assert rank == 1 and mode == "PROXIMITY_WINDOW"
    assert [r["entry"] for r in selected] == [2, 3]


def test_chip_status_is_per_set():
    chips = lambda *pairs: [{"name": n, "event": e} for n, e in pairs]  # noqa: E731
    analyzer = RivalAnalyzer()
    # GW10: only set 1 matters, a chip used in set 1 is gone
    early = analyzer.calculate_chips_status(chips(("wildcard", 5), ("bboost", 8)), 10)
    assert early["chips_remaining"] == ["freehit", "triple_captain"]
    # GW25: set 1 chips (used or not) no longer count; a GW21 free hit is spent, everything else is fresh
    late = analyzer.calculate_chips_status(chips(("wildcard", 5), ("freehit", 21)), 25)
    assert late["chips_remaining"] == ["wildcard_set2", "bench_boost", "triple_captain"]
    assert late["set1_wildcard_expired"] is False
    assert analyzer.calculate_chips_status([], 25)["set1_wildcard_expired"] is True


def test_free_hit_and_bench_boost_picks_are_not_hold_squads(monkeypatch):
    def pick(element, slot, mult):
        return SimpleNamespace(element=element, position=slot, multiplier=mult, is_captain=slot == 1)

    fh = SimpleNamespace(active_chip="freehit", picks=[pick(900, 1, 2)])
    prior = SimpleNamespace(active_chip=None, picks=[pick(e, e, 1) for e in range(1, 16)])
    bb = SimpleNamespace(active_chip="bboost", picks=[pick(e, e, 1) for e in range(1, 16)])
    boot = SimpleNamespace(
        elements=[],
        events=[SimpleNamespace(id=5, finished=True, deadline_time="2026-09-18T17:30:00Z", is_next=False)]
        + [SimpleNamespace(id=6, finished=False, deadline_time="2099-10-10T10:00:00Z", is_next=True)],
    )

    async def fake_picks(entry, gw):
        return {(2, 5): fh, (2, 4): prior, (3, 5): bb}[(entry, gw)], False

    monkeypatch.setattr(fpl_client, "get_manager_picks", fake_picks)
    monkeypatch.setattr(fpl_client, "get_manager_history", AsyncMock(return_value=(SimpleNamespace(chips=[]), False)))
    rows = [dict(entry=2, total=110, rank=1), dict(entry=3, total=105, rank=2), dict(entry=1, total=100, rank=3)]
    res = asyncio.run(RivalAnalyzer().analyze_rivals(rows, 1, 6, boot, points_window=30))
    by_entry = {r["entry_id"]: r for r in res["rival_squads"]}
    assert by_entry[2]["observed_gameweek"] == 4 and {p["element"] for p in by_entry[2]["squad"]} != {900}
    assert sum(p["is_starter"] for p in by_entry[3]["squad"]) == 11
    assert res["rivals_requested"] == 2 and res["rival_points_window"] == 30
