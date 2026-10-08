import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from fpl_oracle.optimise.hit_ledger import HitLedger


def test_actual_hits_settle_without_inventing_expected_gain(monkeypatch):
    from fpl_oracle.api.fpl_client import fpl_client

    ledger = HitLedger()
    records = []
    monkeypatch.setattr(ledger, "read", lambda mid: list(records))
    monkeypatch.setattr(ledger, "write", lambda mid, r: records.append(r))
    monkeypatch.setattr(
        fpl_client,
        "get_manager_history",
        AsyncMock(return_value=(NS(current=[NS(event=6, event_transfers_cost=4)]), False)),
    )
    monkeypatch.setattr(
        fpl_client, "get_manager_transfers", AsyncMock(return_value=([NS(event=6, element_in=2, element_out=1)], False))
    )
    monkeypatch.setattr(
        fpl_client,
        "get_bootstrap_static",
        AsyncMock(return_value=(NS(events=[NS(id=6, finished=True, data_checked=True)]), False)),
    )
    monkeypatch.setattr(
        fpl_client,
        "get_live_gameweek",
        AsyncMock(
            return_value=(
                {"elements": [{"id": 1, "stats": {"total_points": 5}}, {"id": 2, "stats": {"total_points": 7}}]},
                False,
            )
        ),
    )
    result = asyncio.run(ledger.refresh(123))
    assert result["cooldown_until"] == 13
    assert records[0]["status"] == "failed"
    assert records[0]["expected_gain"] is None
    assert not records[0]["matched_recommendation"]


def test_recommendations_are_not_taken_hits():
    ledger = HitLedger()
    captured = []
    ledger.write = lambda mid, r: captured.append(r)
    ledger.recommend(
        123,
        {
            "hit_cost": 4,
            "trajectory": [{"transfers_in": [2], "transfers_out": [1], "hit_cost": 4, "expected_player_gain": 8}],
        },
        6,
    )
    assert captured[0]["kind"] == "recommended"
    assert captured[0]["status"] == "not_verified_taken"
