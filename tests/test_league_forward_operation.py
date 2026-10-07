import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.league.forward import score_pending


def test_scoring_uses_official_cumulative_score_not_double_subtracted_hits(tmp_path, monkeypatch):
    from fpl_oracle.api.fpl_client import fpl_client

    record = dict(
        schema_version=1,
        source_kind="official_forward",
        frozen_at="2026-10-01T00:00:00+00:00",
        deadline_utc="2026-10-02T00:00:00+00:00",
        target_gameweek=6,
        scope="actual_observed_manager_outcome",
        probability=0.6,
        owner_manager_id=1,
        observed_manager_ids=[1, 2],
    )
    identifier = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    forecast = tmp_path / f"{identifier}.json"
    forecast.write_text(json.dumps(record))
    monkeypatch.setattr(
        fpl_client,
        "get_bootstrap_static",
        AsyncMock(
            return_value=(SimpleNamespace(events=[SimpleNamespace(id=6, finished=True, data_checked=True)]), False)
        ),
    )
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([SimpleNamespace(finished=True)], False)))

    async def history(manager, force_refresh=False):
        assert force_refresh
        return SimpleNamespace(
            current=[
                SimpleNamespace(event=6, total_points=50 if manager == 1 else 49, points=10, event_transfers_cost=8)
            ]
        ), False

    monkeypatch.setattr(fpl_client, "get_manager_history", history)
    summary = asyncio.run(score_pending(tmp_path))
    assert summary["count"] == 1
    result = json.loads(forecast.with_suffix(".score.json").read_text())
    assert result["outcome"] == 1
    assert result["scores"] == {"1": 50, "2": 49}
    assert asyncio.run(score_pending(tmp_path))["count"] == 1
    result["brier"] = 0
    forecast.with_suffix(".score.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(score_pending(tmp_path))
    forecast.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        asyncio.run(score_pending(tmp_path))


def test_pending_finalization_does_not_score(tmp_path, monkeypatch):
    from fpl_oracle.api.fpl_client import fpl_client

    record = dict(target_gameweek=6)
    identifier = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    forecast = tmp_path / f"{identifier}.json"
    forecast.write_text(json.dumps(record))
    monkeypatch.setattr(
        fpl_client,
        "get_bootstrap_static",
        AsyncMock(
            return_value=(SimpleNamespace(events=[SimpleNamespace(id=6, finished=True, data_checked=False)]), False)
        ),
    )
    assert asyncio.run(score_pending(tmp_path))["count"] == 0
    assert not forecast.with_suffix(".score.json").exists()


def test_capture_reuses_immutable_index_without_fetching(tmp_path, monkeypatch):
    from fpl_oracle.data.store import data_store
    from fpl_oracle.league.forward import capture_forward

    monkeypatch.setattr(data_store, "get_profile", lambda: SimpleNamespace(manager_id=1, target_league_id=2))
    from fpl_oracle.league.validation import freeze_forecast

    now = datetime.now(UTC)
    saved = freeze_forecast(
        dict(
            deadline_utc=(now + timedelta(hours=1)).isoformat(),
            source_kind="official_forward",
            snapshot_id="test",
            owner_manager_id=1,
            league_id=2,
            observed_manager_ids=[1, 2],
            target_gameweek=6,
            probability=0.5,
        ),
        tmp_path,
        now,
    )
    (tmp_path / "gw6-1-2.index.json").write_text(json.dumps(saved))
    assert asyncio.run(capture_forward(6, tmp_path)) == saved
    with open(saved["path"], "w") as output:
        output.write("{}")
    import pytest

    with pytest.raises(ValueError, match="hash"):
        asyncio.run(capture_forward(6, tmp_path))


def test_capture_rejects_already_passed_deadline(tmp_path, monkeypatch):
    import pytest

    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.data.store import data_store
    from fpl_oracle.league.forward import capture_forward

    monkeypatch.setattr(data_store, "get_profile", lambda: SimpleNamespace(manager_id=1, target_league_id=2))
    deadline = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    monkeypatch.setattr(
        fpl_client,
        "get_bootstrap_static",
        AsyncMock(return_value=(SimpleNamespace(events=[SimpleNamespace(id=6, deadline_time=deadline)]), False)),
    )
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([], False)))
    with pytest.raises(ValueError, match="too late"):
        asyncio.run(capture_forward(6, tmp_path))


def test_legacy_player_freeze_failure_does_not_disable_league_capture(monkeypatch):
    from fpl_oracle.league import forward
    from fpl_oracle.ml import holdout
    from fpl_oracle.server import jobs

    state = SimpleNamespace(stale=False, seconds_to_deadline=3600)
    monkeypatch.setattr(jobs.game_state_manager, "get_game_state", AsyncMock(return_value=state))
    monkeypatch.setattr(jobs.fpl_client, "get_current_and_next_gw", AsyncMock(return_value=(5, 6)))
    monkeypatch.setattr(holdout, "ensure_holdout_log_initialized", lambda: None)
    monkeypatch.setattr(holdout, "freeze_predictions", AsyncMock(side_effect=ValueError("legacy freeze retained")))
    capture = AsyncMock(return_value=dict(status="captured"))
    score = AsyncMock(return_value=dict(count=0))
    monkeypatch.setattr(forward, "capture_forward", capture)
    monkeypatch.setattr(forward, "score_pending", score)
    monkeypatch.setattr(jobs.data_store, "record_job_start", lambda *a: 1)
    monkeypatch.setattr(jobs.data_store, "record_job_finish", lambda *a: None)
    asyncio.run(jobs.holdout_forward_job())
    capture.assert_awaited_once_with(6)
    score.assert_awaited_once()
