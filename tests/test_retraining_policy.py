"""Durable retrain policy tests use mocked candidate training, never fit production weights."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from fpl_oracle.server import retraining


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.setattr(retraining, "STATE_PATH", tmp_path / "state.json")
    boot = SimpleNamespace(
        events=[
            SimpleNamespace(id=5, finished=True, data_checked=True),
            SimpleNamespace(id=6, finished=True, data_checked=False),
        ]
    )
    monkeypatch.setattr(retraining.fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    history = tmp_path / "history.csv"
    history.write_text("fixture\n123\n")
    monkeypatch.setattr(retraining.historical_manager, "output_file", history)
    monkeypatch.setattr(
        retraining.historical_manager,
        "refresh_current_season",
        AsyncMock(return_value={"complete": True, "finalized_gameweeks": [5]}),
    )
    monkeypatch.setattr(retraining.model_registry, "get_active_version", lambda: {"version": "existing"})
    from fpl_oracle.server.analysis import analysis_service

    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    from fpl_oracle.ml import train

    frame = pd.DataFrame({"x": [1]})
    frame.attrs["training_outcome"] = {"promoted": False, "reason": "coverage failed"}
    mock_train = Mock(return_value=(frame, pd.DataFrame()))
    monkeypatch.setattr(train, "train_all_models", mock_train)
    return mock_train


def test_rejected_candidate_is_terminal_durable_and_unfinalized_skipped(policy):
    async def run():
        assert "rejected" in await retraining.retrain_finalized_gameweeks()
        assert "No new" in await retraining.retrain_finalized_gameweeks()

    asyncio.run(run())
    assert policy.call_count == 1
    saved = json.loads(retraining.STATE_PATH.read_text())
    assert saved["gameweeks"]["2026-27:5"]["status"] == "rejected"
    assert "2026-27:6" not in saved["gameweeks"]
    assert saved["gameweeks"]["2026-27:5"]["active_version"] == "existing"


def test_partial_history_never_trains_and_retries_are_bounded(policy, monkeypatch):
    monkeypatch.setattr(
        retraining.historical_manager,
        "refresh_current_season",
        AsyncMock(return_value={"complete": False, "finalized_gameweeks": [5]}),
    )

    async def run():
        with pytest.raises(RuntimeError, match="incomplete"):
            await retraining.retrain_finalized_gameweeks()
        assert "waiting" in await retraining.retrain_finalized_gameweeks()

    asyncio.run(run())
    assert policy.call_count == 0


def test_interrupted_running_record_recovers(policy):
    retraining.STATE_PATH.write_text(
        json.dumps({"schema_version": 1, "gameweeks": {"2026-27:5": {"status": "running"}}})
    )
    asyncio.run(retraining.retrain_finalized_gameweeks())
    assert policy.call_count == 1


def test_committed_promotion_is_reconciled_without_second_fit(policy, monkeypatch):
    retraining.STATE_PATH.write_text(
        json.dumps(
            dict(
                schema_version=1,
                gameweeks={"2026-27:5": dict(status="running", operation_id="same-operation", covered_gameweeks=[5])},
            )
        )
    )
    monkeypatch.setattr(
        retraining.model_registry,
        "get_active_version",
        lambda: dict(version="new", training_provenance=dict(operation_id="same-operation")),
    )
    assert "Recovered" in asyncio.run(retraining.retrain_finalized_gameweeks())
    assert policy.call_count == 0
    assert json.loads(retraining.STATE_PATH.read_text())["gameweeks"]["2026-27:5"]["status"] == "promoted"
