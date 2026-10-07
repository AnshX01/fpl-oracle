"""Forward provenance contracts. Fixtures are research evidence, never live results."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.ml.holdout import (
    ensure_holdout_log_initialized,
    freeze_predictions,
    score_frozen_predictions,
)


def test_initialize_awaiting(tmp_path):
    with (
        patch("fpl_oracle.ml.holdout.HOLDOUT_LOG_PATH", tmp_path / "log.json"),
        patch("fpl_oracle.ml.holdout.REPORTS_DIR", tmp_path),
    ):
        result = ensure_holdout_log_initialized()
        assert result["evaluated_gameweeks"] == [] and result["status"] == "AWAITING_FORWARD_GAMEWEEKS"


def test_freeze_target_deadline_and_snapshot(tmp_path):
    async def run():
        now = datetime.now(UTC)
        boot = BootstrapStatic.model_validate(
            {
                "elements": [],
                "teams": [],
                "element_types": [],
                "events": [{"id": 7, "name": "GW7", "deadline_time": (now + timedelta(days=1)).isoformat()}],
            }
        )
        df = pd.DataFrame([{"element": 1, "expected_points": 4, "web_name": "Fixture", "p10": 1, "p50": 4, "p90": 8}])
        df.attrs["snapshot_id"] = "fixture"
        df.attrs["snapshot_inputs"] = {"source_kind": "test_fixture"}
        with (
            patch("fpl_oracle.ml.holdout.fpl_client.get_bootstrap_static", new=AsyncMock(return_value=(boot, False))),
            patch("fpl_oracle.ml.holdout.fpl_client.get_fixtures", new=AsyncMock(return_value=([], False))),
            patch("fpl_oracle.server.analysis.analysis_service.projections", new=AsyncMock(return_value={7: df})),
            patch(
                "fpl_oracle.server.analysis.analysis_service._snapshots",
                {"fixture": {"snapshot_id": "fixture", "inputs": {"source_kind": "test_fixture"}}},
            ),
        ):
            payload = await freeze_predictions(7, custom_holdout_dir=tmp_path)
            assert payload["schema_version"] == 2 and payload["snapshot_id"] == "fixture"
            assert datetime.fromisoformat(payload["frozen_at"]) < datetime.fromisoformat(payload["deadline_utc"])
            before = (tmp_path / "gw7_predictions.json").read_bytes()
            assert await freeze_predictions(7, force=True, custom_holdout_dir=tmp_path) == payload
            assert before == (tmp_path / "gw7_predictions.json").read_bytes()

    asyncio.run(run())


def test_late_target_refused(tmp_path):
    async def run():
        boot = BootstrapStatic.model_validate(
            {
                "elements": [],
                "teams": [],
                "element_types": [],
                "events": [
                    {"id": 7, "name": "GW7", "deadline_time": (datetime.now(UTC) - timedelta(seconds=1)).isoformat()}
                ],
            }
        )
        with (
            patch("fpl_oracle.ml.holdout.fpl_client.get_bootstrap_static", new=AsyncMock(return_value=(boot, False))),
            patch("fpl_oracle.ml.holdout.fpl_client.get_fixtures", new=AsyncMock(return_value=([], False))),
        ):
            with pytest.raises(ValueError, match="Late/unknown"):
                await freeze_predictions(7, custom_holdout_dir=tmp_path)
            assert not (tmp_path / "gw7_predictions.json").exists()

    asyncio.run(run())


def test_synthetic_score_dry_run_only(tmp_path):
    payload = {
        "gameweek": 6,
        "predictions": [{"element": i, "expected_points": i, "p10": i - 1, "p90": i + 1} for i in range(10)],
    }
    (tmp_path / "gw6_predictions.json").write_text(json.dumps(payload))
    actuals = {i: i + 0.5 for i in range(10)}
    score = score_frozen_predictions(
        6, actuals, dry_run=True, custom_holdout_dir=tmp_path, custom_log_path=tmp_path / "log.json"
    )
    assert score["sample_count"] == 10 and score["ml_mae"] == 0.5
    assert score["actual_source"] == "research_only" and not (tmp_path / "log.json").exists()
    with pytest.raises(ValueError, match="Legacy/synthetic"):
        score_frozen_predictions(6, actuals, custom_holdout_dir=tmp_path)


def test_partial_actuals_not_scored(tmp_path):
    payload = {
        "gameweek": 6,
        "predictions": [{"element": i, "expected_points": i, "p10": i - 1, "p90": i + 1} for i in range(10)],
    }
    (tmp_path / "gw6_predictions.json").write_text(json.dumps(payload))
    assert score_frozen_predictions(6, {1: 2}, dry_run=True, custom_holdout_dir=tmp_path) is None
