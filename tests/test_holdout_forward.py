"""
Unit tests for Requirement G6: Forward Holdout Logging & Honest Evaluation.

Verifies:
1. reports/holdout_log.json initialization and honest "AWAITING_FORWARD_GAMEWEEKS" status.
2. Freezing served predictions to data/holdout/gw<N>_predictions.json with manifest hash.
3. Scoring frozen predictions against synthetic actual points and computing honest MAE, RMSE, Spearman, coverage.
4. Scheduler on-demand wiring for holdout_forward job.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from fpl_oracle.ml.holdout import (
    ensure_holdout_log_initialized,
    freeze_predictions,
    score_frozen_predictions,
)
from fpl_oracle.server.jobs import run_job_on_demand


def test_ensure_holdout_log_initialized(tmp_path: Path):
    """Verify holdout log initialization with honest status when no forward evaluations exist."""
    log_path = tmp_path / "holdout_log.json"
    with (
        patch("fpl_oracle.ml.holdout.HOLDOUT_LOG_PATH", log_path),
        patch("fpl_oracle.ml.holdout.REPORTS_DIR", tmp_path),
    ):
        log_data = ensure_holdout_log_initialized(next_gw=6)
        assert log_path.exists()
        assert log_data["status"] == "AWAITING_FORWARD_GAMEWEEKS"
        assert log_data["evaluated_gameweeks"] == []
        assert "contaminated" in log_data["note"].lower()
        assert log_data["next_scheduled_freeze_gw"] == 6


@pytest.mark.anyio
async def test_freeze_predictions_synthetic(tmp_path: Path):
    """Verify freezing served predictions serializes expected schema and manifest hash."""
    holdout_dir = tmp_path / "holdout"

    # Mock bootstrap, fixtures, and projection engine
    mock_bootstrap = AsyncMock()
    mock_fixtures = AsyncMock()

    class MockRow:
        def __init__(self, elem, name, team, pos, xp, p10, p50, p90):
            self.d = {
                "element": elem,
                "web_name": name,
                "team": team,
                "position": pos,
                "expected_points": xp,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "opponent_difficulty": 3.0,
                "chance_of_playing": 1.0,
            }

        def __getitem__(self, item):
            return self.d[item]

        def get(self, item, default=None):
            return self.d.get(item, default)

    class MockPredsDF:
        empty = False

        def iterrows(self):
            rows = [
                MockRow(1, "Raya", "Arsenal", "GKP", 4.5, 2.0, 4.0, 7.0),
                MockRow(2, "Saliba", "Arsenal", "DEF", 5.0, 2.5, 4.8, 8.0),
                MockRow(3, "Saka", "Arsenal", "MID", 6.8, 3.0, 6.5, 11.0),
                MockRow(4, "Haaland", "Man City", "FWD", 8.2, 4.0, 8.0, 13.0),
            ]
            yield from enumerate(rows)

    with (
        patch("fpl_oracle.ml.holdout.fpl_client.get_bootstrap_static", AsyncMock(return_value=(mock_bootstrap, False))),
        patch("fpl_oracle.ml.holdout.fpl_client.get_fixtures", AsyncMock(return_value=([mock_fixtures], False))),
        patch("fpl_oracle.ml.holdout.projection_engine.predict_gameweek", return_value=MockPredsDF()),
        patch(
            "fpl_oracle.ml.holdout.model_registry.get_active_version",
            return_value={"version": "v1.0.0", "git_commit": "abc1234"},
        ),
        patch("fpl_oracle.ml.holdout.compute_manifest_sha256", return_value="fake_sha256_hash_12345"),
    ):
        res = await freeze_predictions(target_gw=7, custom_holdout_dir=holdout_dir)

    assert res["player_count"] == 4
    target_file = holdout_dir / "gw7_predictions.json"
    assert target_file.exists()

    with open(target_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["gameweek"] == 7
    assert data["model_version"] == "v1.0.0"
    assert data["git_commit"] == "abc1234"
    assert data["manifest_hash"] == "fake_sha256_hash_12345"
    assert data["player_count"] == 4
    assert len(data["predictions"]) == 4
    assert data["predictions"][0]["element"] == 1
    assert data["predictions"][0]["expected_points"] == 4.5
    assert data["predictions"][3]["web_name"] == "Haaland"


def test_score_frozen_predictions_synthetic(tmp_path: Path):
    """Verify scoring frozen predictions computes honest MAE, RMSE, Spearman, coverage."""
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir(parents=True)
    log_path = tmp_path / "holdout_log.json"

    # Create synthetic frozen predictions for GW6
    synthetic_predictions = [
        {"element": 10, "web_name": "Player_10", "expected_points": 5.0, "p10": 2.0, "p90": 8.0},
        {"element": 11, "web_name": "Player_11", "expected_points": 6.0, "p10": 3.0, "p90": 9.0},
        {"element": 12, "web_name": "Player_12", "expected_points": 2.0, "p10": 0.0, "p90": 4.0},
        {"element": 13, "web_name": "Player_13", "expected_points": 7.0, "p10": 4.0, "p90": 10.0},
        {"element": 14, "web_name": "Player_14", "expected_points": 3.0, "p10": 1.0, "p90": 5.0},
        {"element": 15, "web_name": "Player_15", "expected_points": 4.0, "p10": 2.0, "p90": 6.0},
        {"element": 16, "web_name": "Player_16", "expected_points": 8.0, "p10": 5.0, "p90": 12.0},
        {"element": 17, "web_name": "Player_17", "expected_points": 1.5, "p10": 0.0, "p90": 3.5},
        {"element": 18, "web_name": "Player_18", "expected_points": 4.5, "p10": 2.0, "p90": 7.0},
        {"element": 19, "web_name": "Player_19", "expected_points": 5.5, "p10": 3.0, "p90": 8.0},
    ]
    frozen_payload = {
        "gameweek": 6,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model_version": "v1.0.0",
        "git_commit": "def5678",
        "manifest_hash": "manifest_hash_gw6",
        "player_count": 10,
        "predictions": synthetic_predictions,
    }
    with open(holdout_dir / "gw6_predictions.json", "w", encoding="utf-8") as f:
        json.dump(frozen_payload, f)

    # Synthetic actual points:
    # Elements 10 to 19 actuals: [6.0, 5.0, 2.0, 8.0, 3.0, 4.0, 10.0, 1.0, 5.0, 6.0]
    # All are inside their [p10, p90] intervals except element 17 (1.0 is inside [0.0, 3.5]) -> 100% coverage
    actual_pts_map = {
        10: 6.0,
        11: 5.0,
        12: 2.0,
        13: 8.0,
        14: 3.0,
        15: 4.0,
        16: 10.0,
        17: 1.0,
        18: 5.0,
        19: 6.0,
    }

    with patch("fpl_oracle.ml.holdout.HOLDOUT_LOG_PATH", log_path):
        scored = score_frozen_predictions(
            gw=6,
            actual_points_map=actual_pts_map,
            dry_run=False,
            custom_holdout_dir=holdout_dir,
            custom_log_path=log_path,
        )

    assert scored is not None
    assert scored["gameweek"] == 6
    assert scored["sample_count"] == 10
    assert scored["ml_mae"] > 0.0
    assert scored["coverage_80_pct"] == 100.0
    assert scored["manifest_hash"] == "manifest_hash_gw6"

    # Check updated holdout_log.json
    with open(log_path, encoding="utf-8") as f:
        log_data = json.load(f)

    assert log_data["status"] == "ACTIVE_FORWARD_EVALUATION"
    assert len(log_data["evaluated_gameweeks"]) == 1
    assert log_data["evaluated_gameweeks"][0]["gameweek"] == 6


@pytest.mark.anyio
async def test_holdout_forward_scheduler_job_on_demand():
    """Verify that the holdout_forward background job runs on-demand through scheduler."""
    from fpl_oracle.api.fpl_client import fpl_client

    try:
        res = await run_job_on_demand("holdout_forward")
        assert res["success"] is True
        assert res["job_id"] == "holdout_forward"
        assert res["status"] == "SUCCESS"
        assert "Awaiting GW" in res["details"] or "Froze GW" in res["details"]
    finally:
        await fpl_client.aclose()
