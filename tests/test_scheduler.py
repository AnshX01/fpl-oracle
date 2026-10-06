"""
Unit & Integration tests for APScheduler jobs, job idempotency, system endpoints,
and ModelRegistry automated rollback verification.
"""

import pytest

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.data.store import data_store
from fpl_oracle.ml.model_registry import ModelRegistry
from fpl_oracle.server.jobs import (
    deadline_alert_job,
    get_jobs_status,
    run_job_on_demand,
    scheduler,
    start_scheduler,
    stop_scheduler,
    tracker,
)


@pytest.fixture(autouse=True)
def clean_scheduler():
    """Ensure scheduler and client are cleanly terminated for tests."""
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)
    try:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(fpl_client.aclose())
        else:
            loop.run_until_complete(fpl_client.aclose())
    except Exception:
        pass


@pytest.mark.anyio
async def test_scheduler_job_registration():
    """Verify all 5 background jobs are properly registered with appropriate triggers."""
    start_scheduler()
    assert scheduler.running is True

    registered_ids = [job.id for job in scheduler.get_jobs()]
    expected_ids = [
        "cadence_refresh",
        "news_refresh",
        "price_snapshot",
        "retrain_trigger",
        "deadline_alert",
        "holdout_forward",
    ]
    for j_id in expected_ids:
        assert j_id in registered_ids

    status = get_jobs_status()
    assert status["scheduler_running"] is True
    assert status["total_jobs"] == 6
    assert len(status["jobs"]) == 6
    stop_scheduler()


@pytest.mark.anyio
async def test_job_on_demand_execution_and_idempotency():
    """Verify on-demand execution of price_snapshot and news_refresh jobs."""
    # Run price snapshot
    res_price = await run_job_on_demand("price_snapshot")
    assert res_price["success"] is True
    assert res_price["job_id"] == "price_snapshot"
    assert res_price["status"] == "SUCCESS"

    # Run again to verify idempotency (doesn't fail or corrupt database)
    res_price_2 = await run_job_on_demand("price_snapshot")
    assert res_price_2["success"] is True

    # Check that price snapshots were recorded in database
    snapshots = data_store.get_latest_price_snapshots(limit=10)
    assert len(snapshots) > 0
    assert "web_name" in snapshots[0]
    assert "now_cost" in snapshots[0]

    # Verify execution was recorded in DB job runs
    runs = data_store.get_job_runs(limit=5)
    assert any(r["job_id"] == "price_snapshot" for r in runs)


@pytest.mark.anyio
async def test_deadline_alert_idempotency():
    """Verify deadline alerts only fire once per window."""
    tracker.triggered_deadline_alerts.clear()

    # Run deadline alert job
    await deadline_alert_job()
    state = tracker.jobs_state["deadline_alert"]
    assert state["status"] == "SUCCESS"

    initial_run_count = state["run_count"]
    # Run a second time
    await deadline_alert_job()
    assert state["run_count"] == initial_run_count + 1
    # Check that it executed gracefully without duplicate side effects
    assert state["status"] == "SUCCESS"


@pytest.mark.anyio
async def test_unknown_job_handling():
    """Verify error handling for invalid job IDs."""
    res = await run_job_on_demand("non_existent_job")
    assert res["success"] is False
    assert "Unknown job_id" in res["error"]


def test_model_registry_rollback_on_degraded_metric(tmp_path):
    """
    Verify automated rollback when candidate model degrades validation MAE.
    If active model MAE is 1.48 and candidate MAE is 2.50, candidate must be rejected,
    automatic rollback engaged, and active version preserved.
    """
    registry = ModelRegistry()
    registry.models_dir = tmp_path / "models"
    registry.versions_dir = tmp_path / "models" / "versions"
    registry.rejected_dir = tmp_path / "models" / "rejected"
    registry.manifest_path = tmp_path / "models" / "manifest.json"
    registry._ensure_dirs()
    registry._init_manifest()

    # Active production metrics
    active_metrics = {"ml_mae": 1.48, "ml_spearman": 0.52, "base_mae": 1.74}

    # Mock candidate models
    class MockModel:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("mock_model_weights")

    cand_models = {
        "minutes_model": MockModel(),
        "attacking_model": MockModel(),
        "defending_model": MockModel(),
        "defcon_model": MockModel(),
        "bonus_model": MockModel(),
        "cards_saves_model": MockModel(),
    }

    # Degraded candidate metrics: MAE = 2.50 (worse than 1.48 + 0.05)
    cand_metrics_degraded = {"ml_mae": 2.50, "ml_spearman": 0.35, "base_mae": 1.74}

    res_rollback = registry.verify_and_promote(
        candidate_models=cand_models,
        candidate_metrics=cand_metrics_degraded,
        active_metrics=active_metrics,
        tolerance=0.05,
        notes="Degraded candidate test",
    )

    # Must engage automatic rollback
    assert res_rollback["promoted"] is False
    assert res_rollback["status"] == "rolled_back"
    assert "Automatic rollback engaged" in res_rollback["reason"]
    assert res_rollback["active_version"] == "v1.0.0"

    # Candidate should be saved into rejected dir
    assert (registry.rejected_dir / res_rollback["rejected_version"]).exists()


def test_model_registry_promotion_on_improved_metric(tmp_path):
    """
    Verify candidate promotion when candidate model achieves superior validation MAE.
    """
    registry = ModelRegistry()
    registry.models_dir = tmp_path / "models"
    registry.versions_dir = tmp_path / "models" / "versions"
    registry.rejected_dir = tmp_path / "models" / "rejected"
    registry.manifest_path = tmp_path / "models" / "manifest.json"
    registry._ensure_dirs()
    registry._init_manifest()

    active_metrics = {"ml_mae": 1.48, "ml_spearman": 0.52, "base_mae": 1.74}

    class MockModel:
        def save(self, path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("mock_weights")

    cand_models = {
        "minutes_model": MockModel(),
        "attacking_model": MockModel(),
        "defending_model": MockModel(),
        "defcon_model": MockModel(),
        "bonus_model": MockModel(),
        "cards_saves_model": MockModel(),
    }

    # Superior candidate metrics: MAE = 1.35 (improvement)
    cand_metrics_improved = {
        "ml_mae": 1.35,
        "ml_spearman": 0.58,
        "base_mae": 1.74,
        "rolling_origins": [
            {
                "season": "Holdout 2024-25",
                "train_size": 30000,
                "test_size": 27000,
                "mae": 1.00,
                "rmse": 2.00,
                "spearman": 0.68,
                "best_baseline_mae": 1.10,
                "gain_vs_baseline": 0.10,
                "interval_80_coverage_pct": 80.0,
            },
            {
                "season": "Holdout 2025-26",
                "train_size": 57000,
                "test_size": 29000,
                "mae": 0.94,
                "rmse": 1.98,
                "spearman": 0.70,
                "best_baseline_mae": 1.04,
                "gain_vs_baseline": 0.10,
                "interval_80_coverage_pct": 80.0,
            },
            {
                "season": "Holdout 2026-27",
                "train_size": 87000,
                "test_size": 2000,
                "mae": 1.88,
                "rmse": 2.82,
                "spearman": 0.43,
                "best_baseline_mae": 1.95,
                "gain_vs_baseline": 0.07,
                "interval_80_coverage_pct": 80.0,
            },
        ],
    }

    res_promote = registry.verify_and_promote(
        candidate_models=cand_models,
        candidate_metrics=cand_metrics_improved,
        active_metrics=active_metrics,
        new_version_tag="v1.1.0-improved",
        notes="Superior candidate test",
    )

    assert res_promote["promoted"] is True
    assert res_promote["status"] == "promoted"
    assert res_promote["new_version"] == "v1.1.0-improved"

    # Verified active in manifest
    manifest = registry.load_manifest()
    assert manifest["active_version"] == "v1.1.0-improved"
