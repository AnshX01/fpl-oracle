"""
Unit tests for Requirement G5: Evidence-Gated Model Promotion Gate.
Verifies:
1. Shared schema compliance between evaluator and registry.
2. Gate rejects candidate with inferior MAE vs baseline (e.g. MAE 2.0 vs baseline 1.0).
3. Gate rejects candidate with low interval coverage (e.g. 73.5% vs nominal 75%-85% band).
4. Gate rejects candidate with negative mean or latest rolling origin gain.
5. Gate accepts candidate satisfying all criteria.
6. Gate decision matches the documented real evaluation artifact.
"""

import json
from typing import Any

import pytest

from fpl_oracle.config import REPORTS_DIR
from fpl_oracle.ml.eval import EVAL_METRICS_SCHEMA_KEYS, ROLLING_ORIGIN_SCHEMA_KEYS
from fpl_oracle.ml.model_registry import check_promotion_gate


def test_schema_keys_defined():
    """Verify shared schema constants exist and contain required keys."""
    assert "ml_mae" in EVAL_METRICS_SCHEMA_KEYS
    assert "base_mae" in EVAL_METRICS_SCHEMA_KEYS
    assert "rolling_origins" in EVAL_METRICS_SCHEMA_KEYS

    assert "season" in ROLLING_ORIGIN_SCHEMA_KEYS
    assert "mae" in ROLLING_ORIGIN_SCHEMA_KEYS
    assert "best_baseline_mae" in ROLLING_ORIGIN_SCHEMA_KEYS
    assert "gain_vs_baseline" in ROLLING_ORIGIN_SCHEMA_KEYS
    assert "interval_80_coverage_pct" in ROLLING_ORIGIN_SCHEMA_KEYS


def test_gate_rejects_inferior_baseline():
    """Synthetic actual-shape origin: MAE 2.0 vs baseline 1.0 must FAIL."""
    candidate_metrics: dict[str, Any] = {
        "ml_mae": 2.0,
        "base_mae": 1.0,
        "ml_spearman": 0.35,
        "rolling_origins": [
            {
                "season": "Holdout 2026-27",
                "train_size": 80000,
                "test_size": 2000,
                "mae": 2.0,
                "rmse": 3.0,
                "spearman": 0.35,
                "best_baseline_mae": 1.0,
                "gain_vs_baseline": -1.0,
                "interval_80_coverage_pct": 80.0,
            }
        ],
    }
    passed, reason = check_promotion_gate(candidate_metrics)
    assert not passed
    assert "failed baseline gate" in reason.lower() or "gain" in reason.lower()


def test_gate_rejects_low_coverage():
    """Synthetic actual-shape origin: low coverage (e.g. 73.5%) must FAIL."""
    candidate_metrics: dict[str, Any] = {
        "ml_mae": 1.0,
        "base_mae": 1.05,
        "ml_spearman": 0.70,
        "rolling_origins": [
            {
                "season": "Holdout 2024-25",
                "train_size": 30000,
                "test_size": 27000,
                "mae": 1.00,
                "rmse": 2.00,
                "spearman": 0.68,
                "best_baseline_mae": 1.04,
                "gain_vs_baseline": 0.04,
                "interval_80_coverage_pct": 79.5,
            },
            {
                "season": "Holdout 2025-26",
                "train_size": 57000,
                "test_size": 29000,
                "mae": 0.94,
                "rmse": 1.98,
                "spearman": 0.70,
                "best_baseline_mae": 0.99,
                "gain_vs_baseline": 0.05,
                "interval_80_coverage_pct": 78.2,
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
                "interval_80_coverage_pct": 73.5,  # Below 75% nominal floor
            },
        ],
    }
    passed, reason = check_promotion_gate(candidate_metrics, coverage_min=75.0, coverage_max=85.0)
    assert not passed
    assert "coverage" in reason.lower()
    assert "73.50%" in reason or "73.5" in reason


def test_gate_rejects_negative_rolling_origin_gain():
    """Candidate with negative latest origin gain must FAIL."""
    candidate_metrics: dict[str, Any] = {
        "ml_mae": 1.0,
        "base_mae": 1.05,
        "ml_spearman": 0.70,
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
                "mae": 2.00,
                "rmse": 2.82,
                "spearman": 0.43,
                "best_baseline_mae": 1.95,
                "gain_vs_baseline": -0.05,  # Negative gain on latest origin despite positive mean
                "interval_80_coverage_pct": 80.0,
            },
        ],
    }
    passed, reason = check_promotion_gate(candidate_metrics)
    assert not passed
    assert "latest origin gain" in reason.lower()


def test_gate_rejects_negative_mean_rolling_origin_gain():
    """Candidate with negative mean origin gain must FAIL."""
    candidate_metrics: dict[str, Any] = {
        "ml_mae": 1.0,
        "base_mae": 1.05,
        "ml_spearman": 0.70,
        "rolling_origins": [
            {
                "season": "Holdout 2024-25",
                "train_size": 30000,
                "test_size": 27000,
                "mae": 1.10,
                "rmse": 2.00,
                "spearman": 0.68,
                "best_baseline_mae": 1.04,
                "gain_vs_baseline": -0.06,
                "interval_80_coverage_pct": 80.0,
            },
            {
                "season": "Holdout 2025-26",
                "train_size": 57000,
                "test_size": 29000,
                "mae": 1.05,
                "rmse": 1.98,
                "spearman": 0.70,
                "best_baseline_mae": 0.99,
                "gain_vs_baseline": -0.06,
                "interval_80_coverage_pct": 80.0,
            },
            {
                "season": "Holdout 2026-27",
                "train_size": 87000,
                "test_size": 2000,
                "mae": 1.90,
                "rmse": 2.82,
                "spearman": 0.43,
                "best_baseline_mae": 1.95,
                "gain_vs_baseline": 0.05,
                "interval_80_coverage_pct": 80.0,
            },
        ],
    }
    passed, reason = check_promotion_gate(candidate_metrics)
    assert not passed
    assert "mean gain vs baseline" in reason.lower()


def test_gate_accepts_valid_candidate():
    """Candidate satisfying all criteria must PASS."""
    candidate_metrics: dict[str, Any] = {
        "ml_mae": 0.95,
        "base_mae": 1.05,
        "ml_spearman": 0.70,
        "rolling_origins": [
            {
                "season": "Holdout 2024-25",
                "train_size": 30000,
                "test_size": 27000,
                "mae": 1.00,
                "rmse": 2.00,
                "spearman": 0.68,
                "best_baseline_mae": 1.04,
                "gain_vs_baseline": 0.04,
                "interval_80_coverage_pct": 79.5,
            },
            {
                "season": "Holdout 2025-26",
                "train_size": 57000,
                "test_size": 29000,
                "mae": 0.94,
                "rmse": 1.98,
                "spearman": 0.70,
                "best_baseline_mae": 0.99,
                "gain_vs_baseline": 0.05,
                "interval_80_coverage_pct": 81.2,
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
                "interval_80_coverage_pct": 80.5,
            },
        ],
    }
    passed, reason = check_promotion_gate(candidate_metrics)
    assert passed
    assert "passed all promotion gate criteria" in reason.lower()


def test_real_artifact_schema_and_gate_consistency():
    """Verify the real model_eval.json artifact matches schema and gate decision."""
    eval_json_path = REPORTS_DIR / "model_eval.json"
    if not eval_json_path.exists():
        pytest.skip("reports/model_eval.json does not exist yet (run run_eval.py first)")

    with open(eval_json_path, encoding="utf-8") as f:
        artifact = json.load(f)

    # Schema checks
    assert "ml_mae" in artifact
    assert "base_mae" in artifact
    assert "rolling_origins" in artifact
    assert isinstance(artifact["rolling_origins"], list)
    assert len(artifact["rolling_origins"]) >= 3

    for ro in artifact["rolling_origins"]:
        for k in ROLLING_ORIGIN_SCHEMA_KEYS:
            assert k in ro, f"Key '{k}' missing from rolling origin entry: {ro}"

    # Evaluate gate decision
    passed, reason = check_promotion_gate(artifact)
    recorded_verdict = artifact.get("gate_verdict", {})
    if recorded_verdict:
        assert passed == recorded_verdict.get("passed")
        assert reason == recorded_verdict.get("reason")
