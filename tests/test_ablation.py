"""
Unit tests for Requirement G7: True Retraining Feature Ablation Consistency.

Verifies:
1. Every feature group defined in scripts/run_ablation.py contains valid columns existing in FEATURE_COLUMNS.
2. validate_feature_groups fails loudly (ValueError) if any requested column is missing.
3. validate_feature_groups fails loudly (ValueError) if any group resolves to zero columns.
4. compute_grouped_bootstrap_ci computes valid 95% confidence intervals with grouped-by-gameweek resampling.
5. Schema compliance of reports/ablation.json artifact.
"""

import json

import numpy as np
import pytest

from fpl_oracle.config import REPORTS_DIR
from fpl_oracle.data.features import FEATURE_COLUMNS
from scripts.run_ablation import (
    FEATURE_GROUPS_TO_ABLATE,
    compute_grouped_bootstrap_ci,
    validate_feature_groups,
)


def test_feature_groups_valid_and_non_empty():
    """Verify all 5 feature groups exist, are non-empty, and contain only real FEATURE_COLUMNS."""
    assert len(FEATURE_GROUPS_TO_ABLATE) == 5

    expected_groups = {
        "fixture_difficulty_and_context",
        "team_form_attack_defense",
        "player_underlying_metrics",
        "minutes_and_starts",
        "disciplinary_and_rare",
    }
    assert set(FEATURE_GROUPS_TO_ABLATE.keys()) == expected_groups

    avail_set = set(FEATURE_COLUMNS)
    for grp_name, cols in FEATURE_GROUPS_TO_ABLATE.items():
        assert len(cols) > 0, f"Group '{grp_name}' resolves to zero columns!"
        for c in cols:
            assert c in avail_set, f"Column '{c}' in group '{grp_name}' not found in FEATURE_COLUMNS!"


def test_validate_feature_groups_fails_on_missing_column():
    """Verify validation raises ValueError loudly when a feature group specifies an unknown column."""
    bad_groups = {
        "valid_group": ["roll_xG_3"],
        "corrupted_group": ["roll_xG_5", "non_existent_fake_metric_column"],
    }
    with pytest.raises(ValueError, match="specifies missing columns"):
        validate_feature_groups(bad_groups, FEATURE_COLUMNS)


def test_validate_feature_groups_fails_on_zero_columns():
    """Verify validation raises ValueError loudly if any group resolves to zero columns."""
    empty_group = {
        "empty_feature_group": [],
    }
    with pytest.raises(ValueError, match="resolves to 0 columns"):
        validate_feature_groups(empty_group, FEATURE_COLUMNS)


def test_grouped_bootstrap_ci_computation():
    """Verify grouped-by-gameweek bootstrap CI produces valid intervals on grouped synthetic data."""
    rng = np.random.default_rng(123)
    n_gw = 10
    n_per_gw = 50
    n_samples = n_gw * n_per_gw

    gw_labels = np.repeat(np.arange(1, n_gw + 1), n_per_gw)
    actual = rng.uniform(0.0, 10.0, size=n_samples)
    preds_full = actual + rng.normal(0.0, 1.0, size=n_samples)
    # Ablated model is slightly worse (larger noise)
    preds_ablated = actual + rng.normal(0.0, 1.2, size=n_samples)

    ci_low, ci_high = compute_grouped_bootstrap_ci(
        actual=actual,
        preds_full=preds_full,
        preds_ablated=preds_ablated,
        group_labels=gw_labels,
        n_boot=200,
        seed=42,
    )

    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)
    assert ci_low <= ci_high
    assert not np.isnan(ci_low)
    assert not np.isnan(ci_high)


def test_ablation_artifact_schema():
    """Verify reports/ablation.json matches expected schema when artifact exists."""
    artifact_path = REPORTS_DIR / "ablation.json"
    if not artifact_path.exists():
        pytest.skip("reports/ablation.json does not exist yet (run run_ablation.py first)")

    with open(artifact_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("evaluation_type") == "true_retraining_ablation"
    assert "full_model" in data
    assert "mae" in data["full_model"]
    assert "groups" in data
    assert isinstance(data["groups"], list)

    for grp in data["groups"]:
        assert "group_name" in grp
        assert "features_removed" in grp
        assert "feature_count_removed" in grp
        assert grp["feature_count_removed"] > 0
        assert "ablated_mae" in grp
        assert "mae_delta_vs_full" in grp
        assert "mae_gain_pct" in grp
        assert "ci_95" in grp
        assert len(grp["ci_95"]) == 2
        assert grp["ci_95"][0] <= grp["ci_95"][1]
