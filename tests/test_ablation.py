"""
Test Suite for Requirement F3: True Retraining Feature Ablation.
Verifies that:
1. reports/ablation.json exists and conforms to true retraining schema.
2. Every ablated group has features removed and positive ablated MAE.
3. Bootstrap confidence intervals are well-formed (lower <= upper).
4. No stale conflicting numbers (e.g. 17.14% or 2.88%) are present in the authoritative artifact.
"""

import json
from pathlib import Path

import pytest

from fpl_oracle.config import REPORTS_DIR


def test_ablation_artifact_schema_and_contents():
    """Verify reports/ablation.json exists and is structured as true retraining ablation."""
    ablation_file = REPORTS_DIR / "ablation.json"
    assert ablation_file.exists(), "reports/ablation.json must exist"

    with open(ablation_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("evaluation_type") == "true_retraining_ablation"
    assert "full_model" in data
    assert "groups" in data
    assert len(data["groups"]) >= 4, "Must evaluate at least 4 feature groups"

    full_mae = data["full_model"]["mae"]
    assert full_mae > 0.0

    group_names = set()
    for grp in data["groups"]:
        name = grp["group_name"]
        group_names.add(name)
        assert grp["feature_count_removed"] > 0, f"Group {name} must have removed features"
        assert len(grp["features_removed"]) == grp["feature_count_removed"]
        assert grp["ablated_mae"] > 0.0
        assert "ci_95" in grp
        assert len(grp["ci_95"]) == 2
        ci_low, ci_high = grp["ci_95"]
        assert ci_low <= ci_high, f"CI inverted for {name}: [{ci_low}, {ci_high}]"

    expected_groups = {
        "fixture_difficulty_and_context",
        "player_underlying_metrics",
        "minutes_and_starts",
        "disciplinary_and_rare",
    }
    assert expected_groups.issubset(group_names), f"Missing groups: {expected_groups - group_names}"


def test_no_stale_ablation_metrics_in_artifact():
    """Verify that reports/ablation.json does not contain stale 17.14% or 2.88% values."""
    ablation_file = REPORTS_DIR / "ablation.json"
    raw_text = ablation_file.read_text(encoding="utf-8")
    assert "17.14" not in raw_text, "Ablation artifact must not contain stale 17.14% metric"
    assert "2.88" not in raw_text, "Ablation artifact must not contain stale 2.88% metric"
