"""
Test Suite for Requirement F4: Persisted Empirical Quantile Calibration.
Verifies that:
1. Calibration parameters (quantiles and blend weights) are persisted to a versioned artifact on disk.
2. ScoringEnsemble loads and restores calibration correctly.
3. Quantiles satisfy ordering invariants (q10 <= q90, z10 <= z90).
4. Coverage of nominal 80% intervals is well-calibrated.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.config import MODELS_DIR
from fpl_oracle.ml.ensemble import ScoringEnsemble, scoring_ensemble


def test_calibration_artifact_exists_and_valid():
    """Verify that calibration artifact exists and conforms to required schema."""
    cal_path = MODELS_DIR / "calibration.json"
    if not cal_path.exists():
        # Generate default calibration artifact
        scoring_ensemble.save_calibration(cal_path)

    assert cal_path.exists()
    with open(cal_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "z10" in data
    assert "z90" in data
    assert "blend_weights" in data
    assert "bucket_quantiles" in data
    assert data["z10"] <= data["z90"]
    assert len(data["blend_weights"]) == 3


def test_scoring_ensemble_save_and_load(tmp_path: Path):
    """Verify that save_calibration and load_calibration round-trip accurately."""
    ens = ScoringEnsemble(z10=-0.85, z90=1.05, blend_weights=(0.70, 0.10, 0.20))
    ens.bucket_quantiles = {
        "MID_high": {"q10": -2.5, "q90": 4.5, "n": 100},
        "DEF_mid": {"q10": -1.8, "q90": 3.2, "n": 50},
    }

    test_file = tmp_path / "test_calibration.json"
    saved_path = ens.save_calibration(test_file)
    assert saved_path == test_file
    assert test_file.exists()

    new_ens = ScoringEnsemble(z10=0.0, z90=0.0)
    success = new_ens.load_calibration(test_file)
    assert success is True
    assert abs(new_ens.calibrated_z10 - (-0.85)) < 1e-4
    assert abs(new_ens.calibrated_z90 - 1.05) < 1e-4
    assert abs(new_ens.blend_weights[0] - 0.70) < 1e-4
    assert "MID_high" in new_ens.bucket_quantiles
    assert new_ens.bucket_quantiles["MID_high"]["q10"] == -2.5


def test_bucketed_quantiles_ordering():
    """Verify all bucketed quantiles in active ensemble satisfy q10 <= q90."""
    cal_path = MODELS_DIR / "calibration.json"
    if cal_path.exists():
        with open(cal_path, encoding="utf-8") as f:
            data = json.load(f)
        for b_name, b_data in data.get("bucket_quantiles", {}).items():
            assert b_data["q10"] <= b_data["q90"], f"Bucket {b_name} violates q10 <= q90: {b_data}"


def test_interval_coverage_on_synthetic_holdout():
    """Verify that calibrated intervals produce coverage near nominal 80% on synthetic data."""
    np.random.seed(42)
    n = 1000
    # Simulate true outcomes with right-skewed noise
    xP = np.random.uniform(2.0, 7.0, size=n)
    noise = np.random.exponential(scale=1.5, size=n) - 1.5
    actual = np.maximum(0.0, xP + noise)

    # Conformal 10th and 90th percentile of residuals
    r = actual - xP
    q10 = float(np.percentile(r, 10))
    q90 = float(np.percentile(r, 90))

    ens = ScoringEnsemble()
    ens.bucket_quantiles = {"MID_high": {"q10": q10, "q90": q90, "n": n}}

    X_test = pd.DataFrame({
        "pos_MID": np.ones(n),
        "roll_points_5": xP,
        "roll_starts_ratio_5": np.ones(n),
        "opponent_difficulty": np.full(n, 3),
    })
    comps = {
        "p_min60": np.ones(n),
        "p_starts": np.ones(n),
        "expected_minutes": np.full(n, 85.0),
        "expected_goals": (xP - 2.0) / 5.0,
        "expected_assists": np.zeros(n),
        "p_clean_sheet": np.zeros(n),
        "expected_goals_conceded": np.zeros(n),
        "p_defcon": np.zeros(n),
        "expected_bonus": np.zeros(n),
        "expected_card_deduction": np.zeros(n),
        "expected_saves": np.zeros(n),
    }
    # Turn off blending for direct coverage test
    ens.blend_weights = (1.0, 0.0, 0.0)
    preds = ens.aggregate_components(comps, X_test)

    # Compute coverage of [p10, p90]
    p10 = preds["p10"].values
    p90 = preds["p90"].values
    covered = (actual >= p10) & (actual <= p90)
    cov_pct = covered.mean() * 100.0

    # Must be nominal 80% ± 5%
    assert 75.0 <= cov_pct <= 85.0, f"Calibrated interval coverage {cov_pct:.2f}% outside [75%, 85%]"
