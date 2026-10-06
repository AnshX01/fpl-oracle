"""
Test Suite for Requirement F1: Model Provenance and Manifest Lock.
Verifies that:
1. Manifest locks SHA256 hashes of all served weight files.
2. Active manifest explicitly specifies metric_applies_to == "served_weights".
3. Tampering with any model weight file causes verification to fail.
4. No refit is performed after candidate promotion.
"""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.config import MODELS_DIR
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.model_registry import ModelRegistry, model_registry
from fpl_oracle.ml.train import train_all_models


def test_manifest_metric_applies_to_served_weights():
    """Verify that manifest specifies metric_applies_to == 'served_weights' for active version."""
    active = model_registry.get_active_version()
    assert "metric_applies_to" in active, "Manifest active version must have metric_applies_to"
    assert active["metric_applies_to"] == "served_weights"


def test_manifest_file_hashes_match_disk():
    """Verify that every file listed in active manifest file_hashes matches disk SHA256."""
    active = model_registry.get_active_version()
    file_hashes = active.get("file_hashes", {})
    assert len(file_hashes) >= 6, "Must record hashes for at least all 6 component models"

    for filename, expected_hash in file_hashes.items():
        fpath = MODELS_DIR / filename
        assert fpath.exists(), f"Weight file {filename} listed in manifest does not exist on disk"

        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        actual_hash = sha.hexdigest()
        assert actual_hash == expected_hash, (
            f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
        )


def test_model_integrity_verification_success():
    """Verify that verify_weight_integrity succeeds on untouched production weights."""
    valid, msg = model_registry.verify_weight_integrity(MODELS_DIR)
    assert valid is True
    assert "verified" in msg.lower()


def test_weight_tampering_fails_verification(tmp_path: Path):
    """Verify that modifying a byte in a weight file causes verify_weight_integrity to raise ValueError."""
    # Create a copy of weights in tmp_path
    active = model_registry.get_active_version()
    file_hashes = active.get("file_hashes", {})

    for filename in file_hashes:
        src = MODELS_DIR / filename
        if src.exists():
            (tmp_path / filename).write_bytes(src.read_bytes())

    # Tamper with one file
    tampered_file = tmp_path / "minutes_model.pkl"
    if tampered_file.exists():
        content = bytearray(tampered_file.read_bytes())
        content[0] = (content[0] + 1) % 256  # flip first byte
        tampered_file.write_bytes(bytes(content))

    reg = ModelRegistry()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reg.verify_weight_integrity(tmp_path)


def test_missing_weight_file_fails_verification(tmp_path: Path):
    """Verify that a missing weight file causes verify_weight_integrity to raise ValueError."""
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="Weight file missing"):
        reg.verify_weight_integrity(tmp_path)


def test_no_post_promotion_refit_in_source():
    """Statically verify that train_all_models does not call projection_engine.train after verify_and_promote."""
    src = inspect.getsource(train_all_models)
    assert "verify_and_promote" in src

    # Verify that projection_engine.train is not called after verify_and_promote
    promote_idx = src.find("verify_and_promote")
    after_promote = src[promote_idx:]
    assert "projection_engine.train" not in after_promote, (
        "Found projection_engine.train after verify_and_promote! Post-promotion refit must be eliminated."
    )


def test_calibration_tampering_fails_verification(tmp_path: Path):
    """Verify that tampering with calibration.json causes verify_weight_integrity to raise ValueError (G1)."""
    active = model_registry.get_active_version()
    file_hashes = active.get("file_hashes", {})

    for filename in file_hashes:
        src = MODELS_DIR / filename
        if src.exists():
            (tmp_path / filename).write_bytes(src.read_bytes())

    # Tamper with calibration.json
    cal_file = tmp_path / "calibration.json"
    assert cal_file.exists(), "calibration.json must be present in active version"
    content = bytearray(cal_file.read_bytes())
    content[0] = (content[0] + 1) % 256
    cal_file.write_bytes(bytes(content))

    reg = ModelRegistry()
    with pytest.raises(ValueError, match="SHA256 mismatch for calibration.json"):
        reg.verify_weight_integrity(tmp_path)


def test_evaluation_does_not_mutate_production_weights():
    """Verify that running evaluation does not mutate any file under data/models/ (G1)."""
    # Capture initial SHA256 hashes of all files in MODELS_DIR
    initial_hashes = {}
    for p in MODELS_DIR.iterdir():
        if p.is_file():
            initial_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Run ensemble calibrate in-memory with synthetic data
    comps_dummy = {
        "expected_minutes": np.array([75.0, 90.0]),
        "p_starts": np.array([0.9, 0.95]),
        "p_min60": np.array([0.85, 0.9]),
        "expected_goals": np.array([0.4, 0.5]),
        "expected_assists": np.array([0.2, 0.3]),
        "p_clean_sheet": np.array([0.35, 0.35]),
        "expected_goals_conceded": np.array([1.1, 0.9]),
        "p_defcon": np.array([0.2, 0.2]),
        "expected_bonus": np.array([0.5, 0.8]),
        "expected_card_deduction": np.array([0.1, 0.1]),
        "expected_saves": np.array([0.0, 0.0]),
    }
    X_dummy = pd.DataFrame(
        {
            "pos_MID": [1, 1],
            "pos_DEF": [0, 0],
            "pos_GKP": [0, 0],
            "pos_FWD": [0, 0],
        }
    )
    Y_dummy = pd.DataFrame({"target_points": [5.0, 8.0]})

    scoring_ensemble.calibrate(comps_dummy, X_dummy, Y_dummy)

    # Re-verify that NO file in MODELS_DIR changed
    after_hashes = {}
    for p in MODELS_DIR.iterdir():
        if p.is_file():
            after_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    assert initial_hashes == after_hashes, (
        f"Production model files were mutated during evaluation! Differences: "
        f"{set(initial_hashes.items()) ^ set(after_hashes.items())}"
    )
