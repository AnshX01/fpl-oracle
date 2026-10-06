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

import pytest

from fpl_oracle.config import MODELS_DIR
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
        assert actual_hash == expected_hash, f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"


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
