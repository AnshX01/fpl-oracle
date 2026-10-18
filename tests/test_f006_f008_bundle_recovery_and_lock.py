import hashlib
import json
import tempfile
import time
from pathlib import Path

import pytest

from fpl_oracle.data.features import FEATURE_SCHEMA_HASH
from fpl_oracle.ml.model_registry import RECIPE_ID, ModelBundleCorruptError, ModelRegistry

# Restores only accept bundles trained by the current recipe on the current features.
COMPAT = {"recipe_id": RECIPE_ID, "feature_list_hash": FEATURE_SCHEMA_HASH}


def test_corrupt_bundle_auto_recovers_to_previous_version():
    """
    F-006: Corrupting an active weight file must automatically quarantine
    the corrupt file and restore the last verified version from versions/,
    rather than raising an unhandled raw ValueError.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        versions_dir = models_dir / "versions"
        v1_dir = versions_dir / "v1.0.0"
        v1_dir.mkdir(parents=True)

        # Write legitimate v1 weights
        legit_data = b"legitimate_v1_weights_data"
        (v1_dir / "minutes_model.pkl").write_bytes(legit_data)
        v1_hash = hashlib.sha256(legit_data).hexdigest()

        # Active points to v1.0.0, but file on disk is corrupted
        corrupt_data = b"corrupted_tampered_bytes"
        (models_dir / "minutes_model.pkl").write_bytes(corrupt_data)

        manifest_data = {
            "active_version": "v1.0.0",
            "versions": [
                {
                    "version": "v1.0.0",
                    **COMPAT,
                    "file_hashes": {"minutes_model.pkl": v1_hash},
                }
            ],
        }
        (models_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        reg = ModelRegistry()
        reg.models_dir = models_dir
        reg.manifest_path = models_dir / "manifest.json"
        reg.versions_dir = versions_dir

        # Verification must trigger automatic rollback without raw ValueError crash
        success, msg = reg.verify_weight_integrity()
        assert success is True, f"Integrity check failed: {msg}"

        # The active file on disk must now be restored to legitimate data
        active_content = (models_dir / "minutes_model.pkl").read_bytes()
        assert active_content == legit_data, "Corrupt file was not restored to verified version!"

        # Corrupt file must be quarantined
        quarantine_root = models_dir / "quarantine"
        assert quarantine_root.exists(), "Quarantine directory was not created!"
        quarantined_files = list(quarantine_root.rglob("minutes_model.pkl"))
        assert len(quarantined_files) > 0, "Corrupt file was not quarantined!"


def test_no_valid_version_raises_clear_error_not_raw_valueerror():
    """
    F-006: If no valid version exists to recover from, fail with a clear
    ModelBundleCorruptError, not a raw ValueError.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        versions_dir = models_dir / "versions"
        versions_dir.mkdir()

        (models_dir / "minutes_model.pkl").write_bytes(b"bad_bytes")
        manifest_data = {
            "active_version": "v1.0.0",
            "versions": [
                {
                    "version": "v1.0.0",
                    **COMPAT,
                    "file_hashes": {"minutes_model.pkl": "deadbeef1234"},
                }
            ],
        }
        (models_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        reg = ModelRegistry()
        reg.models_dir = models_dir
        reg.manifest_path = models_dir / "manifest.json"
        reg.versions_dir = versions_dir

        from fpl_oracle.ml.model_registry import ModelBundleCorruptError

        with pytest.raises(ModelBundleCorruptError):
            reg.verify_weight_integrity()


def test_tampered_manifest_auto_recovers():
    """
    F-006: Tampered manifest (invalid JSON or corrupt content) must be detected and fail closed:
    archives of unknown recipe are never restored.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        versions_dir = models_dir / "versions"
        v1_dir = versions_dir / "v1.0.0"
        v1_dir.mkdir(parents=True)

        legit_data = b"legit_model_bytes"
        (v1_dir / "minutes_model.pkl").write_bytes(legit_data)

        # Corrupt active weights and tampered manifest syntax
        (models_dir / "minutes_model.pkl").write_bytes(b"tampered_active_bytes")
        (models_dir / "manifest.json").write_text("{CORRUPT_JSON_TAMPERED!", encoding="utf-8")

        reg = ModelRegistry()
        reg.models_dir = models_dir
        reg.manifest_path = models_dir / "manifest.json"
        reg.versions_dir = versions_dir

        # Without a manifest no archive can be shown to match this code's recipe: fail closed, never guess
        with pytest.raises(ModelBundleCorruptError):
            reg.verify_weight_integrity()
        assert legit_data == (v1_dir / "minutes_model.pkl").read_bytes()


def test_cross_process_bundle_lock_mutual_exclusion():
    """
    F-008: Two real OS processes acquiring get_bundle_file_lock must be mutually exclusive.
    """
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as td:
        lock_dir = Path(td)
        log_file = lock_dir / "timing.log"

        worker_script = """
import sys, time
from pathlib import Path
from fpl_oracle.ml.bundle_lock import get_bundle_file_lock

lock_dir = Path(sys.argv[1])
log_file = Path(sys.argv[2])
worker_id = sys.argv[3]

lock = get_bundle_file_lock(lock_dir)
with lock:
    t_start = time.time()
    with open(log_file, "a") as f:
        f.write(f"{worker_id},START,{t_start}\\n")
    time.sleep(0.4)
    t_end = time.time()
    with open(log_file, "a") as f:
        f.write(f"{worker_id},END,{t_end}\\n")
"""
        script_file = lock_dir / "worker.py"
        script_file.write_text(worker_script, encoding="utf-8")

        # Spawn two real OS processes almost simultaneously
        p1 = subprocess.Popen([sys.executable, str(script_file), str(lock_dir), str(log_file), "P1"])
        time.sleep(0.05)
        p2 = subprocess.Popen([sys.executable, str(script_file), str(lock_dir), str(log_file), "P2"])

        p1.wait(timeout=10)
        p2.wait(timeout=10)

        assert log_file.exists(), "timing log was not created"
        lines = [line.strip().split(",") for line in log_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(lines) == 4, f"Expected 4 events, got {len(lines)}: {lines}"

        # Assert no overlap: one process finishes completely before the other starts
        # Format: [ [w, "START", t], [w, "END", t], [w2, "START", t], [w2, "END", t] ]
        first_worker = lines[0][0]
        assert lines[0][1] == "START"
        assert lines[1][0] == first_worker
        assert lines[1][1] == "END"

        second_worker = lines[2][0]
        assert second_worker != first_worker
        assert lines[2][1] == "START"
        assert lines[3][0] == second_worker
        assert lines[3][1] == "END"

        first_end_t = float(lines[1][2])
        second_start_t = float(lines[2][2])
        assert second_start_t >= first_end_t - 0.05, (
            f"Processes overlapped! P1 end={first_end_t}, P2 start={second_start_t}"
        )


def test_cqr_tampering_triggers_quarantine_and_rollback():
    """
    WI-3.8 / G1: Tampering with a CQR file triggers quarantine and rollback to verified restore point.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        versions_dir = models_dir / "versions"
        v1_dir = versions_dir / "v1.0.0"
        (v1_dir / "cqr").mkdir(parents=True)

        legit_cqr_data = b"cqr_quantile_model_data_verified"
        (v1_dir / "cqr" / "cqr_DEF_q10.txt").write_bytes(legit_cqr_data)
        v1_hash = hashlib.sha256(legit_cqr_data).hexdigest()

        # Active models dir has tampered CQR file
        (models_dir / "cqr").mkdir(parents=True)
        (models_dir / "cqr" / "cqr_DEF_q10.txt").write_bytes(b"tampered_cqr_bytes")

        manifest_data = {
            "active_version": "v1.0.0",
            "versions": [
                {
                    "version": "v1.0.0",
                    **COMPAT,
                    "file_hashes": {"cqr/cqr_DEF_q10.txt": v1_hash},
                }
            ],
        }
        (models_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        reg = ModelRegistry()
        reg.models_dir = models_dir
        reg.manifest_path = models_dir / "manifest.json"
        reg.versions_dir = versions_dir

        success, msg = reg.verify_weight_integrity()
        assert success is True, f"Verification failed: {msg}"

        # Verified data restored
        assert (models_dir / "cqr" / "cqr_DEF_q10.txt").read_bytes() == legit_cqr_data
        # Tampered data quarantined
        quarantine_root = models_dir / "quarantine"
        assert quarantine_root.exists()


def test_production_quarantine_and_recover_has_valid_restore_point_and_does_not_raise():
    """
    G1: Active production models have a valid restore point in versions/ and quarantine_and_recover does not raise.
    """
    from fpl_oracle.config import MODELS_DIR

    reg = ModelRegistry()
    active = reg.get_active_version()
    assert active is not None
    # Verify versions_dir has at least one valid restore point
    versions = [d for d in reg.versions_dir.iterdir() if d.is_dir()]
    assert len(versions) >= 1, "Must have at least one valid restore point in data/models/versions/"

    # In a temp copy of production models, simulate quarantine_and_recover
    with tempfile.TemporaryDirectory() as td:
        tmp_models = Path(td) / "models"
        import shutil

        shutil.copytree(MODELS_DIR, tmp_models)

        temp_reg = ModelRegistry()
        temp_reg.models_dir = tmp_models
        temp_reg.manifest_path = tmp_models / "manifest.json"
        temp_reg.versions_dir = tmp_models / "versions"

        # Calling quarantine_and_recover must succeed and not raise ModelBundleCorruptError
        event = temp_reg.quarantine_and_recover(reason="Simulated test corruption")
        assert event["restored_version"] is not None
        assert Path(event["quarantine_dir"]).exists()


def test_rollback_to_previous_bundle_replaces_active_weights():
    """
    Item I verification:
    1. List data/models/versions/: both active (v2026.10.16.1307) and previous (v2026.10.06.1713) exist.
    2. Confirm previous known-good bundle is retained and has distinct weight hashes from active.
    3. Test rolling back to v2026.10.06.1713 actually replaces active weights with the previous version.
    """
    from fpl_oracle.config import MODELS_DIR

    reg = ModelRegistry()
    active_info = reg.get_active_version()
    assert active_info is not None
    active_ver = active_info["version"]
    active_dir = reg.versions_dir / active_ver
    if not active_dir.exists():
        pytest.skip(f"Active version {active_ver} has no local archive")
    active_minutes = hashlib.sha256((active_dir / "minutes_model.pkl").read_bytes()).hexdigest()
    others = sorted(
        d.name
        for d in reg.versions_dir.iterdir()
        if d.is_dir()
        and d.name != active_ver
        and reg.is_compatible(next((v for v in reg.load_manifest()["versions"] if v["version"] == d.name), {}))
        and (d / "minutes_model.pkl").exists()
        and hashlib.sha256((d / "minutes_model.pkl").read_bytes()).hexdigest() != active_minutes
    )
    if not others:
        pytest.skip("No distinct archived bundle to roll back to")
    prev_ver = others[-1]
    prev_dir = reg.versions_dir / prev_ver

    assert prev_dir.exists() and prev_dir.is_dir(), f"Previous version {prev_ver} must be retained in versions/"
    assert active_dir.exists() and active_dir.is_dir(), f"Active version {active_ver} must exist in versions/"

    # Hashes of previous vs active bundle must be distinct
    prev_minutes_hash = hashlib.sha256((prev_dir / "minutes_model.pkl").read_bytes()).hexdigest()
    active_minutes_hash = hashlib.sha256((active_dir / "minutes_model.pkl").read_bytes()).hexdigest()
    assert prev_minutes_hash != active_minutes_hash, "Previous bundle must have distinct weight hashes from active"

    # Simulate rollback in temporary models directory
    with tempfile.TemporaryDirectory() as td:
        tmp_models = Path(td) / "models"
        import shutil

        shutil.copytree(MODELS_DIR, tmp_models)

        temp_reg = ModelRegistry()
        temp_reg.models_dir = tmp_models
        temp_reg.manifest_path = tmp_models / "manifest.json"
        temp_reg.versions_dir = tmp_models / "versions"

        # Verify initial active version in temp directory
        temp_active = temp_reg.get_active_version()
        assert temp_active is not None and temp_active["version"] == active_ver
        curr_active_hash = hashlib.sha256((tmp_models / "minutes_model.pkl").read_bytes()).hexdigest()
        assert curr_active_hash == active_minutes_hash

        # Execute rollback to previous bundle
        res = temp_reg.rollback_to_version(prev_ver)
        assert res["success"] is True
        assert res["active_version"] == prev_ver

        # Verify active version in manifest changed to previous
        temp_active_after = temp_reg.get_active_version()
        assert temp_active_after is not None and temp_active_after["version"] == prev_ver

        # Verify weights in models root now match previous bundle, NOT active bundle
        rolled_back_hash = hashlib.sha256((tmp_models / "minutes_model.pkl").read_bytes()).hexdigest()
        assert rolled_back_hash == prev_minutes_hash
        assert rolled_back_hash != active_minutes_hash
