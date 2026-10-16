import hashlib
import json
import tempfile
import time
from pathlib import Path

import pytest

from fpl_oracle.ml.model_registry import ModelRegistry


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
    F-006: Tampered manifest (invalid JSON or corrupt content) must be detected
    and recovered from disk archives without unhandled crash.
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

        success, msg = reg.verify_weight_integrity()
        assert success is True
        assert (models_dir / "minutes_model.pkl").read_bytes() == legit_data


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
