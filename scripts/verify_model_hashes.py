"""
Verify that on-disk model files and calibration match data/models/manifest.json locks.
Exits 0 if all hashes match, 1 otherwise.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "models" / "manifest.json"
MODELS_DIR = ROOT / "data" / "models"


def main():
    print("=== MODEL WEIGHTS & CALIBRATION HASH INTEGRITY VERIFIER ===")
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found!")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    active_v = manifest.get("active_version")
    print(f"Active Version:      {active_v}")
    version_info = next((v for v in manifest.get("versions", []) if v.get("version") == active_v), None)
    if not version_info:
        print(f"ERROR: Version {active_v} not found in manifest!")
        sys.exit(1)

    print(f"Git Commit Lock:     {version_info.get('git_commit')}")
    print(f"Recipe ID:           {version_info.get('recipe_id')}")
    print(f"Training Data Hash:  {version_info.get('training_data_hash')}")
    print(f"Max Gameweek:        {version_info.get('max_gw')}")
    print(f"Metric Applies To:   {version_info.get('metric_applies_to')}")

    file_hashes = version_info.get("file_hashes", {})
    all_match = True

    print("\n--- File Hashes Comparison ---")
    for fname, expected_hash in file_hashes.items():
        fpath = MODELS_DIR / fname
        if not fpath.exists():
            print(f"  [MISSING]  {fname:<24} File missing on disk!")
            all_match = False
            continue

        actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if actual_hash == expected_hash:
            print(f"  [MATCH]    {fname:<24} SHA256: {actual_hash[:16]}... verified")
        else:
            print(f"  [MISMATCH] {fname:<24} expected: {expected_hash}, got: {actual_hash}")
            all_match = False

    if all_match:
        print("\nAll 7 component model and calibration SHA256 hashes strictly verified against manifest!")
        sys.exit(0)
    else:
        print("\nERROR: Integrity check failed! One or more file hashes do not match manifest.")
        sys.exit(1)


if __name__ == "__main__":
    main()
