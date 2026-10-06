"""
Ledger and Evidence Verifier for FPL Oracle Final Fix Pass (Rule B1, B2).
Verifies that all evidence files exist, are non-empty, and match the ledger status.
Validates SHA256 integrity locks in data/models/manifest.json against weights on disk.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "reports" / "evidence"
LEDGER_PATH = ROOT / "reports" / "final_fix_ledger.md"
MANIFEST_PATH = ROOT / "data" / "models" / "manifest.json"

REQUIRED_EVIDENCE_FILES = [
    "F1-train.txt",
    "F2-eval.txt",
    "F3-ablation.txt",
    "F4-calibration.txt",
    "F5-tests.txt",
    "F6-after.txt",
    "F8-backtest.txt",
    "F9-news.txt",
    "F10-tests.txt",
    "F11-plan.txt",
    "F12-parity.txt",
    "F13-lint.txt",
    "F14-screenshots.txt",
]


def verify_evidence_files() -> tuple[int, int]:
    print("\n--- 1. Verifying Evidence Files in reports/evidence/ ---")
    passed = 0
    failed = 0
    for filename in REQUIRED_EVIDENCE_FILES:
        filepath = EVIDENCE_DIR / filename
        if not filepath.exists():
            print(f"  [MISSING] {filename} does not exist!")
            failed += 1
            continue

        size = filepath.stat().st_size
        if size == 0:
            print(f"  [EMPTY]   {filename} exists but is empty (0 bytes)!")
            failed += 1
            continue

        print(f"  [OK]      {filename:<22} ({size:,} bytes)")
        passed += 1

    return passed, failed


def verify_manifest_hashes() -> bool:
    print("\n--- 2. Verifying Manifest SHA256 Locks Against Disk Weights ---")
    if not MANIFEST_PATH.exists():
        print("  [ERROR] manifest.json missing!")
        return False

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    active_v = manifest.get("active_version")
    version_info = next((v for v in manifest.get("versions", []) if v.get("version") == active_v), None)
    if not version_info:
        print(f"  [ERROR] Active version {active_v} not found in manifest!")
        return False

    file_hashes = version_info.get("file_hashes", {})
    all_match = True
    for fname, expected_hash in file_hashes.items():
        fpath = ROOT / "data" / "models" / fname
        if not fpath.exists():
            print(f"  [MISSING] Model weight file {fname} not found on disk!")
            all_match = False
            continue

        with open(fpath, "rb") as bf:
            actual_hash = hashlib.sha256(bf.read()).hexdigest()

        if actual_hash != expected_hash:
            print(f"  [MISMATCH] {fname}: expected {expected_hash}, got {actual_hash}")
            all_match = False
        else:
            print(f"  [MATCH]   {fname:<24} SHA256 verified")

    return all_match


def verify_ledger_closed_status() -> bool:
    print("\n--- 3. Verifying reports/final_fix_ledger.md Milestones ---")
    if not LEDGER_PATH.exists():
        print("  [ERROR] final_fix_ledger.md missing!")
        return False

    content = LEDGER_PATH.read_text(encoding="utf-8")
    milestones = [f"F{i}" for i in range(1, 15)]

    all_closed = True
    for m in milestones:
        if f"| **{m}** |" in content:
            # Check if line contains CLOSED
            line = next(line_str for line_str in content.splitlines() if f"| **{m}** |" in line_str)
            if "**CLOSED**" in line:
                print(f"  [CLOSED]  {m:<4} verified closed in ledger matrix")
            else:
                print(f"  [OPEN]    {m:<4} not marked CLOSED in ledger!")
                all_closed = False
        else:
            print(f"  [MISSING] {m:<4} not found in ledger matrix!")
            all_closed = False

    return all_closed


def main():
    print("=" * 60)
    print("FPL ORACLE — FINAL FIX PASS LEDGER & EVIDENCE VERIFIER")
    print("=" * 60)

    p_ev, f_ev = verify_evidence_files()
    manifest_ok = verify_manifest_hashes()
    ledger_ok = verify_ledger_closed_status()

    print("\n" + "=" * 60)
    print(
        f"Summary: Evidence Files {p_ev}/{len(REQUIRED_EVIDENCE_FILES)} Passed | "
        f"Manifest Hashes: {'PASS' if manifest_ok else 'FAIL'} | "
        f"Ledger Status: {'PASS' if ledger_ok else 'FAIL'}"
    )
    print("=" * 60)

    if f_ev > 0 or not manifest_ok or not ledger_ok:
        sys.exit(1)
    else:
        print("All ledger items, evidence files, and artifact hashes strictly verified!")
        sys.exit(0)


if __name__ == "__main__":
    main()
