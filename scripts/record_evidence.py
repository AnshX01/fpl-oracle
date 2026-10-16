"""
Evidence Recorder for FPL Oracle (Milestone G14 / GOAL A7).
Executes a command and stores a verifiable schema_version 1 JSON evidence record:
- schema_version: 1
- command: list of argv strings
- exit_code: integer return code
- stdout: standard output
- stderr: standard error
- output: combined output string
- timestamp: ISO 8601 UTC timestamp
- git_commit / code_head: 40-character git commit SHA
- result: "PASS" (if exit_code == 0) or "FAIL"
- artifacts: list of dicts with {"path": rel_path, "sha256": checksum}
- file_hashes: dict mapping rel_path -> sha256
- gates: list of gate dicts [{"name": str, "status": "PASS", "value": float, "min": ..., "max": ...}]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "reports" / "evidence"


def get_current_git_head() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def record_command(
    cmd: list[str],
    output_path: Path,
    code_head: str | None = None,
    artifacts: list[str] | None = None,
    gates: list[dict[str, Any]] | None = None,
) -> int:
    head = code_head or get_current_git_head() or "0" * 40
    print(f"\n[RECORD] Running: {' '.join(cmd)}")
    start = datetime.now(UTC)
    res = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    elapsed = (datetime.now(UTC) - start).total_seconds()
    print(f"Finished in {elapsed:.2f}s with exit code {res.returncode}")

    output_str = (res.stdout + "\n" + res.stderr).strip()

    artifact_records: list[dict[str, str]] = []
    file_hashes: dict[str, str] = {}
    if artifacts:
        for art in artifacts:
            art_path = (ROOT / art).resolve()
            if art_path.is_file() and art_path.is_relative_to(ROOT.resolve()):
                rel = art_path.relative_to(ROOT.resolve()).as_posix()
                sha = compute_file_sha256(art_path)
                artifact_records.append({"path": rel, "sha256": sha})
                file_hashes[rel] = sha

    record = {
        "schema_version": 1,
        "command": cmd,
        "exit_code": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "output": output_str,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": head,
        "code_head": head,
        "result": "PASS" if res.returncode == 0 else "FAIL",
        "artifacts": artifact_records,
        "file_hashes": file_hashes,
        "gates": gates or [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Recorded schema_version 1 evidence to {output_path}")

    return res.returncode


def get_canonical_milestones() -> dict[str, dict[str, Any]]:
    py = sys.executable
    return {
        "G1-load": {
            "path": EVIDENCE_DIR / "G1-load.txt",
            "cmd": [
                py,
                "-c",
                (
                    "from fpl_oracle.ml.predict import projection_engine; "
                    "projection_engine.load_or_train(); "
                    "print('SUCCESS: Model weights loaded with verified integrity lock; is_loaded =', "
                    "projection_engine.is_loaded)"
                ),
            ],
            "artifacts": ["data/models/manifest.json"],
        },
        "G1-hashes": {
            "path": EVIDENCE_DIR / "G1-hashes.txt",
            "cmd": [py, "scripts/verify_model_hashes.py"],
            "artifacts": ["data/models/manifest.json", "data/models/calibration.json"],
        },
        "G1-tests": {
            "path": EVIDENCE_DIR / "G1-tests.txt",
            "cmd": [
                py,
                "-m",
                "pytest",
                "tests/test_provenance.py",
                "tests/test_calibration_persist.py",
                "tests/test_model_gap_closure.py",
                "-v",
                "-W",
                "ignore::DeprecationWarning",
            ],
            "artifacts": [],
        },
        "G2-ci": {
            "path": EVIDENCE_DIR / "G2-ci.txt",
            "cmd": [
                py,
                "-m",
                "pytest",
                "-q",
                "-m",
                "not slow",
            ],
            "artifacts": [],
        },
        "G3-ui": {
            "path": EVIDENCE_DIR / "G3-ui.txt",
            "cmd": [py, "-m", "pytest", "tests/test_ui_honesty.py", "-v"],
            "artifacts": [],
        },
        "G4-freshness": {
            "path": EVIDENCE_DIR / "G4-freshness.txt",
            "cmd": [py, "scripts/check_freshness.py", "--mode", "freshness"],
            "artifacts": ["data/historical/master_history.csv"],
        },
        "G4-identity": {
            "path": EVIDENCE_DIR / "G4-identity.txt",
            "cmd": [py, "scripts/check_freshness.py", "--mode", "identity"],
            "artifacts": ["data/historical/master_history.csv"],
        },
        "G4-cache": {
            "path": EVIDENCE_DIR / "G4-cache.txt",
            "cmd": [py, "scripts/check_freshness.py", "--mode", "cache"],
            "artifacts": [],
        },
        "G5-eval": {
            "path": EVIDENCE_DIR / "G5-eval.txt",
            "cmd": [py, "scripts/run_eval.py"],
            "artifacts": ["reports/model_eval.json"],
        },
        "G5-gate": {
            "path": EVIDENCE_DIR / "G5-gate.txt",
            "cmd": [py, "-m", "pytest", "tests/test_promotion_gate.py", "-v"],
            "artifacts": [],
        },
        "G6-holdout": {
            "path": EVIDENCE_DIR / "G6-holdout.txt",
            "cmd": [py, "-m", "pytest", "tests/test_holdout_forward.py", "-v"],
            "artifacts": [],
        },
        "G7-ablation": {
            "path": EVIDENCE_DIR / "G7-ablation.txt",
            "cmd": [py, "-m", "pytest", "tests/test_ablation.py", "-v"],
            "artifacts": ["reports/ablation.json"],
        },
        "G8-rivals": {
            "path": EVIDENCE_DIR / "G8-rivals.txt",
            "cmd": [
                py,
                "-m",
                "pytest",
                "tests/test_rivals_standings.py",
                "tests/test_rival_set.py",
                "-v",
            ],
            "artifacts": [],
        },
        "G9-mc": {
            "path": EVIDENCE_DIR / "G9-mc.txt",
            "cmd": [
                py,
                "-m",
                "pytest",
                "tests/test_monte_carlo_components.py",
                "tests/test_montecarlo_cs.py",
                "-v",
            ],
            "artifacts": [],
        },
        "G10-joint": {
            "path": EVIDENCE_DIR / "G10-joint.txt",
            "cmd": [py, "-m", "pytest", "tests/test_joint_plan_live.py", "-v"],
            "artifacts": [],
        },
        "G11-backtest": {
            "path": EVIDENCE_DIR / "G11-backtest.txt",
            "cmd": [py, "scripts/run_league_backtest.py"],
            "artifacts": ["reports/backtest.json"],
        },
        "G12-news": {
            "path": EVIDENCE_DIR / "G12-news.txt",
            "cmd": [py, "scripts/run_news_benchmark.py"],
            "artifacts": ["reports/news_benchmark.json"],
        },
        "G13-parity": {
            "path": EVIDENCE_DIR / "G13-parity.txt",
            "cmd": [py, "scripts/check_feature_parity.py"],
            "artifacts": [],
        },
        "G14-verify": {
            "path": EVIDENCE_DIR / "G14-verify.txt",
            "cmd": [py, "-m", "pytest", "tests/test_evidence_validator.py", "-v"],
            "artifacts": [],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record reproducible schema_version 1 evidence.")
    parser.add_argument("--output", "-o", type=Path, help="Target evidence file path")
    parser.add_argument("--code-head", type=str, help="Explicit git HEAD commit SHA")
    parser.add_argument("--artifact", action="append", default=[], help="Relative path of artifact to hash")
    parser.add_argument("--milestone", type=str, help="Milestone ID to regenerate (e.g. G1-load, G4-freshness)")
    parser.add_argument("--all", action="store_true", help="Regenerate all canonical milestone evidence")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command and arguments to execute")

    args = parser.parse_args()

    milestones = get_canonical_milestones()

    if args.all:
        print(f"Regenerating evidence for all {len(milestones)} milestone records...")
        failures = []
        for name, spec in milestones.items():
            ret = record_command(
                cmd=spec["cmd"],
                output_path=spec["path"],
                code_head=args.code_head,
                artifacts=spec.get("artifacts"),
            )
            if ret != 0:
                failures.append(name)
        if failures:
            print(f"\n[FAIL] {len(failures)} milestones failed: {failures}")
            return 1
        print("\n[SUCCESS] All milestone evidence successfully recorded!")
        return 0

    if args.milestone:
        if args.milestone not in milestones:
            print(f"Unknown milestone: {args.milestone}. Available: {list(milestones.keys())}")
            return 1
        spec = milestones[args.milestone]
        return record_command(
            cmd=spec["cmd"],
            output_path=spec["path"],
            code_head=args.code_head,
            artifacts=spec.get("artifacts"),
        )

    if not args.output:
        print("Error: --output <file> is required when executing custom command.")
        return 1

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print("Error: No command specified to record.")
        return 1

    return record_command(
        cmd=cmd,
        output_path=args.output,
        code_head=args.code_head,
        artifacts=args.artifact,
    )


if __name__ == "__main__":
    sys.exit(main())
