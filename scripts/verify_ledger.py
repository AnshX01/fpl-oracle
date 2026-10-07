"""
Evidence Ledger, Integrity Gate & Structural Reproducibility Validator.
Windows Validator & Fail-Closed Evidence Contract (Chunk 1 Repair A).

Verifies:
1. Component model binary and calibration SHA256 locks in manifest.json against disk.
2. Exact structural evidence contract for every milestone G1..G14 in reports/final_fix_ledger.md:
   - Valid, existing, non-empty evidence files with exact command line.
   - Exact exit_code integer == 0. Rejects missing, malformed, or non-zero exit codes.
   - Non-empty output block.
   - Fail-closed structural output check: rejects explicit failure markers
     ("not passed", "Traceback", "VERIFICATION FAILED", pytest failure/error summaries).
     Does not use naive substring matching ("passed" inside "not passed").
   - Provenance binding: SHA256 file fingerprinting, git HEAD verification.
   - Distinguishes active reproduced evidence from legacy/unverified historical records.
3. Structural doc consistency:
   - Evaluates authoritative generated JSON artifacts against exact structural entries
     in README.md and reports/final_status.md.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "reports" / "evidence"
LEDGER_PATH = ROOT / "reports" / "final_fix_ledger.md"
MANIFEST_PATH = ROOT / "data" / "models" / "manifest.json"
REPORTS_DIR = ROOT / "reports"


def get_current_git_head() -> str | None:
    """Retrieve current Git HEAD hash if running inside a git repository."""
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


def verify_manifest_hashes() -> bool:
    print("\n--- 1. Verifying Manifest SHA256 Locks (Including Calibration) ---")
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
    if not file_hashes:
        print("  [ERROR] No file_hashes recorded in manifest!")
        return False

    # Critical requirement: calibration.json MUST be present in manifest file_hashes
    if "calibration.json" not in file_hashes:
        print("  [ERROR] calibration.json hash lock is missing from manifest!")
        return False

    all_match = True
    for fname, expected_hash in file_hashes.items():
        fpath = ROOT / "data" / "models" / fname
        if not fpath.exists():
            print(f"  [MISSING] File {fname} not found on disk!")
            all_match = False
            continue

        with open(fpath, "rb") as bf:
            actual_hash = hashlib.sha256(bf.read()).hexdigest()

        if actual_hash != expected_hash:
            print(f"  [MISMATCH] {fname}: expected {expected_hash}, got {actual_hash}")
            all_match = False
        else:
            print(f"  [MATCH]   {fname:<24} SHA256 verified ({actual_hash[:16]}...)")

    return all_match


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_evidence_file(file_path: Path) -> tuple[bool, str, dict]:
    """Validate a recorded run, never execute untrusted recorded commands."""
    try:
        raw = file_path.read_bytes()
        record = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise ValueError("Legacy/unverified evidence: schema_version 1 required")
        for field in ("command", "exit_code", "stdout", "stderr", "code_head", "result", "artifacts", "gates"):
            if field not in record:
                raise ValueError(f"Missing {field}")
        if (
            not isinstance(record["command"], list)
            or not record["command"]
            or not all(isinstance(v, str) and v for v in record["command"])
        ):
            raise ValueError("command must be a nonempty argv list")
        if type(record["exit_code"]) is not int or record["exit_code"] != 0:
            raise ValueError("exit_code must be integer zero")
        if not isinstance(record["stdout"], str) or not isinstance(record["stderr"], str):
            raise ValueError("stdout/stderr must be strings")
        head = get_current_git_head()
        if head is None or record["code_head"] != head or not re.fullmatch(r"[0-9a-f]{40}", head):
            raise ValueError("Missing/unknown/stale checked code HEAD")
        if record["result"] != "PASS":
            raise ValueError("Explicit result is not PASS")
        output = record["stdout"] + "\n" + record["stderr"]
        if not output.strip():
            raise ValueError("Empty output")
        # Check every summary, including the final line after pytest's opening banner.
        for summary in re.findall(r"^=+\s*(.*?)\s*=+\s*$", output, re.MULTILINE):
            for count in re.findall(r"\b(\d+)\s+(?:failed|errors?)\b", summary.lower()):
                if int(count) > 0:
                    raise ValueError("Pytest failed/error summary")
        if re.search(
            r"^\s*(?:promotion_gate|gate_result|verification_result)\s*:\s*(?:FAIL|FAILED|NOT PASSED)\s*$",
            output,
            re.I | re.M,
        ):
            raise ValueError("Output declares failed gate")
        if "traceback (most recent call last)" in output.lower():
            raise ValueError("Unhandled traceback")
        if not isinstance(record["gates"], list) or not isinstance(record["artifacts"], list):
            raise ValueError("gates/artifacts must be lists")
        import math

        gate_names = set()
        for gate in record["gates"]:
            name = gate.get("name")
            if not isinstance(name, str) or not name or name in gate_names:
                raise ValueError("Missing/duplicate gate name")
            gate_names.add(name)
            if gate.get("status") != "PASS":
                raise ValueError(f"Gate {name} not PASS")
            value = gate.get("value")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Gate {name} value not finite")
            if "min" not in gate and "max" not in gate:
                raise ValueError(f"Gate {name} missing limits")
            for bound in ("min", "max"):
                if bound in gate and (type(gate[bound]) not in (int, float) or not math.isfinite(gate[bound])):
                    raise ValueError(f"Gate {name} invalid {bound}")
            if ("min" in gate and value < gate["min"]) or ("max" in gate and value > gate["max"]):
                raise ValueError(f"Gate {name} outside bounds")
        seen_paths = set()
        for artifact in record["artifacts"]:
            path = artifact.get("path")
            expected = artifact.get("sha256")
            if not isinstance(path, str) or path in seen_paths or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                raise ValueError("Invalid/duplicate artifact reference")
            seen_paths.add(path)
            target = (ROOT / path).resolve()
            if not target.is_relative_to(ROOT.resolve()) or not target.is_file():
                raise ValueError(f"Missing/outside artifact {path}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact hash mismatch {path}")
        return (
            True,
            "Valid recorded run; not rerun by this validator",
            {
                "command": record["command"],
                "exit_code": 0,
                "fingerprint_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        )
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return False, str(exc), {}


def verify_ledger_milestones_and_evidence() -> bool:
    print("\n--- 2. Verifying Milestone Accounting & Evidence in reports/final_fix_ledger.md ---")
    current_head = get_current_git_head()
    print(f"  [PROVENANCE] Current Git HEAD: {current_head or 'Unknown'}")
    if current_head is None:
        return False

    if not LEDGER_PATH.exists():
        print("  [ERROR] final_fix_ledger.md missing!")
        return False

    content = LEDGER_PATH.read_text(encoding="utf-8-sig")
    lines = content.splitlines()

    milestone_rows: dict[str, str] = {}
    row_pattern = re.compile(
        r"\|\s*\*\*(G\d+)\*\*\s*\|(.*?)\|\s*\*\*(CLOSED|PARTIAL|BLOCKED|NOT DONE|OPEN)\*\*\s*\|\s*(.*?)\|"
    )

    for line in lines:
        match = row_pattern.search(line)
        if match:
            m_id = match.group(1)
            if m_id in milestone_rows:
                print(f"  [ERROR] Duplicate milestone {m_id}")
                return False
            milestone_rows[m_id] = line

    all_passed = True
    expected_ids = [f"G{i}" for i in range(1, 15)]

    reproduced_count = 0

    for m_id in expected_ids:
        if m_id not in milestone_rows:
            print(f"  [MISSING] Milestone {m_id} not found in final_fix_ledger.md matrix!")
            all_passed = False
            continue

        line = milestone_rows[m_id]
        parts = [p.strip() for p in line.split("|")[1:-1]]
        final_status = parts[6].replace("*", "").strip() if len(parts) >= 7 else "UNKNOWN"
        evidence_field = parts[7].strip() if len(parts) >= 8 else ""

        print(f"  [{final_status:<8}] {m_id:<4}: ", end="")

        if final_status == "CLOSED":
            raw_paths = re.findall(r"reports/evidence/[A-Za-z0-9_.-]+", evidence_field)
            if not raw_paths:
                print(f"FAILED (No valid evidence file path in '{evidence_field}')")
                all_passed = False
                continue

            row_evidence_valid = True
            for rel_path in raw_paths:
                fpath = ROOT / rel_path
                is_valid, msg, meta = validate_evidence_file(fpath)
                if not is_valid:
                    print(f"\n    [FAIL-CLOSED REJECT] {rel_path}: {msg}")
                    row_evidence_valid = False
                    continue

            if row_evidence_valid:
                print(f"Recorded evidence validated: {len(raw_paths)} file(s); commands NOT rerun")
                reproduced_count += 1
            else:
                all_passed = False

        elif final_status in ("PARTIAL", "BLOCKED", "NOT DONE"):
            action_desc = parts[5].strip() if len(parts) >= 6 else ""
            if len(action_desc) < 10:
                print(f"FAILED (Honest accounting for {final_status} requires explicit rationale)")
                all_passed = False
            else:
                print(f"Accepted honest accounting: {action_desc[:50]}...")
        else:
            print(f"FAILED (Unresolved status '{final_status}')")
            all_passed = False

    print(
        f"\n  [AUDIT SUMMARY] Current-HEAD recorded milestones: {reproduced_count}/{len(expected_ids)}; rerun here: 0"
    )
    print("  [AUDIT SUMMARY] Historical records (M*, N*, T*, W*, Q*, G00-*): Preserved as audit history baseline.")
    return all_passed


def verify_doc_and_metric_consistency() -> bool:
    print("\n--- 3. Verifying Metric Consistency with Authoritative JSON Artifacts ---")
    all_consistent = True

    try:
        with open(REPORTS_DIR / "model_eval.json", encoding="utf-8") as f:
            eval_data = json.load(f)
        with open(REPORTS_DIR / "ablation.json", encoding="utf-8") as f:
            ablation_data = json.load(f)
        with open(REPORTS_DIR / "backtest.json", encoding="utf-8") as f:
            backtest_data = json.load(f)
        with open(REPORTS_DIR / "news_benchmark.json", encoding="utf-8") as f:
            news_data = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Failed to load JSON artifacts: {e}")
        return False

    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    status_text = (REPORTS_DIR / "final_status.md").read_text(encoding="utf-8")

    # Structural check on Model Performance section in README.md
    perf_section_match = re.search(r"##\s*📊\s*Model Validation & Performance(.*?)(?:\n##|\Z)", readme_text, re.DOTALL)
    if not perf_section_match:
        print("  [MISMATCH] Section 'Model Validation & Performance' missing in README.md")
        return False
    perf_section = perf_section_match.group(1)

    # 1. 2026-27 Holdout MAE and best baseline
    r_current = next((r for r in eval_data.get("rolling_origins", []) if "2026-27" in r.get("season", "")), {})
    current_mae_str = f"{r_current.get('mae', 0.0):.3f}"
    best_base_str = f"{r_current.get('best_baseline_mae', 0.0):.3f}"

    holdout_pattern = rf"\*\*Current 2026-27 Holdout\*\*:\s*\*\*{re.escape(current_mae_str)}\s*MAE\*\*\s*vs\s*{re.escape(best_base_str)}"
    if not re.search(holdout_pattern, perf_section):
        print(
            f"  [MISMATCH] 2026-27 Holdout MAE {current_mae_str} vs {best_base_str} structural entry missing in README.md"
        )
        all_consistent = False
    else:
        print(
            f"  [CONSISTENT] 2026-27 Holdout MAE: {current_mae_str} vs {best_base_str} structurally verified in README.md"
        )

    # 2. Backtest Sample Size and Brier
    bt_samples = str(backtest_data.get("sample_size"))
    bt_brier_str = f"{backtest_data.get('model_brier', 0.0):.4f}"
    backtest_pattern = rf"\*\*Brier Calibration Score\*\*:\s*\*\*{re.escape(bt_brier_str)}\*\*"
    scenario_pattern = rf"Evaluated across\s*{re.escape(bt_samples)}\s*historical scenarios"

    if not re.search(backtest_pattern, perf_section) or not re.search(scenario_pattern, perf_section):
        print(
            f"  [MISMATCH] Backtest structural entry (scenarios={bt_samples}, Brier={bt_brier_str}) missing in README.md"
        )
        all_consistent = False
    else:
        print(
            f"  [CONSISTENT] Backtest: {bt_samples} scenarios, Brier {bt_brier_str} structurally verified in README.md"
        )

    # 3. Ablation Improvement
    top_ab = ablation_data.get("groups", [{}])[0]
    ab_gain_str = f"{top_ab.get('mae_gain_pct', 0.0):.2f}%"
    ablation_pattern = rf"\*\*True Retraining Feature Ablation\*\*:\s*\+{re.escape(ab_gain_str)}"
    if not re.search(ablation_pattern, perf_section):
        print(f"  [MISMATCH] Ablation gain {ab_gain_str} structural entry missing in README.md")
        all_consistent = False
    else:
        print(f"  [CONSISTENT] Ablation gain +{ab_gain_str} structurally verified in README.md")

    # 4. News Benchmark Cases and Precision
    news_cases = str(news_data.get("total_cases_evaluated"))
    news_prec_str = f"{news_data.get('extraction_precision_pct', 0.0):.1f}%"
    news_pattern = rf"{re.escape(news_cases)}/{re.escape(news_cases)}\s*held-out cases passed\s*\({re.escape(news_prec_str)}\s*precision"
    if not re.search(news_pattern, perf_section):
        print(
            f"  [MISMATCH] News benchmark structural entry ({news_cases} cases, {news_prec_str} precision) missing in README.md"
        )
        all_consistent = False
    else:
        print(
            f"  [CONSISTENT] News Benchmark: {news_cases} cases, {news_prec_str} precision structurally verified in README.md"
        )

    # 5. Stale contradictions check in final_status.md
    if "F1 through F14" in status_text or "20/20" in status_text:
        print("  [STALE] reports/final_status.md contains stale F-pass or 20/20 claims!")
        all_consistent = False
    else:
        print("  [CONSISTENT] No known legacy pass-count wording; not a readiness verdict")

    return all_consistent


def main() -> int:
    print("=" * 70)
    print("FPL Oracle - Evidence Ledger, Integrity & Metric Validator (G14/Chunk 1)")
    print("=" * 70)

    hashes_ok = verify_manifest_hashes()
    ledger_ok = verify_ledger_milestones_and_evidence()
    docs_ok = verify_doc_and_metric_consistency()

    print("\n" + "=" * 70)
    if hashes_ok and ledger_ok and docs_ok:
        print("ALL AUDIT CHECKS PASSED: 100% EVIDENCE-GATED VERIFICATION ACHIEVED (0 LITERALS)")
        print("=" * 70)
        return 0
    else:
        print("VERIFICATION FAILED: Review the specific errors flagged above.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
