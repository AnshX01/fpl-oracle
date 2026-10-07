"""
Unit Tests for Evidence Validator Fail-Closed Behavior (Chunk 1 Repair A).

Tests:
1. Valid passing evidence fixture is accepted.
2. Explicit exit 9 / "not passed" fixture is rejected (audit defect reproduction).
3. Malformed non-integer exit code is rejected.
4. Missing command line is rejected.
5. Empty output block is rejected.
6. Pytest failure reporting in summary line is rejected.
7. Negative test case (e.g. 'test_rejection PASSED') is correctly accepted, not confused for failure.
8. Metric contradiction between JSON artifact and README is rejected.
"""

import json
from pathlib import Path

from scripts.verify_ledger import (
    validate_evidence_file,
    verify_doc_and_metric_consistency,
)


def test_passing_fixture_accepted(tmp_path: Path):
    passing_file = tmp_path / "passing.txt"
    passing_file.write_text(
        "COMMAND: pytest tests/test_smoke.py\n"
        "EXIT CODE: 0\n"
        "OUTPUT:\n"
        "============================= test session starts =============================\n"
        "tests/test_smoke.py .                                                    [100%]\n"
        "============================== 1 passed in 0.05s ==============================\n",
        encoding="utf-8",
    )
    is_valid, reason, meta = validate_evidence_file(passing_file)
    assert is_valid is True
    assert meta["exit_code"] == 0
    assert "pytest tests/test_smoke.py" in meta["command"]


def test_explicit_exit_9_not_passed_rejected(tmp_path: Path):
    """Directly tests the audit finding: COMMAND: python failing.py, EXIT CODE: 9, OUTPUT: not passed."""
    failing_file = tmp_path / "failing.txt"
    failing_file.write_text(
        "COMMAND: python failing.py\nEXIT CODE: 9\nOUTPUT:\nnot passed\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(failing_file)
    assert is_valid is False
    assert "non-zero exit code: 9" in reason.lower() or "not passed" in reason.lower()


def test_exit_0_with_not_passed_content_rejected(tmp_path: Path):
    """Tests that exit code 0 is still rejected if output contains 'not passed' failure marker."""
    failing_file = tmp_path / "failing_exit_zero.txt"
    failing_file.write_text(
        "COMMAND: python check.py\nEXIT CODE: 0\nOUTPUT:\nAll checks run. Status: not passed.\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(failing_file)
    assert is_valid is False
    assert "not passed" in reason


def test_malformed_exit_code_rejected(tmp_path: Path):
    bad_file = tmp_path / "malformed.txt"
    bad_file.write_text(
        "COMMAND: python run.py\nEXIT CODE: abc\nOUTPUT:\nExecuted.\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(bad_file)
    assert is_valid is False
    assert "Malformed" in reason


def test_missing_command_rejected(tmp_path: Path):
    no_cmd_file = tmp_path / "no_cmd.txt"
    no_cmd_file.write_text(
        "EXIT CODE: 0\nOUTPUT:\nSome random output.\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(no_cmd_file)
    assert is_valid is False
    assert "command" in reason.lower()


def test_empty_output_rejected(tmp_path: Path):
    empty_out_file = tmp_path / "empty_out.txt"
    empty_out_file.write_text(
        "COMMAND: python run.py\nEXIT CODE: 0\nOUTPUT:\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(empty_out_file)
    assert is_valid is False
    assert "output" in reason.lower()


def test_pytest_failure_summary_rejected(tmp_path: Path):
    pytest_fail_file = tmp_path / "pytest_fail.txt"
    pytest_fail_file.write_text(
        "COMMAND: pytest tests/\n"
        "EXIT CODE: 0\n"  # Even if exit code was mistakenly recorded as 0
        "OUTPUT:\n"
        "tests/test_x.py F                                                        [100%]\n"
        "======================== 1 failed, 4 passed in 1.23s ========================\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(pytest_fail_file)
    assert is_valid is False
    assert "failed" in reason.lower()


def test_negative_test_case_accepted(tmp_path: Path):
    """Ensures a test named 'test_reject_bad_hash' that PASSED is not mistaken for a failure."""
    neg_test_file = tmp_path / "neg_test.txt"
    neg_test_file.write_text(
        "COMMAND: pytest tests/test_security.py\n"
        "EXIT CODE: 0\n"
        "OUTPUT:\n"
        "tests/test_security.py::test_rejection_of_invalid_token PASSED          [100%]\n"
        "============================== 1 passed in 0.12s ==============================\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(neg_test_file)
    assert is_valid is True


def test_traceback_output_rejected(tmp_path: Path):
    tb_file = tmp_path / "traceback.txt"
    tb_file.write_text(
        "COMMAND: python crash.py\n"
        "EXIT CODE: 0\n"
        "OUTPUT:\n"
        "Traceback (most recent call last):\n"
        "  File 'crash.py', line 1, in <module>\n"
        "ValueError: unexpected crash\n",
        encoding="utf-8",
    )
    is_valid, reason, _ = validate_evidence_file(tb_file)
    assert is_valid is False
    assert "traceback" in reason.lower()


def test_metric_contradiction_rejected(monkeypatch, tmp_path: Path):
    """Modifies the eval data in memory to create an artificial contradiction and asserts verify fails."""
    from scripts import verify_ledger

    # Point REPORTS_DIR to a temporary directory with contradictory JSON
    test_reports = tmp_path / "reports"
    test_reports.mkdir()

    # Copy actual JSONs and markdown
    for f in ["ablation.json", "backtest.json", "news_benchmark.json", "final_status.md"]:
        orig = Path("reports") / f
        if orig.exists():
            (test_reports / f).write_text(orig.read_text(encoding="utf-8"), encoding="utf-8")

    # Write contradictory model_eval.json with MAE 9.999
    contradictory_eval = {
        "rolling_origins": [
            {
                "season": "Holdout 2026-27 (trained on 2023-24, 2024-25, 2025-26)",
                "mae": 9.999,
                "best_baseline_mae": 1.952,
            }
        ]
    }
    (test_reports / "model_eval.json").write_text(json.dumps(contradictory_eval), encoding="utf-8")

    monkeypatch.setattr(verify_ledger, "REPORTS_DIR", test_reports)
    consistent = verify_doc_and_metric_consistency()
    assert consistent is False
