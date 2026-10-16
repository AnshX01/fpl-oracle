from pathlib import Path
from unittest.mock import patch

from scripts.check_secrets import (
    check_file_contents,
    check_git_tracked_files,
    scan_working_tree,
)


def test_check_file_contents_fails_closed_on_unreadable_file(tmp_path):
    """S1-1e: scripts/check_secrets.py must FAIL CLOSED on unreadable files."""
    test_file = tmp_path / "locked_file.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    # Simulate read error (e.g. PermissionError or OS error)
    with patch.object(Path, "read_text", side_effect=PermissionError("Permission denied")):
        results = check_file_contents(test_file)

    assert len(results) == 1
    line_num, category, snippet = results[0]
    assert category == "Unreadable File Error"
    assert "FAIL CLOSED" in snippet
    assert "Permission denied" in snippet


def test_check_file_contents_detects_aiza_and_aq_keys(tmp_path):
    """S1-1e: Detect AIza and AQ keys."""
    file_aiza = tmp_path / "aiza.py"
    key_aiza = "AIza" + "SyD123456789012345678901234567890123"
    file_aiza.write_text(f"API_KEY = '{key_aiza}'", encoding="utf-8")
    res_aiza = check_file_contents(file_aiza)
    assert len(res_aiza) == 1
    assert "Google Gemini / Cloud API Key" in res_aiza[0][1]

    file_aq = tmp_path / "aq.py"
    key_aq = "AQ." + "AbCdEfGhIjKlMnOpQrStUvWxYz_123456789"
    file_aq.write_text(f"OAUTH = '{key_aq}'", encoding="utf-8")
    res_aq = check_file_contents(file_aq)
    assert len(res_aq) == 1
    assert "AQ prefix" in res_aq[0][1]


def test_check_git_tracked_files_detects_forbidden_patterns():
    """S1-1e: Detects tracked .env, database, and log files."""
    mock_files = "\n".join(
        [
            "src/fpl_oracle/main.py",
            ".env",
            "logs/app.log",
            "data/app.sqlite",
        ]
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = mock_files
        violations = check_git_tracked_files()

    assert any(".env" in v for v in violations)
    assert any("logs/app.log" in v for v in violations)
    assert any("data/app.sqlite" in v for v in violations)


def test_scan_working_tree_passes_clean_repository():
    """S1-1e: Verify clean repository working tree produces 0 violations."""
    violations = scan_working_tree(include_logs=False, include_env=False)
    assert violations == [], f"Expected 0 violations, found: {violations}"
