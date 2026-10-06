import subprocess
from unittest.mock import patch

import pytest

import scripts.check_secrets as cs


def test_subprocess_failure_fails_closed_in_tracked_files():
    """Verify that if git fails, check_git_tracked_files returns a violation (fail-closed)."""
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["git", "ls-files"])):
        violations = cs.check_git_tracked_files()
        assert len(violations) > 0
        assert any("Security audit aborted" in v for v in violations)


def test_subprocess_failure_fails_closed_in_scan_tree():
    """Verify that if git fails during tree scan, scan_working_tree returns a violation (fail-closed)."""
    with patch("subprocess.run", side_effect=FileNotFoundError("git command not found")):
        findings = cs.scan_working_tree()
        assert len(findings) > 0
        assert any("Security audit aborted" in f for f in findings)


def test_main_exits_nonzero_when_subprocess_fails():
    """Verify that main() exits with status 1 if git subprocess fails."""
    with patch("subprocess.run", side_effect=RuntimeError("Subprocess failed")):
        with pytest.raises(SystemExit) as exc_info:
            cs.main()
        assert exc_info.value.code == 1


def test_main_exits_zero_on_clean_repo():
    """Verify that main() exits with status 0 on a clean repository."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "src/fpl_oracle/config.py\nREADME.md\n"
        with patch.object(cs, "check_file_contents", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                cs.main()
            assert exc_info.value.code == 0
