#!/usr/bin/env python3
"""
scripts/check_secrets.py - Pre-commit Security & Secrets Auditor for FPL Oracle.
Scans the working tree and git tracking status for accidental secret leaks,
tokens, private keys, and tracked user-identifiable data files (.env, profile.json, *.db).
"""

import re
import subprocess
import sys
from pathlib import Path

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    (r"AIza[0-9A-Za-z-_]{35}", "Google Gemini / Cloud API Key"),
    (r"sk-[a-zA-Z0-9T3BlbkFJ]{20,}", "OpenAI API Key (Legacy)"),
    (r"sk-proj-[a-zA-Z0-9-_]{20,}", "OpenAI Project API Key"),
    (r"sk-ant-[a-zA-Z0-9-_]{20,}", "Anthropic Claude API Key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
    (r"github_pat_[a-zA-Z0-9_]{50,}", "GitHub Fine-Grained Token"),
    (r"\b\d{9,10}:[a-zA-Z0-9_-]{35}\b", "Telegram Bot Token"),
    (r"tvly-[a-zA-Z0-9]{32,}", "Tavily Search API Key"),
    (r"BSA[a-zA-Z0-9-_]{30,}", "Brave Search API Key"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----", "Private Key Header"),
]

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

FORBIDDEN_TRACKED_PATTERNS = [
    r"^\.env$",
    r"^data/profile\.json$",
    r"^.*\.db$",
    r"^.*\.sqlite$",
    r"^.*\.sqlite3$",
]


def check_git_tracked_files() -> list[str]:
    """Verify that sensitive user data files are not tracked in git."""
    violations = []
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        tracked_files = res.stdout.strip().splitlines()
        for f in tracked_files:
            norm_f = f.replace("\\", "/")
            for pat in FORBIDDEN_TRACKED_PATTERNS:
                if re.search(pat, norm_f):
                    violations.append(f"Forbidden tracked file in git: {f}")
    except Exception as e:
        print(f"[WARN] Could not run git ls-files: {e}")
    return violations


def check_file_contents(file_path: Path) -> list[tuple[int, str, str]]:
    """Scan a single file for secret patterns."""
    findings = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    lines = content.splitlines()
    for line_num, line in enumerate(lines, start=1):
        # Skip comment explanations in docs or this script itself
        if "check_secrets.py" in str(file_path) and "SECRET_PATTERNS" in line:
            continue
        if ".env.example" in str(file_path):
            continue
        for pat, desc in SECRET_PATTERNS:
            if re.search(pat, line):
                # Mask matched string
                findings.append((line_num, desc, line[:40] + "... [MASKED]"))
    return findings


def scan_working_tree() -> list[str]:
    """Scan all tracked and candidate files in repo."""
    findings = []
    try:
        res = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        files = res.stdout.strip().splitlines()
        for f in files:
            p = REPO_ROOT / f
            if p.suffix in {".pkl", ".csv", ".ico", ".png", ".jpg", ".pyc", ".db", ".sqlite"}:
                continue
            results = check_file_contents(p)
            for line_num, desc, snippet in results:
                findings.append(f"Secret detected in {f}:{line_num} ({desc}) -> {snippet}")
    except Exception as e:
        print(f"[WARN] Could not scan files: {e}")
    return findings


def main():
    print("🔒 [FPL Oracle Security Audit] Scanning repository for secrets & sensitive data...")
    tracked_violations = check_git_tracked_files()
    content_violations = scan_working_tree()

    all_violations = tracked_violations + content_violations

    if all_violations:
        print("\n❌ SECURITY CHECK FAILED! Found sensitive leaks:")
        for v in all_violations:
            print(f"  • {v}")
        print("\nPlease fix these issues before committing or pushing!")
        sys.exit(1)
    else:
        print("✅ SECURITY CHECK PASSED! No secrets or forbidden tracked files detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
