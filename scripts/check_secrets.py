#!/usr/bin/env python3
"""
scripts/check_secrets.py - Pre-commit Security & Secrets Auditor for FPL Oracle.
Scans the working tree and git tracking status for accidental secret leaks,
tokens, private keys, and tracked user-identifiable data files (.env, profile.json, *.db).
Fails closed on unreadable files.
"""

import argparse
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
    (r"AQ\.[A-Za-z0-9_-]{20,}", "Google Gemini OAuth / API Key (AQ prefix)"),
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
    r"^logs/.*\.log$",
]

BINARY_EXTENSIONS = {
    ".pkl",
    ".csv",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".pyc",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".zip",
    ".tar",
    ".gz",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
}


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
        )
        tracked_files = res.stdout.strip().splitlines()
        for f in tracked_files:
            norm_f = f.replace("\\", "/")
            for pat in FORBIDDEN_TRACKED_PATTERNS:
                if re.search(pat, norm_f):
                    violations.append(f"Forbidden tracked file in git: {f}")
    except Exception as e:
        print(f"[ERROR] Could not run git ls-files: {e}")
        violations.append(f"FAIL CLOSED: git ls-files failed ({e})")
    return violations


def check_file_contents(file_path: Path) -> list[tuple[int, str, str]]:
    """
    Scan a single file for secret patterns.
    FAIL CLOSED: reports an unreadable file violation if read fails.
    """
    findings: list[tuple[int, str, str]] = []
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return findings

    try:
        content = file_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as err:
            return [(-1, "Unreadable File Error", f"FAIL CLOSED: Could not read {file_path}: {err}")]
    except Exception as err:
        return [(-1, "Unreadable File Error", f"FAIL CLOSED: Could not read {file_path}: {err}")]

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


def scan_working_tree(include_logs: bool = False, include_env: bool = False) -> list[str]:
    """Scan tracked, untracked, and candidate files in repo."""
    findings = []
    try:
        res = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        file_paths = [REPO_ROOT / f for f in res.stdout.strip().splitlines() if f]
    except Exception as e:
        print(f"[ERROR] Could not scan files: {e}")
        return [f"FAIL CLOSED: git ls-files scan failed ({e})"]

    if include_env:
        env_p = REPO_ROOT / ".env"
        if env_p.exists():
            file_paths.append(env_p)

    if include_logs:
        logs_dir = REPO_ROOT / "logs"
        if logs_dir.exists():
            file_paths.extend(logs_dir.glob("*.log"))

    for p in file_paths:
        if not p.exists():
            continue
        results = check_file_contents(p)
        for line_num, desc, snippet in results:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            findings.append(f"Secret detected or file unreadable in {rel}:{line_num} ({desc}) -> {snippet}")
    return findings


def main():
    parser = argparse.ArgumentParser(description="FPL Oracle Secrets & Security Auditor")
    parser.add_argument("--include-logs", action="store_true", help="Also scan ignored logs/ directory")
    parser.add_argument("--include-env", action="store_true", help="Also scan ignored .env file")
    args = parser.parse_args()

    print("🔒 [FPL Oracle Security Audit] Scanning repository for secrets & sensitive data...")
    tracked_violations = check_git_tracked_files()
    content_violations = scan_working_tree(include_logs=args.include_logs, include_env=args.include_env)

    all_violations = tracked_violations + content_violations

    if all_violations:
        print("\n❌ SECURITY CHECK FAILED! Found sensitive leaks or unreadable files:")
        for v in all_violations:
            print(f"  • {v}")
        print("\nPlease fix these issues before committing or pushing!")
        sys.exit(1)
    else:
        print("✅ SECURITY CHECK PASSED! No secrets or forbidden tracked files detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
