"""
scripts/record_g2_evidence.py
Executes all four CI commands (plus check_secrets) and outputs unedited outputs,
exact command lines, and exit codes into reports/evidence/G2-ci.txt.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_FILE = ROOT / "reports" / "evidence" / "G2-ci.txt"

scripts_dir = Path(sys.executable).parent
ruff_cmd = str(scripts_dir / "ruff.exe") if (scripts_dir / "ruff.exe").exists() else "ruff"
mypy_cmd = str(scripts_dir / "mypy.exe") if (scripts_dir / "mypy.exe").exists() else "mypy"
pytest_cmd = str(scripts_dir / "pytest.exe") if (scripts_dir / "pytest.exe").exists() else "pytest"

COMMANDS = [
    [ruff_cmd, "check", "src", "tests", "scripts"],
    [ruff_cmd, "format", "--check", "src", "tests", "scripts"],
    [mypy_cmd, "src/fpl_oracle"],
    [sys.executable, "scripts/check_secrets.py"],
    [pytest_cmd, "-q"],
]

DISPLAY_COMMANDS = [
    "ruff check src tests scripts",
    "ruff format --check src tests scripts",
    "mypy src/fpl_oracle",
    "python scripts/check_secrets.py",
    "pytest -q",
]


def main():
    results = []
    all_success = True
    env = dict(os.environ)
    env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"

    for cmd, disp in zip(COMMANDS, DISPLAY_COMMANDS, strict=True):
        print(f"Running: {disp}")
        res = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"Exit code: {res.returncode}")
        if res.returncode != 0:
            all_success = False

        section = [
            f"=== COMMAND: {disp} ===",
            f"exit code: {res.returncode}",
            "--- STDOUT ---",
            res.stdout.strip(),
            "--- STDERR ---",
            res.stderr.strip() if res.stderr.strip() else "(none)",
            "",
        ]
        results.append("\n".join(section))

    content = "\n".join(results)
    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.write_text(content, encoding="utf-8")
    print(f"Recorded evidence to {EVIDENCE_FILE}")

    if not all_success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
