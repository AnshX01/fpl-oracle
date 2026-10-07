"""
Part C Final Gate Evidence Runner (C3).
Executes each required gate command and records unedited outputs into reports/evidence/G99-*.txt.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "reports" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
PYTHON = sys.executable


def record_command(cmd_args: list[str], display_cmd: str, out_filename: str) -> int:
    print(f"\n{'=' * 70}\n[GATE] Running: {display_cmd}\n{'=' * 70}")
    start = time.time()
    res = subprocess.run(cmd_args, cwd=ROOT, capture_output=True, text=True, errors="replace")
    elapsed = time.time() - start
    print(f"Finished in {elapsed:.2f}s with exit code {res.returncode}")

    output_body = (res.stdout + res.stderr).strip()
    content = f"COMMAND: {display_cmd}\nEXIT CODE: {res.returncode}\nOUTPUT:\n{output_body}\n"
    target = EVIDENCE_DIR / out_filename
    target.write_text(content, encoding="utf-8")
    print(f"Saved evidence to {target}")

    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}!")
    return res.returncode


def main() -> int:
    tasks = [
        # (cmd_args, display_cmd, out_filename)
        (
            [PYTHON, "-c", "import fpl_oracle.server.main; print('fpl_oracle.server.main imported cleanly')"],
            'python -c "import fpl_oracle.server.main"',
            "G99-server.txt",
        ),
        (
            [PYTHON, "scripts/check_freshness.py"],
            "python scripts/check_freshness.py",
            "G99-freshness.txt",
        ),
        (
            [PYTHON, "scripts/check_feature_parity.py"],
            "python scripts/check_feature_parity.py",
            "G99-parity.txt",
        ),
        (
            [PYTHON, "scripts/run_news_benchmark.py"],
            "python scripts/run_news_benchmark.py",
            "G99-news.txt",
        ),
        (
            [PYTHON, "scripts/run_eval.py"],
            "python scripts/run_eval.py",
            "G99-eval.txt",
        ),
        (
            [PYTHON, "scripts/run_league_backtest.py"],
            "python scripts/run_league_backtest.py",
            "G99-backtest.txt",
        ),
        (
            [PYTHON, "-m", "pytest", "tests/test_decision_card.py", "-v"],
            "pytest tests/test_decision_card.py",
            "G99-card.txt",
        ),
        (
            [PYTHON, "scripts/capture_screenshots.py"],
            "python scripts/capture_screenshots.py",
            "G99-screenshots.txt",
        ),
    ]

    failed = 0
    for args, disp, out in tasks:
        code = record_command(args, disp, out)
        if code != 0:
            failed += 1

    print(f"\nFinal gate runner batch finished. Failures: {failed}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
