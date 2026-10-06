#!/usr/bin/env python3
"""
FPL Oracle - Cross-platform runner script
Supports: setup, run, train, test, verify, clean
"""

import os
import shutil
import subprocess
import sys

# Ensure current working directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def get_python_executable():
    """Detect whether to use venv python or sys.executable."""
    venv_python_win = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    venv_python_unix = os.path.join(BASE_DIR, ".venv", "bin", "python")
    if os.path.exists(venv_python_win):
        return venv_python_win
    if os.path.exists(venv_python_unix):
        return venv_python_unix
    return sys.executable


def cmd_setup():
    py = get_python_executable()
    print("=== [FPL Oracle] Running Setup ===")
    os.makedirs(os.path.join(BASE_DIR, "data", "cache"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data", "historical"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

    # Install package in editable mode
    subprocess.run([py, "-m", "pip", "install", "-e", "."], check=True, cwd=BASE_DIR)

    # Download historical data
    print("--- Ingesting historical & live data ---")
    res = subprocess.run(
        [
            py,
            "-c",
            "from fpl_oracle.data.historical import HistoricalDataManager; HistoricalDataManager().ensure_dataset_ready()",
        ],
        cwd=BASE_DIR,
    )
    if res.returncode != 0:
        print("[Warning] Historical data ingestion encountered a warning. Proceeding...")

    print("=== Setup Complete! ===")


def cmd_run():
    py = get_python_executable()
    print("=== [FPL Oracle] Starting Server on http://127.0.0.1:8000 ===")
    try:
        subprocess.run(
            [py, "-m", "uvicorn", "fpl_oracle.server.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=BASE_DIR
        )
    except KeyboardInterrupt:
        print("\n=== Server Stopped ===")


def cmd_train():
    py = get_python_executable()
    print("=== [FPL Oracle] Training ML Models ===")
    subprocess.run([py, "-m", "fpl_oracle.ml.train"], check=True, cwd=BASE_DIR)
    print("=== Training Complete! Reports updated in reports/model_eval.md ===")


def cmd_test():
    py = get_python_executable()
    print("=== [FPL Oracle] Running Test Suite ===")
    res = subprocess.run([py, "-m", "pytest", "tests", "-v"], cwd=BASE_DIR)
    sys.exit(res.returncode)


def cmd_verify():
    py = get_python_executable()
    print("=== [FPL Oracle] Running Verification Suite ===")
    res = subprocess.run(
        [
            py,
            "-c",
            "from fpl_oracle.verify import run_verification; import sys; sys.exit(0 if run_verification() else 1)",
        ],
        cwd=BASE_DIR,
    )
    sys.exit(res.returncode)


def cmd_backtest():
    py = get_python_executable()
    print("=== [FPL Oracle] Running Blind Out-of-Time Backtest ===")
    res = subprocess.run([py, "-m", "fpl_oracle.backtest"], cwd=BASE_DIR)
    sys.exit(res.returncode)


def cmd_cli(cli_args):
    py = get_python_executable()
    res = subprocess.run([py, "-m", "fpl_oracle.cli"] + cli_args, cwd=BASE_DIR)
    sys.exit(res.returncode)


def cmd_clean():
    print("=== Cleaning Cache & Temporary Files ===")
    for root, dirs, _files in os.walk(BASE_DIR):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    print("Clean completed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [setup|run|train|test|verify|backtest|clean|cli]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "setup":
        cmd_setup()
    elif cmd == "run":
        cmd_run()
    elif cmd == "train":
        cmd_train()
    elif cmd == "test":
        cmd_test()
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "backtest":
        cmd_backtest()
    elif cmd == "clean":
        cmd_clean()
    elif cmd == "cli":
        cmd_cli(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        print("Available commands: setup, run, train, test, verify, backtest, clean, cli")
        sys.exit(1)


if __name__ == "__main__":
    main()
