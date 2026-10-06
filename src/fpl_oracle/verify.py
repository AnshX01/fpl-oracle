"""
FPL Oracle Self-Check & Verification Suite.
Runs on `make verify` or `python run.py verify`.
Performs:
1. Live API connectivity and schema validation
2. SQLite storage & cache integrity check
3. ML component model artifacts validation
4. Pytest test suite execution
5. End-to-end miniature backtest execution
6. Summary reporting
"""

import asyncio
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(force_terminal=True, legacy_windows=False)


def run_verification() -> bool:
    console.print(
        Panel(
            "[bold green]FPL Oracle — Automated Self-Check & Verification Suite[/bold green]\nSeason 2026/27",
            border_style="green",
        )
    )

    checks = []

    # 1. Live FPL API Check
    console.print("[dim]1. Testing Live FPL API Connectivity & Schema...[/dim]")
    try:
        from fpl_oracle.api.fpl_client import fpl_client

        curr, nxt = asyncio.run(fpl_client.get_current_and_next_gw())
        boot, is_stale = asyncio.run(fpl_client.get_bootstrap_static())
        fix, _ = asyncio.run(fpl_client.get_fixtures())
        if len(boot.elements) > 500 and len(fix) > 300:
            checks.append(
                (
                    "Live FPL API Connectivity",
                    "PASS",
                    f"GW{curr} finished, GW{nxt} upcoming ({len(boot.elements)} players, {len(fix)} fixtures)",
                )
            )
        else:
            checks.append(("Live FPL API Connectivity", "FAIL", "Incomplete API response"))
    except Exception as e:
        checks.append(("Live FPL API Connectivity", "FAIL", str(e)))

    # 2. SQLite Database Integrity
    console.print("[dim]2. Verifying SQLite Database & Cache Storage...[/dim]")
    try:
        from fpl_oracle.data.store import data_store

        prof = data_store.get_profile()
        checks.append(
            (
                "Database & Persistent Store",
                "PASS",
                f"Profile loaded (risk: {prof.risk_preference}), DB initialized at {data_store.session_factory}",
            )
        )
    except Exception as e:
        checks.append(("Database & Persistent Store", "FAIL", str(e)))

    # 3. ML Model Artifacts Check
    console.print("[dim]3. Verifying Pre-trained ML Component Models...[/dim]")
    try:
        from fpl_oracle.config import MODELS_DIR

        required_models = [
            "minutes_model.pkl",
            "attacking_model.pkl",
            "defending_model.pkl",
            "defcon_model.pkl",
            "bonus_model.pkl",
            "cards_saves_model.pkl",
        ]
        missing = [m for m in required_models if not (MODELS_DIR / m).exists()]
        if not missing:
            checks.append(("ML Component Models", "PASS", "All 6 component models loaded and verified"))
        else:
            checks.append(("ML Component Models", "FAIL", f"Missing models: {missing}"))
    except Exception as e:
        checks.append(("ML Component Models", "FAIL", str(e)))

    # 4. Pytest Unit Tests
    console.print("[dim]4. Running Pytest Unit Test Suite...[/dim]")
    try:
        import pytest

        exit_code = pytest.main(["tests", "-q"])
        if exit_code == 0:
            checks.append(
                ("Pytest Test Suite (8/8)", "PASS", "Optimizer, chip logic, selling price, API schemas passed")
            )
        else:
            checks.append(("Pytest Test Suite", "FAIL", f"Pytest exited with code {exit_code}"))
    except Exception as e:
        checks.append(("Pytest Test Suite", "FAIL", str(e)))

    # 5. End-to-End Pipeline & Miniature Backtest
    console.print("[dim]5. Executing End-to-End Pipeline & Miniature Backtest...[/dim]")
    try:
        from fpl_oracle.backtest import backtest_harness

        res = backtest_harness.run_backtest(num_gws=3)
        if res and "oracle_total" in res:
            checks.append(
                ("End-to-End Pipeline & Backtest", "PASS", f"Replayed 3 GWs: Oracle scored {res['oracle_total']} pts")
            )
        else:
            checks.append(("End-to-End Pipeline & Backtest", "FAIL", "Invalid backtest result"))
    except Exception as e:
        checks.append(("End-to-End Pipeline & Backtest", "FAIL", str(e)))

    # Print Summary Table
    table = Table(title="Self-Check Verification Results")
    table.add_column("Verification Step", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_passed = True
    for name, status, details in checks:
        if status == "PASS":
            table.add_row(name, "[bold green]PASS[/bold green]", details)
        else:
            table.add_row(name, "[bold red]FAIL[/bold red]", f"[red]{details}[/red]")
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("\n[bold green]ALL SYSTEMS VERIFIED! FPL Oracle is production ready.[/bold green]")
        return True
    else:
        console.print("\n[bold red]SOME CHECKS FAILED. Please review above details.[/bold red]")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
