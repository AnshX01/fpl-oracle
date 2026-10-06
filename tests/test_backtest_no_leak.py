"""
Test suite for Requirement F8: Zero-hindsight historical backtest and leakage barrier assertions.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.backtest import assert_no_future_data


def test_assert_no_future_data_catches_future_round():
    """Verify that assert_no_future_data raises ValueError if future rounds are present."""
    df_leak = pd.DataFrame(
        [
            {"element": 1, "round": 5, "total_points": 6},
            {"element": 1, "round": 6, "total_points": 8},  # future round!
        ]
    )

    with pytest.raises(ValueError, match="Future data leak detected"):
        assert_no_future_data(df_leak, eval_round=5)


def test_assert_no_future_data_passes_valid_history():
    """Verify that assert_no_future_data passes when all rounds <= eval_round."""
    df_valid = pd.DataFrame(
        [
            {"element": 1, "round": 4, "total_points": 6},
            {"element": 1, "round": 5, "total_points": 8},
        ]
    )
    # Should not raise
    assert_no_future_data(df_valid, eval_round=5)


def test_no_hindsight_leaks_in_backtest_source():
    """Verify that backtest.py does not contain expected_points = total_points or 338/424."""
    backtest_src = Path("src/fpl_oracle/backtest.py").read_text(encoding="utf-8")
    assert "expected_points = total_points" not in backtest_src
    assert "338.0 / 424.0" not in backtest_src
    assert "338 / 424" not in backtest_src


def test_historical_backtest_execution_and_brier_score():
    """Verify that the historical league backtest runs and produces a well-calibrated Brier score."""
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from scripts.run_league_backtest import run_backtest

    res = run_backtest()
    assert "brier_score" in res
    assert 0.0 <= res["brier_score"] <= 0.25, f"Brier score {res['brier_score']} exceeds benchmark"
    assert len(res["scenarios"]) == 10
    assert Path("reports/fix_pass_backtest.md").exists()
