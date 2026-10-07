"""
Test suite for Milestone G11: Proxy simulation backtest honesty,
strict FPL squad legality, leakage barriers, and calibration metrics.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.backtest import assert_no_future_data
from scripts.run_league_backtest import assert_legal_squad, run_proxy_simulation


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


def test_assert_legal_squad_enforces_rules():
    """Verify that assert_legal_squad rejects invalid squads and accepts valid ones."""
    # 1. Reject invalid starters count
    df_10 = pd.DataFrame([{"position": "DEF", "team": 1, "value": 50}] * 10)
    with pytest.raises(ValueError, match="must have exactly 11 starters"):
        assert_legal_squad(df_10)

    # 2. Reject missing goalkeeper (0 GKP)
    df_no_gk = pd.DataFrame(
        [{"position": "DEF", "team": 1, "value": 50}] * 3
        + [{"position": "DEF", "team": 2, "value": 50}] * 2
        + [{"position": "MID", "team": 3, "value": 50}] * 3
        + [{"position": "MID", "team": 4, "value": 50}] * 2
        + [{"position": "FWD", "team": 5, "value": 50}] * 1
    )
    with pytest.raises(ValueError, match="must have exactly 1 GKP"):
        assert_legal_squad(df_no_gk)

    # 3. Reject invalid formation (e.g. 2 DEF)
    df_invalid_form = pd.DataFrame(
        [{"position": "GKP", "team": 1, "value": 50}] * 1
        + [{"position": "DEF", "team": 2, "value": 50}] * 2
        + [{"position": "MID", "team": 3, "value": 50}] * 3
        + [{"position": "MID", "team": 4, "value": 50}] * 2
        + [{"position": "FWD", "team": 5, "value": 50}] * 3
    )
    with pytest.raises(ValueError, match="must have 3-5 DEF"):
        assert_legal_squad(df_invalid_form)

    # 4. Reject club limit > 3
    df_club_limit = pd.DataFrame(
        [{"position": "GKP", "team": 1, "value": 50}] * 1
        + [{"position": "DEF", "team": 1, "value": 50}] * 4  # 5 players from team 1!
        + [{"position": "MID", "team": 2, "value": 50}] * 4
        + [{"position": "FWD", "team": 3, "value": 50}] * 2
    )
    with pytest.raises(ValueError, match="Club limit exceeded"):
        assert_legal_squad(df_club_limit)

    # 5. Reject budget exceeded
    df_over_budget = pd.DataFrame(
        [{"position": "GKP", "team": 1, "value": 100}] * 1
        + [{"position": "DEF", "team": 2, "value": 100}] * 3
        + [{"position": "MID", "team": 3, "value": 100}] * 4
        + [{"position": "FWD", "team": 4, "value": 100}] * 3
    )  # Total value = 1100 > 1000
    with pytest.raises(ValueError, match="exceeds budget"):
        assert_legal_squad(df_over_budget, max_budget=1000.0)

    # 6. Accept perfectly legal squad
    df_legal = pd.DataFrame(
        [{"position": "GKP", "team": 1, "value": 45}] * 1
        + [{"position": "DEF", "team": 2, "value": 50}] * 3
        + [{"position": "DEF", "team": 3, "value": 50}] * 1
        + [{"position": "MID", "team": 4, "value": 60}] * 3
        + [{"position": "MID", "team": 5, "value": 60}] * 1
        + [{"position": "FWD", "team": 6, "value": 70}] * 2
    )
    assert_legal_squad(df_legal, max_budget=1000.0)


def test_proxy_simulation_execution_and_artifacts():
    """Verify proxy simulation execution, JSON artifact generation, sample size, and Brier metrics."""
    res = run_proxy_simulation(quick=False)

    assert res["simulation_type"] == "proxy_simulation"
    assert res["is_synthetic_league"] is True
    assert res["sample_size"] > 10
    assert len(res["scenarios"]) == res["sample_size"]

    # Verify Brier score calibration
    assert 0.0 <= res["model_brier"] <= 0.40, f"Model Brier {res['model_brier']} out of bounds"
    assert 0.0 <= res["naive_uniform_brier"] <= 1.0
    assert 0.0 <= res["naive_points_lead_brier"] <= 1.0

    # Verify reliability table structure
    rel_table = res["reliability_table"]
    assert len(rel_table) == 5
    for bin_row in rel_table:
        assert "bin" in bin_row
        assert "count" in bin_row
        assert "pred" in bin_row
        assert "actual" in bin_row

    # Verify squad points summary
    pts_sum = res["squad_points_summary"]
    assert "model_mean_pts_per_gw" in pts_sum
    assert "prev_gw_mean_pts_per_gw" in pts_sum
    assert "season_avg_mean_pts_per_gw" in pts_sum
    assert "template_mean_pts_per_gw" in pts_sum

    # Verify disk artifact
    json_path = Path("reports/backtest.json")
    assert json_path.exists()
    saved_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_data["sample_size"] == res["sample_size"]
    assert saved_data["model_brier"] == res["model_brier"]

    md_path = Path("reports/fix_pass_backtest.md")
    assert md_path.exists()
    md_content = md_path.read_text(encoding="utf-8")
    assert "Proxy Simulation" in md_content
    assert "Reliability Calibration Curve" in md_content
