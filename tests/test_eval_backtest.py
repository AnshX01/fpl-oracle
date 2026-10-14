"""
Tests for Walk-Forward Backtester and Baselines Engine (WI-0.1, WI-0.2, WI-0.3).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fpl_oracle.eval.backtest import WalkForwardBacktester, pinball_loss
from fpl_oracle.ml.holdout import save_official_ep_snapshot


def test_pinball_loss_symmetry():
    """Verify pinball loss computation for quantile boundaries and median."""
    y_true = np.array([10.0, 5.0, 2.0])
    y_pred = np.array([8.0, 5.0, 4.0])

    # Median (tau = 0.5) is half MAE
    loss_50 = pinball_loss(y_true, y_pred, 0.5)
    mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(loss_50, 0.5 * mae)

    # Pinball loss is non-negative
    loss_10 = pinball_loss(y_true, y_pred, 0.1)
    loss_90 = pinball_loss(y_true, y_pred, 0.9)
    assert loss_10 >= 0
    assert loss_90 >= 0


def test_save_official_ep_snapshot(tmp_path: Path):
    """WI-0.2: Verify official ep_next / ep_this pre-deadline snapshotting."""
    el1 = MagicMock()
    el1.id = 101
    el1.web_name = "Haaland"
    el1.team = 11
    el1.element_type = 4
    el1.now_cost = 152
    el1.status = "a"
    el1.ep_next = "8.5"
    el1.ep_this = "8.2"

    el2 = MagicMock()
    el2.id = 102
    el2.web_name = "Palmer"
    el2.team = 6
    el2.element_type = 3
    el2.now_cost = 108
    el2.status = "d"
    el2.ep_next = None
    el2.ep_this = "5.0"

    bootstrap = MagicMock()
    bootstrap.elements = [el1, el2]

    snap_file = save_official_ep_snapshot(bootstrap, target_gw=12, custom_snapshots_dir=tmp_path)
    assert snap_file.exists()
    assert snap_file.name == "gw12_official_ep.json"

    with open(snap_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["gameweek"] == 12
    assert data["player_count"] == 2
    assert len(data["elements"]) == 2
    assert data["elements"][0]["element"] == 101
    assert data["elements"][0]["ep_next"] == 8.5
    assert data["elements"][1]["element"] == 102
    assert data["elements"][1]["ep_next"] is None
    assert data["elements"][1]["ep_this"] == 5.0


def test_walk_forward_backtester_structure(tmp_path: Path):
    """WI-0.1: Verify backtester instantiation and output path handling."""
    tester = WalkForwardBacktester(output_base_dir=tmp_path)
    assert tester.output_base_dir == tmp_path
    assert tmp_path.exists()


def test_calc_spearmans_tied_and_untied():
    """Verify tied vs ordinal untied Spearman calculation."""
    from fpl_oracle.eval.backtest import calc_spearmans

    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([2.0, 3.0, 3.0, 5.0])
    sp_tied, sp_noties = calc_spearmans(x, y)
    assert sp_tied == 1.0
    assert sp_noties == 1.0

    # Strictly reversed values without ties
    x_strict = np.array([1.0, 2.0, 3.0, 4.0])
    y_rev = np.array([4.0, 3.0, 2.0, 1.0])
    sp_tied_rev, sp_noties_rev = calc_spearmans(x_strict, y_rev)
    assert sp_tied_rev == -1.0
    assert sp_noties_rev == -1.0


def test_bootstrap_gameweeks():
    import pandas as pd

    from fpl_oracle.eval.backtest import bootstrap_gameweeks

    # Synthetic multi-GW dataframe where model beats baseline consistently
    records = []
    for gw in range(1, 11):
        for _p in range(20):
            records.append({
                "season": "2025-26",
                "round": gw,
                "actual_points": 5.0,
                "model_xp": 4.5,      # error = 0.5
                "b2_roll_5": 3.0,     # error = 2.0 -> model gain = 1.5
            })
    df_syn = pd.DataFrame(records)
    res = bootstrap_gameweeks(df_syn, n_boot=500, seed=42)
    assert res["n_gameweeks"] == 10
    assert np.isclose(res["mean_gain_mae"], 1.5)
    assert res["ci_95_mae"][0] > 0
    assert res["bootstrap_p_value_mae"] == 0.0


def test_odds_devigging_and_snapshot(tmp_path: Path):
    """WI-3.2: Verify power and Shin devigging methods and odds snapshotting."""
    from fpl_oracle.data.odds import devig_power, devig_shin, snapshot_odds

    odds = [2.0, 3.5, 4.0]
    p_power = devig_power(odds)
    assert np.isclose(np.sum(p_power), 1.0)
    assert p_power[0] > p_power[1] > p_power[2]

    p_shin, z = devig_shin(odds)
    assert np.isclose(np.sum(p_shin), 1.0)
    assert z >= 0.0
    assert p_shin[0] > p_shin[1] > p_shin[2]

    snap_file = snapshot_odds(target_gw=8, custom_output_dir=tmp_path)
    assert snap_file.exists()
    with open(snap_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data["gameweek"] == 8
    assert data["matches_count"] > 0
    assert "devigged_p_shin" in data["matches"][0]
    assert "devigged_p_power" in data["matches"][0]

