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
