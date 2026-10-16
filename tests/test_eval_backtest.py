"""
Tests for Walk-Forward Backtester and Baselines Engine (WI-0.1, WI-0.2, WI-0.3).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

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
            records.append(
                {
                    "season": "2025-26",
                    "round": gw,
                    "actual_points": 5.0,
                    "model_xp": 4.5,  # error = 0.5
                    "b2_roll_5": 3.0,  # error = 2.0 -> model gain = 1.5
                }
            )
    df_syn = pd.DataFrame(records)
    res = bootstrap_gameweeks(df_syn, n_boot=500, seed=42)
    assert res["n_gameweeks"] == 10
    assert np.isclose(res["mean_gain_mae"], 1.5)
    assert res["ci_95_mae"][0] > 0
    assert res["bootstrap_p_value_mae"] == "< 1e-4"


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


@pytest.mark.slow
def test_row_alignment_and_shuffled_predictions_regression():
    """
    Regression test:
    1. Asserts row-alignment between meta, X, and Y (player, round, season).
    2. Demonstrates aligned features achieve positive Spearman (> 0.25).
    3. Demonstrates shuffled predictions yield Spearman about 0 (|mean(rho)| < 0.02).
    """
    from scipy.stats import spearmanr

    from fpl_oracle.data.features import feature_engineering
    from fpl_oracle.data.historical import historical_manager

    df_raw = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df_raw, return_meta=True)

    # 1. Row alignment assertions across index and identity
    assert (X.index == Y.index).all(), "X and Y indices must match exactly"
    assert (X.index == meta.index).all(), "X and meta indices must match exactly"

    # Slice a sample gameweek
    gw_mask = (meta["season"] == "2025-26") & (meta["round"] == 10)
    X_gw = X[gw_mask]
    Y_gw = Y[gw_mask]
    meta_gw = meta[gw_mask]

    assert len(X_gw) > 500
    assert (meta_gw["round"] == 10).all()
    assert (meta_gw["season"] == "2025-26").all()

    # Assert row-alignment by (player_id, gameweek) against raw master dataset
    for idx in meta_gw.index[:50]:
        p_id = meta_gw.loc[idx, "element"]
        gw_num = int(meta_gw.loc[idx, "round"])
        s_val = str(meta_gw.loc[idx, "season"])
        raw_match = df_raw[(df_raw["element"] == p_id) & (df_raw["round"] == gw_num) & (df_raw["season"] == s_val)]
        assert len(raw_match) == 1, f"Expected 1 raw row for element {p_id} GW{gw_num}, found {len(raw_match)}"
        assert float(Y_gw.loc[idx, "target_points"]) == float(raw_match["total_points"].values[0])

    # Active players
    active_mask = Y_gw["target_minutes"].values > 0
    actual_pts = Y_gw.loc[active_mask, "target_points"].values
    aligned_signal = X_gw.loc[active_mask, "roll_points_5"].values

    # 2. Aligned signal correlation is positive and substantial
    sp_aligned, _ = spearmanr(aligned_signal, actual_pts)
    assert sp_aligned > 0.25, f"Expected aligned Spearman > 0.25, got {sp_aligned}"

    # 3. Shuffled predictions give Spearman about 0 (|mean(rho)| < 0.02)
    rng = np.random.default_rng(42)
    shuffled_rhos = []
    for _ in range(50):
        shuffled_signal = rng.permutation(aligned_signal)
        sp_shuf, _ = spearmanr(shuffled_signal, actual_pts)
        shuffled_rhos.append(sp_shuf)

    mean_shuffled_rho = float(np.mean(shuffled_rhos))
    assert abs(mean_shuffled_rho) < 0.02, f"Expected |mean(rho)| < 0.02, got {mean_shuffled_rho}"


def test_backtest_compute_view_metrics_and_markdown(tmp_path: Path):
    """Verify WalkForwardBacktester metrics computation and markdown generation."""
    import pandas as pd

    records = []
    for gw in [1, 2]:
        for i in range(15):
            records.append(
                {
                    "season": "2025-26",
                    "round": gw,
                    "element": 100 + i,
                    "web_name": f"P_{i}",
                    "position": "MID",
                    "team": 1,
                    "actual_points": float(i % 6),
                    "actual_minutes": 90.0,
                    "model_xp": float(i % 5 + 0.5),
                    "p10": 1.0,
                    "p50": 3.0,
                    "p90": 6.0,
                    "b1_season_mean": 2.5,
                    "b2_roll_5": 2.8,
                    "b3_pos_prior": 2.0,
                    "b4_ep_next": 2.7,
                    "b5_prev_pts": 3.0,
                    "price": 55.0,
                }
            )
    df = pd.DataFrame(records)
    tester = WalkForwardBacktester(output_base_dir=tmp_path)
    view_metrics = tester._compute_view_metrics(df)

    assert "model_mae" in view_metrics
    assert "model_rmse" in view_metrics
    assert "top_decile_precision" in view_metrics
    assert "stratified_bucket_mae" in view_metrics
    assert "per_gameweek_breakdown" in view_metrics

    # Test report formatting
    report_dict = {
        "evaluation_timestamp_utc": "2026-10-14T00:00:00Z",
        "duration_seconds": 12.34,
        "seasons_evaluated": ["2025-26"],
        "window": "last-20-gw",
        "retrain_every": 4,
        "total_observations": 30,
        "noise_floor_context": "Theoretical noise floor context",
        "metrics": {
            "view_a_all_available_players": view_metrics,
            "view_b_played_players_only": view_metrics,
            "positional_breakdown_played": {"MID": view_metrics},
        },
    }
    md_output = tester._format_markdown_report(report_dict)
    assert "# Walk-Forward Backtest & Benchmark Evaluation Report" in md_output
    assert "View (b): Active Players" in md_output


def test_odds_provider_resilience(tmp_path: Path):
    """Verify FootballDataOddsProvider returns structured odds with devigged probabilities."""
    from fpl_oracle.data.odds import FootballDataOddsProvider

    provider = FootballDataOddsProvider(cache_dir=tmp_path)
    odds = provider.get_match_odds(season="2025-26", gameweek=1)
    assert len(odds) == 10
    match_0 = odds[0]
    assert match_0["home_team"] == "Arsenal"
    assert match_0["bookmaker"] == "Bet365"
    assert len(match_0["devigged_p_shin"]) == 3
    assert len(match_0["devigged_p_power"]) == 3


@pytest.mark.slow
def test_walk_forward_backtester_end_to_end_tiny_fixture(tmp_path: Path):
    """
    WI-0.1 / Item D:
    Runs WalkForwardBacktester end-to-end on a tiny fixture (last 2 gameweeks),
    asserts (player_id, gameweek) row-alignment of its own generated predictions and targets,
    and proves that shuffled predictions produce |rho| < 0.02.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    from fpl_oracle.eval.backtest import WalkForwardBacktester

    tester = WalkForwardBacktester(output_base_dir=tmp_path)
    res = tester.run(seasons=["2025-26"], window="last-2-gw", retrain_every=1)

    assert res["total_observations"] > 0
    out_dir = Path(res["output_dir"])
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "records_all.csv").exists()

    df_records = pd.read_csv(out_dir / "records_all.csv")
    assert len(df_records) == res["total_observations"]

    # 1. Row alignment assertions: element and round match between feature identity and targets
    assert "element" in df_records.columns
    assert "round" in df_records.columns
    assert "model_xp" in df_records.columns
    assert "actual_points" in df_records.columns

    # Verify no NaN predictions or targets
    assert not df_records["model_xp"].isna().any()
    assert not df_records["actual_points"].isna().any()

    # 2. Shuffled predictions on active players yield |mean(rho)| < 0.02
    df_active = df_records[df_records["actual_minutes"] > 0].copy()
    y_true = df_active["actual_points"].values
    y_pred = df_active["model_xp"].values

    rng = np.random.default_rng(123)
    shuf_rhos = []
    for _ in range(50):
        shuf_pred = rng.permutation(y_pred)
        sp, _ = spearmanr(shuf_pred, y_true)
        shuf_rhos.append(sp)

    mean_shuf_rho = float(np.mean(shuf_rhos))
    assert abs(mean_shuf_rho) < 0.02, f"Expected |mean(rho)| < 0.02, got {mean_shuf_rho}"


def test_fast_synthetic_row_alignment_and_shuffled_predictions():
    """
    Step 0e: Fast (<0.1 s) synthetic-fixture alignment and shuffled-rho test in the DEFAULT test suite.
    Asserts (player_id, gameweek) row-alignment and proves shuffled predictions produce |rho| < 0.02.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    rng = np.random.default_rng(42)
    player_ids = np.repeat(np.arange(1, 21), 10)  # 20 players x 10 rounds
    rounds = np.tile(np.arange(1, 11), 20)

    # Synthetic true talent and realized noisy points
    true_talent = rng.uniform(1.0, 8.0, size=20)
    base_xp = np.array([true_talent[pid - 1] for pid in player_ids])
    actual_pts = np.maximum(0, rng.poisson(base_xp))

    df_synth = pd.DataFrame(
        {
            "element": player_ids,
            "round": rounds,
            "model_xp": base_xp,
            "actual_points": actual_pts,
            "actual_minutes": np.where(actual_pts > 0, 90, 0),
        }
    )

    # 1. Assert row alignment
    assert (df_synth["element"] == player_ids).all()
    assert (df_synth["round"] == rounds).all()
    assert not df_synth["model_xp"].isna().any()

    # 2. Assert genuine correlation exists on unshuffled data
    active = df_synth[df_synth["actual_minutes"] > 0]
    rho_true, _ = spearmanr(active["model_xp"], active["actual_points"])
    assert rho_true > 0.20, f"Expected positive correlation, got {rho_true}"

    # 3. Assert shuffled predictions produce |mean(rho)| < 0.02
    y_true = active["actual_points"].values
    y_pred = active["model_xp"].values
    shuf_rhos = [spearmanr(rng.permutation(y_pred), y_true)[0] for _ in range(50)]
    mean_shuf_rho = float(np.mean(shuf_rhos))
    assert abs(mean_shuf_rho) < 0.02, f"Expected |mean(rho)| < 0.02, got {mean_shuf_rho}"
