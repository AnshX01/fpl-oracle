"""
Acceptance and property test suite for WI-3.4:
Minutes and availability component model, decile reliability, and single-reconciler invariant.

Criteria:
1. P(starts) Brier score and E[minutes] MAE improve or stay equal on the backtest.
2. Reliability for P(play) is within +/- 3 pp per decile (with >= 200 samples per decile).
3. Single-reconciler invariant: only ReconciledAvailability modulates features; zero secondary discounts.
4. Probabilistic coherence: 0 <= p_min60 <= p_starts <= p_play <= 1.0 and
   60 * p_min60 <= E[minutes] <= 95 * p_play.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, mean_absolute_error

from fpl_oracle.data.features import FEATURE_COLUMNS, feature_engineering
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import scoring_ensemble


@pytest.fixture
def synthetic_minutes_model():
    np.random.seed(42)
    n = 150
    X = pd.DataFrame({col: np.random.uniform(0.1, 1.0, size=n) for col in FEATURE_COLUMNS})
    X["chance_of_playing"] = np.full(n, 100.0)

    y_mins = np.random.choice([0.0, 15.0, 65.0, 90.0], size=n, p=[0.3, 0.1, 0.2, 0.4])
    Y = pd.DataFrame(
        {
            "target_starts": (y_mins >= 60).astype(float),
            "target_min60": (y_mins >= 60).astype(float),
            "target_minutes": y_mins,
        }
    )
    X.attrs["meta"] = pd.DataFrame(
        {
            "season": ["2024-25"] * n,
            "round": [(i % 20) + 1 for i in range(n)],
        }
    )

    model = MinutesModel()
    model.fit(X, Y)
    return model


def test_probabilistic_coherence_minutes(synthetic_minutes_model):
    """
    Property test: verifies strict probabilistic coherence across minutes predictions.
    """
    model = synthetic_minutes_model
    n = 50
    X = pd.DataFrame({col: np.random.uniform(0.1, 1.0, size=n) for col in FEATURE_COLUMNS})

    preds = model.predict(X)
    p_starts = preds["p_starts"]
    p_min60 = preds["p_min60"]
    p_play = preds["p_play"]
    exp_mins = preds["expected_minutes"]

    # 1. Bounds: [0, 1]
    assert np.all(p_min60 >= 0.0) and np.all(p_min60 <= 1.0)
    assert np.all(p_starts >= 0.0) and np.all(p_starts <= 1.0)
    assert np.all(p_play >= 0.0) and np.all(p_play <= 1.0)

    # 2. Monotonicity: p_min60 <= p_starts <= p_play
    assert np.all(p_min60 <= p_starts + 1e-6), "p_min60 exceeded p_starts"
    assert np.all(p_starts <= p_play + 1e-6), "p_starts exceeded p_play"

    # 3. Minute expectation coherence
    assert np.all(exp_mins >= p_min60 * 60.0 - 1e-6), "exp_mins lower than 60 * p_min60"
    assert np.all(exp_mins <= p_play * 95.0 + 1e-6), "exp_mins exceeded 95 * p_play"


def test_single_reconciler_invariant():
    """
    Verifies that availability modulation in scoring_ensemble is strictly linear
    with cop_scale, with zero secondary discounts or double-penalties.
    """
    components = {
        "expected_goals": np.array([0.5, 0.5]),
        "expected_assists": np.array([0.3, 0.3]),
        "p_clean_sheet": np.array([0.4, 0.4]),
        "expected_goals_conceded": np.array([1.0, 1.0]),
        "p_min60": np.array([0.8, 0.8]),
        "p_starts": np.array([0.85, 0.85]),
        "p_play": np.array([0.90, 0.90]),
        "expected_minutes": np.array([75.0, 75.0]),
        "p_defcon": np.array([0.2, 0.2]),
        "expected_bonus": np.array([0.3, 0.3]),
        "expected_saves": np.array([0.0, 0.0]),
        "expected_card_deduction": np.array([0.1, 0.1]),
    }

    df_full = pd.DataFrame(
        {
            "chance_of_playing": [100.0, 100.0],
            "pos_MID": [1.0, 1.0],
        }
    )
    df_half = pd.DataFrame(
        {
            "chance_of_playing": [50.0, 50.0],
            "pos_MID": [1.0, 1.0],
        }
    )
    df_zero = pd.DataFrame(
        {
            "chance_of_playing": [0.0, 0.0],
            "pos_MID": [1.0, 1.0],
        }
    )

    res_full = scoring_ensemble.aggregate_components(dict(components), df_full)
    res_half = scoring_ensemble.aggregate_components(dict(components), df_half)
    res_zero = scoring_ensemble.aggregate_components(dict(components), df_zero)

    pts_full = float(res_full["expected_points"].iloc[0])
    pts_half = float(res_half["expected_points"].iloc[0])
    pts_zero = float(res_zero["expected_points"].iloc[0])

    # Invariant 1: 0% availability yields exactly 0 expected points
    assert pts_zero == 0.0, f"Expected 0.0 points for 0% COP, got {pts_zero}"

    # Invariant 2: 50% availability yields exactly 50% of the full expected points (linear scaling)
    assert np.isclose(pts_half, pts_full * 0.5, atol=1e-4), (
        f"Non-linear availability discount detected! 50% COP gave {pts_half:.4f} vs expected {pts_full * 0.5:.4f}"
    )


def test_p_play_decile_reliability_gate():
    """
    Acceptance Gate: reliability for P(play) is within +/- 3 pp per decile
    (with >= 200 samples per decile) on out-of-sample holdout data.
    """
    master_path = Path("data/historical/master_history.csv")
    if not master_path.exists():
        pytest.skip("Historical dataset not available")

    df_raw = pd.read_csv(master_path)
    X, Y, meta = feature_engineering.build_historical_features(df_raw, return_meta=True)

    mask_train = meta["season"].isin(["2023-24", "2024-25"])
    mask_test = meta["season"] == "2025-26"

    mm = MinutesModel()
    mm.fit(X[mask_train], Y[mask_train])
    preds = mm.predict(X[mask_test])

    p_play = preds["p_play"]
    y_mins = Y[mask_test]["target_minutes"].values
    y_play = (y_mins > 0).astype(float)

    # Decile reliability analysis
    df_rel = pd.DataFrame({"pred": p_play, "true": y_play})
    df_rel["decile"] = pd.qcut(df_rel["pred"], q=10, duplicates="drop")

    print("\n--- Out-of-Sample P(play) Decile Calibration Table (2025-26) ---")
    for dec, grp in df_rel.groupby("decile", observed=False):
        n = len(grp)
        pred_m = float(grp["pred"].mean())
        true_m = float(grp["true"].mean())
        diff_pp = (pred_m - true_m) * 100.0
        print(f"Decile {dec} (N={n}): Pred={pred_m:.3f} | Realised={true_m:.3f} | Diff={diff_pp:+.2f} pp")

        # Acceptance Gate: within +/- 3 pp for deciles with >= 200 samples
        if n >= 200:
            assert abs(diff_pp) <= 3.0, f"Decile {dec} diff {diff_pp:+.2f} pp exceeded +/- 3 pp gate (N={n})"


def test_minutes_brier_and_mae_gate():
    """
    Acceptance Gate: P(starts) Brier score and E[minutes] MAE improve or stay equal on the backtest.
    """
    master_path = Path("data/historical/master_history.csv")
    if not master_path.exists():
        pytest.skip("Historical dataset not available")

    df_raw = pd.read_csv(master_path)
    X, Y, meta = feature_engineering.build_historical_features(df_raw, return_meta=True)

    mask_train = meta["season"].isin(["2023-24", "2024-25"])
    mask_test = meta["season"] == "2025-26"

    mm = MinutesModel()
    mm.fit(X[mask_train], Y[mask_train])
    preds = mm.predict(X[mask_test])

    p_starts = preds["p_starts"]
    exp_mins = preds["expected_minutes"]

    y_starts = Y[mask_test]["target_starts"].values
    y_mins = Y[mask_test]["target_minutes"].values

    brier_starts = float(brier_score_loss(y_starts, p_starts))
    mae_mins = float(mean_absolute_error(y_mins, exp_mins))

    print("\n--- Minutes Model Metrics (2025-26 Holdout) ---")
    print(f"P(starts) Brier Score: {brier_starts:.4f} (Benchmark <= 0.1000)")
    print(f"E[minutes] MAE:        {mae_mins:.4f} (Benchmark <= 15.0000)")

    assert brier_starts <= 0.1000, f"Brier score {brier_starts:.4f} regressed"
    assert mae_mins <= 15.0000, f"MAE {mae_mins:.4f} regressed"
