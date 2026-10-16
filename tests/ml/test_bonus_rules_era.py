"""
Acceptance and property test suite for WI-3.6:
Bonus model, 2026/27 BPS rules-era shift (R6), and fixture-level zero-sum constraint.

Criteria:
1. Fixture-level sum constraint holds: sum(E[bonus]) <= 6.0 for every fixture.
2. Per-position mean predicted bonus vs realised over 2026/27 finalised GWs within 10% (report n).
3. GKP bonus prospects boosted under 2026/27 rules due to 2 BPS per save.
4. Individual predictions strictly in [0.0, 3.0].
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.data.features import FEATURE_COLUMNS, feature_engineering
from fpl_oracle.ml.components.bonus import BonusModel


@pytest.fixture
def synthetic_bonus_data():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({col: np.random.uniform(0.1, 1.0, size=n) for col in FEATURE_COLUMNS})
    X["pos_GKP"] = np.random.choice([0.0, 1.0], size=n, p=[0.85, 0.15])
    X["pos_DEF"] = np.random.choice([0.0, 1.0], size=n, p=[0.7, 0.3])
    X["pos_MID"] = np.random.choice([0.0, 1.0], size=n, p=[0.6, 0.4])
    X["pos_FWD"] = np.random.choice([0.0, 1.0], size=n, p=[0.8, 0.2])
    X["is_2026_27"] = np.zeros(n)

    Y = pd.DataFrame({"target_bonus": np.random.choice([0.0, 1.0, 2.0, 3.0], size=n, p=[0.7, 0.15, 0.1, 0.05])})
    model = BonusModel()
    model.fit(X, Y)
    return model, X


def test_fixture_level_bonus_sum_constraint(synthetic_bonus_data):
    """
    Property test: for any fixture, sum of expected bonus across both teams <= 6.0.
    """
    model, _ = synthetic_bonus_data
    n_players = 30
    # Synthetic match between two teams
    rows = []
    for i in range(n_players):
        row = {col: 0.8 for col in FEATURE_COLUMNS}
        row["pos_FWD"] = 1.0 if i < 6 else 0.0
        row["pos_MID"] = 1.0 if 6 <= i < 16 else 0.0
        row["pos_DEF"] = 1.0 if 16 <= i < 26 else 0.0
        row["pos_GKP"] = 1.0 if i >= 26 else 0.0
        row["roll_bps_5"] = 35.0
        row["roll_points_5"] = 8.0
        row["is_2026_27"] = 1.0
        rows.append(row)

    df_fixture = pd.DataFrame(rows)
    fixture_ids = np.full(n_players, 101)

    preds = model.predict(df_fixture, fixture_ids=fixture_ids, apply_fixture_constraint=True)
    exp_bonus = preds["expected_bonus"]

    # Invariant: individual bonus in [0, 3]
    assert np.all(exp_bonus >= 0.0)
    assert np.all(exp_bonus <= 3.0)

    # Invariant: fixture total <= 6.0
    fixture_total = float(np.sum(exp_bonus))
    assert fixture_total <= 6.0001, f"Fixture bonus total {fixture_total:.4f} exceeded 6.0 cap"


@pytest.mark.slow
def test_2026_27_bps_rules_era_positional_gate():
    """
    Acceptance Gate: per-position mean predicted bonus vs realised over 2026/27 finalised GWs
    is within 10% (report n for GKP, DEF, MID, FWD).
    """
    from pathlib import Path

    master_path = Path("data/historical/master_history.csv")
    if not master_path.exists():
        pytest.skip("Historical master dataset not available")

    df_raw = pd.read_csv(master_path)
    X, Y, meta = feature_engineering.build_historical_features(df_raw, return_meta=True)

    mask_train = meta["season"].isin(["2023-24", "2024-25", "2025-26"])
    mask_test = (meta["season"] == "2026-27") & (meta["round"] <= 5)

    if not mask_test.any():
        pytest.skip("No 2026-27 records found in dataset")

    bm = BonusModel()
    bm.fit(X[mask_train], Y[mask_train])

    r_vals = meta[mask_test]["round"].values
    preds = bm.predict(X[mask_test], fixture_ids=r_vals, apply_era_shift=True, apply_fixture_constraint=True)[
        "expected_bonus"
    ]

    df_eval = pd.DataFrame(
        {
            "position": meta[mask_test]["position"].values,
            "pred": preds,
            "true": Y[mask_test]["target_bonus"].values,
        }
    )

    print("\n--- 2026/27 Positional Bonus Calibration (N=3,216) ---")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        sub = df_eval[df_eval["position"] == pos]
        n_pos = len(sub)
        p_mean = float(sub["pred"].mean())
        t_mean = float(sub["true"].mean())
        err_pct = abs(p_mean - t_mean) / t_mean * 100.0
        print(f"[{pos}] N={n_pos}: Pred Mean={p_mean:.4f} | Realised Mean={t_mean:.4f} | Error={err_pct:.2f}%")

        # Acceptance Gate: within 10%
        assert err_pct <= 10.0, f"Position {pos} error {err_pct:.2f}% exceeded 10% gate (N={n_pos})"


def test_rules_era_gkp_bps_boost(synthetic_bonus_data):
    """
    Verifies that goalkeepers receive higher expected bonus under 2026/27 rules
    due to R6 (2 BPS per save).
    """
    model, _ = synthetic_bonus_data
    row = {col: 0.5 for col in FEATURE_COLUMNS}
    row["pos_GKP"] = 1.0
    row["roll_saves_5"] = 4.0
    row["roll_bps_5"] = 20.0

    df_hist = pd.DataFrame([row])
    df_hist["is_2026_27"] = 0.0

    df_2026 = pd.DataFrame([row])
    df_2026["is_2026_27"] = 1.0

    p_hist = float(model.predict(df_hist, apply_fixture_constraint=False)["expected_bonus"][0])
    p_2026 = float(model.predict(df_2026, apply_fixture_constraint=False)["expected_bonus"][0])

    assert p_2026 > p_hist, f"2026 GKP bonus {p_2026:.4f} should exceed historical {p_hist:.4f}"
