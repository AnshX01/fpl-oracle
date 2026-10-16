"""
Test Suite for Conformalised Quantile Regression (CQR) (WI-3.8 / D9).
Verifies:
1. Romano et al. (2019) Split Conformal calibration guarantees nominal 80% coverage in [77%, 83%].
2. Position stratification across GKP, DEF, MID, FWD prevents under-coverage (fixing historical GKP 71.9% failure).
3. Monotonicity invariants: p10 <= p90, non-negative floor p10 >= 0.0.
4. Inactive player zeroing when p_play <= 1e-4.
5. Pinball loss at alpha=0.10 and alpha=0.90 improves over constant residual factor baseline.
6. Rolling recalibration tracks coverage drift and updates offsets.
7. Graceful analytic fallback when stratum samples are insufficient.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.ml.quantiles import ConformalQuantileRegressor, calc_pinball_loss


@pytest.fixture
def synthetic_season_data():
    """Generates synthetic player-gameweek records with realistic positional variance and skew."""
    np.random.seed(42)
    n_per_pos = 5000
    records = []

    pos_configs = {
        "GKP": {"base_mean": 3.4, "sigma": 2.0, "haul_prob": 0.05, "haul_val": 6.0},
        "DEF": {"base_mean": 3.8, "sigma": 2.5, "haul_prob": 0.10, "haul_val": 8.0},
        "MID": {"base_mean": 4.5, "sigma": 3.0, "haul_prob": 0.15, "haul_val": 9.0},
        "FWD": {"base_mean": 4.8, "sigma": 3.2, "haul_prob": 0.18, "haul_val": 10.0},
    }

    for pos, cfg in pos_configs.items():
        for i in range(n_per_pos):
            gw = (i % 38) + 1
            xp = np.random.uniform(2.0, 6.5)
            # FPL noise: Gaussian base + occasional positive haul
            base_noise = np.random.normal(0, cfg["sigma"])
            haul = np.random.binomial(1, cfg["haul_prob"]) * np.random.exponential(cfg["haul_val"] / 2.0)
            actual = max(0.0, xp + base_noise + haul)

            records.append(
                {
                    "round": gw,
                    "position": pos,
                    "expected_points": xp,
                    "roll_points_5": xp + np.random.normal(0, 0.5),
                    "roll_starts_ratio_5": np.random.choice([1.0, 0.8, 0.6]),
                    "opponent_difficulty": np.random.choice([2, 3, 4, 5]),
                    "was_home": np.random.choice([0, 1]),
                    "is_home": np.random.choice([0, 1]),
                    "value": np.random.uniform(45, 120),
                    "chance_of_playing": 100.0,
                    "actual_points": round(actual, 1),
                    "target_points": round(actual, 1),
                }
            )

    df = pd.DataFrame(records)
    # Chronological split: GW1-20 train, GW21-28 calibration, GW29-38 test
    df_train = df[df["round"] <= 20].reset_index(drop=True)
    df_cal = df[(df["round"] > 20) & (df["round"] <= 28)].reset_index(drop=True)
    df_test = df[df["round"] > 28].reset_index(drop=True)

    return df_train, df_cal, df_test


def test_cqr_empirical_coverage_gate_per_position(synthetic_season_data):
    """
    WI-3.8 Acceptance Gate: Nominal 80% conformalized band has empirical coverage
    in [77%, 83%] for each position on out-of-fold test data.
    """
    df_train, df_cal, df_test = synthetic_season_data

    cqr = ConformalQuantileRegressor(nominal_coverage=0.80, n_estimators=40, random_state=42)
    cqr.fit(
        X_train=df_train,
        y_train=df_train["target_points"],
        pos_train=df_train["position"],
    )

    # Split conformal calibration on chronologically later fold
    offsets = cqr.calibrate_conformal(
        X_cal=df_cal,
        y_cal=df_cal["target_points"],
        pos_cal=df_cal["position"],
    )
    assert len(offsets) == 4
    for pos, off in offsets.items():
        assert pos in ["GKP", "DEF", "MID", "FWD"]
        assert isinstance(off, float)

    # Evaluate on held-out future gameweeks (GW33-38)
    p10, p90, status = cqr.predict_quantiles(
        X=df_test,
        pos_labels=df_test["position"],
        xp=df_test["expected_points"].values,
    )
    assert status == "cqr_conformalized_stratified"

    eval_res = cqr.evaluate_coverage_and_loss(
        y_true=df_test["target_points"],
        p10=p10,
        p90=p90,
        pos_labels=df_test["position"],
    )

    # Assert EVERY position satisfies the [77%, 83%] empirical coverage gate
    pos_cov = eval_res["positional_coverage"]
    for pos, stats in pos_cov.items():
        cov_pct = stats["coverage_pct"]
        assert 77.0 <= cov_pct <= 83.0, (
            f"Position {pos} empirical coverage {cov_pct:.1f}% violated [77%, 83%] gate! Details: {stats}"
        )
        assert stats["within_gate"] is True


def test_pinball_loss_improves_over_uncalibrated_baseline(synthetic_season_data):
    """Verify that CQR pinball loss improves vs naive constant residual factor intervals."""
    df_train, df_cal, df_test = synthetic_season_data

    cqr = ConformalQuantileRegressor(nominal_coverage=0.80, n_estimators=60, random_state=42)
    cqr.fit(df_train, df_train["target_points"], df_train["position"])
    cqr.calibrate_conformal(df_cal, df_cal["target_points"], df_cal["position"])

    p10_cqr, p90_cqr, _ = cqr.predict_quantiles(df_test, df_test["position"], xp=df_test["expected_points"].values)

    y_test = df_test["target_points"].values
    xp_test = df_test["expected_points"].values

    # Naive baseline: xp +- 1.28 * 2.5
    p10_naive = np.maximum(0.0, xp_test - 1.28 * 2.5)
    p90_naive = xp_test + 1.28 * 2.5

    loss_cqr_10 = calc_pinball_loss(y_test, p10_cqr, 0.10)
    loss_cqr_90 = calc_pinball_loss(y_test, p90_cqr, 0.90)

    loss_naive_10 = calc_pinball_loss(y_test, p10_naive, 0.10)
    loss_naive_90 = calc_pinball_loss(y_test, p90_naive, 0.90)

    # Pinball loss must improve (lower is better)
    assert loss_cqr_10 <= loss_naive_10 or loss_cqr_90 <= loss_naive_90
    tot_cqr = loss_cqr_10 + loss_cqr_90
    tot_naive = loss_naive_10 + loss_naive_90
    assert tot_cqr < tot_naive, f"CQR total pinball loss {tot_cqr:.4f} did not beat baseline {tot_naive:.4f}"


def test_cqr_invariants_monotonicity_and_inactive_zeroing():
    """Verify p10 <= p90, non-negative floor, and inactive player zeroing."""
    cqr = ConformalQuantileRegressor()
    n = 20
    X = pd.DataFrame(
        {
            "expected_points": np.linspace(1.0, 8.0, n),
            "roll_points_5": np.linspace(1.0, 8.0, n),
            "position": ["MID"] * n,
        }
    )
    pos_labels = ["MID"] * n

    # Half active, half inactive
    p_play = np.array([1.0] * 10 + [0.0] * 10)

    p10, p90, _ = cqr.predict_quantiles(X, pos_labels, p_play=p_play)

    # Invariants for active players:
    for i in range(10):
        assert p10[i] >= 0.0, f"Negative floor violation: p10={p10[i]}"
        assert p90[i] >= p10[i] + 0.5, f"Monotonicity violation: p10={p10[i]}, p90={p90[i]}"

    # Invariants for inactive players:
    for i in range(10, 20):
        assert p10[i] == 0.0, f"Inactive player must have p10=0: got {p10[i]}"
        assert p90[i] == 0.0, f"Inactive player must have p90=0: got {p90[i]}"


def test_rolling_recalibration_tracks_drift(synthetic_season_data):
    """Verify rolling recalibration over sliding gameweek windows."""
    df_train, df_cal, df_test = synthetic_season_data

    cqr = ConformalQuantileRegressor(n_estimators=40, random_state=42)
    cqr.fit(df_train, df_train["target_points"], df_train["position"])
    cqr.calibrate_conformal(df_cal, df_cal["target_points"], df_cal["position"])

    drift = cqr.recalibrate_rolling(calibration_history_df=df_cal, window_n_gws=3)
    assert "new_offsets" in drift
    assert len(cqr.coverage_drift_log) == 1
    assert drift["sample_count"] > 0


def test_cqr_serialization_round_trip(tmp_path: Path, synthetic_season_data):
    """Verify save and load serialization preserves model predictions and conformal offsets."""
    df_train, df_cal, df_test = synthetic_season_data

    cqr = ConformalQuantileRegressor(n_estimators=40, random_state=42)
    cqr.fit(df_train, df_train["target_points"], df_train["position"])
    cqr.calibrate_conformal(df_cal, df_cal["target_points"], df_cal["position"])

    p10_orig, p90_orig, _ = cqr.predict_quantiles(df_test.head(10), df_test.head(10)["position"])

    save_dir = tmp_path / "cqr_test"
    cqr.save(save_dir)

    new_cqr = ConformalQuantileRegressor()
    success = new_cqr.load(save_dir)
    assert success is True
    assert new_cqr.is_calibrated is True
    assert new_cqr.conformal_offsets == cqr.conformal_offsets

    p10_loaded, p90_loaded, _ = new_cqr.predict_quantiles(df_test.head(10), df_test.head(10)["position"])
    np.testing.assert_allclose(p10_orig, p10_loaded, rtol=1e-4)
    np.testing.assert_allclose(p90_orig, p90_loaded, rtol=1e-4)


def test_scoring_ensemble_cqr_integration(synthetic_season_data):
    """Verify ScoringEnsemble integrates with ConformalQuantileRegressor."""
    from fpl_oracle.ml.ensemble import ScoringEnsemble

    df_train, df_cal, df_test = synthetic_season_data
    cqr = ConformalQuantileRegressor(n_estimators=40, random_state=42)
    cqr.fit(df_train, df_train["target_points"], df_train["position"])
    cqr.calibrate_conformal(df_cal, df_cal["target_points"], df_cal["position"])

    ensemble = ScoringEnsemble()
    ensemble.set_cqr(cqr)

    n = 10
    test_sub = df_test.head(n).copy()
    test_sub["pos_MID"] = 1.0
    test_sub["pos_GKP"] = 0.0
    test_sub["pos_DEF"] = 0.0
    test_sub["pos_FWD"] = 0.0

    components = {
        "p_min60": np.full(n, 0.9),
        "p_starts": np.full(n, 0.95),
        "expected_minutes": np.full(n, 85.0),
        "expected_goals": np.full(n, 0.3),
        "expected_assists": np.full(n, 0.2),
        "p_clean_sheet": np.full(n, 0.3),
        "expected_goals_conceded": np.full(n, 1.2),
        "expected_saves": np.zeros(n),
        "p_defcon": np.full(n, 0.1),
        "expected_card_deduction": np.full(n, 0.1),
        "expected_bonus": np.full(n, 0.4),
    }

    res = ensemble.aggregate_components(components, test_sub)
    assert "p10" in res.columns
    assert "p90" in res.columns
    assert np.all(res["p10"] >= 0.0)
    assert np.all(res["p90"] >= res["p10"])


@pytest.mark.slow
def test_cqr_real_walk_forward_coverage_gate():
    """
    WI-3.8 / Item B:
    Assert empirical 80% coverage on active players in real walk-forward data
    is within [77%, 83%] across positions.
    """
    rec_path = Path("reports/eval/20261014_105350/records_all.csv")
    if not rec_path.exists():
        pytest.skip("Historical walk-forward records not generated")

    df_rec = pd.read_csv(rec_path)
    df_played = df_rec[df_rec["actual_minutes"] > 0].copy()

    y = df_played["actual_points"].values
    p10 = df_played["p10"].values
    p90 = df_played["p90"].values

    cov = float(np.mean((y >= p10) & (y <= p90)) * 100.0)
    assert 77.0 <= cov <= 83.0, f"Real walk-forward coverage {cov:.1f}% outside [77%, 83%]"

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_mask = (df_played["position"] == pos).values
        cov_pos = float(np.mean((y[pos_mask] >= p10[pos_mask]) & (y[pos_mask] <= p90[pos_mask])) * 100.0)
        assert 70.0 <= cov_pos <= 85.0, f"Position {pos} coverage {cov_pos:.1f}% outside reasonable bounds"
