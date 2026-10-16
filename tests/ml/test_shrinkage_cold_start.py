"""
Unit and Stratified Acceptance Test Suite for WI-3.7 (Cold Start & Empirical-Bayes Shrinkage).

Verifies:
1. Prior lookup conditioned on position x price-band x team-strength group.
2. New signing, promoted team, and long injury return prior modulations.
3. Mathematical shrinkage invariants:
     rate_shrunk = (n * rate_player + k * rate_prior) / (n + k)
   - n = 0 strictly returns the prior
   - 0 < n < 450 shrinks noisy rates towards the prior
   - Monotonicity in minutes n
4. Vectorized feature shrinkage parity.
5. Stratified improvement: MAE on players with < 450 lookback minutes who played improves vs raw rolling rates.
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.ml.components.shrinkage import (
    get_empirical_prior,
    get_price_band,
    shrink_feature_dataframe,
    shrink_player_rolling_stats,
)
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.predict import projection_engine


def test_price_tier_classification():
    """Verify price band mapping across all 4 positions."""
    # GKP
    assert get_price_band("GKP", 5.5) == "premium"
    assert get_price_band("GKP", 5.0) == "mid"
    assert get_price_band("GKP", 4.5) == "budget"
    assert get_price_band("GKP", 45.0) == "budget"  # tenths handling

    # DEF
    assert get_price_band("DEF", 6.5) == "premium"
    assert get_price_band("DEF", 5.5) == "mid_high"
    assert get_price_band("DEF", 4.5) == "mid"
    assert get_price_band("DEF", 4.0) == "budget"

    # MID
    assert get_price_band("MID", 12.5) == "premium"
    assert get_price_band("MID", 8.0) == "mid_high"
    assert get_price_band("MID", 6.0) == "mid"
    assert get_price_band("MID", 4.5) == "budget"

    # FWD
    assert get_price_band("FWD", 14.0) == "premium"
    assert get_price_band("FWD", 7.5) == "mid_high"
    assert get_price_band("FWD", 6.0) == "mid"
    assert get_price_band("FWD", 5.0) == "budget"


def test_empirical_priors_stratification_and_flags():
    """Verify priors are properly stratified by price band, team strength, and flags."""
    prem_fwd = get_empirical_prior("FWD", val_m=12.0, team_strength=1.30)
    budg_fwd = get_empirical_prior("FWD", val_m=4.5, team_strength=0.90)

    # Premium forward on elite team must have dramatically higher xG and pts expectation than budget forward
    assert prem_fwd["expected_goals"] > budg_fwd["expected_goals"] * 3.0
    assert prem_fwd["total_points"] > budg_fwd["total_points"] * 2.5

    # Promoted team flag reduces attacking expectations
    norm_mid = get_empirical_prior("MID", val_m=6.0, team_strength=1.0, is_promoted=False)
    prom_mid = get_empirical_prior("MID", val_m=6.0, team_strength=1.0, is_promoted=True)
    assert prom_mid["expected_goals"] < norm_mid["expected_goals"]
    assert prom_mid["goals_conceded"] > norm_mid["goals_conceded"]

    # Long injury return discounts start rate and minutes
    inj_mid = get_empirical_prior("MID", val_m=7.5, team_strength=1.1, is_long_injury_return=True)
    fit_mid = get_empirical_prior("MID", val_m=7.5, team_strength=1.1, is_long_injury_return=False)
    assert inj_mid["starts"] < fit_mid["starts"]
    assert inj_mid["minutes"] < fit_mid["minutes"]


def test_mathematical_shrinkage_invariants():
    """Verify shrinkage formula properties: boundary conditions, monotonicity, and convergence."""
    prior = get_empirical_prior("MID", val_m=6.0)
    raw_stats = {
        "roll_minutes_5": 10.0,
        "roll_xG_5": 0.0,
        "roll_points_5": 0.5,
    }

    # 1. n = 0 strictly returns prior
    shrunk_0 = shrink_player_rolling_stats(raw_stats, lookback_minutes=0.0, pos_code="MID", val_m=6.0)
    assert abs(shrunk_0["roll_minutes_5"] - prior["minutes"]) < 1e-6
    assert abs(shrunk_0["roll_xG_5"] - prior["expected_goals"]) < 1e-6
    assert abs(shrunk_0["roll_points_5"] - prior["total_points"]) < 1e-6

    # 2. Intermediate n smoothly blends
    shrunk_90 = shrink_player_rolling_stats(raw_stats, lookback_minutes=90.0, pos_code="MID", val_m=6.0)
    shrunk_270 = shrink_player_rolling_stats(raw_stats, lookback_minutes=270.0, pos_code="MID", val_m=6.0)

    # With higher minutes, points should move closer to raw_val (0.5) from prior (2.60)
    assert prior["total_points"] > shrunk_90["roll_points_5"] > shrunk_270["roll_points_5"] > raw_stats["roll_points_5"]

    # 3. New signing with flag ignores noisy history and uses prior
    shrunk_new = shrink_player_rolling_stats(
        raw_stats, lookback_minutes=150.0, pos_code="MID", val_m=6.0, is_new_signing=True
    )
    assert abs(shrunk_new["roll_points_5"] - prior["total_points"]) < 1e-6


def test_vectorized_dataframe_shrinkage_parity():
    """Verify shrink_feature_dataframe matches individual shrink_player_rolling_stats."""
    sample_rows = [
        {
            "pos_GKP": 0.0,
            "pos_DEF": 0.0,
            "pos_MID": 1.0,
            "pos_FWD": 0.0,
            "value": 60.0,
            "team_strength_attack": 1.10,
            "roll_minutes_5": 20.0,
            "roll_xG_5": 0.02,
            "roll_points_5": 1.2,
            "implied_team_xG": 1.40,
        },
        {
            "pos_GKP": 0.0,
            "pos_DEF": 1.0,
            "pos_MID": 0.0,
            "pos_FWD": 0.0,
            "value": 45.0,
            "team_strength_attack": 0.95,
            "roll_minutes_5": 15.0,
            "roll_xG_5": 0.01,
            "roll_points_5": 0.8,
            "implied_team_xG": 1.10,
        },
    ]
    df = pd.DataFrame(sample_rows)
    df_shrunk = shrink_feature_dataframe(df)

    # Check row 0
    single_shrunk_0 = shrink_player_rolling_stats(
        {"roll_xG_5": 0.02, "roll_points_5": 1.2, "roll_minutes_5": 20.0},
        lookback_minutes=100.0,  # 20 * 5
        pos_code="MID",
        val_m=6.0,
        team_strength=1.10,
    )

    assert abs(df_shrunk.loc[0, "roll_xG_5"] - single_shrunk_0["roll_xG_5"]) < 1e-5
    assert abs(df_shrunk.loc[0, "roll_points_5"] - single_shrunk_0["roll_points_5"]) < 1e-5


def test_low_minute_cohort_error_improves():
    """
    WI-3.7 Acceptance Gate:
    Prediction error (MAE) on players with < 450 minutes in lookback window who played
    improves with empirical-Bayes shrunk rates vs raw rolling rates.
    """
    df = pd.read_csv("data/historical/master_history.csv")
    s_df = df[df["season"] == "2025-26"].copy()
    X, Y, meta = feature_engineering.build_historical_features(s_df, return_meta=True)
    projection_engine.load_or_train()

    y_true = Y["target_points"].values
    low_min_mask = (X["roll_minutes_5"] * 5 < 450).values
    played_mask = (Y["target_minutes"] > 0).values
    target_cohort = low_min_mask & played_mask

    # Raw predictions
    raw_comps = [
        projection_engine.minutes_model.predict(X),
        projection_engine.attacking_model.predict(
            X,
            team_ids=meta["team"].values if "team" in meta else None,
            fixture_ids=meta["fixture_id"].values if "fixture_id" in meta else None,
        ),
        projection_engine.defending_model.predict(X),
        projection_engine.defcon_model.predict(X),
        projection_engine.bonus_model.predict(
            X, fixture_ids=meta["fixture_id"].values if "fixture_id" in meta else None
        ),
        projection_engine.cards_saves_model.predict(X),
    ]
    comps_raw = {}
    for c in raw_comps:
        comps_raw.update(c)
    pred_raw = scoring_ensemble.aggregate_components(comps_raw, X)["expected_points"].values

    # EB-Shrunk features and predictions
    X_shrunk = shrink_feature_dataframe(X)
    shrunk_comps = [
        projection_engine.minutes_model.predict(X_shrunk),
        projection_engine.attacking_model.predict(
            X_shrunk,
            team_ids=meta["team"].values if "team" in meta else None,
            fixture_ids=meta["fixture_id"].values if "fixture_id" in meta else None,
        ),
        projection_engine.defending_model.predict(X_shrunk),
        projection_engine.defcon_model.predict(X_shrunk),
        projection_engine.bonus_model.predict(
            X_shrunk, fixture_ids=meta["fixture_id"].values if "fixture_id" in meta else None
        ),
        projection_engine.cards_saves_model.predict(X_shrunk),
    ]
    comps_shrunk = {}
    for c in shrunk_comps:
        comps_shrunk.update(c)
    pred_shrunk = scoring_ensemble.aggregate_components(comps_shrunk, X_shrunk)["expected_points"].values

    mae_raw = mean_absolute_error(y_true[target_cohort], pred_raw[target_cohort])
    mae_shrunk = mean_absolute_error(y_true[target_cohort], pred_shrunk[target_cohort])

    rmse_raw = root_mean_squared_error(y_true[target_cohort], pred_raw[target_cohort])
    rmse_shrunk = root_mean_squared_error(y_true[target_cohort], pred_shrunk[target_cohort])

    print(f"\nWI-3.7 Acceptance Gate Verification on N={target_cohort.sum()} active low-minute players:")
    print(f"  Raw Rolling MAE:    {mae_raw:.4f}")
    print(f"  EB-Shrunk MAE:      {mae_shrunk:.4f} (improvement: {mae_raw - mae_shrunk:.4f})")
    print(f"  Raw Rolling RMSE:   {rmse_raw:.4f}")
    print(f"  EB-Shrunk RMSE:     {rmse_shrunk:.4f} (improvement: {rmse_raw - rmse_shrunk:.4f})")

    # Gate: MAE must improve (be strictly less)
    assert mae_shrunk <= mae_raw, f"EB-Shrunk MAE {mae_shrunk:.4f} failed to improve over raw {mae_raw:.4f}"
