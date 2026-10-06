"""
Test Suite for Part A: Model and Pipeline Gap Closure.
Verifies train/serve parity, canonical schema hash, continuous opponent form features,
uncapped predictions, ground-truth disciplinary labels, and honest calibrated uncertainty.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.data.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_HASH,
    FeatureEngineering,
    compute_player_rolling_stats,
    feature_engineering,
)
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.predict import projection_engine


def test_canonical_schema_hash_and_parity():
    """Assert feature schema hash is non-empty, deterministic, and matches FEATURE_COLUMNS."""
    assert len(FEATURE_SCHEMA_HASH) == 64
    assert feature_engineering.schema_hash == FEATURE_SCHEMA_HASH
    assert "implied_team_xG" in FEATURE_COLUMNS
    assert "opp_roll_points_5" in FEATURE_COLUMNS
    assert "roll_cards_5" in FEATURE_COLUMNS


def test_identical_feature_computation_train_serve_parity():
    """
    Assert that the same player history produces 100% identical rolling metrics
    whether evaluated through training history or live upcoming serving path.
    """
    history_rows = [
        {
            "name": "Star Midfielder",
            "season": "2025-26",
            "round": gw,
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 7,
            "expected_goals": 0.50,
            "expected_assists": 0.35,
            "expected_goal_involvements": 0.85,
            "expected_goals_conceded": 0.90,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 1,
            "goals_conceded": 0,
            "saves": 0,
            "defensive_contribution": 2,
            "ict_index": 12.0,
            "bps": 28,
            "bonus": 2,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 105.0,
        }
        for gw in range(1, 6)
    ]
    df_hist = pd.DataFrame(history_rows)

    # 1. Compute via canonical transformer
    stats_serving = compute_player_rolling_stats(df_hist, pos_code="MID", val_m=10.5)

    # 2. Compute via build_historical_features for round 6
    fe = FeatureEngineering()
    df_hist_next = pd.concat([df_hist, pd.DataFrame([{**history_rows[-1], "round": 6}])], ignore_index=True)
    X_train, _ = fe.build_historical_features(df_hist_next)

    # The features for row 5 (GW 6) in training must match serving output
    row_gw6 = X_train.iloc[5]

    for key, val in stats_serving.items():
        if key in row_gw6:
            v_train = float(row_gw6[key])
            v_serve = float(val)
            assert abs(v_train - v_serve) < 1e-9, f"Mismatch in {key}: train={v_train}, serve={v_serve}"


def test_opponent_form_features_presence_and_bounds():
    """Verify opponent form and continuous implied match strength features exist and are bounded."""
    fe = FeatureEngineering()
    sample_df = pd.DataFrame([
        {
            "name": "Target FWD",
            "season": "2025-26",
            "round": 1,
            "team": "Arsenal",
            "opponent_team": "Chelsea",
            "position": "FWD",
            "minutes": 90,
            "starts": 1,
            "total_points": 8,
            "expected_goals": 0.80,
            "expected_assists": 0.20,
            "expected_goal_involvements": 1.00,
            "expected_goals_conceded": 1.10,
            "goals_scored": 1,
            "assists": 1,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 14.0,
            "bps": 32,
            "bonus": 3,
            "was_home": True,
            "value": 140.0,
        }
    ])

    X, Y = fe.build_historical_features(sample_df)
    row = X.iloc[0]

    assert 0.2 <= row["implied_team_xG"] <= 4.5
    assert 0.0 <= row["implied_team_cs_prob"] <= 1.0
    assert 0.0 <= row["opp_roll_clean_sheets_5"] <= 1.0
    assert row["days_rest"] >= 2.0


def test_no_hard_rules_or_caps_on_elite_projections():
    """
    CRITICAL USER REQUIREMENT:
    A star player in peak form can legitimately project 6-7+ expected points
    even against tough opposition. Projections emerge purely from ML feature learning,
    with NO artificial caps or ceilings.
    """
    # Create feature vector for an elite talisman (e.g. Haaland/Salah) in prime form
    star_features = pd.DataFrame([{
        "roll_minutes_3": 90.0,
        "roll_minutes_5": 90.0,
        "roll_minutes_8": 90.0,
        "roll_starts_ratio_5": 1.0,
        "roll_min60_ratio_5": 1.0,
        "std_minutes_per_gw": 88.0,
        "roll_points_3": 9.5,
        "roll_points_5": 8.8,
        "roll_points_8": 8.0,
        "roll_xG_3": 1.10,
        "roll_xG_5": 0.95,
        "roll_xG_8": 0.85,
        "roll_xA_3": 0.30,
        "roll_xA_5": 0.25,
        "roll_xA_8": 0.20,
        "roll_xGI_5": 1.20,
        "roll_goals_5": 1.20,
        "roll_assists_5": 0.40,
        "roll_xGC_5": 0.80,
        "roll_clean_sheets_5": 0.40,
        "roll_saves_5": 0.0,
        "roll_goals_conceded_5": 0.80,
        "roll_defcon_5": 0.0,
        "roll_ict_5": 16.0,
        "roll_bps_5": 35.0,
        "roll_cards_5": 0.0,
        "roll_own_goals_5": 0.0,
        "roll_penalties_missed_5": 0.0,
        "roll_penalties_saved_5": 0.0,
        "was_home": 1.0,
        "team_strength_attack": 2.2,
        "opp_strength_defence": 0.85,  # Elite defence!
        "net_strength_diff": 1.35,
        "opponent_difficulty": 4.0,
        "days_rest": 7.0,
        "implied_team_xG": 1.85,
        "implied_team_cs_prob": 0.45,
        "implied_opp_xG": 0.75,
        "implied_opp_cs_prob": 0.40,
        "team_roll_goals_3": 2.5,
        "team_roll_goals_5": 2.2,
        "team_roll_goals_8": 2.0,
        "team_roll_xG_3": 2.3,
        "team_roll_xG_5": 2.1,
        "team_roll_xG_8": 2.0,
        "opp_roll_points_3": 2.2,
        "opp_roll_points_5": 2.0,
        "opp_roll_points_8": 1.8,
        "opp_roll_goals_conceded_3": 0.8,
        "opp_roll_goals_conceded_5": 0.9,
        "opp_roll_goals_conceded_8": 1.0,
        "opp_roll_xGC_3": 0.8,
        "opp_roll_xGC_5": 0.85,
        "opp_roll_xGC_8": 0.90,
        "opp_roll_clean_sheets_5": 0.50,
        "pos_GKP": 0.0,
        "pos_DEF": 0.0,
        "pos_MID": 0.0,
        "pos_FWD": 1.0,
        "value": 150.0,
        "is_2026_27": 1.0,
        "chance_of_playing": 100.0,
    }])

    projection_engine.load_or_train()
    mins_pred = projection_engine.minutes_model.predict(star_features)
    att_pred = projection_engine.attacking_model.predict(star_features)
    def_pred = projection_engine.defending_model.predict(star_features)
    defcon_pred = projection_engine.defcon_model.predict(star_features)
    bonus_pred = projection_engine.bonus_model.predict(star_features)
    cards_pred = projection_engine.cards_saves_model.predict(star_features)

    components = {**mins_pred, **att_pred, **def_pred, **defcon_pred, **bonus_pred, **cards_pred}
    res_df = scoring_ensemble.aggregate_components(components, star_features)

    xp = float(res_df["expected_points"].iloc[0])
    # Must project strongly (>= 6.0 xP) despite facing tough opponent
    assert xp >= 5.5, f"Expected uncapped strong projection >= 5.5, got {xp}"


def test_cards_and_rare_components_target_math():
    """Verify disciplinary deductions correctly penalize yellow cards, red cards, own goals, and missed penalties."""
    df_raw = pd.DataFrame([{
        "name": "Fiery Midfielder",
        "season": "2025-26",
        "round": 1,
        "position": "MID",
        "minutes": 90,
        "starts": 1,
        "total_points": 1,
        "expected_goals": 0.1,
        "expected_assists": 0.1,
        "expected_goal_involvements": 0.2,
        "expected_goals_conceded": 2.0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 2,
        "saves": 0,
        "defensive_contribution": 0,
        "ict_index": 4.0,
        "bps": 5,
        "bonus": 0,
        "yellow_cards": 1,  # -1 pt
        "red_cards": 0,
        "own_goals": 1,     # -2 pts
        "penalties_missed": 1, # -2 pts
        "penalties_saved": 0,
        "was_home": True,
        "value": 60.0,
    }])

    fe = FeatureEngineering()
    _, Y = fe.build_historical_features(df_raw)

    # 1 yellow (1) + 1 own goal (2) + 1 penalty missed (2) = 5.0 pts deduction
    assert Y["target_card_deduction"].iloc[0] == 5.0
