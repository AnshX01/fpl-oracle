"""
Leakage Audit Suite for FPL Oracle.
Formally tests that:
1. Feature values at (player, GW_k) use strictly data prior to GW_k kickoff.
2. Perturbing actual post-deadline events (minutes, goals, xG, xA, clean sheets, defcon) in GW_k
   leaves the feature vector for GW_k completely unchanged (diff == 0.0).
3. Perturbing GW_k events properly propagates to GW_(k+1) features (proving exact shift=1 behavior).
4. Career match 0 for any player has 0.0 rolling metrics (no contemporaneous leakage).
"""

import pandas as pd

from fpl_oracle.data.features import FEATURE_COLUMNS, FeatureEngineering


def make_synthetic_player_history(name: str = "Talisman Forward", n_matches: int = 10) -> pd.DataFrame:
    rows = []
    for gw in range(1, n_matches + 1):
        rows.append(
            {
                "name": name,
                "season": "2025-26",
                "round": gw,
                "position": "FWD",
                "minutes": 90,
                "starts": 1,
                "total_points": 6,
                "expected_goals": 0.45,
                "expected_assists": 0.20,
                "expected_goal_involvements": 0.65,
                "expected_goals_conceded": 1.10,
                "goals_scored": 1,
                "assists": 0,
                "clean_sheets": 0,
                "goals_conceded": 1,
                "saves": 0,
                "defensive_contribution": 1,
                "ict_index": 8.5,
                "bps": 22,
                "bonus": 1,
                "was_home": True,
                "value": 115.0,
            }
        )
    return pd.DataFrame(rows)


def test_strict_zero_leakage_perturbation():
    """
    Assert that modifying GW_k's actual match performance has ZERO impact
    on the feature vector computed for GW_k.
    """
    fe = FeatureEngineering()
    df_clean = make_synthetic_player_history("Test Striker", 10)

    X_clean, Y_clean = fe.build_historical_features(df_clean)

    # Let's inspect GW 5 (index 4)
    gw5_features_clean = X_clean.iloc[4].copy()

    # Create perturbed copy where GW 5 actuals are radically changed:
    # Hat-trick, 3 bonus, 4.0 xG, 90 mins, DefCon 15 actions
    df_perturbed = df_clean.copy()
    df_perturbed.loc[4, "total_points"] = 28
    df_perturbed.loc[4, "goals_scored"] = 4
    df_perturbed.loc[4, "expected_goals"] = 3.8
    df_perturbed.loc[4, "assists"] = 2
    df_perturbed.loc[4, "expected_assists"] = 1.5
    df_perturbed.loc[4, "defensive_contribution"] = 15
    df_perturbed.loc[4, "bps"] = 65
    df_perturbed.loc[4, "bonus"] = 3

    # Also perturb future GWs 6-10 to verify future cannot bleed back
    for f_idx in range(5, 10):
        df_perturbed.loc[f_idx, "total_points"] = 15
        df_perturbed.loc[f_idx, "goals_scored"] = 2
        df_perturbed.loc[f_idx, "expected_goals"] = 2.0

    X_perturbed, Y_perturbed = fe.build_historical_features(df_perturbed)
    gw5_features_perturbed = X_perturbed.iloc[4].copy()

    # The features for GW 5 must be 100% IDENTICAL
    diffs = {}
    for col in FEATURE_COLUMNS:
        v_clean = float(gw5_features_clean[col])
        v_pert = float(gw5_features_perturbed[col])
        if abs(v_clean - v_pert) > 1e-9:
            diffs[col] = (v_clean, v_pert)

    assert len(diffs) == 0, f"DATA LEAKAGE DETECTED in GW 5 features! Leaked columns: {diffs}"


def test_shift_propagation_to_subsequent_gw():
    """
    Verify that perturbing GW_k events DOES propagate to GW_(k+1),
    confirming exact shift(1) feature updating rather than disconnected constants.
    """
    fe = FeatureEngineering()
    df_clean = make_synthetic_player_history("Test Striker", 10)
    X_clean, _ = fe.build_historical_features(df_clean)

    df_perturbed = df_clean.copy()
    # In GW 5, player scores 5 goals (previously 1)
    df_perturbed.loc[4, "goals_scored"] = 5
    X_perturbed, _ = fe.build_historical_features(df_perturbed)

    # In GW 6 (index 5), roll_goals_5 must be strictly higher in the perturbed set
    roll_goals_clean = float(X_clean.iloc[5]["roll_goals_5"])
    roll_goals_perturbed = float(X_perturbed.iloc[5]["roll_goals_5"])

    assert roll_goals_perturbed > roll_goals_clean, (
        f"Expected GW 6 to reflect GW 5 changes, but got {roll_goals_clean} vs {roll_goals_perturbed}"
    )


def test_first_career_match_has_no_lookahead():
    """
    Assert that the very first match of a player's career has 0.0 rolling attacking/defensive form,
    proving no lookahead contamination from future matches.
    """
    fe = FeatureEngineering()
    df = make_synthetic_player_history("Rookie Talent", 5)
    X, _ = fe.build_historical_features(df)

    match_0 = X.iloc[0]
    assert match_0["roll_minutes_3"] == 0.0
    assert match_0["roll_points_3"] == 0.0
    assert match_0["roll_xG_3"] == 0.0
    assert match_0["roll_xA_3"] == 0.0
    assert match_0["roll_goals_5"] == 0.0
    assert match_0["roll_assists_5"] == 0.0
    assert match_0["roll_defcon_5"] == 0.0
