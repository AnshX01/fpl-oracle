"""
Test Suite for Strict Value-Based Feature Parity (F12).
Asserts that all 62 feature values produced by the live serving kernel
(extract_live_features_for_upcoming) numerically equal the values produced
by the training kernel (build_historical_features) on identical historical snapshots.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, Team
from fpl_oracle.data.features import FEATURE_COLUMNS, FeatureEngineering


@pytest.fixture
def synthetic_history_and_upcoming():
    """Build multi-player, multi-round historical match data and upcoming GW6 fixtures."""
    fe = FeatureEngineering()

    players_meta = [
        {"id": 1, "name": "Star Midfielder", "team": "Arsenal", "team_id": 1, "opp": "Chelsea", "opp_id": 2, "pos": "MID", "pos_id": 3, "cost": 105.0, "home": True},
        {"id": 2, "name": "Solid Defender", "team": "Chelsea", "team_id": 2, "opp": "Arsenal", "opp_id": 1, "pos": "DEF", "pos_id": 2, "cost": 55.0, "home": False},
        {"id": 3, "name": "Elite Forward", "team": "Arsenal", "team_id": 1, "opp": "Chelsea", "opp_id": 2, "pos": "FWD", "pos_id": 4, "cost": 140.0, "home": True},
        {"id": 4, "name": "Top Goalkeeper", "team": "Chelsea", "team_id": 2, "opp": "Arsenal", "opp_id": 1, "pos": "GKP", "pos_id": 1, "cost": 50.0, "home": False},
    ]

    history_rows = []
    for r in range(1, 6):
        ko = f"2026-08-{10 + r:02d}T15:00:00Z"
        for p in players_meta:
            is_mid = p["pos"] == "MID"
            is_fwd = p["pos"] == "FWD"
            is_gkp = p["pos"] == "GKP"

            history_rows.append({
                "season": "2026-27",
                "round": r,
                "name": p["name"],
                "element": p["id"],
                "team": p["team"],
                "opponent_team": p["opp"],
                "position": p["pos"],
                "minutes": 90,
                "starts": 1,
                "total_points": 8 if is_fwd else (6 if is_mid else 3),
                "expected_goals": 0.6 if is_fwd else (0.3 if is_mid else 0.0),
                "expected_assists": 0.2 if is_fwd else (0.4 if is_mid else 0.0),
                "expected_goal_involvements": 0.8 if is_fwd else (0.7 if is_mid else 0.0),
                "expected_goals_conceded": 0.9 if p["team"] == "Arsenal" else 1.4,
                "goals_scored": 1 if is_fwd else (0 if not is_mid else 1),
                "assists": 0 if is_fwd else (1 if is_mid else 0),
                "clean_sheets": 1 if p["team"] == "Arsenal" else 0,
                "goals_conceded": 0 if p["team"] == "Arsenal" else 1,
                "saves": 3 if is_gkp else 0,
                "defensive_contribution": 2 if p["pos"] in ("DEF", "GKP") else 0,
                "ict_index": 12.0 if is_fwd else 8.0,
                "bps": 28 if is_fwd else 18,
                "bonus": 2 if is_fwd else 0,
                "yellow_cards": 1 if r == 3 else 0,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_missed": 0,
                "penalties_saved": 1 if is_gkp and r == 2 else 0,
                "was_home": bool(r % 2 == 1),
                "value": p["cost"],
                "kickoff_time": ko,
            })

    # GW6 matches for training (to simulate pre-deadline round 6 evaluation)
    gw6_rows = []
    ko_gw6 = "2026-09-12T15:00:00Z"
    for p in players_meta:
        gw6_rows.append({
            "season": "2026-27",
            "round": 6,
            "name": p["name"],
            "element": p["id"],
            "team": p["team"],
            "opponent_team": p["opp"],
            "position": p["pos"],
            "minutes": 90,
            "starts": 1,
            "total_points": 5,
            "expected_goals": 0.3,
            "expected_assists": 0.2,
            "expected_goal_involvements": 0.5,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 2,
            "defensive_contribution": 1,
            "ict_index": 5.0,
            "bps": 15,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": p["home"],
            "value": p["cost"],
            "kickoff_time": ko_gw6,
        })

    hist_df = pd.DataFrame(history_rows)
    train_full_df = pd.concat([hist_df, pd.DataFrame(gw6_rows)], ignore_index=True)

    # Bootstrap representation for serving
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
    ]
    elements = [
        Element(
            id=p["id"],
            web_name=p["name"],
            team=p["team_id"],
            element_type=p["pos_id"],
            now_cost=int(p["cost"]),
            chance_of_playing_next_round=100,
        )
        for p in players_meta
    ]
    el_types = [
        ElementType(id=1, plural_name="Goalkeepers", plural_name_short="GKP", singular_name="Goalkeeper", singular_name_short="GKP"),
        ElementType(id=2, plural_name="Defenders", plural_name_short="DEF", singular_name="Defender", singular_name_short="DEF"),
        ElementType(id=3, plural_name="Midfielders", plural_name_short="MID", singular_name="Midfielder", singular_name_short="MID"),
        ElementType(id=4, plural_name="Forwards", plural_name_short="FWD", singular_name="Forward", singular_name_short="FWD"),
    ]
    boot = BootstrapStatic(teams=teams, elements=elements, element_types=el_types)

    fixtures = [
        Fixture(
            id=101,
            code=101,
            event=6,
            team_h=1,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=3,
            kickoff_time=ko_gw6,
        )
    ]

    return fe, hist_df, train_full_df, boot, fixtures, players_meta


def test_feature_parity_all_62_features_exact_values(synthetic_history_and_upcoming):
    """
    Assert that all 62 feature values produced by the live serving kernel
    match the values produced by the training kernel with absolute error < 1e-4.
    """
    fe, hist_df, train_full_df, boot, fixtures, players_meta = synthetic_history_and_upcoming

    # 1. Training kernel output
    X_train, Y_train = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]

    # 2. Live serving kernel output
    df_serve = fe.extract_live_features_for_upcoming(
        bootstrap=boot,
        fixtures=fixtures,
        target_gw=6,
        history_df=hist_df,
    )

    assert len(FEATURE_COLUMNS) == 62, f"Expected 62 canonical features, found {len(FEATURE_COLUMNS)}"

    # Evaluate parity for each player
    for p in players_meta:
        elem_id = p["id"]

        # Extract GW6 row for this player from training kernel
        p_train_mask = (meta["element"] == elem_id) & (meta["round"] == 6)
        train_row = X_train[p_train_mask].iloc[0]

        # Extract GW6 row for this player from serving kernel
        p_serve_mask = df_serve["element"] == elem_id
        serve_row = df_serve[p_serve_mask].iloc[0]

        divergences = {}
        for col in FEATURE_COLUMNS:
            v_train = float(train_row[col])
            v_serve = float(serve_row[col])
            diff = abs(v_train - v_serve)
            if diff > 1e-4:
                divergences[col] = (v_train, v_serve, diff)

        assert len(divergences) == 0, (
            f"Feature parity failed for player {p['name']} (element {elem_id}) on {len(divergences)} features:\n"
            + "\n".join(f"  {col}: train={vt}, serve={vs}, delta={d}" for col, (vt, vs, d) in divergences.items())
        )


def test_feature_parity_schema_and_types(synthetic_history_and_upcoming):
    """Verify that all 62 columns exist in both dataframes and cast cleanly to float."""
    fe, hist_df, train_full_df, boot, fixtures, _ = synthetic_history_and_upcoming

    X_train, _ = fe.build_historical_features(train_full_df)
    df_serve = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=6, history_df=hist_df)

    for col in FEATURE_COLUMNS:
        assert col in X_train.columns, f"Missing {col} in training features"
        assert col in df_serve.columns, f"Missing {col} in live serving features"
        # Non-null values
        assert not X_train[col].isna().any(), f"NaN in training feature {col}"
        assert not df_serve[col].isna().any(), f"NaN in serving feature {col}"


def test_feature_parity_rest_days_computation(synthetic_history_and_upcoming):
    """Verify that days_rest feature accurately measures days from previous match in both paths."""
    fe, hist_df, train_full_df, boot, fixtures, _ = synthetic_history_and_upcoming

    X_train, _ = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]
    df_serve = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=6, history_df=hist_df)

    p1_train = X_train[(meta["element"] == 1) & (meta["round"] == 6)].iloc[0]
    p1_serve = df_serve[df_serve["element"] == 1].iloc[0]

    # Both must compute the identical rest interval
    assert abs(float(p1_train["days_rest"]) - float(p1_serve["days_rest"])) < 1e-4
    assert 2.0 <= float(p1_serve["days_rest"]) <= 14.0
