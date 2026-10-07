"""
Test Suite for Strict Value-Based Feature Parity (F12).
Asserts that all 62 feature values produced by the live serving kernel
(extract_live_features_for_upcoming) numerically equal the values produced
by the training kernel (build_historical_features) on identical historical snapshots.
"""

import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, Team
from fpl_oracle.config import HISTORICAL_DIR
from fpl_oracle.data.features import FEATURE_COLUMNS, FeatureEngineering


@pytest.fixture
def synthetic_history_and_upcoming():
    """Build multi-player, multi-round historical match data and upcoming GW6 fixtures."""
    fe = FeatureEngineering()

    players_meta = [
        {
            "id": 1,
            "name": "Star Midfielder",
            "team": "Arsenal",
            "team_id": 1,
            "opp": "Chelsea",
            "opp_id": 2,
            "pos": "MID",
            "pos_id": 3,
            "cost": 105.0,
            "home": True,
        },
        {
            "id": 2,
            "name": "Solid Defender",
            "team": "Chelsea",
            "team_id": 2,
            "opp": "Arsenal",
            "opp_id": 1,
            "pos": "DEF",
            "pos_id": 2,
            "cost": 55.0,
            "home": False,
        },
        {
            "id": 3,
            "name": "Elite Forward",
            "team": "Arsenal",
            "team_id": 1,
            "opp": "Chelsea",
            "opp_id": 2,
            "pos": "FWD",
            "pos_id": 4,
            "cost": 140.0,
            "home": True,
        },
        {
            "id": 4,
            "name": "Top Goalkeeper",
            "team": "Chelsea",
            "team_id": 2,
            "opp": "Arsenal",
            "opp_id": 1,
            "pos": "GKP",
            "pos_id": 1,
            "cost": 50.0,
            "home": False,
        },
    ]

    history_rows = []
    for r in range(1, 6):
        ko = f"2026-08-{10 + r:02d}T15:00:00Z"
        for p in players_meta:
            is_mid = p["pos"] == "MID"
            is_fwd = p["pos"] == "FWD"
            is_gkp = p["pos"] == "GKP"

            history_rows.append(
                {
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
                }
            )

    # GW6 matches for training (to simulate pre-deadline round 6 evaluation)
    gw6_rows = []
    ko_gw6 = "2026-09-12T15:00:00Z"
    for p in players_meta:
        gw6_rows.append(
            {
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
            }
        )

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
        ElementType(
            id=1,
            plural_name="Goalkeepers",
            plural_name_short="GKP",
            singular_name="Goalkeeper",
            singular_name_short="GKP",
        ),
        ElementType(
            id=2, plural_name="Defenders", plural_name_short="DEF", singular_name="Defender", singular_name_short="DEF"
        ),
        ElementType(
            id=3,
            plural_name="Midfielders",
            plural_name_short="MID",
            singular_name="Midfielder",
            singular_name_short="MID",
        ),
        ElementType(
            id=4, plural_name="Forwards", plural_name_short="FWD", singular_name="Forward", singular_name_short="FWD"
        ),
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


# ==============================================================================
# Milestone G13: Real Multi-Season Frozen Snapshot Parity Suite
# ==============================================================================

PARITY_DGW_ALLOW_LIST: dict[str, str] = {
    "roll_points_3": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_points_8": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_xA_3": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_xGC_5": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_saves_5": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_defcon_5": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "roll_ict_5": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
    "std_minutes_per_gw": "Intra-gameweek isolation: serving operates strictly pre-deadline where fixture 1 results are unobserved, whereas historical training log sequences matches chronologically.",
}


@pytest.fixture
def real_multiseason_history_and_upcoming():
    """
    Construct a real multi-season historical snapshot containing:
    1. Colliding player IDs across seasons:
       - 2024-25 element 1: Fabio Ferreira Vieira (MID)
       - 2025-26 & 2026-27 element 1: David Raya Martín (GKP)
       - 2023-24 element 1: Folarin Balogun (FWD)
    2. Transferred player across clubs:
       - Cole Palmer: element 362 (Man City) -> element 182/235/154 (Chelsea)
    3. New player appearing only in current season:
       - Christos Tzolis: element in 2026-27, 0 matches in prior seasons
    4. Multi-season match records for Arsenal, Chelsea, Liverpool.
    """
    fe = FeatureEngineering()
    m_df = pd.read_csv(HISTORICAL_DIR / "master_history.csv")

    hist_26 = m_df[m_df["season"] == "2026-27"].copy()
    prior = m_df[
        (m_df["season"].isin(["2024-25", "2025-26"]))
        & (m_df["name"].str.contains("Raya|Palmer|Vieira|Balogun", na=False))
    ].copy()

    history_df = pd.concat([prior, hist_26], ignore_index=True)

    tzolis_row = hist_26[hist_26["name"] == "Christos Tzolis"].iloc[-1]
    tzolis_id = int(tzolis_row["element"])

    # Upcoming GW6 matches for training kernel
    gw6_matches = [
        {
            "season": "2026-27",
            "round": 6,
            "name": "David Raya Martín",
            "element": 1,
            "team": "Arsenal",
            "opponent_team": "Chelsea",
            "position": "GKP",
            "minutes": 90,
            "starts": 1,
            "total_points": 6,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "expected_goal_involvements": 0.0,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 1,
            "goals_conceded": 0,
            "saves": 4,
            "defensive_contribution": 2,
            "ict_index": 4.0,
            "bps": 24,
            "bonus": 1,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 55.0,
            "kickoff_time": "2026-09-26T14:00:00Z",
        },
        {
            "season": "2026-27",
            "round": 6,
            "name": "Cole Palmer",
            "element": 154,
            "team": "Chelsea",
            "opponent_team": "Arsenal",
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 2,
            "expected_goals": 0.2,
            "expected_assists": 0.1,
            "expected_goal_involvements": 0.3,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 5.0,
            "bps": 12,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": False,
            "value": 105.0,
            "kickoff_time": "2026-09-26T14:00:00Z",
        },
        {
            "season": "2026-27",
            "round": 6,
            "name": "Christos Tzolis",
            "element": tzolis_id,
            "team": "Arsenal",
            "opponent_team": "Chelsea",
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 5,
            "expected_goals": 0.3,
            "expected_assists": 0.1,
            "expected_goal_involvements": 0.4,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 6.0,
            "bps": 15,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 50.0,
            "kickoff_time": "2026-09-26T14:00:00Z",
        },
    ]

    train_full_df = pd.concat([history_df, pd.DataFrame(gw6_matches)], ignore_index=True)

    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
        Team(id=3, name="Liverpool", short_name="LIV"),
    ]
    elements = [
        Element(
            id=1,
            web_name="Raya",
            first_name="David",
            second_name="Raya Martín",
            team=1,
            element_type=1,
            now_cost=55,
            chance_of_playing_next_round=100,
        ),
        Element(
            id=154,
            web_name="Palmer",
            first_name="Cole",
            second_name="Palmer",
            team=2,
            element_type=3,
            now_cost=105,
            chance_of_playing_next_round=100,
        ),
        Element(
            id=tzolis_id,
            web_name="Tzolis",
            first_name="Christos",
            second_name="Tzolis",
            team=1,
            element_type=3,
            now_cost=50,
            chance_of_playing_next_round=100,
        ),
    ]
    el_types = [
        ElementType(
            id=1,
            plural_name="Goalkeepers",
            plural_name_short="GKP",
            singular_name="Goalkeeper",
            singular_name_short="GKP",
        ),
        ElementType(
            id=2, plural_name="Defenders", plural_name_short="DEF", singular_name="Defender", singular_name_short="DEF"
        ),
        ElementType(
            id=3,
            plural_name="Midfielders",
            plural_name_short="MID",
            singular_name="Midfielder",
            singular_name_short="MID",
        ),
        ElementType(
            id=4, plural_name="Forwards", plural_name_short="FWD", singular_name="Forward", singular_name_short="FWD"
        ),
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
            kickoff_time="2026-09-26T14:00:00Z",
        ),
    ]

    return fe, history_df, train_full_df, boot, fixtures, tzolis_id


def test_real_multi_season_parity_colliding_ids(real_multiseason_history_and_upcoming):
    """
    Assert 100% numerical parity across all 62 features for David Raya (element 1 in 2026-27),
    verifying complete isolation from historical element 1 collision (Fabio Vieira in 2024-25).
    """
    fe, history_df, train_full_df, boot, fixtures, _ = real_multiseason_history_and_upcoming

    X_train, _ = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]
    df_serve = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=6, history_df=history_df)

    t_row = X_train[(meta["element"] == 1) & (meta["round"] == 6) & (meta["season"] == "2026-27")].iloc[0]
    s_row = df_serve[df_serve["element"] == 1].iloc[0]

    divergences = {}
    for col in FEATURE_COLUMNS:
        v_train = float(t_row[col])
        v_serve = float(s_row[col])
        diff = abs(v_train - v_serve)
        if diff > 1e-4:
            divergences[col] = (v_train, v_serve, diff)

    assert len(divergences) == 0, f"Colliding ID parity failed on {len(divergences)} features:\n" + "\n".join(
        f"  {k}: train={v[0]}, serve={v[1]}, diff={v[2]}" for k, v in divergences.items()
    )


def test_real_multi_season_parity_transferred_player(real_multiseason_history_and_upcoming):
    """
    Assert 100% numerical parity across all 62 features for Cole Palmer (transferred Man City -> Chelsea),
    verifying that personal form follows the player while club stats reflect Chelsea.
    """
    fe, history_df, train_full_df, boot, fixtures, _ = real_multiseason_history_and_upcoming

    X_train, _ = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]
    df_serve = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=6, history_df=history_df)

    t_row = X_train[(meta["element"] == 154) & (meta["round"] == 6) & (meta["season"] == "2026-27")].iloc[0]
    s_row = df_serve[df_serve["element"] == 154].iloc[0]

    divergences = {}
    for col in FEATURE_COLUMNS:
        v_train = float(t_row[col])
        v_serve = float(s_row[col])
        diff = abs(v_train - v_serve)
        if diff > 1e-4:
            divergences[col] = (v_train, v_serve, diff)

    assert len(divergences) == 0, f"Transferred player parity failed on {len(divergences)} features:\n" + "\n".join(
        f"  {k}: train={v[0]}, serve={v[1]}, diff={v[2]}" for k, v in divergences.items()
    )


def test_real_multi_season_parity_new_player(real_multiseason_history_and_upcoming):
    """
    Assert 100% numerical parity across all 62 features for Christos Tzolis (new player in 2026-27).
    """
    fe, history_df, train_full_df, boot, fixtures, tzolis_id = real_multiseason_history_and_upcoming

    X_train, _ = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]
    df_serve = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=6, history_df=history_df)

    t_row = X_train[(meta["element"] == tzolis_id) & (meta["round"] == 6) & (meta["season"] == "2026-27")].iloc[0]
    s_row = df_serve[df_serve["element"] == tzolis_id].iloc[0]

    divergences = {}
    for col in FEATURE_COLUMNS:
        v_train = float(t_row[col])
        v_serve = float(s_row[col])
        diff = abs(v_train - v_serve)
        if diff > 1e-4:
            divergences[col] = (v_train, v_serve, diff)

    assert len(divergences) == 0, f"New player parity failed on {len(divergences)} features:\n" + "\n".join(
        f"  {k}: train={v[0]}, serve={v[1]}, diff={v[2]}" for k, v in divergences.items()
    )


def test_real_parity_blank_gameweek(real_multiseason_history_and_upcoming):
    """
    Verify serving kernel handles Blank Gameweek (0 fixtures scheduled for team) cleanly,
    producing non-null features, is_bgw=1, and expected neutral context.
    """
    fe, history_df, _, boot, _, _ = real_multiseason_history_and_upcoming

    # No fixtures in GW6 (Blank GW for all teams)
    empty_fixtures: list[Fixture] = []
    df_serve = fe.extract_live_features_for_upcoming(boot, empty_fixtures, target_gw=6, history_df=history_df)

    assert len(df_serve) == 3
    assert (df_serve["is_bgw"] == 1).all()
    assert (df_serve["fixture_count"] == 0).all()

    for col in FEATURE_COLUMNS:
        assert not df_serve[col].isna().any(), f"NaN in BGW feature {col}"


def test_real_parity_double_gameweek(real_multiseason_history_and_upcoming):
    """
    Verify parity for Double Gameweek:
    - Match 1 has 100% exact parity across all 62 features.
    - Match 2 matches on all match context features (including rest days, implied xG, clean sheet probability, opponent difficulty, etc.).
    - Divergences on Match 2 are strictly confined to the locked PARITY_DGW_ALLOW_LIST.
    """
    fe, history_df, _, boot, _, _ = real_multiseason_history_and_upcoming

    # Arsenal has DGW in GW6: Match 1 vs Chelsea, Match 2 vs Liverpool
    gw6_matches_dgw = [
        # Match 1: Arsenal vs Chelsea (Sep 26 14:00)
        {
            "season": "2026-27",
            "round": 6,
            "name": "David Raya Martín",
            "element": 1,
            "team": "Arsenal",
            "opponent_team": "Chelsea",
            "position": "GKP",
            "minutes": 90,
            "starts": 1,
            "total_points": 6,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "expected_goal_involvements": 0.0,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 1,
            "goals_conceded": 0,
            "saves": 4,
            "defensive_contribution": 2,
            "ict_index": 4.0,
            "bps": 24,
            "bonus": 1,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 55.0,
            "kickoff_time": "2026-09-26T14:00:00Z",
        },
        {
            "season": "2026-27",
            "round": 6,
            "name": "Cole Palmer",
            "element": 154,
            "team": "Chelsea",
            "opponent_team": "Arsenal",
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 2,
            "expected_goals": 0.2,
            "expected_assists": 0.1,
            "expected_goal_involvements": 0.3,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 5.0,
            "bps": 12,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": False,
            "value": 105.0,
            "kickoff_time": "2026-09-26T14:00:00Z",
        },
        # Match 2: Arsenal vs Liverpool (Sep 29 19:00)
        {
            "season": "2026-27",
            "round": 6,
            "name": "David Raya Martín",
            "element": 1,
            "team": "Arsenal",
            "opponent_team": "Liverpool",
            "position": "GKP",
            "minutes": 90,
            "starts": 1,
            "total_points": 3,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "expected_goal_involvements": 0.0,
            "expected_goals_conceded": 1.5,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 2,
            "saves": 3,
            "defensive_contribution": 1,
            "ict_index": 3.0,
            "bps": 18,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 55.0,
            "kickoff_time": "2026-09-29T19:00:00Z",
        },
        {
            "season": "2026-27",
            "round": 6,
            "name": "Mohamed Salah",
            "element": 200,
            "team": "Liverpool",
            "opponent_team": "Arsenal",
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 8,
            "expected_goals": 0.7,
            "expected_assists": 0.2,
            "expected_goal_involvements": 0.9,
            "expected_goals_conceded": 1.0,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 14.0,
            "bps": 28,
            "bonus": 3,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": False,
            "value": 125.0,
            "kickoff_time": "2026-09-29T19:00:00Z",
        },
    ]

    train_full_df = pd.concat([history_df, pd.DataFrame(gw6_matches_dgw)], ignore_index=True)
    X_train, _ = fe.build_historical_features(train_full_df)
    meta = X_train.attrs["meta"]

    dgw_fixtures = [
        Fixture(
            id=101,
            code=101,
            event=6,
            team_h=1,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=3,
            kickoff_time="2026-09-26T14:00:00Z",
        ),
        Fixture(
            id=102,
            code=102,
            event=6,
            team_h=1,
            team_a=3,
            team_h_difficulty=4,
            team_a_difficulty=4,
            kickoff_time="2026-09-29T19:00:00Z",
        ),
    ]

    df_serve = fe.extract_live_features_for_upcoming(boot, dgw_fixtures, target_gw=6, history_df=history_df)

    t_rows = X_train[(meta["element"] == 1) & (meta["round"] == 6) & (meta["season"] == "2026-27")]
    s_rows = df_serve[df_serve["element"] == 1]

    assert len(t_rows) == 2
    assert len(s_rows) == 2

    # Fixture 1: 100% exact parity across all 62 features
    for col in FEATURE_COLUMNS:
        v_train = float(t_rows.iloc[0][col])
        v_serve = float(s_rows.iloc[0][col])
        assert abs(v_train - v_serve) < 1e-4, f"DGW Fixture 1 divergence on {col}: train={v_train}, serve={v_serve}"

    # Fixture 2: Non-allow-listed features must match within 1e-4
    unauthorized_diffs = {}
    for col in FEATURE_COLUMNS:
        if col in PARITY_DGW_ALLOW_LIST:
            continue
        v_train = float(t_rows.iloc[1][col])
        v_serve = float(s_rows.iloc[1][col])
        if abs(v_train - v_serve) > 1e-4:
            unauthorized_diffs[col] = (v_train, v_serve, abs(v_train - v_serve))

    assert len(unauthorized_diffs) == 0, "Unauthorized DGW Fixture 2 divergences:\n" + "\n".join(
        f"  {k}: train={v[0]}, serve={v[1]}, diff={v[2]}" for k, v in unauthorized_diffs.items()
    )


def test_real_parity_refreshed_live_snapshot(real_multiseason_history_and_upcoming):
    """
    Verify that incrementally refreshing master history with newly completed gameweeks
    updates serving and training features identically for the next gameweek (GW7).
    """
    fe, history_df, train_full_df, boot, _, _ = real_multiseason_history_and_upcoming

    # train_full_df now contains GW1-6. We simulate advancing to GW7.
    gw7_matches = [
        {
            "season": "2026-27",
            "round": 7,
            "name": "David Raya Martín",
            "element": 1,
            "team": "Arsenal",
            "opponent_team": "Chelsea",
            "position": "GKP",
            "minutes": 90,
            "starts": 1,
            "total_points": 5,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "expected_goal_involvements": 0.0,
            "expected_goals_conceded": 1.0,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 3,
            "defensive_contribution": 1,
            "ict_index": 3.0,
            "bps": 16,
            "bonus": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": True,
            "value": 55.0,
            "kickoff_time": "2026-10-03T14:00:00Z",
        },
        {
            "season": "2026-27",
            "round": 7,
            "name": "Cole Palmer",
            "element": 154,
            "team": "Chelsea",
            "opponent_team": "Arsenal",
            "position": "MID",
            "minutes": 90,
            "starts": 1,
            "total_points": 6,
            "expected_goals": 0.4,
            "expected_assists": 0.2,
            "expected_goal_involvements": 0.6,
            "expected_goals_conceded": 1.0,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "saves": 0,
            "defensive_contribution": 0,
            "ict_index": 9.0,
            "bps": 22,
            "bonus": 1,
            "yellow_cards": 0,
            "red_cards": 0,
            "own_goals": 0,
            "penalties_missed": 0,
            "penalties_saved": 0,
            "was_home": False,
            "value": 105.0,
            "kickoff_time": "2026-10-03T14:00:00Z",
        },
    ]

    gw7_train_full = pd.concat([train_full_df, pd.DataFrame(gw7_matches)], ignore_index=True)
    X_train_gw7, _ = fe.build_historical_features(gw7_train_full)
    meta_gw7 = X_train_gw7.attrs["meta"]

    gw7_fixtures = [
        Fixture(
            id=201,
            code=201,
            event=7,
            team_h=1,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=3,
            kickoff_time="2026-10-03T14:00:00Z",
        ),
    ]

    df_serve_gw7 = fe.extract_live_features_for_upcoming(boot, gw7_fixtures, target_gw=7, history_df=train_full_df)

    for eid in [1, 154]:
        t_row = X_train_gw7[
            (meta_gw7["element"] == eid) & (meta_gw7["round"] == 7) & (meta_gw7["season"] == "2026-27")
        ].iloc[0]
        s_row = df_serve_gw7[df_serve_gw7["element"] == eid].iloc[0]

        divergences = {}
        for col in FEATURE_COLUMNS:
            v_train = float(t_row[col])
            v_serve = float(s_row[col])
            diff = abs(v_train - v_serve)
            if diff > 1e-4:
                divergences[col] = (v_train, v_serve, diff)

        assert len(divergences) == 0, (
            f"Refreshed snapshot parity failed for element {eid} on {len(divergences)} features"
        )


def test_parity_allow_list_minimal_and_locked():
    """
    Assert that the intentional divergence allow-list does not expand beyond 8 features,
    and every listed feature has a verified non-empty rationale.
    """
    assert len(PARITY_DGW_ALLOW_LIST) <= 8, f"Allow-list grew to {len(PARITY_DGW_ALLOW_LIST)}! Must not exceed 8."
    for col, reason in PARITY_DGW_ALLOW_LIST.items():
        assert col in FEATURE_COLUMNS, f"Allow-listed {col} not in FEATURE_COLUMNS"
        assert len(reason.strip()) > 20, f"Allow-list entry {col} lacks substantive rationale"
