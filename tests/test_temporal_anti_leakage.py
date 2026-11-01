"""
Comprehensive Unit Test Suite for Strict Temporal Separation, Anti-Leakage Invariants,
and Intra-Gameweek Causality across Single and Double Gameweeks.

Validates:
1. Single Gameweek Strict Anti-Leakage:
   - Features at round t use strictly matches from rounds < t.
   - Perturbing actual match events in round t and t+1..N leaves round t features bit-for-bit identical.
   - Audits specifically: xg_share_30, xa_share_30, finishing_30, assist_conversion_30,
     form (roll_points_3/5/8), roll_ict_5, roll_minutes_3, roll_clean_sheets_5.
   - Exact shift(1) propagation to round t+1 without off-by-one windowing errors.
2. Double Gameweek Intra-GW Isolation (Match 1 vs Match 2):
   - Match 2 features do NOT contain Match 1 results during pre-deadline prediction.
   - Perturbing Match 1 performance leaves Match 2 features 100% unchanged.
   - Perturbing Match 2 performance leaves Match 1 features 100% unchanged.
   - Both matches in round t reflect strictly pre-deadline state (rounds < t).
   - In round t+1, both legs of the DGW propagate into rolling history.
3. Live Serving Temporal Guard & Deadline Enforcement:
   - extract_live_features_for_upcoming strictly filters out any history >= target_gw or >= deadline.
4. Temporal Folds Chronological Invariants:
   - Indivisible gameweek blocks, expanding train windows, strictly ordered test blocks.
5. Production Code Purity:
   - Zero hardcoded player names in src/.
"""

import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, GameweekEvent, Team
from fpl_oracle.data.features import FEATURE_COLUMNS, FeatureEngineering
from fpl_oracle.ml.temporal import _season_round_sort_key, temporal_folds
from fpl_oracle.utils.season import derive_season


def make_multi_gw_player_data(
    n_gws: int = 8,
    dgw_round: int | None = None,
    season: str = "2026-27",
    elem_id: int = 10,
    name: str = "Test Midfielder",
    pos: str = "MID",
    team: str = "Arsenal",
    opp: str = "Chelsea",
) -> pd.DataFrame:
    """Construct multi-round player history with optional Double Gameweek."""
    rows = []
    for r in range(1, n_gws + 1):
        if dgw_round is not None and r == dgw_round:
            # Match 1 of DGW
            rows.append(
                {
                    "season": season,
                    "round": r,
                    "name": name,
                    "element": elem_id,
                    "team": team,
                    "opponent_team": opp,
                    "position": pos,
                    "minutes": 90,
                    "starts": 1,
                    "total_points": 8,
                    "expected_goals": 0.50,
                    "expected_assists": 0.30,
                    "expected_goal_involvements": 0.80,
                    "expected_goals_conceded": 0.80,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 1,
                    "goals_conceded": 0,
                    "saves": 0,
                    "defensive_contribution": 2,
                    "ict_index": 10.5,
                    "bps": 32,
                    "bonus": 3,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_missed": 0,
                    "penalties_saved": 0,
                    "was_home": True,
                    "value": 85.0,
                    "kickoff_time": f"2026-09-{10 + r:02d}T12:30:00Z",
                    "fixture": 100 + r * 2,
                }
            )
            # Match 2 of DGW
            rows.append(
                {
                    "season": season,
                    "round": r,
                    "name": name,
                    "element": elem_id,
                    "team": team,
                    "opponent_team": "Wolves",
                    "position": pos,
                    "minutes": 75,
                    "starts": 1,
                    "total_points": 5,
                    "expected_goals": 0.25,
                    "expected_assists": 0.40,
                    "expected_goal_involvements": 0.65,
                    "expected_goals_conceded": 1.20,
                    "goals_scored": 0,
                    "assists": 1,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "saves": 0,
                    "defensive_contribution": 1,
                    "ict_index": 7.2,
                    "bps": 18,
                    "bonus": 0,
                    "yellow_cards": 1,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_missed": 0,
                    "penalties_saved": 0,
                    "was_home": False,
                    "value": 85.0,
                    "kickoff_time": f"2026-09-{13 + r:02d}T19:45:00Z",
                    "fixture": 100 + r * 2 + 1,
                }
            )
        else:
            rows.append(
                {
                    "season": season,
                    "round": r,
                    "name": name,
                    "element": elem_id,
                    "team": team,
                    "opponent_team": opp,
                    "position": pos,
                    "minutes": 90,
                    "starts": 1,
                    "total_points": 6,
                    "expected_goals": 0.35,
                    "expected_assists": 0.25,
                    "expected_goal_involvements": 0.60,
                    "expected_goals_conceded": 1.00,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "saves": 0,
                    "defensive_contribution": 1,
                    "ict_index": 8.0,
                    "bps": 22,
                    "bonus": 1,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_missed": 0,
                    "penalties_saved": 0,
                    "was_home": bool(r % 2 == 1),
                    "value": 85.0,
                    "kickoff_time": f"2026-08-{10 + r:02d}T15:00:00Z",
                    "fixture": 100 + r,
                }
            )
    return pd.DataFrame(rows)


# ==============================================================================
# 1. Single Gameweek Strict Anti-Leakage & Off-By-One Audit
# ==============================================================================


def test_single_gw_strict_temporal_causality():
    """
    Assert that modifying actual match outcomes in Gameweek t (and future gameweeks t+1..N)
    has ZERO impact on the feature vector computed for Gameweek t.
    Specifically checks:
    - xg_share_30, xa_share_30, finishing_30, assist_conversion_30
    - form (roll_points_3, roll_points_5, roll_points_8)
    - roll_ict_5, roll_minutes_3, roll_clean_sheets_5
    """
    fe = FeatureEngineering()
    df_clean = make_multi_gw_player_data(n_gws=8)

    X_clean, _ = fe.build_historical_features(df_clean)

    # Focus on Gameweek 5 (0-indexed row 4)
    target_idx = 4
    gw5_clean = X_clean.iloc[target_idx].copy()

    # Create heavily perturbed copy in GW5 and all subsequent GWs 6..8
    df_pert = df_clean.copy()
    df_pert.loc[target_idx, "minutes"] = 15
    df_pert.loc[target_idx, "total_points"] = 25
    df_pert.loc[target_idx, "goals_scored"] = 4
    df_pert.loc[target_idx, "assists"] = 3
    df_pert.loc[target_idx, "expected_goals"] = 3.5
    df_pert.loc[target_idx, "expected_assists"] = 2.1
    df_pert.loc[target_idx, "clean_sheets"] = 1
    df_pert.loc[target_idx, "defensive_contribution"] = 14
    df_pert.loc[target_idx, "ict_index"] = 25.0
    df_pert.loc[target_idx, "bps"] = 60
    df_pert.loc[target_idx, "bonus"] = 3

    # Future gameweeks 6..8
    for f_idx in range(5, len(df_pert)):
        df_pert.loc[f_idx, "total_points"] = 18
        df_pert.loc[f_idx, "goals_scored"] = 3
        df_pert.loc[f_idx, "expected_goals"] = 2.5
        df_pert.loc[f_idx, "clean_sheets"] = 1

    X_pert, _ = fe.build_historical_features(df_pert)
    gw5_pert = X_pert.iloc[target_idx].copy()

    # Assert exact numerical equality across all canonical features
    diffs = {}
    for col in FEATURE_COLUMNS:
        v_c = float(gw5_clean[col])
        v_p = float(gw5_pert[col])
        if abs(v_c - v_p) > 1e-9:
            diffs[col] = (v_c, v_p)

    assert len(diffs) == 0, f"Temporal leakage detected in GW5 features! Leaked columns: {diffs}"

    # Verify critical rolling audit features specifically
    critical_cols = [
        "xg_share_30",
        "xa_share_30",
        "finishing_30",
        "assist_conversion_30",
        "roll_points_3",
        "roll_points_5",
        "roll_points_8",
        "roll_ict_5",
        "roll_minutes_3",
        "roll_clean_sheets_5",
    ]
    for col in critical_cols:
        assert float(gw5_clean[col]) == float(gw5_pert[col]), f"Audit feature {col} leaked future data!"


def test_exact_shift_one_propagation():
    """
    Assert that perturbing GW_k events DOES propagate to GW_(k+1),
    confirming exact shift(1) behavior and zero off-by-one window lag.
    """
    fe = FeatureEngineering()
    df_clean = make_multi_gw_player_data(n_gws=6)
    X_clean, _ = fe.build_historical_features(df_clean)

    # Perturb GW 3 (index 2)
    df_pert = df_clean.copy()
    df_pert.loc[2, "goals_scored"] = 5
    df_pert.loc[2, "expected_goals"] = 3.0
    df_pert.loc[2, "total_points"] = 20

    X_pert, _ = fe.build_historical_features(df_pert)

    # GW 3 features must be unchanged
    assert abs(float(X_clean.iloc[2]["roll_points_3"]) - float(X_pert.iloc[2]["roll_points_3"])) < 1e-9

    # GW 4 features (index 3) MUST reflect the perturbation
    assert float(X_pert.iloc[3]["roll_points_3"]) > float(X_clean.iloc[3]["roll_points_3"])
    assert float(X_pert.iloc[3]["roll_xG_3"]) > float(X_clean.iloc[3]["roll_xG_3"])
    assert float(X_pert.iloc[3]["finishing_30"]) > float(X_clean.iloc[3]["finishing_30"])


# ==============================================================================
# 2. Double Gameweek Intra-GW Isolation (Match 1 vs Match 2)
# ==============================================================================


def test_dgw_match2_does_not_contaminate_match1():
    """
    In a Double Gameweek where a team plays two matches in round t:
    Assert that Match 2 actual performance does NOT contaminate Match 1 features during prediction.
    """
    fe = FeatureEngineering()
    df_clean = make_multi_gw_player_data(n_gws=6, dgw_round=4)

    X_clean, _, meta_clean = fe.build_historical_features(df_clean, return_meta=True)

    # GW4 has 2 fixtures for this player
    gw4_mask = meta_clean["round"] == 4
    assert gw4_mask.sum() == 2, "Expected exactly 2 fixtures in DGW round 4"

    m1_idx = np.flatnonzero(gw4_mask)[0]
    m2_idx = np.flatnonzero(gw4_mask)[1]

    m1_clean = X_clean.iloc[m1_idx].copy()

    # Perturb Match 2 actuals drastically (hat-trick, 3 bonus, 4.0 xG)
    df_pert = df_clean.copy()
    df_pert.loc[m2_idx, "goals_scored"] = 4
    df_pert.loc[m2_idx, "total_points"] = 24
    df_pert.loc[m2_idx, "expected_goals"] = 3.8
    df_pert.loc[m2_idx, "clean_sheets"] = 1

    X_pert, _, _ = fe.build_historical_features(df_pert, return_meta=True)
    m1_pert = X_pert.iloc[m1_idx].copy()

    # Match 1 features must remain 100% UNCHANGED
    diffs = {}
    for col in FEATURE_COLUMNS:
        v_c = float(m1_clean[col])
        v_p = float(m1_pert[col])
        if abs(v_c - v_p) > 1e-9:
            diffs[col] = (v_c, v_p)

    assert len(diffs) == 0, f"Match 2 contaminated Match 1 features! Divergent columns: {diffs}"


def test_dgw_match1_does_not_contaminate_match2_predeadline():
    """
    In a Double Gameweek, managers must lock their squads before the Gameweek deadline (Match 1).
    Therefore, pre-deadline predictions for Match 2 must NOT observe Match 1 realized performance.
    Both Match 1 and Match 2 must share the strict pre-deadline rolling state.
    """
    fe = FeatureEngineering()
    df_clean = make_multi_gw_player_data(n_gws=6, dgw_round=4)

    X_clean, _, meta_clean = fe.build_historical_features(df_clean, return_meta=True)
    gw4_mask = meta_clean["round"] == 4
    m1_idx = np.flatnonzero(gw4_mask)[0]
    m2_idx = np.flatnonzero(gw4_mask)[1]

    m2_clean = X_clean.iloc[m2_idx].copy()

    # Perturb Match 1 actuals drastically
    df_pert = df_clean.copy()
    df_pert.loc[m1_idx, "goals_scored"] = 5
    df_pert.loc[m1_idx, "total_points"] = 28
    df_pert.loc[m1_idx, "expected_goals"] = 4.0
    df_pert.loc[m1_idx, "clean_sheets"] = 1

    X_pert, _, _ = fe.build_historical_features(df_pert, return_meta=True)
    m2_pert = X_pert.iloc[m2_idx].copy()

    # Match 2 pre-deadline features must NOT see Match 1's realized outcome
    player_rolling_cols = [
        "roll_minutes_3",
        "roll_minutes_5",
        "roll_points_3",
        "roll_points_5",
        "roll_points_8",
        "roll_xG_3",
        "roll_xG_5",
        "roll_xA_3",
        "roll_xGI_5",
        "xg90_30",
        "xa90_30",
        "xg_share_30",
        "xa_share_30",
        "finishing_30",
        "assist_conversion_30",
        "roll_clean_sheets_5",
        "roll_ict_5",
    ]
    diffs = {}
    for col in player_rolling_cols:
        v_c = float(m2_clean[col])
        v_p = float(m2_pert[col])
        if abs(v_c - v_p) > 1e-9:
            diffs[col] = (v_c, v_p)

    assert len(diffs) == 0, f"Match 1 leaked into Match 2 pre-deadline features! Divergent columns: {diffs}"


def test_dgw_both_matches_propagate_to_subsequent_gw():
    """
    Verify that in round t+1, BOTH Match 1 and Match 2 results are integrated into rolling statistics.
    """
    fe = FeatureEngineering()
    df_clean = make_multi_gw_player_data(n_gws=6, dgw_round=4)
    X_clean, _, meta_clean = fe.build_historical_features(df_clean, return_meta=True)

    gw4_indices = np.flatnonzero(meta_clean["round"] == 4)
    gw5_idx = np.flatnonzero(meta_clean["round"] == 5)[0]

    # Perturb Match 1
    df_pert1 = df_clean.copy()
    df_pert1.loc[gw4_indices[0], "goals_scored"] = 3
    df_pert1.loc[gw4_indices[0], "total_points"] = 15
    X_pert1, _, _ = fe.build_historical_features(df_pert1, return_meta=True)

    # Perturb Match 2
    df_pert2 = df_clean.copy()
    df_pert2.loc[gw4_indices[1], "goals_scored"] = 3
    df_pert2.loc[gw4_indices[1], "total_points"] = 15
    X_pert2, _, _ = fe.build_historical_features(df_pert2, return_meta=True)

    # Both perturbations must increase GW5 form
    pts_clean = float(X_clean.iloc[gw5_idx]["roll_points_3"])
    pts_pert1 = float(X_pert1.iloc[gw5_idx]["roll_points_3"])
    pts_pert2 = float(X_pert2.iloc[gw5_idx]["roll_points_3"])

    assert pts_pert1 > pts_clean, "Match 1 did not propagate to GW5!"
    assert pts_pert2 > pts_clean, "Match 2 did not propagate to GW5!"


# ==============================================================================
# 3. Live Serving Temporal Guard & Deadline Enforcement
# ==============================================================================


def test_extract_live_features_strictly_filters_future_gameweeks():
    """
    Assert that extract_live_features_for_upcoming cleanly drops any history rows
    at or after target_gw in current season, preventing future contamination.
    """
    curr_season = derive_season(format="hyphen")
    fe = FeatureEngineering()

    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Chelsea", short_name="CHE"),
    ]
    elements = [
        Element(
            id=10,
            web_name="Saka",
            first_name="Bukayo",
            second_name="Saka",
            team=1,
            element_type=3,
            now_cost=100,
            chance_of_playing_next_round=100,
        ),
    ]
    el_types = [
        ElementType(
            id=3,
            plural_name="Midfielders",
            plural_name_short="MID",
            singular_name="Midfielder",
            singular_name_short="MID",
        ),
    ]
    events = [
        GameweekEvent(id=5, name="Gameweek 5", deadline_time="2026-09-19T10:00:00Z", is_next=True),
        GameweekEvent(id=6, name="Gameweek 6", deadline_time="2026-09-26T10:00:00Z", is_next=False),
    ]
    boot = BootstrapStatic(teams=teams, elements=elements, element_types=el_types, events=events)

    fixtures = [
        Fixture(
            id=50,
            code=50,
            event=5,
            team_h=1,
            team_a=2,
            team_h_difficulty=3,
            team_a_difficulty=3,
            kickoff_time="2026-09-19T14:00:00Z",
        ),
    ]

    # Clean history up to GW4
    clean_hist = make_multi_gw_player_data(n_gws=4, season=curr_season, elem_id=10, name="Bukayo Saka")
    df_serve_clean = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=5, history_df=clean_hist)
    saka_clean = df_serve_clean.iloc[0].copy()

    # Contaminated history containing GW5 and GW6 matches with massive stats
    contam_hist = make_multi_gw_player_data(n_gws=6, season=curr_season, elem_id=10, name="Bukayo Saka")
    # Mark GW5 with 20 points, 3 goals
    contam_hist.loc[contam_hist["round"] == 5, "total_points"] = 20
    contam_hist.loc[contam_hist["round"] == 5, "goals_scored"] = 3
    contam_hist.loc[contam_hist["round"] == 5, "expected_goals"] = 2.5

    df_serve_contam = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=5, history_df=contam_hist)
    saka_contam = df_serve_contam.iloc[0].copy()

    # Features for GW5 must be 100% IDENTICAL because the temporal guard stripped GW5 and GW6
    diffs = {}
    for col in FEATURE_COLUMNS:
        v_c = float(saka_clean[col])
        v_p = float(saka_contam[col])
        if abs(v_c - v_p) > 1e-4:
            diffs[col] = (v_c, v_p)

    assert len(diffs) == 0, f"Future history leaked into extract_live_features! Leaked columns: {diffs}"


def test_extract_live_features_deadline_timestamp_guard():
    """
    Assert that if a fixture in current season kicks off at or after the target gameweek's
    deadline timestamp, it is cleanly excluded by the temporal deadline guard.
    """
    curr_season = derive_season(format="hyphen")
    fe = FeatureEngineering()

    teams = [Team(id=1, name="Arsenal", short_name="ARS"), Team(id=2, name="Chelsea", short_name="CHE")]
    elements = [
        Element(id=10, web_name="Saka", first_name="Bukayo", second_name="Saka", team=1, element_type=3, now_cost=100)
    ]
    el_types = [
        ElementType(id=3, plural_name="MID", plural_name_short="MID", singular_name="MID", singular_name_short="MID")
    ]
    events = [GameweekEvent(id=5, name="Gameweek 5", deadline_time="2026-09-19T10:00:00Z", is_next=True)]
    boot = BootstrapStatic(teams=teams, elements=elements, element_types=el_types, events=events)
    fixtures = [Fixture(id=50, code=50, event=5, team_h=1, team_a=2, kickoff_time="2026-09-19T14:00:00Z")]

    clean_hist = make_multi_gw_player_data(n_gws=4, season=curr_season, elem_id=10, name="Bukayo Saka")
    df_clean = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=5, history_df=clean_hist)

    # Add a match with kickoff_time after deadline
    post_deadline_hist = clean_hist.copy()
    leaked_row = post_deadline_hist.iloc[-1].copy()
    leaked_row["kickoff_time"] = "2026-09-19T11:00:00Z"  # 1 hour after 10:00 deadline
    leaked_row["total_points"] = 25
    leaked_row["goals_scored"] = 4
    post_deadline_hist = pd.concat([post_deadline_hist, pd.DataFrame([leaked_row])], ignore_index=True)

    df_guarded = fe.extract_live_features_for_upcoming(boot, fixtures, target_gw=5, history_df=post_deadline_hist)

    for col in ["roll_points_3", "roll_points_5", "roll_goals_5"]:
        assert abs(float(df_clean.iloc[0][col]) - float(df_guarded.iloc[0][col])) < 1e-4


# ==============================================================================
# 4. Temporal Folds Chronological Invariants in temporal.py
# ==============================================================================


def test_temporal_folds_strict_chronological_ordering():
    """
    Assert that temporal_folds strictly enforces:
    1. max(train gameweeks) < min(test gameweeks) for every fold.
    2. Zero gameweek overlap between train and test (indivisible GW blocks).
    3. Expanding train window across successive folds.
    4. Slicing with non-consecutive or filtered indices aligns correctly.
    """
    # Create dataset spanning 3 seasons: 2024-25, 2025-26, 2026-27
    meta_rows = []
    for s in ["2024-25", "2025-26", "2026-27"]:
        for r in range(1, 39 if s != "2026-27" else 11):
            # 5 players per gameweek
            for p in range(1, 6):
                meta_rows.append({"season": s, "round": r, "player": p})

    meta_df = pd.DataFrame(meta_rows)
    X = pd.DataFrame({"feat1": np.arange(len(meta_df))})
    X.attrs["meta"] = meta_df

    folds = list(temporal_folds(X, n_splits=4))
    assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}"

    prev_train_len = 0
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_meta = meta_df.iloc[train_idx]
        test_meta = meta_df.iloc[test_idx]

        # Extract (season, round) sets
        train_gws = set(zip(train_meta["season"], train_meta["round"], strict=True))
        test_gws = set(zip(test_meta["season"], test_meta["round"], strict=True))

        # Invariant 1: Indivisible GW blocks - no intersection
        assert len(train_gws.intersection(test_gws)) == 0, f"Fold {fold_idx}: GW overlap between train and test!"

        # Invariant 2: Index disjointness
        assert len(set(train_idx).intersection(set(test_idx))) == 0, f"Fold {fold_idx}: Overlapping row indices!"

        # Invariant 3: Strict chronological ordering
        max_train_key = max(_season_round_sort_key(g) for g in train_gws)
        min_test_key = min(_season_round_sort_key(g) for g in test_gws)
        assert max_train_key < min_test_key, (
            f"Fold {fold_idx}: Temporal leakage! max_train {max_train_key} >= min_test {min_test_key}"
        )

        # Invariant 4: Expanding window
        assert len(train_idx) > prev_train_len, f"Fold {fold_idx}: Train window did not expand!"
        prev_train_len = len(train_idx)


def test_temporal_folds_handles_filtered_slices():
    """
    Assert that temporal_folds handles a sliced dataframe where meta must align by index.
    """
    meta_df = pd.DataFrame({"season": ["2025-26"] * 20, "round": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5})
    X = pd.DataFrame({"feat": np.arange(20)})
    X.attrs["meta"] = meta_df

    # Sliced X (e.g. only even rows)
    X_sliced = X.iloc[::2].copy()
    # In pandas, X_sliced.attrs inherits the unsliced meta_df
    X_sliced.attrs["meta"] = meta_df

    folds = list(temporal_folds(X_sliced, n_splits=2))
    assert len(folds) >= 1
    for tr, te in folds:
        assert len(tr) > 0
        assert len(te) > 0
        assert len(set(tr).intersection(set(te))) == 0


# ==============================================================================
# 5. Production Code Purity: 0 Hardcoded Player Names in src/
# ==============================================================================


def test_production_code_purity_zero_player_names_in_src():
    """
    Assert that no hardcoded player names exist anywhere in src/.
    """
    import os
    from pathlib import Path

    src_dir = Path("src")
    assert src_dir.exists(), "src/ directory not found"

    # Common canonical player test names that must never be hardcoded in production source
    forbidden = [
        "haaland",
        "salah",
        "saka",
        "palmer",
        "watkins",
        "son",
        "raya",
        "tzolis",
        "balogun",
        "vieira",
        "rashford",
        "foden",
        "isak",
        "trent",
    ]

    excluded_files = {"held_out_benchmark_cases.py"}

    violations = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py"):
                if f in excluded_files:
                    continue
                path = Path(root) / f
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
                for name in forbidden:
                    # Match whole word occurrences in code (ignore comments if trivial, but check whole words)
                    import re

                    pattern = rf"\b{re.escape(name)}\b"
                    matches = list(re.finditer(pattern, text))
                    if matches:
                        # Exclude harmless occurrences in docstrings or comments mentioning external citations
                        violations.append((str(path), name, len(matches)))

    assert len(violations) == 0, f"Found hardcoded player names in src/: {violations}"
