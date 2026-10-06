"""Tests for Player Identity Resolution, Collision Elimination & Prediction Caching (G4)."""

import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, GameweekEvent, Team
from fpl_oracle.config import HISTORICAL_DIR
from fpl_oracle.data.player_identity import normalize_player_name, player_identity_resolver
from fpl_oracle.ml.predict import projection_engine

MASTER_CSV = HISTORICAL_DIR / "master_history.csv"


@pytest.fixture(scope="module")
def master_df():
    assert MASTER_CSV.exists(), f"Missing {MASTER_CSV}"
    return pd.read_csv(MASTER_CSV)


def test_normalize_player_name():
    """Verify name normalization strips diacritics and unifies spelling."""
    assert normalize_player_name("David Raya Martín") == "david raya martin"
    assert normalize_player_name("David Raya Martin") == "david raya martin"
    assert normalize_player_name("Gabriel dos Santos Magalhães") == "gabriel dos santos magalhaes"
    assert normalize_player_name("Erling Haaland") == "erling haaland"
    assert normalize_player_name("  KDB  ") == "kdb"
    assert normalize_player_name(None) == ""


def test_no_cross_season_element_id_contamination(master_df):
    """
    CRITICAL G4 TEST:
    Element ID 1 mapped to Balogun (2023-24), Vieira (2024-25), and Raya (2025-26 & 2026-27).
    Verify that resolving Raya (element 1 in 2026-27) yields ONLY David Raya matches,
    with ZERO matches belonging to Balogun or Vieira.
    """
    player_identity_resolver.load(history_df=master_df)

    raya_hist = player_identity_resolver.get_player_history(
        elem_id=1,
        season="2026-27",
        full_name="David Raya Martín",
        web_name="Raya",
    )

    assert not raya_hist.empty, "David Raya history must not be empty"

    # All rows in Raya's history must belong to David Raya
    norm_names = raya_hist["name"].apply(normalize_player_name).unique()
    assert len(norm_names) == 1, f"Expected single canonical identity, got {norm_names}"
    assert norm_names[0] == "david raya martin"

    # Must contain games from Brentford (2023-24) and Arsenal (2024-25, 2025-26, 2026-27)
    seasons_found = set(raya_hist["season"].unique())
    assert {"2023-24", "2024-25", "2025-26", "2026-27"}.issubset(seasons_found)

    # Must NOT contain Balogun or Vieira
    raw_names = set(raya_hist["name"].unique())
    for name in raw_names:
        assert "Balogun" not in name, f"Balogun contaminated Raya history: {name}"
        assert "Vieira" not in name, f"Vieira contaminated Raya history: {name}"


def test_haaland_multi_season_identity(master_df):
    """
    Verify Haaland's trajectory is correctly assembled across 4 seasons
    despite changing element IDs (355 -> 351 -> 430 -> 411).
    """
    player_identity_resolver.load(history_df=master_df)

    haaland_hist = player_identity_resolver.get_player_history(
        elem_id=411,
        season="2026-27",
        full_name="Erling Haaland",
        web_name="Haaland",
    )

    assert not haaland_hist.empty
    norm_names = haaland_hist["name"].apply(normalize_player_name).unique()
    assert norm_names[0] == "erling haaland"

    seasons = set(haaland_hist["season"].unique())
    assert {"2023-24", "2024-25", "2025-26", "2026-27"}.issubset(seasons)
    # 38 + 38 + 38 + 5 = 119 matches
    assert len(haaland_hist) == 119


def test_zero_nulls_in_critical_columns(master_df):
    """Verify master_history.csv contains zero nulls in critical operational columns."""
    critical_cols = [
        "season",
        "name",
        "element",
        "position",
        "team",
        "round",
        "minutes",
        "total_points",
    ]
    for col in critical_cols:
        null_count = master_df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null values in master_history.csv"


def test_content_based_prediction_caching():
    """
    Verify content-based prediction caching:
    1. Key incorporates model_version, season, target_gw, feature hash, and availability hash.
    2. Identical feature inputs hit cache without re-running models.
    3. Modified availability generates a distinct cache key and re-evaluates.
    """
    # Create synthetic bootstrap & fixture
    elem = Element(
        id=1,
        web_name="Raya",
        first_name="David",
        second_name="Raya Martín",
        team=1,
        element_type=1,
        now_cost=60,
        selected_by_percent="40.0",
        form="6.0",
        points_per_game="5.5",
        total_points=30,
        status="a",
        minutes=450,
        goals_scored=0,
        assists=0,
        clean_sheets=3,
        goals_conceded=4,
        own_goals=0,
        penalties_saved=1,
        penalties_missed=0,
        yellow_cards=0,
        red_cards=0,
        saves=10,
        bonus=3,
        bps=100,
        influence="100.0",
        creativity="0.0",
        threat="0.0",
        ict_index="10.0",
        starts=5,
        expected_goals="0.00",
        expected_assists="0.00",
        expected_goal_involvements="0.00",
        expected_goals_conceded="4.00",
        defensive_contribution=0,
        code=154561,
    )
    event = GameweekEvent(id=6, name="Gameweek 6", deadline_time="2026-09-26T10:00:00Z", is_current=False, is_next=True)
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS", strength=4),
        Team(id=2, name="Aston Villa", short_name="AVL", strength=3),
    ]
    elem_types = [
        ElementType(
            id=1,
            singular_name="Goalkeeper",
            singular_name_short="GKP",
            plural_name="Goalkeepers",
            plural_name_short="GKP",
        ),
    ]
    bootstrap = BootstrapStatic(events=[event], elements=[elem], teams=teams, element_types=elem_types)
    fixtures = [
        Fixture(id=1, code=1001, event=6, team_h=1, team_a=2, kickoff_time="2026-09-26T14:00:00Z"),
    ]

    # Ensure engine is loaded
    projection_engine.load_or_train()
    projection_engine._cache.clear()

    # Run 1: Base prediction
    res1 = projection_engine.predict_gameweek(
        target_gw=6,
        bootstrap=bootstrap,
        fixtures=fixtures,
        reconciled_availabilities=None,
    )
    assert not res1.empty
    assert len(projection_engine._cache) == 1

    key1 = list(projection_engine._cache.keys())[0]
    # Key must be namespace-isolated: model_version:season:gw:feature_hash:avail_hash
    parts = key1.split(":")
    assert len(parts) >= 4
    assert "2026-27" in parts[1]
    assert "gw6" in parts[2]

    # Run 2: Exact same inputs -> must hit cache (same key, no new entries)
    res2 = projection_engine.predict_gameweek(
        target_gw=6,
        bootstrap=bootstrap,
        fixtures=fixtures,
        reconciled_availabilities=None,
    )
    assert len(projection_engine._cache) == 1
    assert res1["expected_points"].iloc[0] == res2["expected_points"].iloc[0]

    # Run 3: Availability override -> must generate distinct key
    res3 = projection_engine.predict_gameweek(
        target_gw=6,
        bootstrap=bootstrap,
        fixtures=fixtures,
        reconciled_availabilities={1: 0.25},
    )
    assert len(projection_engine._cache) == 2
    key2 = [k for k in projection_engine._cache.keys() if k != key1][0]
    assert key2 != key1
    # Scaled availability must produce lower expected points
    assert res3["expected_points"].iloc[0] < res1["expected_points"].iloc[0]
