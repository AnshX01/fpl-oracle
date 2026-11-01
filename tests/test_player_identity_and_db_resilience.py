"""
Unit tests for Player Identity, Cross-Season ID Disambiguation,
Fuzzy Matching Surname Gating & SQLite Concurrency Integrity.
"""

import threading
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from fpl_oracle.api.models import Element
from fpl_oracle.data.fuzzy_match import FuzzyPlayerMatcher
from fpl_oracle.data.historical import (
    STANDARD_COLUMNS,
    merge_cross_season_player_records,
)
from fpl_oracle.data.player_identity import PlayerIdentityResolver
from fpl_oracle.data.store import data_store, retry_on_db_lock

# ============================================================================
# 1. Cross-Season Player ID Integrity & Disambiguation
# ============================================================================


def test_standard_columns_include_permanent_code():
    """Verify that permanent FPL global code is included in standard historical columns."""
    assert "code" in STANDARD_COLUMNS, "Permanent 'code' must be in STANDARD_COLUMNS"
    idx_elem = STANDARD_COLUMNS.index("element")
    idx_code = STANDARD_COLUMNS.index("code")
    assert idx_code == idx_elem + 1, "'code' should follow 'element' in STANDARD_COLUMNS"


def test_merge_cross_season_player_records_joins_on_permanent_code():
    """
    In FPL, seasonal element ID 120 in 2024/25 can be Player A (code 118748),
    while seasonal element ID 120 in 2026/27 can be Player B (code 224466).
    Merging must strictly join on permanent global `code` to prevent attribute collision.
    """
    df_2024 = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "name": "Player Alpha",
                "element": 120,
                "code": 118748,
                "total_points": 180,
                "minutes": 2700,
                "goals_scored": 12,
            },
            {
                "season": "2024-25",
                "name": "Player Gamma",
                "element": 350,
                "code": 999001,
                "total_points": 90,
                "minutes": 1500,
                "goals_scored": 3,
            },
        ]
    )

    df_2026 = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "name": "Player Beta",
                "element": 120,  # SAME seasonal element ID, DIFFERENT human!
                "code": 224466,
                "total_points": 45,
                "minutes": 600,
                "goals_scored": 1,
            },
            {
                "season": "2026-27",
                "name": "Player Alpha",
                "element": 415,  # Player Alpha received new element ID 415
                "code": 118748,  # Permanent code remains 118748!
                "total_points": 140,
                "minutes": 2200,
                "goals_scored": 9,
            },
        ]
    )

    merged = merge_cross_season_player_records(df_2024, df_2026, how="inner")

    # Only Player Alpha (code 118748) should match across both seasons
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["code"] == 118748
    assert row["name_past"] == "Player Alpha"
    assert row["name_current"] == "Player Alpha"
    assert row["element_past"] == 120
    assert row["element_current"] == 415
    # Confirm Player Beta (element 120 in 2026/27) was NOT conflated with Player Alpha (element 120 in 2024/25)
    assert row["total_points_past"] == 180
    assert row["total_points_current"] == 140


def test_merge_cross_season_refuses_naked_element_join_across_different_seasons():
    """
    If permanent `code` is missing and seasons differ, joining on transient `element`
    must raise ValueError rather than silently creating corrupt collisions.
    """
    df_2024 = pd.DataFrame([{"season": "2024-25", "element": 120, "points": 100}])
    df_2026 = pd.DataFrame([{"season": "2026-27", "element": 120, "points": 50}])

    with pytest.raises(ValueError, match="Cannot merge cross-season player records"):
        merge_cross_season_player_records(df_2024, df_2026)


def test_player_identity_resolver_isolates_by_permanent_code():
    """
    Verify PlayerIdentityResolver queries match histories strictly by permanent code
    when provided, completely bypassing seasonal element ID collisions.
    """
    history_df = pd.DataFrame(
        [
            {
                "season": "2024-25",
                "name": "Player One",
                "element": 50,
                "code": 100500,
                "round": 1,
                "team": "Team A",
                "opponent_team": "Team B",
                "minutes": 90,
                "total_points": 8,
                "expected_goals": 0.5,
            },
            {
                "season": "2025-26",
                "name": "Player Two",
                "element": 50,  # Colliding element ID in next season
                "code": 200500,
                "round": 1,
                "team": "Team C",
                "opponent_team": "Team D",
                "minutes": 45,
                "total_points": 1,
                "expected_goals": 0.1,
            },
            {
                "season": "2025-26",
                "name": "Player One",
                "element": 75,  # New element ID for Player One
                "code": 100500,
                "round": 1,
                "team": "Team A",
                "opponent_team": "Team C",
                "minutes": 90,
                "total_points": 10,
                "expected_goals": 0.8,
            },
        ]
    )

    resolver = PlayerIdentityResolver()
    resolver.load(history_df=history_df)

    # Query specifically for Player One by code 100500
    p1_history = resolver.get_player_history(code=100500)
    assert len(p1_history) == 2
    assert set(p1_history["code"]) == {100500}
    assert set(p1_history["element"]) == {50, 75}
    # Player Two's element 50 row must NOT appear
    assert 200500 not in p1_history["code"].values

    # Query specifically for Player Two by code 200500
    p2_history = resolver.get_player_history(code=200500)
    assert len(p2_history) == 1
    assert p2_history.iloc[0]["code"] == 200500
    assert p2_history.iloc[0]["total_points"] == 1


# ============================================================================
# 2. Surnames & Homonym Disambiguation in fuzzy_match.py
# ============================================================================


@pytest.fixture
def homonym_elements():
    """
    Population with multiple real-world homonyms across Premier League squads:
    - Timber (Jurrien vs Quinten)
    - Traore (Adama vs Boubacar)
    - Silva (Bernardo vs Fabio)
    - Williams (Neco vs Nico)
    """
    return [
        # Jurriën Timber (Arsenal, DEF, team 1)
        Element(
            id=101,
            web_name="Timber",
            first_name="Jurrien",
            second_name="Timber",
            team=1,
            element_type=2,
            status="a",
            now_cost=55,
        ),
        # Quinten Timber (Feyenoord/Other, MID, team 5)
        Element(
            id=102,
            web_name="Q.Timber",
            first_name="Quinten",
            second_name="Timber",
            team=5,
            element_type=3,
            status="a",
            now_cost=60,
        ),
        # Adama Traoré (Fulham, MID, team 9)
        Element(
            id=201,
            web_name="Adama",
            first_name="Adama",
            second_name="Traore",
            team=9,
            element_type=3,
            status="a",
            now_cost=50,
        ),
        # Boubacar Traoré (Wolves, MID, team 20)
        Element(
            id=202,
            web_name="B.Traore",
            first_name="Boubacar",
            second_name="Traore",
            team=20,
            element_type=3,
            status="a",
            now_cost=45,
        ),
        # Bernardo Silva (Man City, MID, team 11)
        Element(
            id=301,
            web_name="Bernardo",
            first_name="Bernardo",
            second_name="Silva",
            team=11,
            element_type=3,
            status="a",
            now_cost=70,
        ),
        # Fabio Silva (Wolves, FWD, team 20)
        Element(
            id=302,
            web_name="Silva",
            first_name="Fabio",
            second_name="Silva",
            team=20,
            element_type=4,
            status="a",
            now_cost=50,
        ),
        # Neco Williams (Nottingham Forest, DEF, team 15)
        Element(
            id=401,
            web_name="N.Williams",
            first_name="Neco",
            second_name="Williams",
            team=15,
            element_type=2,
            status="a",
            now_cost=45,
        ),
        # Nico Williams (Athletic/Other, FWD, team 30)
        Element(
            id=402,
            web_name="Williams",
            first_name="Nico",
            second_name="Williams",
            team=30,
            element_type=4,
            status="a",
            now_cost=80,
        ),
    ]


@pytest.fixture
def club_map():
    return {
        1: "Arsenal",
        5: "Feyenoord",
        9: "Fulham",
        11: "Man City",
        15: "Nott'm Forest",
        20: "Wolves",
        30: "Athletic",
    }


def test_shared_surname_without_context_triggers_ambiguity(homonym_elements, club_map):
    matcher = FuzzyPlayerMatcher()

    # Querying "Timber" with no club or position hint
    res_timber = matcher.match_single_name("Timber", homonym_elements, team_map=club_map)
    assert res_timber["matched"] is None, "Ambiguous surname must not resolve to an arbitrary player"
    assert res_timber["ambiguity_detected"] is True
    assert res_timber["confidence"] < 0.55
    alt_ids = [a["element"] for a in res_timber["alternatives"]]
    assert 101 in alt_ids and 102 in alt_ids

    # Querying "Traore" without context
    res_traore = matcher.match_single_name("Traore", homonym_elements, team_map=club_map)
    assert res_traore["matched"] is None
    assert res_traore["ambiguity_detected"] is True

    # Querying "Silva" without context
    res_silva = matcher.match_single_name("Silva", homonym_elements, team_map=club_map)
    assert res_silva["matched"] is None
    assert res_silva["ambiguity_detected"] is True


def test_fuzzy_matching_strictly_gates_by_team_id_and_club_name(homonym_elements, club_map):
    matcher = FuzzyPlayerMatcher()

    # 1. Timber at Arsenal (by club name string)
    res1 = matcher.match_single_name("Timber", homonym_elements, team_map=club_map, club="Arsenal")
    assert res1["matched"] is not None
    assert res1["matched"]["element"] == 101  # Jurrien
    assert res1["confidence"] >= 0.85

    # 2. Timber at Feyenoord (by team ID integer)
    res2 = matcher.match_single_name("Timber", homonym_elements, team_map=club_map, club=5)
    assert res2["matched"] is not None
    assert res2["matched"]["element"] == 102  # Quinten
    assert res2["confidence"] >= 0.85

    # 3. Traore at Wolves (team ID 20) vs Traore at Fulham (club name)
    res_wolves = matcher.match_single_name("Traore", homonym_elements, team_map=club_map, club=20)
    assert res_wolves["matched"] is not None
    assert res_wolves["matched"]["element"] == 202  # Boubacar

    res_fulham = matcher.match_single_name("Traore", homonym_elements, team_map=club_map, club="Fulham")
    assert res_fulham["matched"] is not None
    assert res_fulham["matched"]["element"] == 201  # Adama


def test_fuzzy_matching_strictly_gates_by_position(homonym_elements, club_map):
    matcher = FuzzyPlayerMatcher()

    # Silva: Bernardo is MID (type 3), Fabio is FWD (type 4)
    res_mid = matcher.match_single_name("Silva", homonym_elements, team_map=club_map, position="MID")
    assert res_mid["matched"] is not None
    assert res_mid["matched"]["element"] == 301  # Bernardo Silva

    res_fwd = matcher.match_single_name("Silva", homonym_elements, team_map=club_map, position="FWD")
    assert res_fwd["matched"] is not None
    assert res_fwd["matched"]["element"] == 302  # Fabio Silva

    # Williams: Neco is DEF (type 2), Nico is FWD (type 4)
    res_def = matcher.match_single_name("Williams", homonym_elements, team_map=club_map, position="DEF")
    assert res_def["matched"] is not None
    assert res_def["matched"]["element"] == 401  # Neco Williams


def test_web_name_equals_surname_cannot_bypass_gating(homonym_elements, club_map):
    """
    Even when an element's web_name is exactly 'Timber' or 'Silva',
    it must NOT bypass gating or claim 1.0 confidence when querying the wrong club!
    """
    matcher = FuzzyPlayerMatcher()

    # Element 101 has web_name="Timber" (Arsenal). If query asks for club="Feyenoord",
    # Element 101 MUST be gated out, and Element 102 must be returned!
    res_fey = matcher.match_single_name("Timber", homonym_elements, team_map=club_map, club="Feyenoord")
    assert res_fey["matched"] is not None
    assert res_fey["matched"]["element"] == 102  # Quinten

    # If query asks for an unrelated club e.g. "Chelsea", neither matches -> None
    res_none = matcher.match_single_name("Timber", homonym_elements, team_map=club_map, club="Chelsea")
    assert res_none["matched"] is None
    assert res_none["confidence"] == 0.0


# ============================================================================
# 3. SQLite Concurrency & Lock Resilience in store.py
# ============================================================================


def test_sqlite_wal_mode_and_busy_timeout_configured():
    """Verify active SQLite connection has WAL mode and 30000ms busy timeout enabled."""
    with data_store.get_session() as session:
        conn = session.connection()
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        busy_timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()

        assert str(journal_mode).lower() == "wal", f"Expected WAL mode, got {journal_mode}"
        assert int(busy_timeout) >= 30000, f"Expected busy timeout >= 30000ms, got {busy_timeout}"


def test_sqlite_concurrent_multithreaded_writes():
    """
    Verify SQLite store handles high-frequency concurrent multithreaded writes
    without raising `OperationalError: database is locked` or corrupting data.
    """
    num_threads = 16
    writes_per_thread = 10
    errors = []

    def worker(thread_idx: int):
        try:
            for i in range(writes_per_thread):
                key = f"concurrent_test_key_{thread_idx}_{i}"
                payload = {"thread": thread_idx, "iter": i, "timestamp": datetime.now(UTC).isoformat()}
                # 1. Cache write
                data_store.set_cache(key, payload, ttl_seconds=60)
                # 2. Snapshot write
                data_store.save_snapshot("concurrent_test_endpoint", payload)
                # 3. Cache read back
                entry = data_store.get_cache_entry(key)
                if not entry or entry[0]["thread"] != thread_idx:
                    errors.append(f"Cache verification failed for {key}")
        except Exception as ex:
            errors.append(f"Thread {thread_idx} raised {type(ex).__name__}: {ex}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent database writes encountered errors: {errors}"


def test_sqlite_transaction_rollback_on_failure():
    """Verify transaction context manager rolls back dirty state on exception."""
    test_key = "test_rollback_cache_key"
    data_store.delete_cache_entry(test_key)

    # Intentionally fail inside transaction
    with pytest.raises(RuntimeError, match="Simulated write failure"):
        with data_store.transaction() as session:
            from fpl_oracle.data.store import APICacheEntry

            entry = APICacheEntry(
                key=test_key,
                data_json='{"status": "should_be_rolled_back"}',
                updated_at=datetime.now(UTC),
                expires_at=datetime.now(UTC),
            )
            session.add(entry)
            session.flush()
            raise RuntimeError("Simulated write failure")

    # Confirm key was rolled back and does not exist in store
    assert data_store.get_cache_entry(test_key) is None


def test_retry_on_db_lock_decorator_recovers_from_transient_busy():
    """Verify retry_on_db_lock catches operational lock and succeeds on retry."""
    call_count = 0

    @retry_on_db_lock(max_retries=3, base_delay=0.01, max_delay=0.05)
    def flaky_write():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OperationalError("database is locked", {}, Exception())
        return "SUCCESS"

    result = flaky_write()
    assert result == "SUCCESS"
    assert call_count == 3
