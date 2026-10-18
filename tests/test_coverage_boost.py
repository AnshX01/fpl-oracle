"""
Real unit tests targeting new and core modules to push test coverage above the >= 75.8% gate.
Modules tested:
- fpl_oracle.utils.season (derive_season, _format_season with env, config, fixtures, naive dates)
- fpl_oracle.utils.logging (RedactingFormatter formatException, dict args filtering, empty text, startup secrets)
- fpl_oracle.ml.model_registry (single-GW bootstrap, secondary metric failure, version history)
- fpl_oracle.data.historical (clean_and_standardize, vaastav fetch, live season fetch, refresh, dataset ready)
"""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.data.historical import HistoricalDataManager
from fpl_oracle.ml.model_registry import ModelRegistry, paired_block_bootstrap_gw
from fpl_oracle.utils.logging import RedactingFormatter, SecretRedactingFilter, redact_text, setup_logging
from fpl_oracle.utils.season import _format_season, derive_season


def test_format_season_fallbacks_and_non_integer():
    """Verify _format_season handles invalid start year and formatting cleanly."""
    res_hyphen = _format_season("invalid-season", "hyphen")
    assert res_hyphen == "invalid-season"
    res_slash = _format_season("invalid-season", "slash")
    assert res_slash == "invalid/season"


def test_derive_season_from_env_var(monkeypatch):
    """Verify derive_season respects FPL_SEASON environment variable."""
    monkeypatch.setenv("FPL_SEASON", "2024-25")
    assert derive_season(format="slash") == "2024/25"
    assert derive_season(format="hyphen") == "2024-25"


def test_derive_season_from_fixtures():
    """Verify derive_season correctly derives season from fixture kickoff dates."""
    # String kickoff time in October 2025 -> 2025/26 season
    f1 = SimpleNamespace(kickoff_time="2025-10-15T15:00:00Z")
    assert derive_season(fixtures=[f1], format="slash") == "2025/26"

    # Datetime kickoff time in January 2026 (month < 7) -> 2025/26 season
    f2 = SimpleNamespace(kickoff_time=datetime(2026, 1, 10, 15, 0, tzinfo=UTC))
    assert derive_season(fixtures=[f2], format="hyphen") == "2025-26"

    # Naive datetime fixture
    f3 = SimpleNamespace(kickoff_time=datetime(2026, 8, 12, 12, 0))
    assert derive_season(fixtures=[f3], format="slash") == "2026/27"


def test_derive_season_with_naive_datetime():
    """Verify derive_season accepts a naive datetime and localizes to UTC."""
    naive_dt = datetime(2027, 3, 1)  # March 2027 -> 2026/27 season
    assert derive_season(dt=naive_dt, format="hyphen") == "2026-27"


def test_redact_text_empty_and_falsy():
    """Verify redact_text handles empty and None inputs safely."""
    assert redact_text("") == ""
    assert redact_text(None) is None


def test_redacting_formatter_format_exception():
    """Verify RedactingFormatter scrubs secrets from formatException tracebacks."""
    formatter = RedactingFormatter(fmt="%(message)s")
    fake_key = "AIza" + "SyFakeSecretKeyForTestingFormatExc123"
    try:
        raise ValueError(f"Connection failed: https://api.fpl.internal?key={fake_key}")
    except ValueError:
        import sys

        ei = sys.exc_info()
        formatted_exc = formatter.formatException(ei)
        assert fake_key not in formatted_exc
        assert "[REDACTED]" in formatted_exc


def test_secret_redacting_filter_with_dict_args():
    """Verify SecretRedactingFilter scrubs secrets when record.args is a dictionary."""
    filter_obj = SecretRedactingFilter()
    secret_val = "AQ." + "TestSecretValue1234567890"
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="API response payload: %(key)s",
        args={"key": secret_val, "status": 200},
        exc_info=None,
    )
    filter_obj.filter(record)
    assert secret_val not in record.args["key"]
    assert "REDACTED" in record.args["key"]
    assert record.args["status"] == 200


def test_setup_logging_stream_handler():
    """Verify setup_logging attaches stream handler properly."""
    logger = setup_logging()
    assert logger is not None
    assert logger.name == "fpl_oracle"


def test_block_bootstrap_comparison_single_gw_fallback():
    """Verify block_bootstrap_comparison handles single-gameweek input with fallback."""
    cand = np.array([2.0, 3.0, 4.0])
    act = np.array([1.5, 2.5, 3.5])
    y = np.array([2.0, 3.0, 4.0])
    gws = np.array([1, 1, 1])  # Only 1 unique GW

    res = paired_block_bootstrap_gw(cand, act, y, gws)
    assert res["decision_rule"] == "insufficient_gws_fallback"
    assert res["promoted"] is True
    assert "Insufficient gameweeks" in res["reason"]


def test_block_bootstrap_comparison_secondary_failure():
    """Verify candidate is rejected when secondary metric fails."""
    cand = np.array([2.0, 2.0, 2.0, 2.0])
    act = np.array([2.0, 2.0, 2.0, 2.0])
    y = np.array([2.0, 2.0, 2.0, 2.0])
    gws = np.array([1, 1, 2, 2])

    res = paired_block_bootstrap_gw(
        cand,
        act,
        y,
        gws,
        candidate_secondary=2.5,
        active_secondary=2.0,  # candidate secondary worse (higher MAE)
    )
    assert res["promoted"] is False
    assert res["decision_rule"] == "rejected"
    assert "secondary metric failed" in res["reason"]


def test_model_registry_version_history():
    """Verify ModelRegistry.get_version_history returns a list."""
    reg = ModelRegistry()
    hist = reg.get_version_history()
    assert isinstance(hist, list)


def test_historical_dataset_manager_ensure_ready():
    """Verify ensure_dataset_ready returns master history when file exists."""
    mgr = HistoricalDataManager()
    if mgr.output_file.exists():
        df = mgr.ensure_dataset_ready(force_refresh=False)
        assert not df.empty
        assert "element" in df.columns


def test_format_season_value_error_branch():
    """Verify ValueError in _format_season when first 4 chars are non-digits."""
    assert _format_season("abcd-ef", "hyphen") == "abcd-ef"
    assert _format_season("abcd-ef", "slash") == "abcd/ef"


def test_derive_season_from_config_and_invalid_fixture(monkeypatch):
    """Verify derive_season falls back past invalid fixture and reads config."""
    import fpl_oracle.utils.season as season_mod
    from fpl_oracle.config import SETTINGS

    monkeypatch.delenv("FPL_SEASON", raising=False)
    monkeypatch.setitem(SETTINGS, "season", "2023-24")

    bad_fixture = SimpleNamespace(kickoff_time="not-a-valid-timestamp")
    derived = season_mod.derive_season(fixtures=[bad_fixture], format="hyphen")
    assert derived == "2023-24"


def test_setup_logging_console_handler_creation(monkeypatch):
    """Verify setup_logging adds console StreamHandler when none present."""
    import fpl_oracle.utils.logging as log_mod

    monkeypatch.setattr(log_mod, "_is_configured", False)
    root_l = logging.getLogger()
    saved_handlers = list(root_l.handlers)
    try:
        root_l.handlers = [h for h in root_l.handlers if not isinstance(h, logging.StreamHandler)]
        logger = setup_logging()
        assert any(isinstance(h, logging.StreamHandler) for h in root_l.handlers)
        assert logger is not None
    finally:
        root_l.handlers = saved_handlers


def test_check_promotion_gate_error_and_degradation_branches():
    """Verify check_promotion_gate branch logic for missing fields, degradation, and origins."""
    from fpl_oracle.ml.model_registry import check_promotion_gate

    # 1. Missing MAE fields
    ok, reason = check_promotion_gate({})
    assert not ok
    assert "missing required MAE fields" in reason

    # 2. Baseline gate failure
    ok, reason = check_promotion_gate({"mae": 2.5, "base_mae": 2.0})
    assert not ok
    assert "failed baseline gate" in reason

    # 3. Nonfinite active MAE
    ok, reason = check_promotion_gate({"mae": 1.5, "base_mae": 2.0}, active_metrics={"mae": float("nan")})
    assert not ok
    assert "Active MAE is missing/nonfinite" in reason

    # 4. Active production degradation
    ok, reason = check_promotion_gate({"mae": 1.8, "base_mae": 2.0}, active_metrics={"mae": 1.5})
    assert not ok
    assert "degraded vs active production MAE" in reason

    # 5. Missing schema key in origins
    origins_missing_key = [{"season": "2023-24"}, {"season": "2024-25"}, {"season": "2025-26"}]
    ok, reason = check_promotion_gate({"mae": 1.5, "base_mae": 2.0}, rolling_origins=origins_missing_key)
    assert not ok
    assert "missing required schema key" in reason

    # 6. Non-positive mean gain in origins
    origins_negative_gain = [
        {
            "season": f"202{i}-2{i + 1}",
            "train_size": 1000,
            "test_size": 200,
            "mae": 2.0,
            "best_baseline_mae": 1.5,
            "gain_vs_baseline": -0.5,
            "interval_80_coverage_pct": 80.0,
            "spearman": 0.4,
            "rmse": 2.5,
        }
        for i in range(3)
    ]
    ok, reason = check_promotion_gate({"mae": 1.5, "base_mae": 2.0}, rolling_origins=origins_negative_gain)
    assert not ok
    assert "mean gain vs baseline" in reason


def test_init_startup_secrets_exception_and_redaction(monkeypatch):
    """Verify _init_startup_secrets populates secrets and safely catches file errors."""
    import pathlib

    import fpl_oracle.utils.logging as log_mod

    # Case 1: unreadable file raises OSError, function must catch without raising and retain set
    def mock_raise(*args, **kwargs):
        raise OSError("Simulated unreadable file")

    monkeypatch.setattr(pathlib.Path, "read_text", mock_raise)
    log_mod._init_startup_secrets()
    assert isinstance(log_mod._STARTUP_SECRETS, set)

    # Case 2: secret in environment is registered and redacted
    test_secret = "SecretKeyTestingValue12345"
    monkeypatch.setenv("FPL_API_SECRET_TOKEN", test_secret)
    log_mod._init_startup_secrets()
    assert test_secret in log_mod._STARTUP_SECRETS

    scrubbed = log_mod.redact_text(f"Authorization: Bearer {test_secret}")
    assert test_secret not in scrubbed
    assert "[REDACTED]" in scrubbed


# ==============================================================================
# Comprehensive HistoricalDataManager Unit Tests
# ==============================================================================


def test_historical_clean_and_standardize():
    """Verify _clean_and_standardize maps positions, adds missing columns, coerces numerics, and sorts."""
    mgr = HistoricalDataManager()

    raw_data = pd.DataFrame(
        [
            {
                "season": "2025-26",
                "round": 2,
                "element": 102,
                "position": 1,  # Numeric GKP
                "was_home": 1,
                "goals_scored": "3",  # string number
                "threat": None,  # null
            },
            {
                "season": "2024-25",
                "round": 1,
                "element": 50,
                "position": "GK",  # string GK -> GKP
                "was_home": False,
                "goals_scored": 0,
            },
            {
                "season": "2025-26",
                "round": 1,
                "element": 101,
                "position": "UNKNOWN",  # unknown -> MID
                "was_home": True,
                "goals_scored": 1,
            },
        ]
    )

    clean_df = mgr._clean_and_standardize(raw_data)

    # All standard columns present
    for col in [
        "season",
        "name",
        "element",
        "position",
        "team",
        "round",
        "fixture",
        "was_home",
        "minutes",
        "goals_scored",
        "threat",
        "total_points",
    ]:
        assert col in clean_df.columns

    # Positions standardized
    assert clean_df.loc[clean_df["element"] == 50, "position"].iloc[0] == "GKP"
    assert clean_df.loc[clean_df["element"] == 102, "position"].iloc[0] == "GKP"
    assert clean_df.loc[clean_df["element"] == 101, "position"].iloc[0] == "MID"

    # Numeric coercion and fillna
    assert clean_df.loc[clean_df["element"] == 102, "goals_scored"].iloc[0] == 3.0
    assert clean_df.loc[clean_df["element"] == 102, "threat"].iloc[0] == 0.0

    # was_home boolean conversion
    assert isinstance(clean_df.loc[clean_df["element"] == 102, "was_home"].iloc[0], (bool, np.bool_))

    # Chronologically sorted by season, round, element
    assert clean_df["season"].tolist() == ["2024-25", "2025-26", "2025-26"]
    assert clean_df["round"].tolist() == [1, 1, 2]
    assert clean_df["element"].tolist() == [50, 101, 102]


def test_get_last_finished_gameweek(tmp_path, monkeypatch):
    """Verify get_last_finished_gameweek returns maximum GW correctly across edge cases."""
    mgr = HistoricalDataManager()

    # Case 1: output file does not exist
    mgr.output_file = tmp_path / "nonexistent.csv"
    assert mgr.get_last_finished_gameweek() == 0

    # Case 2: output file exists with current season (e.g. 2026-27 up to GW 6)
    test_csv = tmp_path / "master.csv"
    df = pd.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2026-27", "2026-27"],
            "round": [37, 38, 5, 6],
            "element": [1, 2, 1, 2],
        }
    )
    df.to_csv(test_csv, index=False)
    mgr.output_file = test_csv

    import fpl_oracle.utils.season as season_mod

    monkeypatch.setattr(season_mod, "derive_season", lambda format="hyphen": "2026-27")
    assert mgr.get_last_finished_gameweek() == 6

    # Case 3: current season has no rows in file, falls back to overall max
    monkeypatch.setattr(season_mod, "derive_season", lambda format="hyphen": "2027-28")
    assert mgr.get_last_finished_gameweek() == 38

    # Case 4: read_csv raises exception -> returns 0
    def mock_bad_read(*args, **kwargs):
        raise ValueError("Corrupted CSV")

    monkeypatch.setattr(pd, "read_csv", mock_bad_read)
    assert mgr.get_last_finished_gameweek() == 0


def test_fetch_vaastav_season(tmp_path, monkeypatch):
    """Verify _fetch_vaastav_season handles cached files, network downloads, and failures."""
    import urllib.request

    from fpl_oracle.data import historical as hist_mod

    mgr = HistoricalDataManager()
    monkeypatch.setattr(hist_mod, "HISTORICAL_DIR", tmp_path)

    # 1. Local cache hit
    cached_file = tmp_path / "2023-24_merged_gw.csv"
    cached_df = pd.DataFrame({"element": [1, 2], "round": [1, 1], "total_points": [5, 2]})
    cached_df.to_csv(cached_file, index=False)

    df_cached = mgr._fetch_vaastav_season("2023-24")
    assert df_cached is not None
    assert len(df_cached) == 2
    assert "season" in df_cached.columns
    assert df_cached["season"].iloc[0] == "2023-24"

    # 2. Corrupted cache falls back to download
    corrupt_file = tmp_path / "2024-25_merged_gw.csv"
    corrupt_file.write_text("bad,data\n1,2,3,4,5\nextra\n")  # will trigger warning

    # Mock successful network download
    mock_csv_bytes = b"element,round,total_points\n10,1,6\n11,1,8\n"

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return mock_csv_bytes

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30: MockResponse())
    df_downloaded = mgr._fetch_vaastav_season("2024-25")
    assert df_downloaded is not None
    assert len(df_downloaded) == 2
    assert df_downloaded["season"].iloc[0] == "2024-25"

    # 3. Network failure returns None
    def mock_fail_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", mock_fail_urlopen)
    df_fail = mgr._fetch_vaastav_season("2025-26")
    assert df_fail is None


@pytest.mark.anyio
async def test_fetch_live_season_async_and_sync(monkeypatch):
    """Verify _fetch_live_season_async ingests elements and tracks failures, and sync wrapper."""
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.api.models import BootstrapStatic, Element, ElementSummary, ElementType, PlayerMatchHistory, Team

    mgr = HistoricalDataManager()

    # Mock bootstrap
    t1 = Team(id=1, name="Arsenal", short_name="ARS")
    et1 = ElementType(
        id=3, singular_name="Midfielder", singular_name_short="MID", plural_name="Midfielders", plural_name_short="MID"
    )
    p1 = Element(
        id=10, first_name="Martin", second_name="Odegaard", web_name="Odegaard", team=1, element_type=3, now_cost=85
    )
    p2 = Element(id=20, first_name="Declan", second_name="Rice", web_name="Rice", team=1, element_type=3, now_cost=65)
    boot = BootstrapStatic(events=[], elements=[p1, p2], teams=[t1], element_types=[et1])

    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))

    # Mock element summary: p1 succeeds, p2 raises error
    hist1 = PlayerMatchHistory(
        element=10,
        fixture=1,
        opponent_team=2,
        total_points=8,
        was_home=True,
        kickoff_time="2026-08-15T14:00:00Z",
        round=1,
        minutes=90,
        goals_scored=1,
        assists=0,
        clean_sheets=1,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        yellow_cards=0,
        red_cards=0,
        saves=0,
        bonus=2,
        bps=30,
        influence="35.0",
        creativity="40.0",
        threat="20.0",
        ict_index="9.5",
        starts=1,
        expected_goals="0.45",
        expected_assists="0.30",
        expected_goal_involvements="0.75",
        expected_goals_conceded="0.20",
        value=85,
    )
    summary1 = ElementSummary(history=[hist1])

    async def mock_get_summary(eid, force_refresh=True):
        if eid == 10:
            return summary1, False
        raise RuntimeError("API timeout on player 20")

    monkeypatch.setattr(fpl_client, "get_element_summary", mock_get_summary)

    live_df = await mgr._fetch_live_season_async()
    assert len(live_df) == 1
    assert live_df["name"].iloc[0] == "Martin Odegaard"
    assert live_df["total_points"].iloc[0] == 8
    assert 20 in mgr._last_failed_player_ids


def test_fetch_live_season_sync_wrapper(monkeypatch):
    """Verify _fetch_live_season_sync calls asyncio.run and handles exceptions."""
    mgr = HistoricalDataManager()
    dummy_df = pd.DataFrame([{"element": 1, "round": 1}])

    async def mock_async():
        return dummy_df

    monkeypatch.setattr(mgr, "_fetch_live_season_async", mock_async)
    sync_res = mgr._fetch_live_season_sync()
    assert sync_res is not None
    assert len(sync_res) == 1

    async def mock_fail():
        raise RuntimeError("Network boom")

    monkeypatch.setattr(mgr, "_fetch_live_season_async", mock_fail)
    assert mgr._fetch_live_season_sync() is None


@pytest.mark.anyio
async def test_refresh_current_season_lifecycle(tmp_path, monkeypatch):
    """Verify refresh_current_season handles stale checks, awaiting finalization, and atomic merge."""
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.api.models import BootstrapStatic, Fixture, GameweekEvent

    mgr = HistoricalDataManager()
    test_master = tmp_path / "master_history.csv"
    mgr.output_file = test_master

    # Pre-populate master with old seasons and old GW1 rows
    old_df = pd.DataFrame(
        [
            {"season": "2025-26", "round": 38, "element": 1, "fixture": 380, "name": "Player 1", "total_points": 5},
            {"season": "2026-27", "round": 1, "element": 1, "fixture": 1, "name": "Player 1", "total_points": 2},
        ]
    )
    old_df.to_csv(test_master, index=False)

    ev1 = GameweekEvent(id=1, name="GW1", deadline_time="2026-08-15T11:00:00Z", finished=True, data_checked=True)
    ev2 = GameweekEvent(id=2, name="GW2", deadline_time="2026-08-22T11:00:00Z", finished=True, data_checked=True)
    boot = BootstrapStatic(events=[ev1, ev2], elements=[], teams=[], element_types=[])
    f1 = Fixture(id=1, code=101, team_h=1, team_a=2, event=1, finished=True)
    f2 = Fixture(id=2, code=102, team_h=2, team_a=1, event=2, finished=True)

    # 1. Stale bootstrap raises RuntimeError
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, True)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([f1, f2], False)))
    with pytest.raises(RuntimeError, match="needs fresh official context"):
        await mgr.refresh_current_season()

    # 2. Fresh bootstrap and fixtures
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([f1, f2], False)))

    # Empty fresh live returns error
    monkeypatch.setattr(mgr, "_fetch_live_season_async", AsyncMock(return_value=pd.DataFrame()))
    with pytest.raises(RuntimeError, match="No fresh current-season rows"):
        await mgr.refresh_current_season()

    # Awaiting finalization (no rows match finalized events)
    unfinalized_fresh = pd.DataFrame([{"round": 99, "fixture": 999, "element": 1, "total_points": 0}])
    monkeypatch.setattr(mgr, "_fetch_live_season_async", AsyncMock(return_value=unfinalized_fresh))
    res_awaiting = await mgr.refresh_current_season()
    assert res_awaiting["status"] == "awaiting_finalization"

    # Successful fresh rows for GW1 and GW2
    fresh_rows = pd.DataFrame(
        [
            {"season": "2026-27", "round": 1, "element": 1, "fixture": 1, "name": "Player 1", "total_points": 8},
            {"season": "2026-27", "round": 2, "element": 1, "fixture": 2, "name": "Player 1", "total_points": 6},
        ]
    )
    monkeypatch.setattr(mgr, "_fetch_live_season_async", AsyncMock(return_value=fresh_rows))

    meta = await mgr.refresh_current_season()
    assert meta["status"] == "refreshed"
    assert meta["complete"] is True
    assert meta["rows"] == 2

    # Verify updated CSV on disk
    saved_df = pd.read_csv(test_master)
    # Replaced GW1 row now has updated points 8 (not 2)
    p1_gw1 = saved_df[(saved_df["season"] == "2026-27") & (saved_df["round"] == 1)]
    assert p1_gw1["total_points"].iloc[0] == 8.0
    # GW2 row was appended
    p1_gw2 = saved_df[(saved_df["season"] == "2026-27") & (saved_df["round"] == 2)]
    assert len(p1_gw2) == 1
    assert p1_gw2["total_points"].iloc[0] == 6.0


def test_ensure_dataset_ready_force_refresh(tmp_path, monkeypatch):
    """Verify ensure_dataset_ready(force_refresh=True) merges past seasons and live season."""
    mgr = HistoricalDataManager()
    test_master = tmp_path / "master.csv"
    mgr.output_file = test_master

    # Case 1: no season data available raises RuntimeError
    monkeypatch.setattr(mgr, "_fetch_vaastav_season", lambda s: None)
    monkeypatch.setattr(mgr, "_fetch_live_season_sync", lambda: None)
    with pytest.raises(RuntimeError, match="Failed to build historical dataset"):
        mgr.ensure_dataset_ready(force_refresh=True)

    # Case 2: successfully builds and standardizes dataset
    s1 = pd.DataFrame([{"season": "2023-24", "round": 1, "element": 1, "total_points": 5}])
    live = pd.DataFrame([{"season": "2026-27", "round": 1, "element": 1, "total_points": 10}])

    monkeypatch.setattr(mgr, "_fetch_vaastav_season", lambda s: s1 if s == "2023-24" else None)
    monkeypatch.setattr(mgr, "_fetch_live_season_sync", lambda: live)

    df_built = mgr.ensure_dataset_ready(force_refresh=True)
    assert len(df_built) == 2
    assert test_master.exists()
    assert set(df_built["season"]) == {"2023-24", "2026-27"}
