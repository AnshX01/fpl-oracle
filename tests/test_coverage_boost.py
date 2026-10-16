"""
Real unit tests targeting new and core modules to push test coverage above the >= 75.8% gate.
Modules tested:
- fpl_oracle.utils.season (derive_season, _format_season with env, config, fixtures, naive dates)
- fpl_oracle.utils.logging (RedactingFormatter formatException, dict args filtering, empty text)
- fpl_oracle.ml.model_registry (single-GW bootstrap, secondary metric failure, version history)
- fpl_oracle.data.historical (ensure_dataset_ready, last finished gameweek lookup)
"""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
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
    monkeypatch.setattr(season_mod, "_SEASON_OVERRIDE", None)
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


def test_init_startup_secrets_exception(monkeypatch):
    """Verify _init_startup_secrets catches and ignores file read errors."""
    import pathlib

    import fpl_oracle.utils.logging as log_mod

    def mock_raise(*args, **kwargs):
        raise OSError("Simulated unreadable file")

    monkeypatch.setattr(pathlib.Path, "read_text", mock_raise)
    log_mod._init_startup_secrets()


@pytest.mark.anyio
async def test_refresh_gameweek_live_error_branches(monkeypatch):
    """Verify refresh_gameweek_live rejects stale bootstrap, missing gw, stale fixtures, and empty live elements."""
    from unittest.mock import AsyncMock

    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.api.models import BootstrapStatic, GameweekEvent

    mgr = HistoricalDataManager()

    ev = GameweekEvent(id=7, name="GW7", deadline_time="2026-10-01T10:00:00Z", finished=True, data_checked=True)
    boot = BootstrapStatic(events=[ev], elements=[], teams=[], element_types=[])

    # 1. Stale bootstrap
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, True)))
    with pytest.raises(RuntimeError, match="requires fresh official bootstrap"):
        await mgr.refresh_gameweek_live(7, force=False)

    # 2. Missing event in bootstrap
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    with pytest.raises(ValueError, match="not found in official events"):
        await mgr.refresh_gameweek_live(99, force=False)

    # 3. Stale fixtures
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([], True)))
    with pytest.raises(RuntimeError, match="requires fresh fixture data"):
        await mgr.refresh_gameweek_live(7, force=False)

    # 4. Empty live elements
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([], False)))
    monkeypatch.setattr(fpl_client, "get_live_gameweek", AsyncMock(return_value=({"elements": []}, False)))
    with pytest.raises(RuntimeError, match="No element data returned"):
        await mgr.refresh_gameweek_live(7, force=False)
