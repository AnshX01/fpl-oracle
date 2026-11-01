"""
Unit tests for Deadline API Resilience, Cloudflare 503/429 Error Protection,
Local Disk Caching Fallback, and Price Change Microstructure with Wildcard Isolation.
Strictly maintains 0 hardcoded player names.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import httpx

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import BootstrapStatic, Element, GameweekEvent, SquadPicks
from fpl_oracle.optimise.price_change import (
    calculate_fall_threshold,
    calculate_hours_to_cutoff,
    calculate_rise_threshold,
    estimate_wildcard_factor,
    price_change_predictor,
)


class MockCloudflareChallengeTransport(httpx.AsyncBaseTransport):
    """Simulates Cloudflare 503 / 403 HTML challenge pages and 429 rate limits."""

    def __init__(self, mode: str = "503_html", retry_after: str | None = None, fail_times: int = 999):
        self.mode = mode
        self.retry_after = retry_after
        self.fail_times = fail_times
        self.attempts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.attempts > self.fail_times:
            # Succeed after fail_times
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=b'{"events": [], "elements": [{"id": 1, "web_name": "Player1", "team": 1, "element_type": 1, "now_cost": 50}]}',
                request=request,
            )

        if self.mode == "503_html":
            html_challenge = (
                b"<!DOCTYPE html><html lang='en'><head><title>Just a moment...</title></head>"
                b"<body><h1>Checking your browser before accessing fantasy.premierleague.com</h1></body></html>"
            )
            return httpx.Response(
                status_code=503,
                headers={"Content-Type": "text/html; charset=UTF-8"},
                content=html_challenge,
                request=request,
            )
        elif self.mode == "200_html_challenge":
            html_challenge = (
                b"<!DOCTYPE html><html><head><title>Attention Required! | Cloudflare</title></head>"
                b"<body>Please complete the security check to continue.</body></html>"
            )
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "text/html"},
                content=html_challenge,
                request=request,
            )
        elif self.mode == "429_throttled":
            headers = {"Content-Type": "application/json"}
            if self.retry_after:
                headers["Retry-After"] = self.retry_after
            return httpx.Response(
                status_code=429,
                headers=headers,
                content=b'{"detail": "Request throttled. Expected available in 1 second."}',
                request=request,
            )
        elif self.mode == "520_cloudflare":
            return httpx.Response(
                status_code=520,
                headers={"Content-Type": "text/html"},
                content=b"<!DOCTYPE html><html><body>520 Web server is returning an unknown error</body></html>",
                request=request,
            )
        return httpx.Response(status_code=500, content=b"Server Error", request=request)


# ====================================================================
# PART 1: API RESILIENCE & CLOUDFLARE PROTECTION TESTS
# ====================================================================


def test_cloudflare_503_html_clean_handling_and_stale_fallback(monkeypatch):
    """
    Verify that when Cloudflare returns HTTP 503 with an HTML challenge,
    the client catches it cleanly without crashing on JSONDecodeError,
    retries with jitter, and falls back to stale cache.
    """

    async def _run():
        sleep_mock = AsyncMock()
        monkeypatch.setattr("fpl_oracle.api.fpl_client.asyncio.sleep", sleep_mock)
        monkeypatch.setattr(fpl_client, "rate_delay", 0)

        # Seed cache
        cached_payload = {
            "events": [{"id": 1, "is_current": True, "name": "Gameweek 1", "deadline_time": "2026-08-15T10:00:00Z"}],
            "elements": [{"id": 101, "web_name": "P101", "team": 1, "element_type": 3, "now_cost": 75}],
        }
        cache_manager.set("bootstrap-static", cached_payload, ttl_seconds=300)

        transport = MockCloudflareChallengeTransport(mode="503_html")
        fpl_client.set_transport(transport)
        try:
            boot, is_stale = await fpl_client.get_bootstrap_static(force_refresh=True)
            assert is_stale is True, "Expected is_stale=True when falling back from Cloudflare 503"
            assert len(boot.elements) == 1
            assert boot.elements[0].id == 101
            assert sleep_mock.await_count >= 2
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_cloudflare_200_html_challenge_detection(monkeypatch):
    """
    Verify that if Cloudflare returns HTTP 200 with HTML (e.g. Turnstile / challenge),
    content-type and body validation prevents JSONDecodeError and triggers retry/fallback.
    """

    async def _run():
        sleep_mock = AsyncMock()
        monkeypatch.setattr("fpl_oracle.api.fpl_client.asyncio.sleep", sleep_mock)
        monkeypatch.setattr(fpl_client, "rate_delay", 0)

        # Seed fallback in cache
        cached_payload = {
            "events": [],
            "elements": [{"id": 202, "web_name": "P202", "team": 2, "element_type": 2, "now_cost": 60}],
        }
        cache_manager.set("bootstrap-static", cached_payload, ttl_seconds=300)

        transport = MockCloudflareChallengeTransport(mode="200_html_challenge")
        fpl_client.set_transport(transport)
        try:
            boot, is_stale = await fpl_client.get_bootstrap_static(force_refresh=True)
            assert is_stale is True
            assert boot.elements[0].id == 202
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_cloudflare_429_retry_after_and_eventual_recovery(monkeypatch):
    """
    Verify that 429 throttling respects Retry-After header and succeeds
    when the server recovers on a subsequent attempt.
    """

    async def _run():
        sleep_mock = AsyncMock()
        monkeypatch.setattr("fpl_oracle.api.fpl_client.asyncio.sleep", sleep_mock)
        monkeypatch.setattr(fpl_client, "rate_delay", 0)

        # Fail twice with 429 then succeed
        transport = MockCloudflareChallengeTransport(mode="429_throttled", retry_after="2", fail_times=2)
        fpl_client.set_transport(transport)
        try:
            boot, is_stale = await fpl_client.get_bootstrap_static(force_refresh=True)
            assert is_stale is False, "Expected fresh data on recovery"
            assert len(boot.elements) == 1
            assert transport.attempts == 3
            assert sleep_mock.await_count == 2
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_local_disk_cache_fallback(monkeypatch):
    """
    Verify that when in-memory and SQLite cache are cold, the client
    falls back to local disk cache in data/cache/.
    """

    async def _run():
        monkeypatch.setattr("fpl_oracle.api.fpl_client.asyncio.sleep", AsyncMock())
        monkeypatch.setattr(fpl_client, "rate_delay", 0)

        test_key = "test-disk-endpoint"
        cache_manager.invalidate(test_key)

        disk_data = {"custom_data": 42, "status": "disk_persisted"}
        fpl_client._save_disk_cache(test_key, disk_data)

        loaded = fpl_client._load_disk_cache(test_key)
        assert loaded is not None
        assert loaded[0]["status"] == "disk_persisted"

        transport = MockCloudflareChallengeTransport(mode="503_html")
        fpl_client.set_transport(transport)
        try:
            data, is_stale = await fpl_client._fetch_json(
                test_key, test_key, ttl_seconds=60, retries=1, force_refresh=True
            )
            assert is_stale is True
            assert data["custom_data"] == 42
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_get_manager_team_resilience():
    """Verify get_manager_team is available, fetches picks, and handles parameters gracefully."""

    async def _run():
        sample_picks = {
            "active_chip": "wildcard",
            "entry_history": {"event": 5, "points": 65, "total_points": 320, "bank": 15, "value": 1010},
            "picks": [
                {"element": 10, "position": 1, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
                {"element": 20, "position": 2, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
            ],
        }
        cache_manager.set("entry:99999:event:5:picks", sample_picks, ttl_seconds=300)

        picks, is_stale = await fpl_client.get_manager_team(manager_id=99999, gw=5)
        assert isinstance(picks, SquadPicks)
        assert picks.active_chip == "wildcard"
        assert len(picks.picks) == 2
        assert picks.picks[1].is_captain is True

    asyncio.run(_run())


def test_get_current_and_next_gw_never_crashes(monkeypatch):
    """Verify get_current_and_next_gw never crashes even when API is completely down."""

    async def _run():
        monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(side_effect=RuntimeError("API Outage")))
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        assert curr_gw is None
        assert next_gw is None

    asyncio.run(_run())


# ====================================================================
# PART 2: PRICE CHANGE MICROSTRUCTURE & WILDCARD ISOLATION TESTS
# ====================================================================


def test_wildcard_factor_estimation_peak_windows():
    """Verify wildcard factor accounts for peak windows (GW8-10, GW19-20, GW30-32)."""
    assert estimate_wildcard_factor(4) == 0.10
    assert estimate_wildcard_factor(15) == 0.10

    # Peak autumn international break wildcard window (GW8-10)
    assert estimate_wildcard_factor(8) == 0.22
    assert estimate_wildcard_factor(9) == 0.22
    assert estimate_wildcard_factor(10) == 0.22

    # Autumn shoulder gameweeks
    assert estimate_wildcard_factor(7) == 0.15
    assert estimate_wildcard_factor(11) == 0.15

    # 1st Wildcard expiry window (GW19-20)
    assert estimate_wildcard_factor(19) == 0.20
    assert estimate_wildcard_factor(20) == 0.20

    # 2nd Wildcard peak window (GW30-32)
    assert estimate_wildcard_factor(30) == 0.22
    assert estimate_wildcard_factor(31) == 0.22
    assert estimate_wildcard_factor(32) == 0.22


def test_hours_to_cutoff_calculation():
    """
    Verify calculation of hours remaining to 01:30 UK time.
    Tests various times of day in London.
    """
    tz_london = ZoneInfo("Europe/London")

    # 1. 23:30 London time -> 2.0 hours until 01:30 next morning
    t1 = datetime(2026, 10, 20, 23, 30, tzinfo=tz_london)
    h1 = calculate_hours_to_cutoff(t1)
    assert abs(h1 - 2.0) < 0.01

    # 2. 00:30 London time -> 1.0 hour until 01:30
    t2 = datetime(2026, 10, 21, 0, 30, tzinfo=tz_london)
    h2 = calculate_hours_to_cutoff(t2)
    assert abs(h2 - 1.0) < 0.01

    # 3. 02:00 London time -> 23.5 hours until 01:30 tomorrow
    t3 = datetime(2026, 10, 21, 2, 0, tzinfo=tz_london)
    h3 = calculate_hours_to_cutoff(t3)
    assert abs(h3 - 23.5) < 0.01

    # 4. 13:30 London time -> 12.0 hours until 01:30 tomorrow
    t4 = datetime(2026, 10, 21, 13, 30, tzinfo=tz_london)
    h4 = calculate_hours_to_cutoff(t4)
    assert abs(h4 - 12.0) < 0.01


def test_dynamic_thresholds_asymmetry_and_status():
    """Verify rise vs fall dynamic thresholds and flagged player protection."""
    # Low ownership vs high ownership rise threshold
    t_rise_low = calculate_rise_threshold(1.0, 10_500_000)
    t_rise_high = calculate_rise_threshold(40.0, 10_500_000)
    assert t_rise_low >= 25_000.0
    assert t_rise_high > t_rise_low
    assert t_rise_high <= 125_000.0

    # Fall threshold: fit vs injured (status='i')
    t_fall_fit = calculate_fall_threshold(15.0, status="a", total_players=10_500_000)
    t_fall_injured = calculate_fall_threshold(15.0, status="i", total_players=10_500_000)
    # Injured players have 3x higher fall threshold to reflect flag protection
    assert abs(t_fall_injured - 3.0 * t_fall_fit) < 1.0


def test_price_change_analysis_with_wildcard_isolation_and_velocity():
    """
    Verify complete price change prediction with Wildcard isolation and transfer velocity.
    Compares standard GW vs peak Wildcard GW discounting.
    """
    # Create mock bootstrap static with 2 synthetic players (0 hardcoded real names)
    elem_rising = Element.model_validate(
        {
            "id": 10,
            "web_name": "DynamicAssetA",
            "team": 1,
            "element_type": 3,
            "now_cost": 85,
            "selected_by_percent": "20.0",
            "transfers_in_event": 30000,
            "transfers_out_event": 10000,  # Raw net = +20,000
            "price_change_hourly_rate": 500,  # Velocity = +500/hr
            "status": "a",
        }
    )
    elem_falling = Element.model_validate(
        {
            "id": 20,
            "web_name": "DynamicAssetB",
            "team": 2,
            "element_type": 2,
            "now_cost": 55,
            "selected_by_percent": "15.0",
            "transfers_in_event": 5000,
            "transfers_out_event": 45000,  # Raw net = -40,000
            "price_change_hourly_rate": -800,  # Velocity = -800/hr
            "status": "a",
        }
    )

    gw_event = GameweekEvent(id=9, name="Gameweek 9", deadline_time="2026-10-24T10:00:00Z", is_current=True)
    bootstrap = BootstrapStatic(events=[gw_event], elements=[elem_rising, elem_falling])

    # Case 1: Standard gameweek (10% wildcard discount) at 21:30 London (4.0 hours to 01:30 cutoff)
    tz_london = ZoneInfo("Europe/London")
    now_dt = datetime(2026, 10, 20, 21, 30, tzinfo=tz_london)
    preds_standard = price_change_predictor.analyze_price_changes(bootstrap, now_dt=now_dt, wildcard_factor=0.10)

    p_rise = next(p for p in preds_standard if p["element"] == 10)
    assert p_rise["net_transfers_event"] == 20000
    assert p_rise["wildcard_factor"] == 0.10
    # Effective net = 20,000 * 0.9 = 18,000
    assert p_rise["net_transfers_effective"] == 18000.0
    # Effective velocity = 500 * 0.9 = 450/hr
    assert p_rise["effective_hourly_rate"] == 450.0
    assert p_rise["hours_to_cutoff"] == 4.0
    # Projected net = 18,000 + (450 * 4.0) = 19,800
    assert p_rise["projected_net_at_cutoff"] == 19800.0
    assert p_rise["calibrated_price_forecast"] is True

    # Case 2: Peak Wildcard Gameweek (GW9: 22% wildcard discount)
    preds_peak = price_change_predictor.analyze_price_changes(bootstrap, now_dt=now_dt, wildcard_factor=0.22)
    p_rise_peak = next(p for p in preds_peak if p["element"] == 10)
    assert p_rise_peak["wildcard_factor"] == 0.22
    # Effective net = 20,000 * 0.78 = 15,600
    assert p_rise_peak["net_transfers_effective"] == 15600.0
    # Effective velocity = 500 * 0.78 = 390/hr
    assert p_rise_peak["effective_hourly_rate"] == 390.0
    # Projected net = 15,600 + (390 * 4.0) = 17,160
    assert p_rise_peak["projected_net_at_cutoff"] == 17160.0
    # In peak wildcard week, projected transfers and progress are strictly lower
    assert p_rise_peak["projected_net_at_cutoff"] < p_rise["projected_net_at_cutoff"]
    assert p_rise_peak["progress_pct"] < p_rise["progress_pct"]
    assert p_rise_peak["urgency_score"] < p_rise["urgency_score"]


def test_falling_player_direction_and_bounds():
    """Verify falling player progress, urgency score, and direction bounds."""
    elem_falling = Element.model_validate(
        {
            "id": 99,
            "web_name": "DroppingAsset",
            "team": 3,
            "element_type": 4,
            "now_cost": 100,
            "selected_by_percent": "10.0",
            "transfers_in_event": 1000,
            "transfers_out_event": 75000,  # Raw net = -74,000
            "price_change_hourly_rate": -3000,
            "status": "a",
        }
    )
    gw_event = GameweekEvent(id=4, name="Gameweek 4", deadline_time="2026-09-12T10:00:00Z", is_current=True)
    bootstrap = BootstrapStatic(events=[gw_event], elements=[elem_falling])

    tz_london = ZoneInfo("Europe/London")
    now_dt = datetime(2026, 9, 10, 23, 30, tzinfo=tz_london)  # 2 hours to cutoff
    preds = price_change_predictor.analyze_price_changes(bootstrap, now_dt=now_dt)
    p_fall = preds[0]

    assert p_fall["direction"] in ("FALL_IMMINENT", "LIKELY_FALL")
    assert p_fall["urgency_score"] <= -50.0
    assert p_fall["projected_net_at_cutoff"] < 0
    assert p_fall["target_threshold"] > 0
    assert -100.0 <= p_fall["urgency_score"] <= 100.0
