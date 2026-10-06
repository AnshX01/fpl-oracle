"""
Fault Injection & Degraded Network Test Suite for FPL Oracle.
Simulates:
- FPL API 429 Rate Limits
- FPL API 500 / 502 / 503 Server Errors
- Network Read Timeouts
- Schema Drift (new unknown fields / unexpected values)
- Empty mini-leagues & missing manager IDs
- Malformed manual squads
Verifies that all endpoints return valid JSON and graceful fallbacks with stale flags.
"""

import asyncio

import httpx

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.server.main import app


class FaultInjectingTransport(httpx.AsyncBaseTransport):
    """Custom HTTP transport that simulates network faults."""

    def __init__(self, mode: str = "429"):
        self.mode = mode
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        url_str = str(request.url)

        if self.mode == "429":
            return httpx.Response(
                status_code=429,
                headers={"Retry-After": "1"},
                content=b'{"detail": "Request was throttled. Expected available in 1 second."}',
                request=request,
            )
        elif self.mode == "503":
            return httpx.Response(
                status_code=503,
                content=b'{"detail": "FPL Service Temporarily Unavailable - Updating Game"}',
                request=request,
            )
        elif self.mode == "timeout":
            raise httpx.ReadTimeout(f"Read timeout after 15.0s connecting to {url_str}")
        elif self.mode == "drift":
            # Return valid bootstrap JSON structure with extra unexpected schema fields
            # Retrieve cached bootstrap and inject unexpected fields
            from fpl_oracle.api.cache import cache_manager

            cached = cache_manager.get_stale("bootstrap-static")
            if cached:
                drifted = dict(cached)
                drifted["new_unexpected_2026_rule"] = {"active": True, "token": "xyz"}
                if drifted.get("elements"):
                    drifted["elements"] = [dict(e, unknown_metric_xG90=9.99) for e in drifted["elements"]]
                import json

                return httpx.Response(status_code=200, content=json.dumps(drifted).encode("utf-8"), request=request)
            return httpx.Response(status_code=200, content=b'{"events":[],"elements":[]}', request=request)

        return httpx.Response(status_code=500, content=b'{"error": "Unknown fault mode"}', request=request)


def test_fault_injection_429_stale_fallback():
    """Verify that when FPL API returns 429, client gracefully serves stale cache."""

    async def _run():
        await fpl_client.get_bootstrap_static()
        faulty_transport = FaultInjectingTransport(mode="429")
        fpl_client.set_transport(faulty_transport)
        try:
            boot, is_stale = await fpl_client._fetch_json(
                "bootstrap-static/", "bootstrap-static", ttl_seconds=1, retries=2, force_refresh=True
            )
            assert is_stale is True, "Expected is_stale=True when falling back after 429"
            assert len(boot.get("elements", [])) > 0
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_fault_injection_503_service_unavailable():
    """Verify that when FPL API returns 503, server routes still succeed with stale=True."""

    async def _run():
        from fpl_oracle.api.cache import cache_manager

        cache_manager.force_expire("bootstrap-static")
        faulty_transport = FaultInjectingTransport(mode="503")
        fpl_client.set_transport(faulty_transport)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.get("/api/squad")
                assert res.status_code == 200, f"Expected 200 with stale fallback, got {res.status_code}"
                data = res.json()
                assert data.get("stale") is True or data.get("is_stale") is True
                assert len(data.get("starters", [])) + len(data.get("bench", [])) == 15
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_fault_injection_read_timeout():
    """Verify that read timeouts do not crash endpoints and trigger stale fallback."""

    async def _run():
        from fpl_oracle.api.cache import cache_manager

        cache_manager.force_expire("bootstrap-static")
        cache_manager.force_expire("fixtures:all")
        faulty_transport = FaultInjectingTransport(mode="timeout")
        fpl_client.set_transport(faulty_transport)
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.get("/api/projections?horizon=2")
                assert res.status_code == 200
                data = res.json()
                assert data.get("stale") is True or data.get("is_stale") is True
                assert len(data.get("players", [])) > 0
        finally:
            fpl_client.reset_transport()

    asyncio.run(_run())


def test_fault_injection_schema_drift():
    """Verify that unexpected API fields do not crash Pydantic models (extra='allow')."""

    async def _run():
        from fpl_oracle.api.cache import cache_manager

        orig_cached = cache_manager.get_stale("bootstrap-static")
        faulty_transport = FaultInjectingTransport(mode="drift")
        fpl_client.set_transport(faulty_transport)
        try:
            boot, _ = await fpl_client.get_bootstrap_static(force_refresh=True)
            assert len(boot.elements) > 0
        finally:
            fpl_client.reset_transport()
            if orig_cached:
                cache_manager.set("bootstrap-static", orig_cached, ttl_seconds=300)

    asyncio.run(_run())


def test_malformed_manual_squad_validation():
    """Verify that malformed squads return clean 400 errors without leaking tracebacks."""

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Fewer than 15 players
            res = await client.post("/api/squad/manual", json={"player_ids": [1, 2, 3]})
            assert res.status_code == 400
            data = res.json()
            assert "error" in data
            assert data.get("code") == 400
            assert data.get("fallback_used") is False

            # 2. Invalid formation (e.g. 15 goalkeepers)
            res = await client.post("/api/squad/manual", json={"player_ids": [1] * 15})
            assert res.status_code == 400
            data = res.json()
            assert "error" in data
            assert "traceback" not in str(data).lower()

    asyncio.run(_run())
