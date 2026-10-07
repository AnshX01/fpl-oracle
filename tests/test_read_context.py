"""Concurrent refresh cannot alter repeated upstream reads within one request."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.api.fpl_client import FPLClient
from fpl_oracle.api.read_context import read_context


def test_pinned_reads_are_copies_and_keep_original_revision(monkeypatch):
    client = FPLClient()
    monkeypatch.setattr(client, "_do_fetch_json", AsyncMock(return_value=({"ids": [1]}, False)))

    async def run():
        token = read_context.set({})
        try:
            first, _ = await client._fetch_json("test/", "unique-test-context", 1, force_refresh=True)
            first["ids"].append(2)
            client._cache_timestamps["unique-test-context"] = datetime.now(UTC)
            second, _ = await client._fetch_json("test/", "unique-test-context", 1)
            assert second["ids"] == [1]
            assert read_context.get()["unique-test-context"][2] is None
        finally:
            read_context.reset(token)

    asyncio.run(run())


@pytest.mark.parametrize(
    "path,method",
    [("/api/squad", "GET"), ("/api/decision-card", "GET"), ("/api/transfers", "POST"), ("/api/optimize", "POST")],
)
def test_http_guard_rejects_revision_change(monkeypatch, path, method):
    from starlette.requests import Request
    from starlette.responses import Response

    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.api.read_context import remember
    from fpl_oracle.server.main import upstream_snapshot_guard

    async def run():
        async def changed(request):
            stamp = datetime(2026, 10, 1, tzinfo=UTC)
            remember("guard-test", {}, False, stamp)
            fpl_client._cache_timestamps["guard-test"] = datetime.now(UTC)
            return Response("{}", media_type="application/json")

        request = Request({"type": "http", "path": path, "method": method, "headers": []})
        response = await upstream_snapshot_guard(request, changed)
        assert response.status_code == 409
        assert b"Upstream data changed" in response.body
        fpl_client._cache_timestamps.pop("guard-test", None)

    asyncio.run(run())
