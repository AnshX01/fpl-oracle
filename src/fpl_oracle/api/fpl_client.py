"""
Async FPL API Client.
Features:
- Persistent httpx.AsyncClient with connection pooling
- Concurrency limiting via asyncio.Semaphore
- Exponential backoff with random jitter on 429/5xx status codes
- Request coalescing (deduplication of in-flight requests)
- In-memory & SQLite cache with stale-while-revalidate
- Snapshot persistence for offline ML retraining and price modeling
- Pluggable transport for fault-injection testing
- Polite batch fetching for element summaries without thundering herd
"""

import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import Any

import httpx

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.models import (
    BootstrapStatic,
    ClassicLeagueResponse,
    ElementSummary,
    Fixture,
    ManagerEntry,
    ManagerHistory,
    SquadPicks,
    TransferHistoryItem,
)
from fpl_oracle.config import SETTINGS
from fpl_oracle.data.store import data_store

logger = logging.getLogger("fpl_oracle.api")
logging.basicConfig(level=logging.INFO)


class FPLClient:
    def __init__(self):
        fpl_cfg = SETTINGS.get("fpl", {})
        self.base_url = fpl_cfg.get("base_url", "https://fantasy.premierleague.com/api").rstrip("/")
        self.headers = {
            "User-Agent": fpl_cfg.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
            "Accept": "application/json",
        }
        self.timeout = float(fpl_cfg.get("request_timeout_seconds", 15))
        self.max_concurrency = int(fpl_cfg.get("max_concurrency", 5))
        self.semaphore = None
        self._semaphore_loop = None
        self._client_loop = None
        self.rate_delay = float(fpl_cfg.get("rate_limit_delay_seconds", 0.05))
        self.revision = 0
        self.is_stale_mode = False
        self.last_sync_time: datetime | None = None
        self._cache_timestamps: dict[str, datetime] = {}

        # Request coalescing: endpoint -> in-flight asyncio.Task
        self._in_flight: dict[str, asyncio.Task] = {}

        # Persistent client session
        self._custom_transport: httpx.AsyncBaseTransport | None = None
        self._client: httpx.AsyncClient | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self.semaphore is None or self._semaphore_loop != loop:
            self._semaphore_loop = loop
            self.semaphore = asyncio.Semaphore(self.max_concurrency)
        return self.semaphore

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or initialize persistent AsyncClient with connection pooling."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._client is None or self._client.is_closed or self._client_loop != loop:
            self._client_loop = loop
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            self._client = httpx.AsyncClient(
                headers=self.headers, timeout=self.timeout, limits=limits, transport=self._custom_transport
            )
        return self._client

    def set_transport(self, transport: httpx.AsyncBaseTransport | None):
        """Inject a custom transport (used for fault-injection and offline testing)."""
        self._custom_transport = transport
        if self._client is not None and not self._client.is_closed:
            asyncio.create_task(self._client.aclose())
        self._client = None

    def reset_transport(self):
        """Reset transport to default live networking."""
        self.set_transport(None)

    async def aclose(self):
        """Close persistent HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def get_data_as_of(self, cache_key: str) -> str | None:
        """Return ISO timestamp of when data for key was fetched or last updated."""
        dt = self._cache_timestamps.get(cache_key)
        return dt.isoformat() if dt else None

    async def _fetch_json(
        self, endpoint: str, cache_key: str, ttl_seconds: int, retries: int = 3, force_refresh: bool = False
    ) -> tuple[dict[str, Any], bool]:
        """
        Fetch JSON from endpoint with coalescing, retries with jitter, and stale cache fallback.
        Returns (data, is_stale).
        """
        from fpl_oracle.api.read_context import recalled, remember

        if not hasattr(self, "_source_requests"):
            self._source_requests = {}
        self._source_requests[cache_key] = (endpoint, ttl_seconds)

        pinned = recalled(cache_key)
        if pinned is not None and not force_refresh:
            return pinned
        if not force_refresh:
            cache_res = cache_manager.get_with_meta(cache_key)
            if cache_res is not None:
                data, upd = cache_res
                self._cache_timestamps[cache_key] = upd
                remember(cache_key, data, False, upd)
                return data, False

        # Request coalescing: if an identical request is already in flight, await it
        if cache_key in self._in_flight:
            try:
                result = await self._in_flight[cache_key]
                remember(cache_key, result[0], result[1], self._cache_timestamps.get(cache_key))
                return result
            except Exception:
                pass

        # Create coalesced task
        task = asyncio.create_task(self._do_fetch_json(endpoint, cache_key, ttl_seconds, retries))
        self._in_flight[cache_key] = task
        try:
            res = await task
            remember(cache_key, res[0], res[1], self._cache_timestamps.get(cache_key))
            return res
        finally:
            self._in_flight.pop(cache_key, None)

    async def _do_fetch_json(
        self, endpoint: str, cache_key: str, ttl_seconds: int, retries: int
    ) -> tuple[dict[str, Any], bool]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        backoff = 1.0

        for attempt in range(retries):
            async with self._get_semaphore():
                if self.rate_delay > 0:
                    await asyncio.sleep(self.rate_delay)
                try:
                    client = self.get_http_client()
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        now_dt = datetime.now(UTC)
                        # Save in cache and snapshot store
                        cache_manager.set(cache_key, data, ttl_seconds)
                        data_store.save_snapshot(endpoint, data)
                        self.revision += 1
                        self.last_sync_time = now_dt
                        self._cache_timestamps[cache_key] = now_dt
                        self.is_stale_mode = False
                        return data, False
                    elif resp.status_code in [429, 500, 502, 503, 504]:
                        logger.warning(f"FPL API error {resp.status_code} for {url}. Attempt {attempt + 1}/{retries}")
                        # Exponential backoff with random jitter (0.8x to 1.2x)
                        jittered_delay = backoff * (0.8 + 0.4 * random.random())
                        await asyncio.sleep(jittered_delay)
                        backoff *= 2.0
                    else:
                        logger.error(f"FPL API client error {resp.status_code} for {url}: {resp.text[:100]}")
                        break
                except Exception as e:
                    logger.warning(f"Network error accessing {url}: {e}. Attempt {attempt + 1}/{retries}")
                    jittered_delay = backoff * (0.8 + 0.4 * random.random())
                    await asyncio.sleep(jittered_delay)
                    backoff *= 2.0

        # Fallback 1: Stale cache entry
        stale_entry = cache_manager.get_stale_with_meta(cache_key)
        if stale_entry is not None:
            data, upd = stale_entry
            logger.warning(f"Using stale cached data for {cache_key} (timestamp: {upd})")
            self.is_stale_mode = True
            self._cache_timestamps[cache_key] = upd
            return data, True

        # Fallback 2: Latest raw snapshot from SQLite
        snapshot_entry = data_store.get_latest_snapshot(endpoint)
        if snapshot_entry is not None:
            data, upd = snapshot_entry
            logger.warning(f"Using historic raw snapshot for {endpoint} (timestamp: {upd})")
            self.is_stale_mode = True
            self._cache_timestamps[cache_key] = upd
            return data, True

        raise RuntimeError(f"FPL API unreachable for {url} and no cached fallback or snapshot exists.")

    async def get_bootstrap_static(self, force_refresh: bool = False) -> tuple[BootstrapStatic, bool]:
        cache_key = "bootstrap-static"
        ttl = SETTINGS.get("cache", {}).get("bootstrap_ttl_seconds", 300)
        raw, is_stale = await self._fetch_json("bootstrap-static/", cache_key, ttl, force_refresh=force_refresh)
        return BootstrapStatic.model_validate(raw), is_stale

    async def get_fixtures(self, event: int | None = None, force_refresh: bool = False) -> tuple[list[Fixture], bool]:
        cache_key = f"fixtures:event:{event}" if event else "fixtures:all"
        endpoint = f"fixtures/?event={event}" if event else "fixtures/"
        ttl = SETTINGS.get("cache", {}).get("fixtures_ttl_seconds", 900)
        raw, is_stale = await self._fetch_json(endpoint, cache_key, ttl, force_refresh=force_refresh)
        fixtures = [Fixture.model_validate(item) for item in raw]
        return fixtures, is_stale

    async def get_element_summary(self, element_id: int, force_refresh: bool = False) -> tuple[ElementSummary, bool]:
        cache_key = f"element-summary:{element_id}"
        ttl = SETTINGS.get("cache", {}).get("summary_ttl_seconds", 1800)
        raw, is_stale = await self._fetch_json(
            f"element-summary/{element_id}/", cache_key, ttl, force_refresh=force_refresh
        )
        return ElementSummary.model_validate(raw), is_stale

    async def get_element_summaries_batch(
        self, element_ids: list[int], batch_size: int = 8, polite_delay: float = 0.05
    ) -> dict[int, tuple[ElementSummary, bool]]:
        """Polite batch fetch of player summaries to prevent rate limits."""
        results: dict[int, tuple[ElementSummary, bool]] = {}
        for i in range(0, len(element_ids), batch_size):
            chunk = element_ids[i : i + batch_size]
            tasks = [self.get_element_summary(eid) for eid in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            for eid, res in zip(chunk, chunk_results, strict=False):
                if isinstance(res, BaseException):
                    logger.warning(f"Failed element summary for {eid}: {res}")
                elif isinstance(res, tuple):
                    results[eid] = res
            if polite_delay > 0 and i + batch_size < len(element_ids):
                await asyncio.sleep(polite_delay)
        return results

    async def get_live_gameweek(self, gw: int, force_refresh: bool = False) -> tuple[dict[str, Any], bool]:
        cache_key = f"event:{gw}:live"
        ttl = SETTINGS.get("cache", {}).get("live_gw_ttl_seconds", 60)
        return await self._fetch_json(f"event/{gw}/live/", cache_key, ttl, force_refresh=force_refresh)

    async def get_manager_entry(self, manager_id: int, force_refresh: bool = False) -> tuple[ManagerEntry, bool]:
        cache_key = f"entry:{manager_id}"
        ttl = 300
        raw, is_stale = await self._fetch_json(f"entry/{manager_id}/", cache_key, ttl, force_refresh=force_refresh)
        return ManagerEntry.model_validate(raw), is_stale

    async def get_manager_history(self, manager_id: int, force_refresh: bool = False) -> tuple[ManagerHistory, bool]:
        cache_key = f"entry:{manager_id}:history"
        ttl = 300
        raw, is_stale = await self._fetch_json(
            f"entry/{manager_id}/history/", cache_key, ttl, force_refresh=force_refresh
        )
        return ManagerHistory.model_validate(raw), is_stale

    async def get_manager_picks(self, manager_id: int, gw: int, force_refresh: bool = False) -> tuple[SquadPicks, bool]:
        cache_key = f"entry:{manager_id}:event:{gw}:picks"
        ttl = 300
        raw, is_stale = await self._fetch_json(
            f"entry/{manager_id}/event/{gw}/picks/", cache_key, ttl, force_refresh=force_refresh
        )
        return SquadPicks.model_validate(raw), is_stale

    async def get_manager_transfers(
        self, manager_id: int, force_refresh: bool = False
    ) -> tuple[list[TransferHistoryItem], bool]:
        cache_key = f"entry:{manager_id}:transfers"
        ttl = 300
        raw, is_stale = await self._fetch_json(
            f"entry/{manager_id}/transfers/", cache_key, ttl, force_refresh=force_refresh
        )
        transfers = [TransferHistoryItem.model_validate(item) for item in raw]
        return transfers, is_stale

    async def get_classic_league_standings(self, league_id: int, page: int = 1) -> tuple[ClassicLeagueResponse, bool]:
        cache_key = f"league-classic:{league_id}:page:{page}"
        ttl = 600
        raw, is_stale = await self._fetch_json(
            f"leagues-classic/{league_id}/standings/?page_standings={page}", cache_key, ttl
        )
        return ClassicLeagueResponse.model_validate(raw), is_stale

    async def get_event_status(self) -> tuple[dict[str, Any], bool]:
        cache_key = "event-status"
        ttl = 60
        return await self._fetch_json("event-status/", cache_key, ttl)

    async def get_dream_team(self, gw: int) -> tuple[dict[str, Any], bool]:
        cache_key = f"dream-team:{gw}"
        ttl = 3600
        return await self._fetch_json(f"dream-team/{gw}/", cache_key, ttl)

    async def get_set_piece_notes(self) -> tuple[dict[str, Any], bool]:
        cache_key = "team-set-piece-notes"
        ttl = 3600
        return await self._fetch_json("team/set-piece-notes/", cache_key, ttl)

    async def get_current_and_next_gw(self) -> tuple[int | None, int | None]:
        """Always derive current and next gameweeks live from API."""
        bootstrap, _ = await self.get_bootstrap_static()
        curr_gw = None
        next_gw = None
        for ev in bootstrap.events:
            if ev.is_current:
                curr_gw = ev.id
            if ev.is_next:
                next_gw = ev.id
        if next_gw is None and curr_gw is not None and curr_gw < 38:
            next_gw = curr_gw + 1
        return curr_gw, next_gw


fpl_client = FPLClient()
