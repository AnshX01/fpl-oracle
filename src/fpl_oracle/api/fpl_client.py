"""
Async FPL API Client.
Features:
- Concurrency limiting via asyncio.Semaphore
- Exponential backoff & jitter on 429/5xx status codes
- In-memory & SQLite cache
- Snapshot persistence for offline ML retraining
- Graceful fallback with staleness detection
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
import httpx
from datetime import datetime, timezone

from fpl_oracle.config import SETTINGS
from fpl_oracle.api.models import (
    BootstrapStatic, Fixture, ElementSummary, ManagerEntry,
    ManagerHistory, SquadPicks, TransferHistoryItem, ClassicLeagueResponse
)
from fpl_oracle.api.cache import cache_manager
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
        self.semaphore = asyncio.Semaphore(int(fpl_cfg.get("max_concurrency", 5)))
        self.rate_delay = float(fpl_cfg.get("rate_limit_delay_seconds", 0.1))
        self.is_stale_mode = False
        self.last_sync_time: Optional[datetime] = None

    async def _fetch_json(self, endpoint: str, cache_key: str, ttl_seconds: int, retries: int = 3) -> Tuple[Dict[str, Any], bool]:
        """
        Fetch JSON from endpoint with caching, retry logic and fallback.
        Returns (data, is_stale).
        """
        # 1. Check cache first
        cached = cache_manager.get(cache_key)
        if cached is not None:
            return cached, False

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        backoff = 1.0

        for attempt in range(retries):
            async with self.semaphore:
                if self.rate_delay > 0:
                    await asyncio.sleep(self.rate_delay)
                try:
                    async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            data = resp.json()
                            # Cache in store
                            cache_manager.set(cache_key, data, ttl_seconds)
                            data_store.save_snapshot(endpoint, data)
                            self.last_sync_time = datetime.now(timezone.utc)
                            self.is_stale_mode = False
                            return data, False
                        elif resp.status_code in [429, 500, 502, 503, 504]:
                            logger.warning(f"FPL API error {resp.status_code} for {url}. Attempt {attempt+1}/{retries}")
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                        else:
                            logger.error(f"FPL API client error {resp.status_code} for {url}: {resp.text[:100]}")
                            break
                except Exception as e:
                    logger.warning(f"Network error accessing {url}: {e}. Attempt {attempt+1}/{retries}")
                    await asyncio.sleep(backoff)
                    backoff *= 2.0

        # Fallback to stale cache if API failed
        stale = cache_manager.get_stale(cache_key)
        if stale is not None:
            logger.warning(f"Using stale cached data for {cache_key}")
            self.is_stale_mode = True
            return stale, True

        raise RuntimeError(f"Failed to fetch data from {url} and no cached fallback exists.")

    async def get_bootstrap_static(self, force_refresh: bool = False) -> Tuple[BootstrapStatic, bool]:
        cache_key = "bootstrap-static"
        if force_refresh:
            cache_manager.invalidate(cache_key)
        ttl = SETTINGS.get("cache", {}).get("bootstrap_ttl_seconds", 300)
        raw, is_stale = await self._fetch_json("bootstrap-static/", cache_key, ttl)
        return BootstrapStatic.model_validate(raw), is_stale

    async def get_fixtures(self, event: Optional[int] = None, force_refresh: bool = False) -> Tuple[List[Fixture], bool]:
        cache_key = f"fixtures:event:{event}" if event else "fixtures:all"
        if force_refresh:
            cache_manager.invalidate(cache_key)
        endpoint = f"fixtures/?event={event}" if event else "fixtures/"
        ttl = SETTINGS.get("cache", {}).get("fixtures_ttl_seconds", 900)
        raw, is_stale = await self._fetch_json(endpoint, cache_key, ttl)
        fixtures = [Fixture.model_validate(item) for item in raw]
        return fixtures, is_stale

    async def get_element_summary(self, element_id: int) -> Tuple[ElementSummary, bool]:
        cache_key = f"element-summary:{element_id}"
        ttl = SETTINGS.get("cache", {}).get("summary_ttl_seconds", 1800)
        raw, is_stale = await self._fetch_json(f"element-summary/{element_id}/", cache_key, ttl)
        return ElementSummary.model_validate(raw), is_stale

    async def get_live_gameweek(self, gw: int) -> Tuple[Dict[str, Any], bool]:
        cache_key = f"event:{gw}:live"
        ttl = SETTINGS.get("cache", {}).get("live_gw_ttl_seconds", 60)
        return await self._fetch_json(f"event/{gw}/live/", cache_key, ttl)

    async def get_manager_entry(self, manager_id: int) -> Tuple[ManagerEntry, bool]:
        cache_key = f"entry:{manager_id}"
        ttl = 300
        raw, is_stale = await self._fetch_json(f"entry/{manager_id}/", cache_key, ttl)
        return ManagerEntry.model_validate(raw), is_stale

    async def get_manager_history(self, manager_id: int) -> Tuple[ManagerHistory, bool]:
        cache_key = f"entry:{manager_id}:history"
        ttl = 300
        raw, is_stale = await self._fetch_json(f"entry/{manager_id}/history/", cache_key, ttl)
        return ManagerHistory.model_validate(raw), is_stale

    async def get_manager_picks(self, manager_id: int, gw: int) -> Tuple[SquadPicks, bool]:
        cache_key = f"entry:{manager_id}:event:{gw}:picks"
        ttl = 300
        raw, is_stale = await self._fetch_json(f"entry/{manager_id}/event/{gw}/picks/", cache_key, ttl)
        return SquadPicks.model_validate(raw), is_stale

    async def get_manager_transfers(self, manager_id: int) -> Tuple[List[TransferHistoryItem], bool]:
        cache_key = f"entry:{manager_id}:transfers"
        ttl = 300
        raw, is_stale = await self._fetch_json(f"entry/{manager_id}/transfers/", cache_key, ttl)
        transfers = [TransferHistoryItem.model_validate(item) for item in raw]
        return transfers, is_stale

    async def get_classic_league_standings(self, league_id: int, page: int = 1) -> Tuple[ClassicLeagueResponse, bool]:
        cache_key = f"league-classic:{league_id}:page:{page}"
        ttl = 600
        raw, is_stale = await self._fetch_json(f"leagues-classic/{league_id}/standings/?page_standings={page}", cache_key, ttl)
        return ClassicLeagueResponse.model_validate(raw), is_stale

    async def get_event_status(self) -> Tuple[Dict[str, Any], bool]:
        cache_key = "event-status"
        ttl = 60
        return await self._fetch_json("event-status/", cache_key, ttl)

    async def get_dream_team(self, gw: int) -> Tuple[Dict[str, Any], bool]:
        cache_key = f"dream-team:{gw}"
        ttl = 3600
        return await self._fetch_json(f"dream-team/{gw}/", cache_key, ttl)

    async def get_set_piece_notes(self) -> Tuple[Dict[str, Any], bool]:
        cache_key = "team-set-piece-notes"
        ttl = 3600
        return await self._fetch_json("team/set-piece-notes/", cache_key, ttl)

    async def get_current_and_next_gw(self) -> Tuple[Optional[int], Optional[int]]:
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
