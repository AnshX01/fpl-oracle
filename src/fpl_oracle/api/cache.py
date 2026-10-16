"""
Caching layer for FPL API responses.
Combines fast in-memory LRU cache with SQLite persistent cache.
Tracks update timestamps and cache age for stale-while-revalidate.
"""

import time
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.data.store import data_store


class CacheManager:
    def __init__(self):
        # key -> {"data": data, "expires_at": float, "updated_at": datetime}
        self._mem_cache: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Any | None:
        res = self.get_with_meta(key)
        return res[0] if res else None

    def get_with_meta(self, key: str) -> tuple[Any, datetime] | None:
        """Return (data, updated_at) if valid cache entry exists."""
        now = time.time()
        # Check in-memory first
        if key in self._mem_cache:
            entry = self._mem_cache[key]
            if entry["expires_at"] > now:
                return entry["data"], entry["updated_at"]
            del self._mem_cache[key]

        # Check SQLite cache
        db_entry = data_store.get_cache_entry(key)
        if db_entry is not None:
            data, upd, exp = db_entry
            # Rehydrate in memory preserving remaining TTL (expires_at) from SQLite
            self._mem_cache[key] = {"data": data, "expires_at": exp.timestamp(), "updated_at": upd}
            return data, upd
        return None

    def get_stale(self, key: str) -> Any | None:
        res = self.get_stale_with_meta(key)
        return res[0] if res else None

    def get_stale_with_meta(self, key: str) -> tuple[Any, datetime] | None:
        """Return (data, updated_at) for stale cached data if API is currently down."""
        if key in self._mem_cache:
            return self._mem_cache[key]["data"], self._mem_cache[key]["updated_at"]
        return data_store.get_stale_cache_entry(key)

    def set(self, key: str, data: Any, ttl_seconds: int):
        now_ts = time.time()
        now_dt = datetime.now(UTC)
        self._mem_cache[key] = {"data": data, "expires_at": now_ts + ttl_seconds, "updated_at": now_dt}
        data_store.set_cache(key, data, ttl_seconds)

    def _candidate_keys(self, key: str) -> list[str]:
        from fpl_oracle.utils.season import derive_season

        season = derive_season(format="hyphen")
        if key.startswith(f"{season}:"):
            return [key]
        return [f"{season}:{key}", key]

    def invalidate(self, key: str):
        for k in self._candidate_keys(key):
            if k in self._mem_cache:
                del self._mem_cache[k]

    def force_expire(self, key: str):
        """Force cache entry to be expired so get() misses but get_stale() succeeds."""
        keys = self._candidate_keys(key)
        for k in keys:
            if k in self._mem_cache:
                del self._mem_cache[k]
        from fpl_oracle.data.store import APICacheEntry

        with data_store.get_session() as session:
            for k in keys:
                entry = session.query(APICacheEntry).filter(APICacheEntry.key == k).first()
                if entry:
                    entry.expires_at = datetime.fromtimestamp(0, tz=UTC)  # type: ignore[assignment]
            session.commit()

    def get_cache_age_seconds(self, key: str) -> float | None:
        """Return age of cached entry in seconds, or None if not cached."""
        now_dt = datetime.now(UTC)
        for k in self._candidate_keys(key):
            if k in self._mem_cache:
                upd = self._mem_cache[k]["updated_at"]
                return max(0.0, (now_dt - upd).total_seconds())
            entry = data_store.get_cache_entry(k) or data_store.get_stale_cache_entry(k)
            if entry:
                upd = entry[1]
                return max(0.0, (now_dt - upd).total_seconds())
        return None

    def get_all_cache_ages(self) -> dict[str, float]:
        """Return age in seconds for key datasets."""
        tracked_keys = ["bootstrap-static", "fixtures:all", "event-status", "team-set-piece-notes"]
        ages = {}
        for k in tracked_keys:
            age = self.get_cache_age_seconds(k)
            if age is not None:
                ages[k] = round(age, 1)
        return ages


cache_manager = CacheManager()
