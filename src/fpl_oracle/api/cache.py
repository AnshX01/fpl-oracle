"""
Caching layer for FPL API responses.
Combines fast in-memory LRU cache with SQLite persistent cache.
"""

from typing import Any, Optional, Dict
import time
from fpl_oracle.data.store import data_store
from fpl_oracle.config import SETTINGS

class CacheManager:
    def __init__(self):
        self._mem_cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        # Check in-memory first
        if key in self._mem_cache:
            entry = self._mem_cache[key]
            if entry["expires_at"] > now:
                return entry["data"]
            del self._mem_cache[key]
        
        # Check SQLite cache
        data = data_store.get_cache(key)
        if data is not None:
            # Rehydrate in memory
            ttl = SETTINGS.get("cache", {}).get("bootstrap_ttl_seconds", 300)
            self._mem_cache[key] = {"data": data, "expires_at": now + ttl}
            return data
        return None

    def get_stale(self, key: str) -> Optional[Any]:
        """Return stale cached data if API is currently down."""
        if key in self._mem_cache:
            return self._mem_cache[key]["data"]
        return data_store.get_stale_cache(key)

    def set(self, key: str, data: Any, ttl_seconds: int):
        now = time.time()
        self._mem_cache[key] = {"data": data, "expires_at": now + ttl_seconds}
        data_store.set_cache(key, data, ttl_seconds)

    def invalidate(self, key: str):
        if key in self._mem_cache:
            del self._mem_cache[key]

cache_manager = CacheManager()
