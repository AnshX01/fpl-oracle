"""Shared chat/news Gemini concurrency and server-directed rate-limit cooldown."""

import asyncio
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


class GeminiRateLimitError(RuntimeError):
    pass


class GeminiGate:
    def __init__(self):
        self.lock = asyncio.Lock()
        self._next_allowed: dict[str, float] = {}

    @property
    def next_allowed(self):
        # Backwards compatibility: return max timestamp
        if not self._next_allowed:
            return 0.0
        return max(self._next_allowed.values())

    @next_allowed.setter
    def next_allowed(self, value):
        self._next_allowed["default"] = value

    @staticmethod
    def retry_seconds(value):
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    return max(0.0, (date - datetime.now(UTC)).total_seconds())
                except (ValueError, TypeError):
                    pass
        return 5.0

    async def post(self, client, url, **kwargs):
        # One shared in-flight Gemini request. Local optimiser tools do not hold
        # this lock. A cooldown applies to both chat and news extraction.
        async with self.lock:
            for attempt in range(2):
                target_allowed = max(self._next_allowed.get(url, 0.0), self._next_allowed.get("default", 0.0))
                delay = max(0.0, target_allowed - time.monotonic())
                if delay > 30:
                    raise GeminiRateLimitError("Gemini rate limit cooldown")
                if delay:
                    await asyncio.sleep(delay)
                try:
                    response = await client.post(url, **kwargs)
                except httpx.TimeoutException:
                    if attempt == 1:
                        raise
                    await asyncio.sleep(1.0)
                    continue
                self._next_allowed[url] = time.monotonic() + 2.0
                if response.status_code == 429:
                    wait = self.retry_seconds(getattr(response, "headers", {}).get("Retry-After"))
                    self._next_allowed[url] = time.monotonic() + wait
                    if attempt == 1 or wait > 30:
                        raise GeminiRateLimitError("Gemini rate limited")
                    continue
                if response.status_code in (503, 504) and attempt == 0:
                    await asyncio.sleep(2.0)
                    continue
                return response


gemini_gate = GeminiGate()
