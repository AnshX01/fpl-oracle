"""Shared chat/news Gemini concurrency and server-directed rate-limit cooldown."""

import asyncio
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


class GeminiRateLimitError(RuntimeError):
    pass


class GeminiGate:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.next_allowed = 0.0

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
                delay = max(0.0, self.next_allowed - time.monotonic())
                if delay > 30:
                    raise GeminiRateLimitError("Gemini rate limit cooldown")
                if delay:
                    await asyncio.sleep(delay)
                response = await client.post(url, **kwargs)
                self.next_allowed = time.monotonic() + 2.0
                if response.status_code != 429:
                    return response
                wait = self.retry_seconds(getattr(response, "headers", {}).get("Retry-After"))
                self.next_allowed = time.monotonic() + wait
                if attempt == 1 or wait > 30:
                    raise GeminiRateLimitError("Gemini rate limited")
            raise GeminiRateLimitError("Gemini rate limited")


gemini_gate = GeminiGate()
