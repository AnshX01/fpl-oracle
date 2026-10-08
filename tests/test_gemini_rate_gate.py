import asyncio
from types import SimpleNamespace

import pytest

from fpl_oracle.llm.gemini_requests import GeminiGate, GeminiRateLimitError


def test_retry_after_zero_retries_once():
    async def run():
        gate = GeminiGate()
        calls = []

        class Client:
            async def post(self, *a, **kw):
                calls.append(1)
                return SimpleNamespace(status_code=429 if len(calls) == 1 else 200, headers={"Retry-After": "0"})

        assert (await gate.post(Client(), "https://example.invalid")).status_code == 200
        assert len(calls) == 2

    asyncio.run(run())


def test_long_retry_after_sets_shared_cooldown_without_hammering():
    async def run():
        gate = GeminiGate()
        calls = []

        class Client:
            async def post(self, *a, **kw):
                calls.append(1)
                return SimpleNamespace(status_code=429, headers={"Retry-After": "120"})

        for _ in range(2):
            with pytest.raises(GeminiRateLimitError):
                await gate.post(Client(), "https://example.invalid")
        assert len(calls) == 1

    asyncio.run(run())


def test_repeated_429_is_bounded():
    async def run():
        calls = []

        class Client:
            async def post(self, *a, **kw):
                calls.append(1)
                return SimpleNamespace(status_code=429, headers={"Retry-After": "0"})

        with pytest.raises(GeminiRateLimitError):
            await GeminiGate().post(Client(), "https://example.invalid")
        assert len(calls) == 2

    asyncio.run(run())


def test_shared_gate_caps_concurrent_chat_and_news(monkeypatch):
    async def run():
        gate = GeminiGate()
        active = 0
        maximum = 0

        async def yield_only(delay):
            await original_sleep(0)

        original_sleep = asyncio.sleep
        monkeypatch.setattr(asyncio, "sleep", yield_only)

        class Client:
            async def post(self, *a, **kw):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                await original_sleep(0.001)
                active -= 1
                return SimpleNamespace(status_code=200, headers={})

        await asyncio.gather(*(gate.post(Client(), "https://example.invalid") for _ in range(4)))
        assert maximum == 1

    asyncio.run(run())
