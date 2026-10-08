import asyncio
import logging
from unittest.mock import AsyncMock

from fpl_oracle.llm.agent import ExpertAgent
from fpl_oracle.llm.tools import chat_tool_cache, tool_executor


def test_chat_has_no_whole_tool_timeout_or_duplicate_tool_work(monkeypatch):
    from fpl_oracle.data.store import data_store
    from fpl_oracle.llm import agent

    monkeypatch.setattr(data_store, "add_chat_message", lambda **kw: None)
    monkeypatch.setattr(data_store, "get_chat_history", lambda **kw: [])
    execute = AsyncMock(return_value={"plan": "keep full horizon"})
    monkeypatch.setattr(tool_executor, "_execute", execute)

    class Provider:
        async def chat(self, **kw):
            # The provider's network timeouts are separate. Local tools must not
            # be wrapped in the old 80-second conversation-wide wait_for.
            first = await tool_executor.execute("optimise_transfers", {})
            await asyncio.sleep(0)
            second = await tool_executor.execute("optimise_transfers", {})
            assert first == second
            return "Save the transfer."

    monkeypatch.setattr(agent, "get_llm_provider", lambda: Provider())

    async def forbidden_wait_for(*a, **kw):
        raise AssertionError("Whole-chat timeout must not cancel local planning")

    monkeypatch.setattr(asyncio, "wait_for", forbidden_wait_for)
    assert asyncio.run(ExpertAgent().answer("transfers?")) == "Save the transfer."
    execute.assert_awaited_once()
    assert chat_tool_cache.get() is None


def test_empty_timeout_logs_type_and_trace_then_fallback(monkeypatch, caplog):
    from fpl_oracle.data.store import data_store
    from fpl_oracle.llm import agent, provider

    monkeypatch.setattr(data_store, "add_chat_message", lambda **kw: None)
    monkeypatch.setattr(data_store, "get_chat_history", lambda **kw: [])

    class Provider:
        async def chat(self, **kw):
            raise TimeoutError()

    monkeypatch.setattr(agent, "get_llm_provider", lambda: Provider())
    monkeypatch.setattr(provider.OfflineExpertProvider, "chat", AsyncMock(return_value="Save the transfer."))
    with caplog.at_level(logging.ERROR):
        assert asyncio.run(ExpertAgent().answer("transfers?")) == "Save the transfer."
    assert "type=TimeoutError" in caplog.text
    assert "Traceback" in caplog.text


def test_rate_limited_offline_reply_is_labelled(monkeypatch):
    from fpl_oracle.data.store import data_store
    from fpl_oracle.llm import agent, provider
    from fpl_oracle.llm.gemini_requests import GeminiRateLimitError

    monkeypatch.setattr(data_store, "add_chat_message", lambda **kw: None)
    monkeypatch.setattr(data_store, "get_chat_history", lambda **kw: [])

    class Provider:
        async def chat(self, **kw):
            raise GeminiRateLimitError("busy")

    monkeypatch.setattr(agent, "get_llm_provider", lambda: Provider())
    monkeypatch.setattr(provider.OfflineExpertProvider, "chat", AsyncMock(return_value="Captain Saka."))
    answer = asyncio.run(ExpertAgent().answer("captain?"))
    assert "without Gemini" in answer
    assert "Captain Saka" in answer
