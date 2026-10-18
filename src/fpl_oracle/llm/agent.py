"""
FPL Conversational Expert Agent.
Coordinates multi-turn dialogue, tool calling, conversation memory,
and factual citation generation.
"""

import logging

import httpx

from fpl_oracle.data.store import data_store
from fpl_oracle.llm.gemini_requests import GeminiRateLimitError
from fpl_oracle.llm.provider import OFFLINE_MODE_PREFIX, build_system_prompt, get_llm_provider

logger = logging.getLogger("fpl_oracle.llm.agent")

HISTORY_TURNS = 20


def _failure_note(provider: object, error: Exception) -> str:
    name = type(provider).__name__.removesuffix("Provider") or "The AI model"
    if isinstance(error, GeminiRateLimitError):
        reason = "busy"
    elif isinstance(error, (httpx.TimeoutException, TimeoutError)):
        reason = "timed out"
    else:
        reason = type(error).__name__
    return f"{name} unavailable ({reason}). This answer uses your FPL data without {name}.\n\n"


class ExpertAgent:
    async def answer(self, user_message: str, session_id: str = "default") -> str:
        """
        Process user message, retrieve session history, execute LLM or offline expert,
        and save turn in database.
        """
        # Save user message
        data_store.add_chat_message(role="user", content=user_message, session_id=session_id)

        # Recent history ending with this turn. Providers need the first turn to be the user's,
        # and offline banners are not part of what an earlier assistant "said".
        history = data_store.get_chat_history(session_id=session_id, limit=HISTORY_TURNS)
        history = [
            {**m, "content": m["content"].removeprefix(OFFLINE_MODE_PREFIX)} if m["role"] == "assistant" else m
            for m in history[:-1]
        ] + [{"role": "user", "content": user_message}]
        while history[0]["role"] != "user":
            history.pop(0)

        provider = get_llm_provider()
        system_prompt = await build_system_prompt()

        from fpl_oracle.llm.tools import chat_tool_cache

        tools_token = chat_tool_cache.set({})
        try:
            response_text = await provider.chat(messages=history, system_prompt=system_prompt)
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("Provider returned no answer")
        except Exception as e:
            logger.exception(
                "LLM chat failed: provider=%s type=%s; using offline answer", type(provider).__name__, type(e).__name__
            )
            from fpl_oracle.llm.provider import OfflineExpertProvider

            try:
                response_text = _failure_note(provider, e) + await OfflineExpertProvider().chat(
                    messages=history, system_prompt=system_prompt
                )
            except Exception:
                logger.exception("Offline answer failed")
                response_text = "Advice unavailable. Refresh and try again."
        finally:
            chat_tool_cache.reset(tools_token)
        if not isinstance(response_text, str) or not response_text.strip():
            response_text = "No answer. Try again."

        # Save assistant message
        data_store.add_chat_message(role="assistant", content=response_text, session_id=session_id)
        return response_text


expert_agent = ExpertAgent()
