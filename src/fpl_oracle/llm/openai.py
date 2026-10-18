"""
OpenAI LLM Provider with a bounded multi-round tool loop.
"""

import json
import logging
import os

import httpx

from fpl_oracle.llm.tools import MAX_TOOL_ROUNDS, TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger("fpl_oracle.llm.openai")

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider:
    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = (model or os.environ.get("OPENAI_MODEL", "")).strip() or DEFAULT_MODEL
        self.url = "https://api.openai.com/v1/chat/completions"

    async def chat(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        tools = [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
            }
            for t in TOOL_DEFINITIONS
        ]
        conv = [{"role": "system", "content": system_prompt}] + messages
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await client.post(
                    self.url, json={"model": self.model, "messages": conv, "tools": tools}, headers=headers
                )
                if resp.status_code != 200:
                    # Do not log or raise API response bodies.
                    raise RuntimeError(f"OpenAI error {resp.status_code}")
                msg = resp.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    text = (msg.get("content") or "").strip()
                    if not text:
                        raise RuntimeError("OpenAI returned no answer text")
                    return text
                conv.append(msg)
                for call in calls:
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    out = await tool_executor.execute(call["function"]["name"], args)
                    conv.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out, default=str)})
        raise RuntimeError("OpenAI tool-round limit reached")
