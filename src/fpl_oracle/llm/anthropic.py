"""
Anthropic Claude LLM Provider with a bounded multi-round tool loop.
"""

import json
import logging
import os
from typing import Any

import httpx

from fpl_oracle.llm.tools import MAX_TOOL_ROUNDS, TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger("fpl_oracle.llm.anthropic")

DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicProvider:
    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = (model or os.environ.get("ANTHROPIC_MODEL", "")).strip() or DEFAULT_MODEL
        self.url = "https://api.anthropic.com/v1/messages"

    async def chat(self, messages: list[dict[str, Any]], system_prompt: str) -> str:
        tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_DEFINITIONS
        ]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        convo = list(messages)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(MAX_TOOL_ROUNDS):
                payload = {
                    "model": self.model,
                    "system": system_prompt,
                    "messages": convo,
                    "tools": tools,
                    "max_tokens": 4096,
                }
                resp = await client.post(self.url, json=payload, headers=headers)
                if resp.status_code != 200:
                    # Do not log or raise API response bodies.
                    raise RuntimeError(f"Anthropic error {resp.status_code}")
                content = resp.json().get("content", [])
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                if not tool_uses:
                    text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
                    if not text:
                        raise RuntimeError("Anthropic returned no answer text")
                    return text
                results = []
                for block in tool_uses:
                    out = await tool_executor.execute(block["name"], block.get("input") or {})
                    results.append(
                        {"type": "tool_result", "tool_use_id": block["id"], "content": json.dumps(out, default=str)}
                    )
                convo += [{"role": "assistant", "content": content}, {"role": "user", "content": results}]
        raise RuntimeError("Anthropic tool-round limit reached")
