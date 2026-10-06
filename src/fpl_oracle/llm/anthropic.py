"""
Anthropic Claude LLM Provider with tool calling.
"""

import logging
from typing import Any

import httpx

from fpl_oracle.llm.tools import TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger("fpl_oracle.llm.anthropic")


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.anthropic.com/v1/messages"

    async def chat(self, messages: list[dict[str, Any]], system_prompt: str) -> str:
        formatted_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_DEFINITIONS
        ]

        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "tools": formatted_tools,
            "max_tokens": 2048,
            "temperature": 0.2,
        }

        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.url, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text}")

            res = resp.json()
            content = res.get("content", [])

            # Check for tool use
            for block in content:
                if block.get("type") == "tool_use":
                    fn_name = block["name"]
                    fn_args = block.get("input", {})
                    tool_res = await tool_executor.execute(fn_name, fn_args)

                    import json

                    followup_messages = list(messages)
                    followup_messages.append({"role": "assistant", "content": content})
                    followup_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": block["id"], "content": json.dumps(tool_res)}
                            ],
                        }
                    )

                    resp2 = await client.post(
                        self.url,
                        json={
                            "model": self.model,
                            "system": system_prompt,
                            "messages": followup_messages,
                            "max_tokens": 2048,
                        },
                        headers=headers,
                    )
                    if resp2.status_code == 200:
                        res2 = resp2.json()
                        text_blocks = [b.get("text", "") for b in res2.get("content", []) if b.get("type") == "text"]
                        return "\n".join(text_blocks)
                elif block.get("type") == "text":
                    return block.get("text", "")

            return "Analysis complete."
