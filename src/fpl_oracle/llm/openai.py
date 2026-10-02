"""
OpenAI LLM Provider with tool calling.
"""

import logging

import httpx

from fpl_oracle.llm.tools import TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger("fpl_oracle.llm.openai")

class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.openai.com/v1/chat/completions"

    async def chat(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            }
            for t in TOOL_DEFINITIONS
        ]

        conv = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": self.model,
            "messages": conv,
            "tools": formatted_tools,
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                self.url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")

            res = resp.json()
            msg = res["choices"][0]["message"]

            if msg.get("tool_calls"):
                tool_call = msg["tool_calls"][0]
                fn_name = tool_call["function"]["name"]
                import json
                fn_args = json.loads(tool_call["function"].get("arguments", "{}"))

                tool_res = await tool_executor.execute(fn_name, fn_args)

                conv.append(msg)
                conv.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_res)
                })

                resp2 = await client.post(
                    self.url,
                    json={"model": self.model, "messages": conv, "temperature": 0.2},
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                )
                if resp2.status_code == 200:
                    return resp2.json()["choices"][0]["message"]["content"]

            return msg.get("content", "")
