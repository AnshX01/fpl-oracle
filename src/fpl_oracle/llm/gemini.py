"""Gemini chat with grounded, bounded tool rounds and honest failures."""

import json
import logging

import httpx

from fpl_oracle.llm.tools import TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger(__name__)


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def chat(self, messages, system_prompt):
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages
        ]
        declarations = [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in TOOL_DEFINITIONS
        ]
        grounded = False
        async with httpx.AsyncClient(timeout=45.0) as client:
            for _ in range(5):
                payload = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "tools": [{"function_declarations": declarations}],
                    "tool_config": {"function_calling_config": {"mode": "AUTO" if grounded else "ANY"}},
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
                }
                resp = await client.post(self.base_url, json=payload, headers={"x-goog-api-key": self.api_key})
                if resp.status_code != 200:
                    # Do not log API response bodies or secret-bearing URLs.
                    raise RuntimeError(f"Gemini returned status {resp.status_code}")
                data = resp.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    reason = data.get("promptFeedback", {}).get("blockReason", "empty_candidates")
                    raise RuntimeError("Gemini returned no candidate: " + str(reason))
                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                logger.info(
                    "Gemini chat response: round=%s finish_reason=%s function_calls=%s",
                    _ + 1,
                    candidate.get("finishReason", "unspecified"),
                    sum("functionCall" in p for p in parts),
                )
                calls = [p["functionCall"] for p in parts if "functionCall" in p]
                if calls:
                    contents.append({"role": "model", "parts": parts})
                    responses = []
                    for call in calls:
                        if call["name"] not in {d["name"] for d in declarations}:
                            raise RuntimeError("Unknown data request")
                        result = await tool_executor.execute(call["name"], call.get("args", {}))
                        result = json.loads(json.dumps(result, default=str))
                        responses.append({"functionResponse": {"name": call["name"], "response": result}})
                    contents.append({"role": "user", "parts": responses})
                    grounded = True
                    continue
                text = "\n".join(p["text"] for p in parts if "text" in p).strip()
                if grounded and text:
                    return text
                raise RuntimeError("No grounded Gemini answer returned")
        raise RuntimeError("Gemini data-request limit reached")
