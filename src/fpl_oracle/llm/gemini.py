"""Gemini chat with grounded, bounded tool rounds and honest failures."""

import json
import logging
import os

import httpx

from fpl_oracle.llm.gemini_requests import gemini_gate
from fpl_oracle.llm.tools import MAX_TOOL_ROUNDS, TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger(__name__)


class GeminiProvider:
    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        model = (model if model is not None else os.getenv("GEMINI_MODEL", "")).strip() or "gemini-flash-lite-latest"
        self.model = model

    async def _chat_with_model(self, model_name: str, messages, system_prompt, timeout_secs: float):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages
        ]
        declarations = [
            {"name": t["name"], "description": t["description"]}
            | ({"parameters": t["parameters"]} if t["parameters"].get("properties") else {})
            for t in TOOL_DEFINITIONS
        ]
        for round_no in range(MAX_TOOL_ROUNDS):
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "tools": [{"function_declarations": declarations}],
                "tool_config": {"function_calling_config": {"mode": "AUTO"}},
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
            }
            # A fresh client per request: one held open across a long tool loop goes stale.
            async with httpx.AsyncClient(timeout=timeout_secs) as client:
                resp = await gemini_gate.post(client, url, json=payload, headers={"x-goog-api-key": self.api_key})
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
                "Gemini chat response: model=%s round=%s finish_reason=%s function_calls=%s",
                model_name,
                round_no + 1,
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
                continue
            text = "\n".join(p["text"] for p in parts if "text" in p).strip()
            if text:
                return text
            raise RuntimeError("Gemini returned no answer text")
        raise RuntimeError("Gemini data-request limit reached")

    async def chat(self, messages, system_prompt):
        from fpl_oracle.llm.gemini_requests import GeminiRateLimitError

        timeout_secs = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60.0"))
        candidate_models = [self.model]
        for fallback in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for cand in candidate_models:
            try:
                return await self._chat_with_model(cand, messages, system_prompt, timeout_secs)
            except (GeminiRateLimitError, RuntimeError) as e:
                is_rate_limit = isinstance(e, GeminiRateLimitError) or "429" in str(e)
                if is_rate_limit and cand != candidate_models[-1]:
                    logger.warning("Gemini model %s rate limited, attempting fallback to next model...", cand)
                    last_error = e
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("No Gemini response returned")
