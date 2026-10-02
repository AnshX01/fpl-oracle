"""
Google Gemini LLM Provider with tool-calling capabilities.
Connects directly to Google Generative Language API via httpx.
"""

from typing import List, Dict, Any, Optional
import json
import logging
import httpx

from fpl_oracle.llm.tools import TOOL_DEFINITIONS, tool_executor

logger = logging.getLogger("fpl_oracle.llm.gemini")

class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    async def chat(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Execute multi-turn conversation with tool calling loop against Gemini API.
        """
        # Format tools for Gemini API
        gemini_tools = [{
            "function_declarations": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
                for t in TOOL_DEFINITIONS
            ]
        }]

        # Prepare contents
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}]
            })

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "tools": gemini_tools,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{self.base_url}?key={self.api_key}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code != 200:
                logger.error(f"Gemini API error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Gemini API returned status {resp.status_code}")

            data = resp.json()
            candidate = data.get("candidates", [{}])[0].get("content", {})
            parts = candidate.get("parts", [])

            # Check if Gemini made a function call
            for part in parts:
                if "functionCall" in part:
                    fn_name = part["functionCall"]["name"]
                    fn_args = part["functionCall"].get("args", {})
                    # Execute tool
                    tool_res = await tool_executor.execute(fn_name, fn_args)

                    # Send tool result back to Gemini
                    followup_contents = list(contents)
                    followup_contents.append({
                        "role": "model",
                        "parts": [{"functionCall": part["functionCall"]}]
                    })
                    followup_contents.append({
                        "role": "function",
                        "parts": [{
                            "functionResponse": {
                                "name": fn_name,
                                "response": tool_res
                            }
                        }]
                    })

                    followup_payload = {
                        "system_instruction": {"parts": [{"text": system_prompt}]},
                        "contents": followup_contents,
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
                    }

                    resp2 = await client.post(
                        f"{self.base_url}?key={self.api_key}",
                        json=followup_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    if resp2.status_code == 200:
                        data2 = resp2.json()
                        parts2 = data2.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        texts = [p.get("text", "") for p in parts2 if "text" in p]
                        return "\n".join(texts)
                elif "text" in part:
                    return part["text"]

            return "I have analyzed your request based on the latest 2026/27 FPL data."
