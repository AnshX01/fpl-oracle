"""
FPL Conversational Expert Agent.
Coordinates multi-turn dialogue, tool calling, conversation memory,
and factual citation generation.
"""

import logging

from fpl_oracle.data.store import data_store
from fpl_oracle.llm.provider import SYSTEM_PROMPT, get_llm_provider

logger = logging.getLogger("fpl_oracle.llm.agent")

class ExpertAgent:
    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    async def answer(self, user_message: str, session_id: str = "default") -> str:
        """
        Process user message, retrieve session history, execute LLM or offline expert,
        and save turn in database.
        """
        # Save user message
        data_store.add_chat_message(role="user", content=user_message, session_id=session_id)

        # Retrieve recent history
        history = data_store.get_chat_history(session_id=session_id, limit=20)

        # Get LLM Provider
        provider = get_llm_provider()

        try:
            response_text = await provider.chat(messages=history, system_prompt=self.system_prompt)
        except Exception as e:
            logger.error(f"Error during LLM chat generation: {e}. Falling back to offline engine...")
            from fpl_oracle.llm.provider import OfflineExpertProvider
            response_text = await OfflineExpertProvider().chat(messages=history, system_prompt=self.system_prompt)

        # Save assistant message
        data_store.add_chat_message(role="assistant", content=response_text, session_id=session_id)
        return response_text

expert_agent = ExpertAgent()
