"""
Run sample chat test questions through FPL Oracle Expert Agent
and write transcript to reports/chat_examples.md.
"""

import asyncio
import sys

# Configure stdout encoding for Windows UTF-8 support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fpl_oracle.config import REPORTS_DIR
from fpl_oracle.llm.agent import expert_agent

QUESTIONS = [
    "Should I take a -4 for Haaland?",
    "When should I use my Bench Boost?",
    "Who's the best differential midfielder under 6.5?",
    "Is it time to wildcard?",
    "What happens if I Free Hit in the blank GW?",
    "How do I catch the person above me in my mini league?",
    "Explain why you picked Saka over Palmer this week.",
    "Who should I captain for Gameweek 6?",
    "Which players are likely to rise in price tonight?",
    "Give me my transfer roadmap for the next 4 gameweeks.",
]


async def run_chat_examples():
    output_path = REPORTS_DIR / "chat_examples.md"
    print("=== Running FPL Oracle Chat Evaluation (10 Sample Questions) ===")

    content = "# FPL Oracle — Expert Chat Evaluation & Transcripts\n\n"
    content += "This document contains verbatim transcripts from test interactions with the FPL Oracle Conversational Expert Agent, demonstrating tool calling, factual grounding, 2026/27 rule verification, and reasoning standards.\n\n"

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"[{idx}/{len(QUESTIONS)}] Asking: '{q}'...")
        session_id = f"test_session_{idx}"
        ans = await expert_agent.answer(user_message=q, session_id=session_id)
        content += f'## Question {idx}: "{q}"\n\n'
        content += f"### Assistant Response:\n\n{ans}\n\n---\n\n"

    output_path.write_text(content, encoding="utf-8")
    print(f"=== Chat examples successfully saved to {output_path} ===")


if __name__ == "__main__":
    asyncio.run(run_chat_examples())
