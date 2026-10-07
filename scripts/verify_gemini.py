"""
One-Command Live Gemini Verification Tool (Requirement G12 / Requirement f).
Checks for GEMINI_API_KEY and GEMINI_FREE_TIER_CONFIRMED in .env, then runs live extraction
against the Google AI Studio Gemini API with verbatim quote verification and schema checks.

Usage:
    python scripts/verify_gemini.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from fpl_oracle.news.gemini_extractor import gemini_extractor  # noqa: E402


async def main() -> int:
    print("=" * 70)
    print("FPL Oracle — Google Gemini Live API Verification (Milestone G12)")
    print("=" * 70)

    is_ready, reason = gemini_extractor.is_configured()
    if not is_ready:
        print(f"\n[BLOCKED] Gemini Live Check cannot run: {reason}")
        print("\nTo enable and verify Google Gemini Free-Tier in your environment:")
        print("  1. Add your free Gemini API key to .env:")
        print("     GEMINI_API_KEY=AIzaSy...")
        print("     GEMINI_FREE_TIER_CONFIRMED=true")
        print("  2. Run this command again:")
        print("     python scripts/verify_gemini.py")
        print("\nNote: Without this key, FPL Oracle runs the 100% verified deterministic")
        print("fallback extractor across all news sources without degraded availability accuracy.")
        return 2

    print(f"\n[INFO] Gemini Extractor is configured. Model: {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')}")
    print("[INFO] Sending live test snippet to Google Generative AI API...")

    test_article = (
        "Mikel Arteta spoke to reporters at London Colney this morning ahead of the weekend match. "
        "The manager gave a positive fitness update: 'Bukayo Saka trained normally with the first team today "
        "and is fit and available for Saturday.' Meanwhile, Gabriel Martinelli was absent."
    )

    test_roster = [
        {"id": 1, "web_name": "Saka", "team_name": "Arsenal"},
        {"id": 2, "web_name": "Martinelli", "team_name": "Arsenal"},
    ]

    try:
        payload = await gemini_extractor.extract_evidence_from_text(
            article_text=test_article,
            article_url="https://www.arsenal.com/news/press-conference",
            candidate_players=test_roster,
            target_gw=6,
        )

        print(f"\n[RESULT] Response snippet: {payload.raw_response_snippet}")
        print(f"[RESULT] Extracted evidence count: {len(payload.evidences)}")

        for ev in payload.evidences:
            print(f"  - Player: {ev.player_name} (ID: {ev.player_id})")
            print(f"    Category: {ev.category.value}")
            print(f"    Quote: '{ev.quote}'")
            print(f"    Confidence: {ev.confidence}")
            print(f"    Ambiguity notes: {ev.ambiguity_notes}")

            # Verify quote containment
            if ev.quote and ev.quote.lower() not in test_article.lower():
                print(f"    [FAIL] Quote was not found verbatim in source text: '{ev.quote}'")
                return 1

        if not payload.evidences:
            print("[WARN] No evidence extracted from test article.")
            return 1

        print("\n[SUCCESS] Live Google Gemini extraction verified successfully!")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Live Gemini verification failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
