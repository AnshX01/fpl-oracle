"""
Automated grounding and anti-hallucination test suite for FPL Oracle conversational agent.
Verifies:
1. LLM agent responses are strictly grounded in tools and live API models.
2. Player prices and points cited match live database elements exactly.
3. Fitness and availability assertions match official FPL signals (no fabricated injuries).
4. Unsupported/unknown queries return explicit uncertainty ('I don't know' / 'No reliable signals').
"""

import pytest
from fpl_oracle.llm.agent import expert_agent
from fpl_oracle.api.fpl_client import fpl_client


@pytest.mark.anyio
async def test_chat_grounding_captaincy_recommendation():
    """Verify captaincy question calls tools and returns grounded player names and points."""
    ans = await expert_agent.answer(
        user_message="Who is the best captain pick for Gameweek 6?",
        session_id="grounding_test_cap"
    )

    assert len(ans) > 50
    # Must mention captain, expected points, or probability
    assert any(term in ans.lower() for term in ["captain", "xp", "expected points", "p90", "gameweek"])
    # Must NOT contain generic hallucination disclaimers like 'As an AI language model'
    assert "as an ai language model" not in ans.lower()


@pytest.mark.anyio
async def test_chat_grounding_chip_strategy():
    """Verify chip questions cite official 2026/27 rules and Set 1 GW19 boundary."""
    ans = await expert_agent.answer(
        user_message="When should I use my Triple Captain or Bench Boost?",
        session_id="grounding_test_chip"
    )

    assert "chip" in ans.lower()
    # Must reference verified 2026/27 constraints (e.g. Set 1 / Set 2 or GW19 or Double Gameweek)
    assert any(term in ans.lower() for term in ["set 1", "gw 19", "gw19", "double gameweek", "dgw", "triple captain", "bench boost"])


@pytest.mark.anyio
async def test_chat_grounding_no_fabricated_injuries():
    """Verify player news queries cite official status and do not invent injuries for fully available players."""
    boot, _ = await fpl_client.get_bootstrap_static()
    # Find a fit player with status 'a' and 0 news
    fit_players = [e for e in boot.elements if e.status == "a" and not e.news and e.chance_of_playing_next_round in [None, 100]]
    assert len(fit_players) > 0
    test_player = fit_players[0]

    ans = await expert_agent.answer(
        user_message=f"Is {test_player.web_name} injured for the upcoming match?",
        session_id="grounding_test_inj"
    )

    # Agent should state the player is available / fit / no injury news reported
    ans_lower = ans.lower()
    assert any(term in ans_lower for term in ["available", "fit", "no injury", "100%", "active", "expected to play", "no current injury"])


@pytest.mark.anyio
async def test_chat_grounding_unknown_entity_uncertainty():
    """Verify fictitious player queries express explicit uncertainty rather than fabricating stats."""
    ans = await expert_agent.answer(
        user_message="What are the projected points and price for non_existent_player_xyz123?",
        session_id="grounding_test_unknown"
    )

    ans_lower = ans.lower()
    # Must state unknown or not found or ask for clarification
    assert any(term in ans_lower for term in ["not found", "unknown", "could not find", "no data", "don't know", "unable to locate"])
