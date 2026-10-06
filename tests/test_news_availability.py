"""
Tests for News Ingestion, Gemini Free-Tier Extractor, and Single Availability Reconciliation Layer.
Verifies:
1. SSRF URL safety validator.
2. Free-tier safeguard (fails closed when confirmation is false).
3. Budget limiter (request and token bounds).
4. Single availability reconciliation (zero double-counting, negation handling, cup isolation, loan ineligibility, shadow mode).
5. Elimination of crude heuristics in analyse.py.
6. Absence of secret/ID input fields in web/index.html.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fpl_oracle.api.models import Element
from fpl_oracle.news.gemini_extractor import GeminiBudgetManager, GeminiEvidenceExtractor
from fpl_oracle.news.ingest import is_safe_external_url
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence, RecommendationMode
from fpl_oracle.news.reconcile import AvailabilityReconciler


def test_is_safe_external_url():
    """Verify SSRF protection blocks local, private, and invalid URLs."""
    # Disallowed loopback and localhost
    assert not is_safe_external_url("http://localhost/feed")
    assert not is_safe_external_url("http://127.0.0.1/rss")
    assert not is_safe_external_url("http://0.0.0.0/rss")
    assert not is_safe_external_url("http://[::1]/rss")

    # Disallowed private RFC-1918 IPs
    assert not is_safe_external_url("http://10.0.0.1/feed.xml")
    assert not is_safe_external_url("http://192.168.1.50/news")
    assert not is_safe_external_url("http://172.16.0.22/rss")

    # Disallowed link-local / cloud metadata IP
    assert not is_safe_external_url("http://169.254.169.254/latest/meta-data/")

    # Disallowed non-HTTP schemes
    assert not is_safe_external_url("file:///etc/passwd")
    assert not is_safe_external_url("ftp://ftp.example.com/feed")
    assert not is_safe_external_url("javascript:alert(1)")

    # Allowed public domains
    assert is_safe_external_url("https://feeds.bbci.co.uk/sport/football/rss.xml")
    assert is_safe_external_url("http://www.premierleague.com/news/rss")
    assert is_safe_external_url("https://theathletic.com/rss")


def test_gemini_is_configured_safeguards(monkeypatch):
    """Verify Gemini extractor fails closed unless both API key and explicit confirmation are present."""
    extractor = GeminiEvidenceExtractor()

    # Case 1: No key, no confirmation
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_FREE_TIER_CONFIRMED", raising=False)
    configured, msg = extractor.is_configured()
    assert not configured
    assert "GEMINI_API_KEY not set" in msg

    # Case 2: Key present, but confirmation missing
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "false")
    configured, msg = extractor.is_configured()
    assert not configured
    assert "GEMINI_FREE_TIER_CONFIRMED is not true" in msg

    # Case 3: Key present and confirmation is true
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")
    configured, msg = extractor.is_configured()
    assert configured
    assert msg == "Ready"


def test_gemini_budget_manager(tmp_path: Path):
    """Verify daily request and token budget enforcement with persistence."""
    cache_file = tmp_path / "test_budget.json"
    budget = GeminiBudgetManager(cache_file=cache_file)

    # Initial check
    status = budget.get_status()
    assert status["requests_used"] == 0

    # Consume some requests
    allowed, _ = budget.check_and_increment(estimated_tokens=2000)
    assert allowed
    assert budget.get_status()["requests_used"] == 1
    assert budget.get_status()["tokens_used"] == 2000

    # Simulate hitting max requests
    state = budget._load_budget_state()
    state["requests_count"] = 150
    budget._save_budget_state(state)

    allowed, reason = budget.check_and_increment(estimated_tokens=500)
    assert not allowed
    assert "Daily free request budget exceeded" in reason


@pytest.mark.anyio
async def test_gemini_extractor_fallback_when_unconfigured(monkeypatch):
    """Verify graceful fallback without raising exceptions when unconfigured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    extractor = GeminiEvidenceExtractor()

    payload = await extractor.extract_evidence_from_text(
        article_text="Erling Haaland has picked up an ankle injury and will miss the match.",
        article_url="https://example.com/test",
    )
    assert payload is not None
    assert payload.evidences == []
    assert "Skipped" in (payload.raw_response_snippet or "")


def test_availability_reconciler_no_double_discounting():
    """Verify single-adjustment invariant: official 50% chance is not multiplied again by candidate doubt."""
    reconciler = AvailabilityReconciler(mode="gated_active")

    element = MagicMock(spec=Element)
    element.id = 101
    element.web_name = "Isak"
    element.team = 15
    element.status = "d"
    element.chance_of_playing_this_round = 50.0
    element.chance_of_playing_next_round = 50.0
    element.news = "Groin injury - 50% chance of playing"
    element.scout_risks = []

    # Candidate evidence also notes doubt
    candidate = [
        PlayerEvidence(
            player_id=101,
            player_name="Isak",
            category=EvidenceCategory.DOUBTFUL,
            quote="Isak faces late fitness test",
            confidence=0.90,
        )
    ]

    reconciled = reconciler.reconcile_player_fixture(
        element=element,
        target_gw=28,
        candidate_evidence=candidate,
    )

    # Effective probability must remain 50%, NOT multiplied 0.5 * 0.5 = 0.25 (25%)
    assert reconciled.effective_chance_of_playing == 50.0
    assert reconciled.p_available == 0.50


def test_availability_reconciler_negation_handling():
    """Verify negated statements ('not injured') do not rule out players."""
    reconciler = AvailabilityReconciler(mode="gated_active")

    element = MagicMock(spec=Element)
    element.id = 202
    element.web_name = "Palmer"
    element.team = 6
    element.status = "a"
    element.chance_of_playing_this_round = 100.0
    element.chance_of_playing_next_round = 100.0
    element.news = ""
    element.scout_risks = []

    candidate = [
        PlayerEvidence(
            player_id=202,
            player_name="Palmer",
            category=EvidenceCategory.RULED_OUT,
            quote="Cole Palmer is not ruled out of Sunday's clash",
            is_negated=True,
            confidence=0.95,
        )
    ]

    reconciled = reconciler.reconcile_player_fixture(
        element=element,
        target_gw=28,
        candidate_evidence=candidate,
    )

    # Negation must cause abstention; player stays 100% available
    assert reconciled.effective_chance_of_playing == 100.0
    assert reconciled.p_available == 1.0
    assert any("Negated statement" in s for s in reconciled.rejected_signals)


def test_availability_reconciler_cup_competition_isolation():
    """Verify cup competition quotes do not affect Premier League gameweek availability."""
    reconciler = AvailabilityReconciler(mode="gated_active")

    element = MagicMock(spec=Element)
    element.id = 303
    element.web_name = "Saka"
    element.team = 1
    element.status = "a"
    element.chance_of_playing_this_round = 100.0
    element.chance_of_playing_next_round = 100.0
    element.news = ""
    element.scout_risks = []

    candidate = [
        PlayerEvidence(
            player_id=303,
            player_name="Saka",
            category=EvidenceCategory.RULED_OUT,
            quote="Saka rested for Carabao Cup tie on Wednesday",
            match_context="Carabao Cup",
            confidence=0.90,
        )
    ]

    reconciled = reconciler.reconcile_player_fixture(
        element=element,
        target_gw=28,
        candidate_evidence=candidate,
    )

    # Cup quote must be rejected for PL GW 28
    assert reconciled.effective_chance_of_playing == 100.0
    assert any("Cup competition quote" in s for s in reconciled.rejected_signals)


def test_availability_reconciler_scout_loan_ineligibility():
    """Verify scout risk loan ineligibility zeros availability for the targeted GW."""
    reconciler = AvailabilityReconciler(mode="api_only")

    element = MagicMock(spec=Element)
    element.id = 404
    element.web_name = "Sterling"
    element.team = 1
    element.status = "a"
    element.chance_of_playing_this_round = 100.0
    element.chance_of_playing_next_round = 100.0
    element.news = ""
    element.scout_risks = [{"property": "loan_ineligible", "gameweek": 28}]

    reconciled_gw28 = reconciler.reconcile_player_fixture(element=element, target_gw=28)
    assert reconciled_gw28.effective_chance_of_playing == 0.0
    assert reconciled_gw28.p_available == 0.0

    # For GW 29, loan ineligibility against that specific opponent does not apply
    reconciled_gw29 = reconciler.reconcile_player_fixture(element=element, target_gw=29)
    assert reconciled_gw29.effective_chance_of_playing == 100.0


def test_availability_reconciler_shadow_mode_isolation():
    """Verify shadow mode computes overrides for logging/eval but keeps production values at baseline."""
    shadow_reconciler = AvailabilityReconciler(mode="shadow")

    element = MagicMock(spec=Element)
    element.id = 505
    element.web_name = "Salah"
    element.team = 11
    element.status = "a"
    element.chance_of_playing_this_round = 100.0
    element.chance_of_playing_next_round = 100.0
    element.news = ""
    element.scout_risks = []

    candidate = [
        PlayerEvidence(
            player_id=505,
            player_name="Salah",
            category=EvidenceCategory.RULED_OUT,
            quote="Salah ruled out with hamstring strain",
            confidence=0.95,
        )
    ]

    reconciled = shadow_reconciler.reconcile_player_fixture(
        element=element,
        target_gw=28,
        candidate_evidence=candidate,
    )

    # In shadow mode, is_shadow_override is recorded for eval, but production value remains official baseline
    assert reconciled.is_shadow_override is True
    assert reconciled.applied_to_production is False
    assert reconciled.effective_chance_of_playing == 100.0  # Production value unmutated
    assert reconciled.mode == RecommendationMode.SHADOW


def test_news_analyse_no_crude_heuristics():
    """Verify that crude 50% or 85% availability multipliers are completely removed from analyse.py."""
    analyse_path = Path(__file__).resolve().parent.parent / "src" / "fpl_oracle" / "news" / "analyse.py"
    with open(analyse_path, encoding="utf-8") as f:
        content = f.read()

    # Crude multiplier patterns previously present:
    # "0.5 if status == 'd' else 0.85" or similar arbitrary hardcoded multipliers
    assert "0.5 if" not in content
    assert "0.85 if" not in content
    assert "availability_reconciler" in content


def test_web_index_no_secret_input_forms():
    """Verify web/index.html contains zero input forms for entering manager ID, league ID, or API keys."""
    index_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    with open(index_path, encoding="utf-8") as f:
        content = f.read()

    # Form inputs for credentials/IDs must NOT exist
    assert '<input type="password"' not in content
    assert 'name="gemini_api_key"' not in content
    assert 'name="fpl_team_id"' not in content
    assert 'name="fpl_league_id"' not in content
    assert 'v-model="fplTeamId"' not in content
    assert 'v-model="geminiApiKey"' not in content
