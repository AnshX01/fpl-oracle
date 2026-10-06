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

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
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


def test_controlled_mock_xp_shadow_vs_gated_active():
    """
    N2 CONTROLLED MOCK TEST:
    Asserts that a manager quote ruling a player out:
    1. Changes xP to 0.0 in gated_active mode (candidate evidence applied to production).
    2. Leaves xP completely unchanged in shadow mode (official baseline preserved).
    """
    from fpl_oracle.ml.ensemble import scoring_ensemble

    shadow_reconciler = AvailabilityReconciler(mode="shadow")
    gated_reconciler = AvailabilityReconciler(mode="gated_active")

    element = MagicMock(spec=Element)
    element.id = 101
    element.web_name = "Haaland"
    element.team = 1
    element.status = "a"
    element.chance_of_playing_this_round = 100.0
    element.chance_of_playing_next_round = 100.0
    element.news = ""
    element.scout_risks = []

    quote_evidence = [
        PlayerEvidence(
            player_id=101,
            player_name="Haaland",
            category=EvidenceCategory.RULED_OUT,
            quote="Haaland twisted his ankle in training and is definitely out for the weekend.",
            confidence=0.95,
        )
    ]

    # Reconcile under SHADOW mode
    res_shadow = shadow_reconciler.reconcile_player_fixture(
        element=element, target_gw=10, candidate_evidence=quote_evidence
    )
    assert res_shadow.applied_to_production is False
    assert res_shadow.is_shadow_override is True
    assert res_shadow.effective_chance_of_playing == 100.0  # Official baseline preserved

    # Reconcile under GATED_ACTIVE mode
    res_gated = gated_reconciler.reconcile_player_fixture(
        element=element, target_gw=10, candidate_evidence=quote_evidence
    )
    assert res_gated.applied_to_production is True
    assert res_gated.is_shadow_override is True
    assert res_gated.effective_chance_of_playing == 0.0  # Overridden by verified evidence

    # Verify xP propagation through ScoringEnsemble
    # Create synthetic component predictions representing a peak-form forward
    import numpy as np

    comps = {
        "expected_minutes": np.array([88.0]),
        "p_starts": np.array([0.95]),
        "p_min60": np.array([0.92]),
        "expected_goals": np.array([0.85]),
        "expected_assists": np.array([0.25]),
        "p_clean_sheet": np.array([0.45]),
        "expected_goals_conceded": np.array([0.80]),
        "p_defcon": np.array([0.0]),
        "expected_bonus": np.array([1.80]),
        "expected_card_deduction": np.array([0.10]),
        "expected_saves": np.array([0.0]),
    }

    # Evaluate shadow mode (chance_of_playing = 100.0)
    df_shadow_features = pd.DataFrame([{
        "pos_FWD": 1.0,
        "pos_MID": 0.0,
        "pos_DEF": 0.0,
        "pos_GKP": 0.0,
        "chance_of_playing": res_shadow.effective_chance_of_playing,
    }])
    df_shadow_pred = scoring_ensemble.aggregate_components(comps, df_shadow_features)
    shadow_xp = df_shadow_pred["expected_points"].iloc[0]

    # Evaluate gated_active mode (chance_of_playing = 0.0)
    df_gated_features = pd.DataFrame([{
        "pos_FWD": 1.0,
        "pos_MID": 0.0,
        "pos_DEF": 0.0,
        "pos_GKP": 0.0,
        "chance_of_playing": res_gated.effective_chance_of_playing,
    }])
    df_gated_pred = scoring_ensemble.aggregate_components(comps, df_gated_features)
    gated_xp = df_gated_pred["expected_points"].iloc[0]

    # Verification assertions:
    assert shadow_xp > 5.0, f"Expected unmutated shadow xP > 5.0, got {shadow_xp}"
    assert gated_xp == 0.0, f"Expected gated active xP == 0.0, got {gated_xp}"
    assert shadow_xp - gated_xp > 5.0, "Delta between shadow and gated_active must be significant"


def test_reconcile_named_probability_settings():
    """
    N3 NAMED SETTINGS VERIFICATION:
    Asserts that AvailabilityReconciler respects configurable ReconcileProbabilitySettings.
    """
    from fpl_oracle.news.reconcile import ReconcileProbabilitySettings

    custom_settings = ReconcileProbabilitySettings(
        prob_start_high_base=0.95,
        prob_start_low_base=0.40,
        prob_start_minutes_limit=0.60,
        prob_avail_returned_training=0.80,
        prob_start_returned_training=0.55,
        prob_avail_doubtful=0.45,
        prob_start_doubtful=0.35,
    )

    reconciler = AvailabilityReconciler(mode="gated_active", prob_settings=custom_settings)

    element = MagicMock(spec=Element)
    element.id = 202
    element.web_name = "Player"
    element.team = 2
    element.status = "d"
    element.chance_of_playing_next_round = 25.0
    element.news = ""
    element.scout_risks = []

    ev = [
        PlayerEvidence(
            player_id=202,
            player_name="Player",
            category=EvidenceCategory.RETURNED_TO_TRAINING,
            quote="Back in full training yesterday.",
            confidence=0.90,
        )
    ]

    res = reconciler.reconcile_player_fixture(element, target_gw=12, candidate_evidence=ev)
    assert res.p_available == 0.80
    assert res.p_start_given_available == 0.55

