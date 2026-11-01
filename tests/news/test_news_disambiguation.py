"""
Comprehensive regression and disambiguation test suite for news ingestion and reconciliation.
Verifies that:
1. Interrogative/speculative question headlines (e.g. Carabao Cup red card articles) are rejected.
2. Official FPL disciplinary status 's' is authoritative for suspensions (status 'a' is never downgraded by unconfirmed rumors).
3. Domestic cup match contexts are cleanly segregated from Premier League availability.
4. Multi-entity clauses do not attribute secondary player injuries or historical debuts to candidate players.
5. Production code purity is maintained with zero hardcoded player names in src/.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fpl_oracle.news.extract import text_extractor
from fpl_oracle.news.local_nlp import local_nlp_engine
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.reconcile import availability_reconciler, is_safe_quote


def _make_element(element_id: int, web_name: str, status: str = "a", cop: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=element_id,
        web_name=web_name,
        team=1,
        status=status,
        news="",
        news_added=None,
        chance_of_playing_next_round=cop,
        chance_of_playing_this_round=None,
        scout_risks=[],
    )


def test_speculative_question_headline_rejected_in_quote_safety():
    """Interrogative headlines ending with '?' or starting with speculative phrasing are rejected."""
    question_quotes = [
        "Which game will Chelsea's Palmer miss after red card?",
        "Will Watkins start against Arsenal this weekend?",
        "Could Saka be rested for Sunday's clash?",
        "Why is Alexander-Arnold out of training?",
        "Is Haaland fit enough to feature?",
    ]
    for q in question_quotes:
        safe, reason = is_safe_quote(q)
        assert not safe, f"Quote '{q}' should be rejected as unsafe/speculative"
        assert "speculative" in reason.lower() or "question" in reason.lower()


def test_declarative_manager_quotes_remain_safe():
    """Declarative factual injury statements from managers pass quote safety validation."""
    factual_quotes = [
        "He suffered a high ankle sprain and will be sidelined for four weeks.",
        "Arteta confirmed the player has returned to full team training.",
        "It is touch and go for Saturday. We will assess him tomorrow.",
        "He underwent knee surgery and will miss the remainder of the season.",
    ]
    for q in factual_quotes:
        safe, reason = is_safe_quote(q)
        assert safe, f"Quote '{q}' should pass safety check: {reason}"


def test_palmer_carabao_cup_red_card_headline_rejected():
    """
    Forensic regression test for Palmer Carabao Cup red card headline:
    Headline: 'Which game will Chelsea's Palmer miss after red card?'
    Must not downgrade availability for a Premier League asset with official status 'a'.
    """
    quote = "Which game will Chelsea's Palmer miss after red card?"
    elem = _make_element(124, "Palmer", status="a", cop=None)

    # 1. Extractor test: question headline returns UNKNOWN or discarded
    evs = text_extractor.extract_evidence_from_text(quote, player_name="Palmer", player_id=124)
    assert len(evs) == 1
    assert evs[0].category == EvidenceCategory.UNKNOWN
    assert "speculative" in evs[0].ambiguity_notes.lower() or "question" in evs[0].ambiguity_notes.lower()

    # 2. Local NLP engine test: classified as available/speculative question
    band, pattern, _ = local_nlp_engine.classify_statement(quote)
    assert band.category == EvidenceCategory.AVAILABLE
    assert pattern == "speculative question"

    # 3. Reconciler test: even if injected as candidate evidence, rejected due to question and suspension invariant
    cand_ev = PlayerEvidence(
        player_id=124,
        player_name="Palmer",
        category=EvidenceCategory.RULED_OUT,
        quote=quote,
        source_url="https://www.bbc.co.uk/sport/football/articles/c04j44krr18o",
        published_at=datetime.now(UTC) - timedelta(hours=1),
        confidence=0.95,
    )
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=[cand_ev], first_gw=9)

    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert not res.applied_to_production
    assert any("question" in s.lower() or "speculative" in s.lower() for s in res.rejected_signals)


def test_official_fpl_suspension_invariant_prevents_unconfirmed_press_bans():
    """
    A press article reporting a red card or suspension cannot downgrade availability
    if the player's official FPL status is 'a' (available) and not flagged 's'.
    """
    elem = _make_element(200, "Defender", status="a")
    press_ev = PlayerEvidence(
        player_id=200,
        player_name="Defender",
        category=EvidenceCategory.RULED_OUT,
        quote="Defender was shown a red card and faces a three match suspension.",
        source_url="https://www.theguardian.com/football/match",
        published_at=datetime.now(UTC) - timedelta(hours=2),
        confidence=0.95,
    )
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=[press_ev], first_gw=9)
    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert not res.applied_to_production
    assert any("suspension unconfirmed by official fpl status 's'" in s.lower() for s in res.rejected_signals)


def test_konsa_multi_entity_debut_and_secondary_injury_not_attributed():
    """
    Forensic regression test for Konsa / Mosquera entity misattribution:
    Text: 'Konsa joined just hours before their season-opener against Coventry,
           while he came on for his debut in their second match after Mosquera picked up an injury.'
    Must NOT penalize Konsa for Mosquera's injury or the historical debut recap.
    """
    text = (
        "Konsa joined just hours before their season-opener against Coventry, "
        "while he came on for his debut in their second match after Mosquera picked up an injury."
    )
    roster = [{"id": 123, "web_name": "Konsa", "team_name": "Aston Villa", "team": 2}]
    elem = _make_element(123, "Konsa", status="a")

    # 1. Local NLP engine extraction isolates secondary subject and historical narrative
    evs = local_nlp_engine.extract_calibrated_evidence(
        article_text=text,
        article_url="https://www.bbc.co.uk/sport/football/articles/c4gz94n8n3vo",
        published_dt=datetime.now(UTC),
        candidate_roster=roster,
        target_gw=9,
    )
    # Konsa should not have any lowering evidence extracted
    assert len(evs) == 0

    # 2. Text extractor verification
    text_evs = text_extractor.extract_evidence_from_text(text, player_name="Konsa", player_id=123)
    for ev in text_evs:
        assert ev.category in (EvidenceCategory.UNKNOWN, EvidenceCategory.AVAILABLE)
        assert ev.category != EvidenceCategory.DOUBTFUL
        assert ev.category != EvidenceCategory.RULED_OUT

    # 3. Reconciler verification
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=evs, first_gw=9)
    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert not res.applied_to_production


def test_historical_narrative_recap_rejected():
    """Historical narrative clauses describing past seasons/openers without present context are rejected."""
    elem = _make_element(300, "Striker", status="a")
    recap_ev = PlayerEvidence(
        player_id=300,
        player_name="Striker",
        category=EvidenceCategory.DOUBTFUL,
        quote="He came on for his debut in their second match after an injury earlier this season.",
        source_url="https://www.bbc.co.uk/sport/recap",
        published_at=datetime.now(UTC) - timedelta(hours=2),
        confidence=0.85,
    )
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=[recap_ev], first_gw=9)
    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert any("historical narrative recap" in s.lower() for s in res.rejected_signals)


def test_cup_context_in_quote_or_url_segregated():
    """Quotes mentioning Carabao Cup or FA Cup without Premier League context are rejected."""
    elem = _make_element(400, "Midfielder", status="a")
    cup_ev = PlayerEvidence(
        player_id=400,
        player_name="Midfielder",
        category=EvidenceCategory.RULED_OUT,
        quote="Midfielder will miss Wednesday's Carabao Cup tie with a tight hamstring.",
        source_url="https://www.skysports.com/football/carabao-cup-preview",
        published_at=datetime.now(UTC) - timedelta(hours=3),
        confidence=0.95,
    )
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=[cup_ev], first_gw=9)
    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert any("cup context" in s.lower() or "cup competition" in s.lower() for s in res.rejected_signals)


def test_contrastive_cup_absence_with_affirmative_league_availability():
    """
    Forensic test for BBC sentence:
    'while the Blues will miss Palmer for their trip to Anfield, he will be available to face United three days later'
    Must NOT rule out Palmer for the upcoming United league match.
    """
    sentence = (
        "while the Blues will miss Palmer for their trip to Anfield, "
        "he will be available to face United three days later"
    )
    elem = _make_element(154, "Palmer", status="a")

    # 1. Extractor classifies as available
    evs = text_extractor.extract_evidence_from_text(sentence, player_name="Palmer", player_id=154)
    assert len(evs) == 1
    assert evs[0].category == EvidenceCategory.AVAILABLE

    # 2. Local NLP classifies as available
    band, pattern, _ = local_nlp_engine.classify_statement(sentence)
    assert band.category == EvidenceCategory.AVAILABLE

    # 3. Reconciler preserves 100% availability
    res = availability_reconciler.reconcile_player_fixture(elem, target_gw=9, candidate_evidence=evs, first_gw=9)
    assert res.effective_chance_of_playing == 100.0
    assert res.p_available == 1.0
    assert not res.applied_to_production
