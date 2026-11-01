"""
Unit tests for LocalCalibratedNLPEngine.
Verifies offline probabilistic statement classification, calibrated probability bands,
negation overrides, and multi-player clause isolation.
"""

from datetime import UTC, datetime

from fpl_oracle.news.local_nlp import (
    BAND_IN_CONTENTION,
    BAND_MAJOR_DOUBT,
    BAND_RULED_OUT,
    BAND_TOUCH_AND_GO,
    local_nlp_engine,
)
from fpl_oracle.news.models import EvidenceCategory


def test_classify_ruled_out_statements():
    samples = [
        "The defender has been ruled out of this weekend's match following surgery.",
        "He suffered a torn hamstring and will be sidelined for several weeks.",
        "The striker has no chance of playing after fracturing his ankle in training.",
    ]
    for s in samples:
        band, pattern, quote = local_nlp_engine.classify_statement(s)
        assert band == BAND_RULED_OUT
        assert band.p_play == 0.0
        assert band.p_starts == 0.0
        assert band.category == EvidenceCategory.RULED_OUT


def test_classify_major_doubt_statements():
    samples = [
        "He is a major doubt for the fixture and has not trained with the group all week.",
        "The midfielder felt a twinge in recovery and is struggling to make it.",
    ]
    for s in samples:
        band, pattern, quote = local_nlp_engine.classify_statement(s)
        assert band == BAND_MAJOR_DOUBT
        assert band.p_play == 0.25
        assert band.category == EvidenceCategory.DOUBTFUL


def test_classify_touch_and_go_statements():
    samples = [
        "It is touch and go for Saturday. We will make a decision tomorrow after assessment.",
        "He faces a late fitness test to see how he responds in the morning.",
    ]
    for s in samples:
        band, pattern, quote = local_nlp_engine.classify_statement(s)
        assert band == BAND_TOUCH_AND_GO
        assert band.p_play == 0.50
        assert band.category == EvidenceCategory.DOUBTFUL


def test_classify_in_contention_and_negation():
    samples = [
        "He suffered a slight knock but trained today and should be fine.",
        "The winger is back in training with the squad and in contention.",
    ]
    for s in samples:
        band, pattern, quote = local_nlp_engine.classify_statement(s)
        assert band == BAND_IN_CONTENTION
        assert band.p_play == 0.75

    # Negation test: "not ruled out"
    negated = "The manager confirmed he is not ruled out despite the earlier rumors."
    band_neg, _, _ = local_nlp_engine.classify_statement(negated)
    assert band_neg == BAND_IN_CONTENTION


def test_multi_player_clause_isolation():
    text = (
        "PlayerA has undergone knee surgery and will miss the next two months. "
        "Meanwhile, PlayerB passed fit and is available for selection."
    )
    roster = [
        {"id": 101, "web_name": "PlayerA", "team_name": "TeamA", "team": 1},
        {"id": 102, "web_name": "PlayerB", "team_name": "TeamA", "team": 1},
    ]
    evidences = local_nlp_engine.extract_calibrated_evidence(
        article_text=text,
        article_url="https://example.com/presser",
        published_dt=datetime.now(UTC),
        candidate_roster=roster,
        target_gw=10,
    )

    by_id = {ev.player_id: ev for ev in evidences}
    assert 101 in by_id
    assert by_id[101].category == EvidenceCategory.RULED_OUT
    assert "surgery" in by_id[101].quote.lower()
