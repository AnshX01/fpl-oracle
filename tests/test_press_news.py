"""Press / team-news: disambiguation, safety rules for applying evidence, 72h window. No network."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

from fpl_oracle.api.models import Element
from fpl_oracle.news import press, press_log
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.reconcile import AvailabilityReconciler

NOW = datetime.now(UTC)
BBC = "https://www.bbc.co.uk/sport/football/articles/abc"


def _boot():
    teams = [NS(id=1, name="Chelsea", short_name="CHE"), NS(id=2, name="Everton", short_name="EVE"),
             NS(id=3, name="Aston Villa", short_name="AVL"), NS(id=4, name="Man Utd", short_name="MUN")]  # fmt: skip
    mk = lambda i, n, t, f="X": NS(id=i, web_name=n, team=t, first_name=f, second_name=n)  # noqa: E731
    return NS(
        teams=teams,
        elements=[mk(10, "Martinez", 1), mk(11, "Martinez", 3), mk(12, "Haaland", 2), mk(13, "Silva", 4, "Bernardo")],
    )


def _art(title, summary="", link=BBC, age_h=2):
    return {"title": title, "summary": summary, "link": link, "published_dt": NOW - timedelta(hours=age_h)}


def _elem(**kw):
    base = dict(id=10, web_name="Martinez", team=1, element_type=2, status="a", chance_of_playing_next_round=100,
                news="", now_cost=50)  # fmt: skip
    return Element(**{**base, **kw})


def _ev(cat=EvidenceCategory.RULED_OUT, age_h=2, url=BBC, quote="Martinez is out for six weeks.", **kw):
    return PlayerEvidence(player_id=10, player_name="Martinez", category=cat, quote=quote, source_url=url,
                          published_at=NOW - timedelta(hours=age_h), confidence=0.9, **kw)  # fmt: skip


def _rec(evs, element=None, target=8, first=8):
    return AvailabilityReconciler(mode="press").reconcile_player_fixture(
        element or _elem(), target, candidate_evidence=evs, first_gw=first
    )


def test_shared_surname_needs_club_mention():
    idx = press.PlayerIndex(_boot())
    assert press.extract_article_evidence(_art("Martinez is ruled out for the weekend"), "", idx) == []
    evs = press.extract_article_evidence(_art("Chelsea defender Martinez is ruled out for the weekend"), "", idx)
    assert [(e.player_id, e.category) for e in evs] == [(10, EvidenceCategory.RULED_OUT)]


def test_unique_surname_needs_no_club():
    evs = press.extract_article_evidence(_art("Haaland is ruled out for the weekend"), "", press.PlayerIndex(_boot()))
    assert [e.player_id for e in evs] == [12]


def test_manager_with_same_surname_is_not_the_player():
    evs = press.extract_article_evidence(
        _art("Man Utd: Marco Silva says he will be without a key man"), "", press.PlayerIndex(_boot())
    )
    assert evs == []


def test_jaw_article_marks_chelseas_martinez_out_for_next_gw():
    art = _art("Martinez breaks jaw in Chelsea win at Everton")
    evs = press.extract_article_evidence(art, "", press.PlayerIndex(_boot()))
    assert [e.player_id for e in evs] == [10]
    r = _rec(evs)
    assert r.applied_to_production and r.effective_chance_of_playing == 0 and r.source_url == BBC
    assert "jaw" in r.source_quote
    assert not _rec(evs, target=9, first=8).applied_to_production  # later GW keeps the official decay


def test_press_never_raises_official_chance():
    el = _elem(chance_of_playing_next_round=25, status="d")
    r = _rec([_ev(EvidenceCategory.DOUBTFUL, quote="Martinez is a doubt for Saturday.")], el)  # 50 > 25
    assert not r.applied_to_production and r.effective_chance_of_playing == 25
    r = _rec(
        [_ev(EvidenceCategory.AVAILABLE, quote="Martinez is fully fit.")],
        _elem(chance_of_playing_next_round=0, status="i"),
    )
    assert not r.applied_to_production and r.effective_chance_of_playing == 0


def test_categories_map_to_chances():
    assert (
        _rec([_ev(EvidenceCategory.DOUBTFUL, quote="Martinez faces a late fitness test.")]).effective_chance_of_playing
        == 50
    )
    assert _rec([_ev(EvidenceCategory.DOUBTFUL, quote="Martinez is a major doubt.")]).effective_chance_of_playing == 25
    r = _rec([_ev(EvidenceCategory.MINUTES_LIMIT, quote="Martinez is only fit for the bench.", minutes_restriction=30)])
    assert r.applied_to_production and r.expected_minutes_limit == 30 and r.effective_chance_of_playing == 100


def test_72h_cutoff_and_untrusted_and_missing_quote():
    assert not _rec([_ev(age_h=73)]).applied_to_production
    assert _rec([_ev(age_h=71)]).applied_to_production
    assert not _rec([_ev(url="https://example.com/x")]).applied_to_production
    assert not _rec([_ev(quote="")]).applied_to_production
    assert [a["title"] for a in press.recent_articles([{"title": "old", "published_at": (NOW - timedelta(hours=80)).isoformat()},
                                                       {"title": "new", "published_at": (NOW - timedelta(hours=5)).isoformat()},
                                                       {"title": "undated"}])] == ["new"]  # fmt: skip


def test_official_flag_change_after_article_wins():
    el = _elem(news="Knock - 75% chance of playing", news_added=(NOW - timedelta(hours=1)).isoformat())
    assert not _rec([_ev(age_h=3)], el).applied_to_production
    el = _elem(news="Knock", news_added=(NOW - timedelta(hours=10)).isoformat(), chance_of_playing_next_round=75)
    assert _rec([_ev(age_h=3)], el).effective_chance_of_playing == 0


def test_shadow_mode_applies_nothing():
    r = AvailabilityReconciler(mode="shadow").reconcile_player_fixture(
        _elem(), 8, candidate_evidence=[_ev()], first_gw=8
    )
    assert not r.applied_to_production


def test_applied_signals_logged_and_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(press_log.config, "DATA_DIR", tmp_path)
    sig = dict(element_id=10, web_name="Martinez", team_name="Chelsea", signal_type="PRESS_RULED_OUT", chance_of_playing=0.0,
               baseline_chance=100.0, quote="Martinez is out.", source_url=BBC, applied_to_production=True)  # fmt: skip
    assert press_log.log_applied([sig], 8) == 1 and press_log.log_applied([sig], 8) == 0
    rep = press_log.score_gameweek(8, {10: 0})
    assert rep["ruled_out_n"] == 1 and rep["ruled_out_precision"] == 1.0
    note = press.applied_press_notes([sig], {10})[0]["text"]
    assert note == 'Press: Martinez ruled out — "Martinez is out." (BBC)'


def test_known_false_positives_from_live_feeds_are_not_evidence():
    idx = press.PlayerIndex(_boot())
    for text in (
        "There was an injury-time winner from, miracle of miracles, Haaland.",  # 'acl' inside 'miracle'
        "Haaland had his first two strikes ruled out for offside.",
        '"Hopefully that is the case," said Haaland, "but you just keep working."',
        "Everton were reduced to ten men after a red card for Haaland.",
    ):
        assert press.extract_article_evidence(_art(text), "", idx) == [], text
