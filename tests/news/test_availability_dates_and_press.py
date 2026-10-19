"""Return-date availability, press duration / supersession / strict ruled-out, extractor precision and name matching."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from fpl_oracle.news.extract import surname_token, text_extractor
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.press import PlayerIndex, extract_article_evidence
from fpl_oracle.news.reconcile import AvailabilityReconciler, absence_gameweeks, official_chance_of_playing

EVENTS = [
    SimpleNamespace(id=1, deadline_time="2026-08-21T17:30:00Z"),
    SimpleNamespace(id=7, deadline_time="2026-10-17T10:00:00Z"),
    SimpleNamespace(id=8, deadline_time="2026-10-23T17:30:00Z"),
    SimpleNamespace(id=9, deadline_time="2026-10-31T11:00:00Z"),
    SimpleNamespace(id=10, deadline_time="2026-11-06T18:30:00Z"),
    SimpleNamespace(id=11, deadline_time="2026-11-21T11:00:00Z"),
]


def _player(status, news, cop=0, **kw):
    return SimpleNamespace(
        id=kw.pop("id", 1),
        web_name="X",
        team=1,
        status=status,
        news=news,
        news_added=None,
        chance_of_playing_next_round=cop,
        chance_of_playing_this_round=None,
        scout_risks=[],
        **kw,
    )


def _horizon(element, events=EVENTS):
    return [official_chance_of_playing(element, gw, 8, events) for gw in range(8, 12)]


def test_injury_with_stated_date_is_zero_until_the_gameweek_after_it_then_ramps():
    porro = _player("i", "Hamstring injury - Expected back 24 Oct")
    assert _horizon(porro) == [0.0, 50.0, 75.0, 100.0]


def test_suspension_returns_at_full_chance_on_the_stated_date():
    janelt = _player("s", "Suspended until 31 Oct")
    assert _horizon(janelt) == [0.0, 100.0, 100.0, 100.0]


def test_unknown_return_date_stays_out_for_the_whole_horizon():
    szoboszlai = _player("i", "Knee injury - Unknown return date")
    assert _horizon(szoboszlai) == [0.0] * 4
    assert _horizon(_player("i", "Hamstring injury - Expected back 24 Oct"), events=None) == [0.0] * 4


def test_date_already_passed_with_flag_still_on_is_stale_and_ramps_from_next_gameweek():
    tomiyasu = _player("i", "Calf injury - Expected back 18 Oct")
    assert _horizon(tomiyasu) == [50.0, 75.0, 100.0, 100.0]
    assert _horizon(_player("s", "Suspended until 15 Oct")) == [50.0, 75.0, 100.0, 100.0]


def test_loaned_players_are_never_revived_by_a_date():
    lynch = _player("u", "Has joined Oxford United on loan until January")
    assert _horizon(lynch) == [0.0] * 4


def test_doubts_are_unchanged_by_events():
    assert official_chance_of_playing(_player("d", "Knee injury - 75% chance of playing", cop=75), 9, 8, EVENTS) == 87.5


# --- extractor precision ---------------------------------------------------------------------------------------------


def _category(text, name):
    return text_extractor.extract_evidence_from_text(text, name, 1)[0].category


@pytest.mark.parametrize(
    "text, name",
    [
        ("There is no suggestion Alexander Isak will miss the game against Chelsea.", "Isak"),
        ("Alexander Isak undoubtedly looks sharp in training.", "Isak"),
        ("Jack Grealish scored a stunning knockdown volley to make it 2-0.", "Grealish"),
        ("There is no sign of a setback for Martin Odegaard.", "Odegaard"),
        ("Ollie Watkins is not expected to be out for long, Unai Emery said.", "Watkins"),
        ("Dominic Solanke has not been ruled out of Saturday's game, according to Ange Postecoglou.", "Solanke"),
        ("Antoine Semenyo is no longer a doubt for the visit of Wolves.", "Semenyo"),
    ],
)
def test_extractor_false_positives_are_not_ruled_out_or_doubtful(text, name):
    assert _category(text, name) in (EvidenceCategory.AVAILABLE, EvidenceCategory.UNKNOWN, EvidenceCategory.DOUBTFUL)
    assert _category(text, name) != EvidenceCategory.RULED_OUT


@pytest.mark.parametrize(
    "text, name, expected",
    [
        ("Alexander Isak undoubtedly looks sharp in training.", "Isak", EvidenceCategory.UNKNOWN),
        ("Jack Grealish scored a stunning knockdown volley to make it 2-0.", "Grealish", EvidenceCategory.UNKNOWN),
        ("Mohamed Salah is a doubt for Sunday's game.", "Salah", EvidenceCategory.DOUBTFUL),
        ("Eberechi Eze faces up to six weeks out after damaging his hamstring.", "Eze", EvidenceCategory.RULED_OUT),
    ],
)
def test_extractor_whole_word_matching(text, name, expected):
    assert _category(text, name) == expected


# --- name matching ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "web_name, token",
    [
        ("B.Fernandes", "fernandes"),
        ("N'Diaye", "diaye"),
        ("Gibbs-White", "white"),
        ("Kroupi.Jr", "kroupi"),
        ("Salah", "salah"),
    ],
)
def test_surname_token_handles_initials_and_apostrophes(web_name, token):
    assert surname_token(web_name) == token


def _index():
    teams = [
        SimpleNamespace(id=1, name="Man Utd", short_name="MUN"),
        SimpleNamespace(id=2, name="Brentford", short_name="BRE"),
        SimpleNamespace(id=3, name="Spurs", short_name="TOT"),
    ]
    elements = [
        SimpleNamespace(id=10, web_name="B.Fernandes", first_name="Bruno", second_name="Fernandes", team=1),
        SimpleNamespace(id=11, web_name="Fernandes", first_name="Rui", second_name="Fernandes", team=2),
        SimpleNamespace(id=12, web_name="N'Diaye", first_name="Habib", second_name="Diaye", team=3),
    ]
    return PlayerIndex(SimpleNamespace(teams=teams, elements=elements))


def _evidence(body):
    art = {"title": body, "summary": "", "link": "https://www.bbc.co.uk/x", "published_dt": datetime.now(UTC)}
    return {ev.player_id: ev for ev in extract_article_evidence(art, "", _index())}


def test_dotted_and_apostrophe_names_match_and_keep_the_club_check():
    ev = _evidence("Manchester United will be without Bruno Fernandes, who is ruled out for six weeks.")
    assert ev[10].category == EvidenceCategory.RULED_OUT and 11 not in ev  # Brentford's Fernandes stays unmatched
    assert (
        _evidence("Tottenham midfielder Habib N'Diaye is ruled out of the derby.")[12].category
        == EvidenceCategory.RULED_OUT
    )
    assert _evidence("Fernandes is ruled out for the weekend.") == {}  # no club named: ambiguous, no match


# --- press: duration, supersession, strict ruled-out -----------------------------------------------------------------


def _press(category, quote, hours_ago, url="https://www.bbc.co.uk/sport/a"):
    return PlayerEvidence(
        player_id=1,
        player_name="X",
        category=category,
        quote=quote,
        source_url=url,
        published_at=datetime.now(UTC) - timedelta(hours=hours_ago),
        confidence=0.95,
    )


def _applied(element, evidence, gw=8, first_gw=8):
    rec = AvailabilityReconciler(mode="press").reconcile_player_fixture(
        element, gw, candidate_evidence=evidence, first_gw=first_gw, events=EVENTS
    )
    return rec.effective_chance_of_playing


OUT = "Isak is out for six weeks with a hamstring tear."
BBC, GUARDIAN = "https://www.bbc.co.uk/sport/a", "https://www.theguardian.com/football/b"


@pytest.mark.parametrize(
    "text, weeks",
    [
        ("He is out for six weeks.", 6),
        ("He will be out for a month.", 4),
        ("He will miss the next three games.", 3),
        ("Out for two months.", 8),
        ("He is out for the weekend.", 1),
    ],
)
def test_absence_gameweeks(text, weeks):
    assert absence_gameweeks(text) == weeks


def test_stated_absence_spans_its_period_and_then_reverts_to_the_official_figure():
    player = _player("a", "", cop=None)
    evidence = [_press(EvidenceCategory.RULED_OUT, OUT, 2, BBC), _press(EvidenceCategory.RULED_OUT, OUT, 3, GUARDIAN)]
    assert [_applied(player, evidence, gw) for gw in range(8, 15)] == [0.0] * 6 + [100.0]


def test_doubt_and_unstated_ruled_out_apply_to_the_next_gameweek_only():
    player = _player("d", "", cop=75)
    evidence = [_press(EvidenceCategory.RULED_OUT, "Isak is ruled out of Saturday's game.", 2)]
    assert _applied(player, evidence, 8) == 0.0
    assert _applied(player, evidence, 9) == 87.5


def test_ruled_out_from_one_source_with_no_fpl_flag_is_only_a_doubt():
    player = _player("a", "", cop=None)
    one = [_press(EvidenceCategory.RULED_OUT, OUT, 2, BBC)]
    assert _applied(player, one) == 75.0
    assert _applied(player, one + [_press(EvidenceCategory.RULED_OUT, OUT, 3, GUARDIAN)]) == 0.0
    assert _applied(player, one + [_press(EvidenceCategory.RULED_OUT, OUT, 3, BBC.replace("/a", "/c"))]) == 75.0


def test_ruled_out_from_one_source_with_an_fpl_doubt_goes_to_zero():
    assert _applied(_player("d", "", cop=75), [_press(EvidenceCategory.RULED_OUT, OUT, 2)]) == 0.0


def test_newer_fit_news_cancels_an_older_doubt_but_not_a_newer_one():
    player = _player("a", "", cop=None)
    doubt = _press(EvidenceCategory.DOUBTFUL, "Isak is a doubt for Saturday.", 30)
    fit = _press(EvidenceCategory.AVAILABLE, "Isak is fully fit.", 5, GUARDIAN)
    assert _applied(player, [doubt, fit]) == 100.0
    assert _applied(player, [_press(EvidenceCategory.DOUBTFUL, "Isak is a doubt for Saturday.", 2), fit]) == 50.0


def test_negated_fit_news_still_counts_as_fit_news():
    player = _player("a", "", cop=None)
    doubt = _press(EvidenceCategory.DOUBTFUL, "Isak is a doubt for Saturday.", 30)
    fit = _press(EvidenceCategory.AVAILABLE, "There is no suggestion Isak will miss the game.", 5, GUARDIAN)
    fit.is_negated = True
    assert _applied(player, [doubt, fit]) == 100.0
