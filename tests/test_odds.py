"""Bookmaker odds -> expected goals: the Poisson fit, fixture lookup and market-implied team ratings."""

import json
import math
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import fpl_oracle.data.odds as odds


def test_implied_goals_reproduce_the_market():
    home, away = odds.implied_goals(1.2, 7.0, 13.0, 1.57, 2.38)  # heavy home favourite, goals expected
    assert 2.2 < home < 2.9 and 0.4 < away < 0.8
    even_home, even_away = odds.implied_goals(2.7, 3.3, 2.7, 2.0, 1.85)
    assert abs(even_home - even_away) < 0.15
    assert odds.implied_goals("", 3, 3, 2, 2) is None


def test_lookup_and_ratings_use_names_and_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    rows = ["Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5"]
    teams = ["Arsenal", "Man United", "Tottenham", "Hull", "Chelsea", "Everton"]
    day = 1
    for i, h in enumerate(teams):  # every team plays every other: enough prices for ratings
        for a in teams[i + 1 :] + teams[:i]:
            price = "1.5,4.2,6.0" if h == "Arsenal" else ("6.0,4.2,1.5" if a == "Arsenal" else "2.6,3.3,2.8")
            rows.append(f"{day:02d}/09/2026,{h},{a},{price},1.9,1.9")
            day = day % 28 + 1
    (tmp_path / "2026-27.csv").write_text("\n".join(rows))
    assert odds.market_goals("Arsenal", "Man Utd", "2026-09-01T14:00:00Z") is not None  # FPL name matches
    assert odds.market_goals("Arsenal", "Man Utd", "2026-12-01T14:00:00Z") is None  # different date: not that game
    strong = odds.projected_market_goals("Arsenal", "Hull City", datetime(2026, 11, 1))
    weak = odds.projected_market_goals("Hull City", "Arsenal", datetime(2026, 11, 1))
    assert strong and weak and strong[0] > weak[0] and math.isfinite(strong[1])


HEADER = "Div,Date,Time,HomeTeam,AwayTeam," + ",".join(odds.FIELDS[3:])
NOW = datetime(2026, 10, 19, tzinfo=UTC)
SEASON_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def _csv(*matches: tuple[str, str, str]) -> str:
    return "\n".join([HEADER] + [f"E0,{d},12:30,{h},{a},1.5,4.2,6.0,1.9,1.9" for d, h, a in matches]) + "\n"


def _serve(monkeypatch, tmp_path, pages: dict[str, httpx.Response]):
    """Answer football-data requests from canned responses through a real httpx client (redirects included)."""
    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    monkeypatch.setattr(odds, "UPCOMING_FILE", tmp_path / "upcoming.csv")
    monkeypatch.setattr(odds, "refresh_scorer_odds", lambda: 0)
    monkeypatch.setattr("fpl_oracle.config.ODDS_API_KEY", "")
    real = httpx.Client

    def get(url, **kwargs):
        with real(transport=httpx.MockTransport(lambda r: pages[str(r.url)])) as client:
            return client.get(url, **kwargs)

    monkeypatch.setattr(odds.httpx, "get", get)


def test_refresh_follows_the_redirect_to_the_bare_domain(tmp_path, monkeypatch):
    redirect = httpx.Response(302, headers={"location": "https://football-data.co.uk/x"})
    page = httpx.Response(200, text=_csv(("24/10/2026", "Arsenal", "Hull")))
    _serve(monkeypatch, tmp_path, {SEASON_URL: redirect, FIXTURES_URL: redirect, "https://football-data.co.uk/x": page})
    assert odds.refresh_odds(NOW) == {"2026-27": 1, "upcoming": 1, "scorers": 0}
    assert odds.market_goals("Arsenal", "Hull", "2026-10-24T14:00:00Z") is not None


def test_html_or_smaller_downloads_never_replace_the_odds_file(tmp_path, monkeypatch):
    good = _csv(("01/09/2026", "Arsenal", "Hull"), ("02/09/2026", "Chelsea", "Everton"))
    (tmp_path / "2026-27.csv").write_text(good)
    html = httpx.Response(200, text="<html><body>Please sign in</body></html>")
    empty = httpx.Response(200, text=_csv())
    _serve(monkeypatch, tmp_path, {SEASON_URL: html, FIXTURES_URL: empty})
    with pytest.raises(RuntimeError, match="no usable"):
        odds.refresh_odds(NOW)
    smaller = httpx.Response(200, text=_csv(("01/09/2026", "Arsenal", "Hull")))
    _serve(monkeypatch, tmp_path, {SEASON_URL: smaller, FIXTURES_URL: empty})
    with pytest.raises(RuntimeError):
        odds.refresh_odds(NOW)
    assert (tmp_path / "2026-27.csv").read_text() == good
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_downloads_are_logged_and_partial_success_is_reported(tmp_path, monkeypatch, caplog):
    _serve(monkeypatch, tmp_path, {SEASON_URL: httpx.Response(503), FIXTURES_URL: httpx.Response(404)})
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        odds.refresh_odds(NOW)
    assert "Season odds download failed" in caplog.text and "Upcoming odds download failed" in caplog.text
    fixtures = httpx.Response(200, text=_csv(("24/10/2026", "Arsenal", "Hull")))
    _serve(monkeypatch, tmp_path, {SEASON_URL: httpx.Response(503), FIXTURES_URL: fixtures})
    assert odds.refresh_odds(NOW) == {"upcoming": 1, "scorers": 0}


def test_upcoming_odds_are_used_for_next_gameweek_fixtures(tmp_path, monkeypatch):
    """FPL spellings resolve to a priced upcoming row, which beats the ratings fallback."""
    from fpl_oracle.data.features import MARKET_USE, with_market_odds

    monkeypatch.setattr(odds, "ODDS_DIR", tmp_path)
    rows = _csv(("24/10/2026", "Tottenham", "Ipswich"), ("25/10/2026", "Nott'm Forest", "Coventry"))
    (tmp_path / "upcoming.csv").write_text(rows)
    MARKET_USE.update(odds=0, ratings=0, model=0)
    dc = {"implied_team_xG": 9.0}
    for home, away, day in (("Spurs", "Ipswich", 24), ("Nott'm Forest", "Coventry", 25)):
        assert with_market_odds(dc, home, away, f"2026-10-{day}T14:00:00Z", True)["implied_team_xG"] != 9.0
    assert with_market_odds(dc, "Spurs", "Ipswich", "2026-11-24T14:00:00Z", True) is dc  # no odds, no ratings
    assert MARKET_USE == {"odds": 2, "ratings": 0, "model": 1}


def test_scorer_prices_older_than_a_week_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(odds, "SCORERS_FILE", tmp_path / "scorers.json")

    def save(age_days: float | None):
        payload: dict = {"fixtures": {"77": {"5": 0.4}}}
        if age_days is not None:
            payload["fetched_at"] = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
        (tmp_path / "scorers.json").write_text(json.dumps(payload))

    save(2)
    assert odds.scorer_probabilities() == {(77, 5): 0.4}
    save(8)
    assert odds.scorer_probabilities() == {}
    save(None)
    assert odds.scorer_probabilities() == {}
