"""Bookmaker odds -> expected goals: the Poisson fit, fixture lookup and market-implied team ratings."""

import math
from datetime import datetime

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
