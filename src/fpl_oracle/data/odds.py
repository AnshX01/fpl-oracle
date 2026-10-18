"""
Bookmaker odds -> expected goals per team per match (docs/decision_model.md section 1).

Betting markets are the most accurate public estimate of how many goals each side will score. For every
Premier League match we take the market-average 1X2 and over/under-2.5 prices, remove the bookmaker
margin, and fit the two Poisson goal rates (home, away) that reproduce those probabilities. The same
lookup feeds training (past matches) and live advice (upcoming matches); without odds for a match the
caller keeps its own team-strength model.

Free sources, no key needed: football-data.co.uk season files (odds collected on Friday / Tuesday
afternoon, i.e. around the FPL deadline) and its fixtures.csv for upcoming matches. With ODDS_API_KEY set,
The Odds API adds upcoming odds earlier in the week.
"""

import csv
import json
import logging
import math
import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from fpl_oracle.config import DATA_DIR

logger = logging.getLogger("fpl_oracle.data.odds")

ODDS_DIR = DATA_DIR / "odds"
UPCOMING_FILE = ODDS_DIR / "upcoming.csv"
SCORERS_FILE = ODDS_DIR / "scorers.json"
# Anytime-goalscorer prices averaged over several bookmakers for the next gameweek's fixtures, keyed by FPL
# fixture and player ids (embedded in the page's data). Free, no key.
SCORERS_URL = "https://kickoffscore.com/fpl/odds"
FIELDS = ("Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5")

# football-data / The Odds API names -> the names FPL uses
ALIASES = {
    "man united": "man utd",
    "manchester united": "man utd",
    "manchester city": "man city",
    "tottenham": "spurs",
    "tottenham hotspur": "spurs",
    "nottingham forest": "nott'm forest",
    "newcastle united": "newcastle",
    "wolverhampton wanderers": "wolves",
    "west ham united": "west ham",
    "brighton and hove albion": "brighton",
    "sheffield united": "sheffield utd",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "luton town": "luton",
    "afc bournemouth": "bournemouth",
}


def team_key(name: str) -> str:
    """Comparable team name: lower case, FPL spelling, no 'City'/'Town' suffix."""
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    key = ALIASES.get(key, key)
    return re.sub(r" (city|town|fc|afc)$", "", key)


def implied_goals(home: Any, draw: Any, away: Any, over: Any, under: Any) -> tuple[float, float] | None:
    """Poisson goal rates (home, away) matching margin-free 1X2 and over/under-2.5 probabilities."""
    try:
        inv = [1 / float(o) for o in (home, draw, away)]
        ou = [1 / float(over), 1 / float(under)]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(v) and v > 0 for v in inv + ou):
        return None
    p_h, p_d, p_a = (v / sum(inv) for v in inv)
    p_over = ou[0] / sum(ou)
    goals = np.arange(11)
    lower = np.tril(np.ones((11, 11)), -1).astype(bool)  # home goals > away goals
    low_total = np.add.outer(goals, goals) <= 2

    def loss(rates):
        joint = np.outer(poisson.pmf(goals, rates[0]), poisson.pmf(goals, rates[1]))
        home_win, drawn = joint[lower].sum(), np.trace(joint)
        over = 1.0 - joint[low_total].sum()
        return (home_win - p_h) ** 2 + (drawn - p_d) ** 2 + (1 - home_win - drawn - p_a) ** 2 + (over - p_over) ** 2

    fit = minimize(loss, x0=[1.4, 1.1], bounds=[(0.1, 5.0), (0.1, 5.0)], method="L-BFGS-B")
    return (float(fit.x[0]), float(fit.x[1])) if fit.success else None


def _parse_date(text: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip()[:10], fmt)
        except ValueError:
            continue
    return None


@lru_cache(maxsize=1)
def _index(version: float) -> dict[tuple[str, str], list[tuple[datetime, float, float]]]:
    """(home, away) -> [(date, home rate, away rate)] from every odds file on disk."""
    index: dict[tuple[str, str], list[tuple[datetime, float, float]]] = {}
    for path in sorted(ODDS_DIR.glob("*.csv")):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                when = _parse_date(row.get("Date", ""))
                rates = implied_goals(*(row.get(k) for k in FIELDS[3:])) if when else None
                if rates and when:
                    key = (team_key(row["HomeTeam"]), team_key(row["AwayTeam"]))
                    index.setdefault(key, []).append((when, *rates))
    return index


def market_goals(home: str, away: str, kickoff: datetime | str | None) -> tuple[float, float] | None:
    """Market expected goals (home, away) for this fixture, or None if no odds are held for it."""
    version = max((p.stat().st_mtime for p in ODDS_DIR.glob("*.csv")), default=0.0)
    rows = _index(version).get((team_key(home), team_key(away)))
    if not rows:
        return None
    if isinstance(kickoff, str):
        try:
            kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError:
            kickoff = None
    if kickoff is None:
        return None
    day = kickoff.replace(tzinfo=None)
    near = [r for r in rows if abs(r[0] - day) <= timedelta(days=3)]  # same fixture, not the reverse one
    if not near:
        return None
    _, rate_home, rate_away = min(near, key=lambda r: abs(r[0] - day))
    return rate_home, rate_away


def _season_code(now: datetime) -> tuple[str, str]:
    start = now.year if now.month >= 7 else now.year - 1
    return f"{start}-{str(start + 1)[2:]}", f"{str(start)[2:]}{str(start + 1)[2:]}"


def _keep_columns(text: str) -> list[dict[str, str]]:
    return [
        {k: row.get(k, "") for k in FIELDS}
        for row in csv.DictReader(StringIO(text.lstrip("﻿")))
        if row.get("Div", "E0") == "E0"
    ]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _odds_api_rows(api_key: str) -> list[dict[str, str]]:
    """Upcoming EPL odds from The Odds API (free key), averaged across bookmakers, in football-data columns."""
    resp = httpx.get(
        "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
        params={"apiKey": api_key, "regions": "uk", "markets": "h2h,totals", "oddsFormat": "decimal"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for event in resp.json():
        prices: dict[str, list[float]] = {}
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                for out in market.get("outcomes", []):
                    if market["key"] == "h2h":
                        label = {event["home_team"]: "AvgH", event["away_team"]: "AvgA"}.get(out["name"], "AvgD")
                    elif market["key"] == "totals" and float(out.get("point", 0)) == 2.5:
                        label = "Avg>2.5" if out["name"] == "Over" else "Avg<2.5"
                    else:
                        continue
                    prices.setdefault(label, []).append(float(out["price"]))
        if len(prices) == 5:
            when = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            row = {"Date": when.strftime("%d/%m/%Y"), "HomeTeam": event["home_team"], "AwayTeam": event["away_team"]}
            row.update({k: f"{sum(v) / len(v):.3f}" for k, v in prices.items()})
            rows.append(row)
    return rows


def refresh_odds(now: datetime | None = None) -> dict[str, int]:
    """Download this season's odds and the upcoming fixtures' odds. Returns rows written per file."""
    from fpl_oracle.config import ODDS_API_KEY

    now = now or datetime.now(UTC)
    season, code = _season_code(now)
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    season_rows = _keep_columns(httpx.get(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", timeout=30).text)
    if season_rows:
        _write(ODDS_DIR / f"{season}.csv", season_rows)
        written[season] = len(season_rows)
    upcoming = _keep_columns(httpx.get("https://www.football-data.co.uk/fixtures.csv", timeout=30).text)
    if ODDS_API_KEY:
        try:
            upcoming += _odds_api_rows(ODDS_API_KEY)
        except Exception as error:  # the free sources still stand
            logger.warning("The Odds API unavailable: %s", type(error).__name__)
    _write(UPCOMING_FILE, upcoming)
    written["upcoming"] = len(upcoming)
    try:  # player prices are a bonus: the team odds above stand without them
        written["scorers"] = refresh_scorer_odds()
    except Exception as error:
        logger.warning("Goalscorer odds unavailable: %s", type(error).__name__)
    return written


def parse_scorer_page(html: str) -> dict[int, dict[int, float]]:
    """{FPL fixture id: {FPL player id: bookmaker probability of scoring}} from the page's embedded data.
    Untrusted input: only well-formed numbers in range are kept."""
    text = html.replace('\\"', '"')
    decoder = json.JSONDecoder()
    fixtures: dict[int, dict[int, float]] = {}
    for match in re.finditer(r'\{"fixture":(\d+),', text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        rows = {}
        for s in obj.get("scorers") or []:
            element, implied = s.get("element"), s.get("implied")
            if isinstance(element, int) and isinstance(implied, int | float) and 0.0 < implied < 1.0:
                rows[element] = float(implied)
        if rows:
            fixtures[int(obj["fixture"])] = rows
    return fixtures


def refresh_scorer_odds() -> int:
    """Download the next gameweek's anytime-goalscorer prices; returns the number of player prices kept."""
    resp = httpx.get(SCORERS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    fixtures = parse_scorer_page(resp.text)
    if not fixtures:
        raise ValueError("no goalscorer prices found on the page")
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SCORERS_FILE.with_suffix(".json.tmp")
    payload = {
        "source": SCORERS_URL,
        "fetched_at": datetime.now(UTC).isoformat(),
        "fixtures": {str(f): {str(e): p for e, p in rows.items()} for f, rows in fixtures.items()},
    }
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(SCORERS_FILE)
    return sum(len(rows) for rows in fixtures.values())


def scorer_probabilities() -> dict[tuple[int, int], float]:
    """(fixture, player) -> bookmaker chance that he scores if he plays (margin still in; the attacking model
    removes it per team against the match odds)."""
    if not SCORERS_FILE.exists():
        return {}
    data = json.loads(SCORERS_FILE.read_text(encoding="utf-8"))
    return {(int(f), int(e)): float(p) for f, rows in data.get("fixtures", {}).items() for e, p in rows.items()}


@lru_cache(maxsize=64)
def _ratings(version: float, as_of: datetime) -> dict[str, float] | None:
    """Team attack/defence ratings fitted to market goal rates of the last RATING_WINDOW_DAYS before as_of:
    log rate(home) = mu + home + att[home] - def[away]; log rate(away) = mu + att[away] - def[home].
    Ridge-regularised least squares, so a team with few priced matches stays near average."""
    rows = [
        (h, a, when, rh, ra)
        for (h, a), games in _index(version).items()
        for when, rh, ra in games
        if as_of - timedelta(days=RATING_WINDOW_DAYS) <= when < as_of
    ]
    teams = sorted({t for h, a, *_ in rows for t in (h, a)})
    if len(rows) < 20:
        return None
    col = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    design, target = [], []
    for h, a, when, rh, ra in rows:
        weight = 0.5 ** ((as_of - when).days / RATING_HALF_LIFE_DAYS)
        for attacker, defender, rate, home in ((h, a, rh, 1.0), (a, h, ra, 0.0)):
            x = np.zeros(2 * n + 2)
            x[0], x[1], x[2 + col[attacker]], x[2 + n + col[defender]] = 1.0, home, 1.0, -1.0
            design.append(x * math.sqrt(weight))
            target.append(math.log(rate) * math.sqrt(weight))
    a_mat, b_vec = np.array(design), np.array(target)
    ridge = np.eye(2 * n + 2) * 0.5
    ridge[0, 0] = ridge[1, 1] = 0.0
    coef = np.linalg.solve(a_mat.T @ a_mat + ridge, a_mat.T @ b_vec)
    ratings = {"mu": coef[0], "home": coef[1]}
    for t in teams:
        ratings[f"att:{t}"], ratings[f"def:{t}"] = coef[2 + col[t]], coef[2 + n + col[t]]
    return ratings


RATING_WINDOW_DAYS = 240  # about 25 league weeks of prices, across a summer break when needed
RATING_HALF_LIFE_DAYS = 60  # recent prices count more (form, injuries, managers)


def projected_market_goals(home: str, away: str, kickoff: datetime | str | None) -> tuple[float, float] | None:
    """Expected goals for a fixture without its own odds, from market ratings as of today."""
    version = max((p.stat().st_mtime for p in ODDS_DIR.glob("*.csv")), default=0.0)
    if isinstance(kickoff, str):
        try:
            kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError:
            return None
    if kickoff is None:
        return None
    as_of = min(datetime.now(UTC).replace(tzinfo=None), kickoff.replace(tzinfo=None)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    ratings = _ratings(version, as_of)
    h, a = team_key(home), team_key(away)
    if not ratings or f"att:{h}" not in ratings or f"att:{a}" not in ratings:
        return None
    rate_h = math.exp(ratings["mu"] + ratings["home"] + ratings[f"att:{h}"] - ratings[f"def:{a}"])
    rate_a = math.exp(ratings["mu"] + ratings[f"att:{a}"] - ratings[f"def:{h}"])
    return rate_h, rate_a
