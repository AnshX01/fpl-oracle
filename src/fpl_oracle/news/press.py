"""
Press / team-news helpers: recent-article window, trusted article body fetch, player name
disambiguation and sentence-level evidence extraction. No network access except fetch_bodies.
"""

import logging
import re
import unicodedata
import urllib.parse
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup

from fpl_oracle.news.extract import text_extractor
from fpl_oracle.news.ingest import is_safe_external_url
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.reconcile import extract_publisher_domain, get_trusted_publishers, normalize_publisher_group

logger = logging.getLogger("fpl_oracle.news.press")

MAX_FETCHES_PER_REFRESH = 60
MAX_BODY_CHARS = 30000
_BODY_CACHE: dict[str, str] = {}  # url -> body text ("" = fetch failed); process lifetime

# Surnames that are also everyday words or very short: always need the club in the article.
COMMON_WORDS = {
    "young", "king", "cook", "gray", "grey", "white", "brown", "wood", "mount", "bell", "hall", "long", "reed",
    "rice", "green", "ward", "love", "strand", "back", "best", "fish", "gap", "case", "page", "ball", "will",
}  # fmt: skip

CLUB_ALIASES = {
    "man utd": ["manchester united", "man united", "man utd"],
    "man city": ["manchester city", "man city"],
    "spurs": ["tottenham hotspur", "tottenham", "spurs"],
    "nott'm forest": ["nottingham forest", "nott'm forest", "forest"],
    "wolves": ["wolverhampton wanderers", "wolverhampton", "wolves"],
    "newcastle": ["newcastle united", "newcastle"],
    "brighton": ["brighton and hove albion", "brighton & hove albion", "brighton"],
    "leeds": ["leeds united", "leeds"],
    "west ham": ["west ham united", "west ham"],
    "aston villa": ["aston villa", "villa"],
    "crystal palace": ["crystal palace", "palace"],
}

_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?\n]+|$)")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text.lower().replace("’", "'").replace("‘", "'")


def parse_published(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def recent_articles(
    articles: list[dict[str, Any]], hours: int = 72, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Articles published within `hours`; each gets `published_dt`. Undated articles are dropped."""
    now = now or datetime.now(UTC)
    out = []
    for art in articles:
        dt = parse_published(art.get("published_at"))
        if dt and timedelta(0) - timedelta(hours=1) <= now - dt <= timedelta(hours=hours):
            out.append({**art, "published_dt": dt})
    return out


def _body_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
        tag.decompose()
    root = soup.find("article") or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    return re.sub(r"\s+", " ", " ".join(p for p in paras if len(p) > 25))[:MAX_BODY_CHARS]


async def fetch_bodies(articles: list[dict[str, Any]], max_fetches: int = MAX_FETCHES_PER_REFRESH) -> dict[str, str]:
    """Body text for trusted-publisher articles (plain GET, 10s timeout, cached by URL, sequential)."""
    trusted = get_trusted_publishers()
    fetched = 0
    async with httpx.AsyncClient(
        timeout=10.0, follow_redirects=False, headers={"User-Agent": "Mozilla/5.0 FPLOracle/1.0"}
    ) as client:
        for art in articles:
            url = art.get("link", "")
            if not url or url in _BODY_CACHE or extract_publisher_domain(url) not in trusted:
                continue
            if fetched >= max_fetches:
                break
            fetched += 1
            body = ""
            try:
                cur = url
                for _ in range(3):
                    if not is_safe_external_url(cur) or extract_publisher_domain(cur) not in trusted:
                        break
                    resp = await client.get(cur)
                    if resp.is_redirect and resp.headers.get("location"):
                        cur = urllib.parse.urljoin(cur, resp.headers["location"])
                        continue
                    if resp.status_code == 200:
                        body = _body_text(resp.text)
                    break
            except Exception as e:
                logger.debug("Body fetch failed for %s: %s", url, e)
            _BODY_CACHE[url] = body
    return {a["link"]: _BODY_CACHE.get(a["link"], "") for a in articles if a.get("link")}


def _surname_key(web_name: str) -> str:
    """Same token the deterministic extractor matches on."""
    tokens = [t for t in re.split(r"[\s\-]+", web_name.strip()) if len(t) >= 3]
    return _norm(tokens[-1] if tokens else web_name.strip())


class PlayerIndex:
    """Surname -> players, with club-mention disambiguation."""

    def __init__(self, bootstrap: Any):
        self.teams = {t.id: t for t in bootstrap.teams}
        self.by_key: dict[str, list[Any]] = {}
        self.key_of: dict[int, str] = {}
        for e in bootstrap.elements:
            key = _surname_key(e.web_name)
            self.by_key.setdefault(key, []).append(e)
            self.key_of[int(e.id)] = key

    def is_ambiguous(self, key: str) -> bool:
        return len(self.by_key.get(key, [])) > 1 or key in COMMON_WORDS or len(key) <= 3

    def club_terms(self, team_id: int) -> list[str]:
        team = self.teams.get(team_id)
        if not team:
            return []
        name = _norm(team.name)
        return CLUB_ALIASES.get(name, [name]) + [_norm(team.short_name)] * (len(team.short_name) > 3)

    def club_mentioned(self, team_id: int, text: str) -> bool:
        low = _norm(text)
        return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in self.club_terms(team_id))

    def resolve(self, key: str, article_text: str, sentence: str) -> list[Any]:
        """Players the token refers to: all of them if unique, else only those whose club is named."""
        players = self.by_key.get(key, [])
        if not self.is_ambiguous(key):
            return players
        in_article = [p for p in players if self.club_mentioned(p.team, article_text)]
        if len(in_article) > 1:
            in_article = [p for p in in_article if self.club_mentioned(p.team, sentence)]
        return in_article if len(in_article) == 1 else []

    def club_ok(self, player_id: int, article_text: str) -> bool:
        """For evidence from other extractors: ambiguous names need their club in the article."""
        key = self.key_of.get(player_id)
        if key is None:
            return False
        if not self.is_ambiguous(key):
            return True
        player = next(p for p in self.by_key[key] if int(p.id) == player_id)
        return self.club_mentioned(player.team, article_text)


def _other_given_name(sentence: str, key: str, player: Any) -> bool:
    """True if every use of the surname follows a different capitalised given name (e.g. 'Marco Silva')."""
    own = {_norm(n) for n in re.split(r"[\s\-]+", f"{player.first_name} {player.second_name}") if n}
    seen = False
    for m in re.finditer(
        r"(?:((?-i:\b[A-Z])[\w'’-]*)\s+)?\b" + re.escape(key) + r"\b", _norm_keep_case(sentence), re.IGNORECASE
    ):
        seen = True
        prev = m.group(1)
        if not prev or prev.lower().endswith(("'s", "s'")) or _norm(prev) in own:
            return False
    return seen


def _speaker_or_coach(sentence: str, key: str) -> bool:
    """True if every use of the surname is a manager / quoted speaker rather than the player."""
    low = _norm(sentence)
    uses = list(re.finditer(r"\b" + re.escape(key) + r"\b", low))
    return bool(uses) and all(
        re.search(r"(?:manager|boss|coach|head coach)\s+(?:\w+\s+)?$|[-\u2013\u2014:]\s*$", low[: m.start()])
        or re.match(
            r"\s+(?:said|says|admitted|confirmed|revealed|insisted|told|added|explained|believes|feels|hopes|expects)\b",
            low[m.end() :],
        )
        for m in uses
    )


def _norm_keep_case(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]


def extract_article_evidence(article: dict[str, Any], body: str, index: PlayerIndex) -> list[PlayerEvidence]:
    """Deterministic evidence for every roster player named in title + summary + body."""
    title = text_extractor.clean_html(article.get("title", ""))
    summary = text_extractor.clean_html(article.get("summary", ""))
    sents = [title] if title else []
    sents += sentences(summary) + sentences(body)
    full = " ".join([title, summary, body])
    out: dict[tuple[int, str], PlayerEvidence] = {}
    for sent in sents:
        for key in set(re.findall(r"[a-z]{3,}", _norm(sent))) & index.by_key.keys():
            for player in index.resolve(key, full, sent):
                if _other_given_name(sent, key, player) or _speaker_or_coach(sent, key):
                    continue
                for ev in text_extractor.extract_evidence_from_text(sent, player.web_name, int(player.id)):
                    if ev.category == EvidenceCategory.UNKNOWN or (int(player.id), ev.quote) in out:
                        continue
                    if "red card" in ev.quote.lower() and not re.search(
                        r"suspen|\bban|miss|ruled out|without", ev.quote, re.I
                    ):
                        continue  # a sending-off alone is not an availability statement; FPL flags suspensions itself
                    ev.source_url = article.get("link", "")
                    ev.published_at = article.get("published_dt")
                    team = index.teams.get(player.team)
                    ev.team_name = team.name if team else ""
                    out[(int(player.id), ev.quote)] = ev
    return list(out.values())


_PUBLISHERS = {"bbc": "BBC", "guardian": "The Guardian", "sky": "Sky Sports", "talksport": "talkSPORT"}


def applied_press_notes(signals: list[dict[str, Any]], element_ids: set[int]) -> list[dict[str, Any]]:
    """Applied press signals for the given players, ready for the decision card / chat."""
    notes = []
    for s in signals:
        if not (s.get("applied_to_production") and str(s.get("signal_type", "")).startswith("PRESS_")):
            continue
        if s["element_id"] not in element_ids:
            continue
        chance, limit = s["chance_of_playing"], s.get("expected_minutes_limit")
        state = (
            "ruled out"
            if chance == 0
            else "major doubt"
            if chance <= 25
            else "doubtful"
            if chance < (s.get("baseline_chance") or 100)
            else f"limited to about {limit:.0f} minutes"
        )
        group = normalize_publisher_group(extract_publisher_domain(s["source_url"]))
        publisher = _PUBLISHERS.get(group, group)
        notes.append(
            {
                "player": s["web_name"],
                "state": state,
                "chance_of_playing": chance,
                "quote": s["quote"],
                "publisher": publisher,
                "source_url": s["source_url"],
                "text": f'Press: {s["web_name"]} {state} — "{s["quote"]}" ({publisher})',
            }
        )
    return notes
