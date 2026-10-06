"""
News and injury ingestion engine.
Fetches official FPL injury and status data, and pulls live RSS feeds from
verified football news outlets with SSRF protection, SHA256 deduplication,
and bounded rate limits.
"""

import hashlib
import ipaddress
import logging
import socket
import urllib.parse
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.config import NEWS_SOURCES

logger = logging.getLogger("fpl_oracle.news.ingest")

SAFE_ALLOWED_SCHEMES = {"http", "https"}


def is_safe_ip(ip_str: str) -> bool:
    """Validate whether an IP address is safe (not private, loopback, metadata, or reserved)."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_reserved
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
        # Block cloud metadata addresses explicitly
        if ip_str in ("169.254.169.254", "fd00:ec2::254"):
            return False
        return True
    except ValueError:
        return False


def is_safe_external_url(url: str, resolve_dns: bool = True) -> bool:
    """SSRF Protection: Block private network, loopback, cloud metadata, and disallowed schemes with DNS resolution (N6)."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in SAFE_ALLOWED_SCHEMES:
            return False
        hostname = (parsed.hostname or "").lower()
        if not hostname or hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"):
            return False

        # If hostname is an IP literal
        try:
            ipaddress.ip_address(hostname)
            return is_safe_ip(hostname)
        except ValueError:
            pass  # It is a domain name, proceed to DNS resolution

        # Resolve hostname via DNS to prevent rebinding
        if resolve_dns:
            try:
                addr_info = socket.getaddrinfo(hostname, None)
                if not addr_info:
                    return False
                for _family, _, _, _, sockaddr in addr_info:
                    ip_str = sockaddr[0]
                    if not is_safe_ip(ip_str):
                        return False
            except socket.gaierror:
                return False

        return True
    except Exception:
        return False


class NewsIngestion:
    def __init__(self):
        self.sources_cfg = NEWS_SOURCES.get("rss_feeds", [])
        self.dead_feeds: set[str] = set()
        self.last_fetch_time: datetime | None = None
        self.last_articles_count: int = 0
        self._seen_article_hashes: set[str] = set()

    def get_news_status(self) -> dict[str, Any]:
        """Return diagnostic metrics for news pipeline."""
        total_feeds = len(self.sources_cfg)
        active_feeds = total_feeds - len(self.dead_feeds)
        return {
            "last_fetch": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            "total_configured_feeds": total_feeds,
            "active_feeds": active_feeds,
            "dead_feeds_count": len(self.dead_feeds),
            "dead_feed_ids": list(self.dead_feeds),
            "last_articles_count": self.last_articles_count,
        }

    def fetch_official_fpl_signals(self, bootstrap: BootstrapStatic) -> list[dict[str, Any]]:
        """
        Extract authoritative injury, suspension, and availability updates from FPL bootstrap.
        Distinguishes source publication time from local fetch time.
        """
        updates = []
        team_map = {t.id: t.name for t in bootstrap.teams}
        now_iso = datetime.now(UTC).isoformat()

        for elem in bootstrap.elements:
            has_news = bool(elem.news and elem.news.strip())
            not_fully_available = elem.status != "a" or (
                elem.chance_of_playing_next_round is not None and elem.chance_of_playing_next_round < 100
            )

            if has_news or not_fully_available:
                # Check for scout risks like loan ineligibility
                risks_list = getattr(elem, "scout_risks", None) or []
                risk_types = [
                    (r.property if hasattr(r, "property") else r.get("property", ""))
                    for r in risks_list
                ]

                updates.append(
                    {
                        "element_id": elem.id,
                        "web_name": elem.web_name,
                        "team_id": elem.team,
                        "team_name": team_map.get(elem.team, f"Team {elem.team}"),
                        "status": elem.status,
                        "news": elem.news,
                        "news_added": elem.news_added,
                        "chance_of_playing_next_round": elem.chance_of_playing_next_round,
                        "chance_of_playing_this_round": elem.chance_of_playing_this_round,
                        "scout_risks": risk_types,
                        "source": "Official FPL API",
                        "confidence": 1.0,
                        "fetched_at": now_iso,
                        "published_at": elem.news_added or now_iso,
                    }
                )

        return updates

    async def fetch_rss_articles(self) -> list[dict[str, Any]]:
        """
        Fetch articles from configured RSS feeds with SSRF safety, bounded timeouts,
        and content deduplication.
        """
        articles = []
        now_iso = datetime.now(UTC).isoformat()

        for feed in self.sources_cfg:
            feed_id = feed.get("id")
            if not feed.get("enabled", True) or feed_id in self.dead_feeds:
                continue
            feed_url = feed.get("url")
            feed_name = feed.get("name")
            feed_weight = float(feed.get("weight", 0.8))

            if not is_safe_external_url(feed_url):
                logger.warning(f"Rejecting unsafe RSS feed URL: {feed_url}")
                continue

            try:
                async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
                    current_url = feed_url
                    resp = None
                    for _ in range(5):
                        if not is_safe_external_url(current_url):
                            logger.warning(f"Rejecting unsafe URL or redirect hop: {current_url}")
                            resp = None
                            break
                        resp = await client.get(current_url, headers={"User-Agent": "Mozilla/5.0 FPLOracle/1.0"})
                        if resp.is_redirect:
                            loc = resp.headers.get("location")
                            if not loc:
                                break
                            current_url = urllib.parse.urljoin(current_url, loc)
                        else:
                            break

                    if resp is not None and resp.status_code == 200:
                        parsed = feedparser.parse(resp.text)
                        for entry in parsed.entries[:15]:
                            title = getattr(entry, "title", "").strip()
                            summary = getattr(entry, "summary", "").strip()
                            link = getattr(entry, "link", "").strip()
                            published = getattr(entry, "published", None)

                            # Deduplication by content hash
                            content_str = f"{link}|{title}|{summary}"
                            c_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
                            if c_hash in self._seen_article_hashes:
                                continue
                            self._seen_article_hashes.add(c_hash)

                            articles.append(
                                {
                                    "source_id": feed_id,
                                    "source_name": feed_name,
                                    "weight": feed_weight,
                                    "title": title,
                                    "summary": summary,
                                    "link": link,
                                    "content_hash": c_hash,
                                    "published_at": published,
                                    "fetched_at": now_iso,
                                }
                            )
                    else:
                        logger.warning(
                            f"Dropping dead RSS feed {feed_name} ({feed_url}) - HTTP status {resp.status_code}"
                        )
                        self.dead_feeds.add(feed_id)
            except Exception as e:
                logger.warning(f"Dropping unreachable RSS feed {feed_name} ({feed_url}): {e}")
                self.dead_feeds.add(feed_id)

        self.last_fetch_time = datetime.now(UTC)
        self.last_articles_count = len(articles)
        return articles


news_ingestion = NewsIngestion()
