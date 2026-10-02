"""
News and injury ingestion engine.
Fetches official FPL injury and status data, and pulls live RSS feeds from
verified football news outlets (BBC, Sky Sports, The Guardian).
"""

from typing import List, Dict, Any, Optional, Set
import logging
from datetime import datetime, timezone
import feedparser
import httpx

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.config import NEWS_SOURCES

logger = logging.getLogger("fpl_oracle.news.ingest")

class NewsIngestion:
    def __init__(self):
        self.sources_cfg = NEWS_SOURCES.get("rss_feeds", [])
        self.dead_feeds: Set[str] = set()

    def fetch_official_fpl_signals(self, bootstrap: BootstrapStatic) -> List[Dict[str, Any]]:
        """
        Extract authoritative injury, suspension, and availability updates from FPL bootstrap.
        """
        updates = []
        team_map = {t.id: t.name for t in bootstrap.teams}

        for elem in bootstrap.elements:
            # If player has news, injury, or non-100% chance of playing
            has_news = bool(elem.news and elem.news.strip())
            not_fully_available = elem.status != "a" or (elem.chance_of_playing_next_round is not None and elem.chance_of_playing_next_round < 100)

            if has_news or not_fully_available:
                updates.append({
                    "element_id": elem.id,
                    "web_name": elem.web_name,
                    "team_id": elem.team,
                    "team_name": team_map.get(elem.team, f"Team {elem.team}"),
                    "status": elem.status,
                    "news": elem.news,
                    "news_added": elem.news_added,
                    "chance_of_playing_next_round": elem.chance_of_playing_next_round,
                    "chance_of_playing_this_round": elem.chance_of_playing_this_round,
                    "source": "Official FPL API",
                    "confidence": 1.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        return updates

    async def fetch_rss_articles(self) -> List[Dict[str, Any]]:
        """
        Fetch articles from configured RSS feeds with graceful error handling and automatic dead feed dropping.
        """
        articles = []
        for feed in self.sources_cfg:
            feed_id = feed.get("id")
            if not feed.get("enabled", True) or feed_id in self.dead_feeds:
                continue
            feed_url = feed.get("url")
            feed_name = feed.get("name")
            feed_weight = float(feed.get("weight", 0.8))

            try:
                # Use httpx with short timeout to prevent hanging
                async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                    resp = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0 FPLOracle/1.0"})
                    if resp.status_code == 200:
                        parsed = feedparser.parse(resp.text)
                        for entry in parsed.entries[:10]:
                            title = getattr(entry, "title", "")
                            summary = getattr(entry, "summary", "")
                            link = getattr(entry, "link", "")
                            published = getattr(entry, "published", datetime.now(timezone.utc).isoformat())

                            articles.append({
                                "source_id": feed_id,
                                "source_name": feed_name,
                                "weight": feed_weight,
                                "title": title,
                                "summary": summary,
                                "link": link,
                                "published": published
                            })
                    else:
                        logger.warning(f"Dropping dead RSS feed {feed_name} ({feed_url}) - HTTP status {resp.status_code}")
                        self.dead_feeds.add(feed_id)
            except Exception as e:
                logger.warning(f"Dropping unreachable RSS feed {feed_name} ({feed_url}): {e}")
                self.dead_feeds.add(feed_id)

        return articles

news_ingestion = NewsIngestion()
