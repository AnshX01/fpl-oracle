"""
Mini-league standings fetcher and parser.
Paginates through classic leagues and extracts competitor rankings.
Supports full multi-page pagination (up to 10 pages / 500 managers).
"""

import logging
from typing import Any

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import ClassicStandingResult

logger = logging.getLogger("fpl_oracle.league.standings")


class LeagueStandingsManager:
    def __init__(self):
        pass

    async def get_league_standings(self, league_id: int, max_pages: int = 50) -> dict[str, Any]:
        """
        Fetch standings for a classic mini-league up to max_pages (up to 500 teams).
        Paginates until standings.has_next is False or max_pages is reached.
        """
        all_results: list[ClassicStandingResult] = []
        league_info: dict[str, Any] = {}

        for page in range(1, max_pages + 1):
            try:
                resp, _ = await fpl_client.get_classic_league_standings(league_id, page=page)
                if page == 1:
                    league_info = resp.league
                all_results.extend(resp.standings.results)
                if not resp.standings.has_next:
                    break
            except Exception as e:
                logger.warning(f"Error fetching league standings page {page}: {e}")
                break

        return {
            "league_id": league_id,
            "league_name": league_info.get("name", f"League #{league_id}"),
            "total_teams": len(all_results),
            "standings": [r.model_dump() for r in all_results],
        }


league_standings_manager = LeagueStandingsManager()
