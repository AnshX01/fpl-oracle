"""
Mini-league standings fetcher and parser.
Paginates through classic leagues and extracts competitor rankings.
Supports full multi-page pagination without artificial 50-page caps.
Exposes coverage information (pages fetched, total entries, partial status).
"""

import asyncio
import logging
from typing import Any

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import ClassicStandingResult
from fpl_oracle.config import MAX_STANDINGS_PAGES

logger = logging.getLogger("fpl_oracle.league.standings")


def current_manager_rank(standings_data, manager_id):
    """Current official mini-league rank, never a simulation or overall-rank fallback."""
    if not manager_id:
        return {"current_league_rank": None, "league_rank_status": "manager_unconfigured"}
    for row in standings_data.get("standings", []):
        if row.get("entry") == manager_id:
            rank = row.get("rank")
            if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
                return {"current_league_rank": rank, "league_rank_status": "available"}
            return {"current_league_rank": None, "league_rank_status": "unavailable"}
    partial = standings_data.get("coverage", {}).get("partial", False)
    return {"current_league_rank": None, "league_rank_status": "unavailable" if partial else "not_listed"}


class LeagueStandingsManager:
    def __init__(self):
        pass

    async def get_league_standings(
        self,
        league_id: int,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """
        Fetch standings for a classic mini-league without arbitrary truncation.
        Paginates until standings.has_next is False or high safety limit (max_pages) is reached.
        If safety limit is hit or page fetch fails after retries, sets coverage.partial = True.
        """
        safety_max = max_pages if max_pages is not None else MAX_STANDINGS_PAGES
        all_results: list[ClassicStandingResult] = []
        league_info: dict[str, Any] = {}
        pages_fetched = 0
        is_partial = False
        any_stale = False
        partial_reason: str | None = None

        page = 1
        while page <= safety_max:
            page_success = False
            last_err: Exception | None = None

            # Retry with exponential backoff on transient network or API errors
            for attempt in range(3):
                try:
                    resp, page_stale = await fpl_client.get_classic_league_standings(league_id, page=page)
                    any_stale = any_stale or page_stale
                    if page == 1:
                        league_info = resp.league if isinstance(resp.league, dict) else resp.league.model_dump()
                    all_results.extend(resp.standings.results)
                    pages_fetched += 1
                    page_success = True
                    has_next = resp.standings.has_next
                    break
                except Exception as ex:
                    last_err = ex
                    logger.warning(
                        f"Error fetching league standings page {page} for league {league_id} "
                        f"(attempt {attempt + 1}/3): {ex}"
                    )
                    await asyncio.sleep(0.1 * (2**attempt))

            if not page_success:
                logger.error(
                    f"Failed to fetch league standings page {page} for league {league_id} after 3 attempts: {last_err}"
                )
                is_partial = True
                partial_reason = f"Failed to fetch page {page}: {last_err}"
                break

            if not has_next:
                # Natural end of standings reached
                break

            page += 1
            if page > safety_max and has_next:
                is_partial = True
                partial_reason = f"Reached safety limit of {safety_max} pages before end of league."
                break

        total_entries = (
            league_info.get("total_entries")
            if isinstance(league_info, dict)
            else getattr(league_info, "total_entries", None)
        )
        if total_entries is None:
            total_entries = len(all_results)

        league_name = (
            league_info.get("name") if isinstance(league_info, dict) else getattr(league_info, "name", None)
        ) or f"League #{league_id}"

        return {
            "league_id": league_id,
            "is_stale": any_stale,
            "league_name": league_name,
            "total_teams": len(all_results),
            "coverage": {
                "pages_fetched": pages_fetched,
                "total_entries": total_entries,
                "partial": is_partial,
                "reason": partial_reason,
            },
            "standings": [r.model_dump() if hasattr(r, "model_dump") else r for r in all_results],
        }


league_standings_manager = LeagueStandingsManager()
