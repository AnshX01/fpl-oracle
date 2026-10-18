"""
Odds provider interface and pre-deadline odds snapshotting into data/snapshots/odds/ (WI-3.2).
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fpl_oracle.config import SNAPSHOTS_DIR

logger = logging.getLogger("fpl_oracle.data.odds")

ODDS_SNAPSHOTS_DIR = SNAPSHOTS_DIR / "odds"


class OddsProvider(ABC):
    """Abstract interface for bookmaker and betting market odds."""

    @abstractmethod
    def get_match_odds(self, season: str, gameweek: int | None = None) -> list[dict[str, Any]]:
        """Fetch match-level 1X2 and over/under odds."""
        pass


class FootballDataOddsProvider(OddsProvider):
    """football-data.co.uk adapter: it only publishes historical closing odds, so there are no live odds to return."""

    def get_match_odds(self, season: str, gameweek: int | None = None) -> list[dict[str, Any]]:
        return []


def snapshot_odds(
    target_gw: int | None = None,
    custom_output_dir: Path | None = None,
    provider: OddsProvider | None = None,
) -> Path:
    """
    WI-3.2: Snapshots current pre-deadline market odds to data/snapshots/odds/.
    Creates timestamped raw JSON archives to accumulate training history.
    Raises RuntimeError when the provider returns no real odds.
    """
    out_dir = custom_output_dir or ODDS_SNAPSHOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    active_provider = provider or FootballDataOddsProvider()
    gw_num = target_gw or 1
    odds_data = active_provider.get_match_odds(season="2026-27", gameweek=gw_num)
    if not odds_data:
        # Never persist invented odds as if they were a market snapshot.
        raise RuntimeError(f"No odds source configured for GW{gw_num}; refusing to write an empty odds snapshot")

    ts_now = datetime.now(UTC)
    timestamp_str = ts_now.strftime("%Y%m%d_%H%M%S")

    payload = {
        "gameweek": gw_num,
        "snapshot_at_utc": ts_now.isoformat(),
        "provider": active_provider.__class__.__name__,
        "matches_count": len(odds_data),
        "matches": odds_data,
    }

    target_path = out_dir / f"gw{gw_num}_odds_{timestamp_str}.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[Odds] Pre-deadline odds snapshot saved to {target_path}")
    return target_path
