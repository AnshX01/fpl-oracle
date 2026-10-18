"""
Dynamic Premier League season determination.

Derives season representation ('2026/27' or '2026-27') dynamically from
the live FPL API fixture dates, current UTC time, or configuration overrides.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.config import SETTINGS


def _format_years(start_year: int, target_format: str) -> str:
    end_year_short = str(start_year + 1)[-2:]
    if target_format == "hyphen":
        return f"{start_year}-{end_year_short}"
    return f"{start_year}/{end_year_short}"


def _format_season(season_str: str, target_format: str) -> str:
    cleaned = season_str.replace("/", "-")
    parts = cleaned.split("-")
    if len(parts) == 2 and len(parts[0]) == 4:
        try:
            start_year = int(parts[0])
            return _format_years(start_year, target_format)
        except ValueError:
            pass
    if target_format == "hyphen":
        return season_str.replace("/", "-")
    return season_str.replace("-", "/")


def derive_season(
    dt: datetime | None = None,
    fixtures: list[Any] | None = None,
    format: str = "slash",  # "slash" -> "YYYY/YY", "hyphen" -> "YYYY-YY"
) -> str:
    """
    Derive the Premier League season dynamically.

    Priority:
    1. FPL_SEASON environment variable
    2. settings.yaml 'season' configuration
    3. Earliest kickoff timestamp among provided fixtures
    4. Provided datetime (or current UTC time) with Premier League year rollover (July-June)
    """
    env_season = os.environ.get("FPL_SEASON")
    if env_season:
        return _format_season(env_season, format)

    cfg_season = SETTINGS.get("season")
    if cfg_season:
        return _format_season(str(cfg_season), format)

    # 3. Check fixture dates if available
    if fixtures:
        for f in fixtures:
            ko = getattr(f, "kickoff_time", None)
            if ko:
                try:
                    if isinstance(ko, str):
                        ko_dt = datetime.fromisoformat(ko.replace("Z", "+00:00"))
                    elif isinstance(ko, datetime):
                        ko_dt = ko
                    else:
                        continue
                    if ko_dt.tzinfo is None:
                        ko_dt = ko_dt.replace(tzinfo=UTC)
                    start_yr = ko_dt.year if ko_dt.month >= 7 else ko_dt.year - 1
                    return _format_years(start_yr, format)
                except Exception:
                    pass

    # 4. Check UTC date
    now = dt or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    start_year = now.year if now.month >= 7 else now.year - 1
    return _format_years(start_year, format)
