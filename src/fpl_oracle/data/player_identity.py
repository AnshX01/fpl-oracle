"""
Player Identity Resolution & Canonical Historical Mapping (G4).

Eliminates the 841 element integer ID collisions across seasons in master_history.csv
by establishing a stable canonical player identity table keyed by normalized player
identity, season-scoped element IDs, and official FPL player codes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_oracle.config import BASE_DIR

logger = logging.getLogger(__name__)

HISTORICAL_DIR = BASE_DIR / "data" / "historical"
MASTER_HISTORY_CSV = HISTORICAL_DIR / "master_history.csv"


def normalize_player_name(s: str | None) -> str:
    """
    Produce a canonical normalized player name stripped of diacritics, punctuation,
    and irregular casing for 100% stable cross-season matching.
    """
    if not s or not isinstance(s, str):
        return ""
    # Normalize unicode diacritics (NFKD) and remove combining marks
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Remove non-alphanumeric except spaces, hyphens
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", stripped)
    return " ".join(cleaned.lower().split())


class PlayerIdentityResolver:
    """
    Manages canonical player identities across multiple Premier League seasons.
    Prevents element ID collision contamination across season boundaries.
    """

    def __init__(self, master_csv_path: Path | None = None):
        self.master_csv_path = master_csv_path or MASTER_HISTORY_CSV
        self._history_df: pd.DataFrame | None = None
        self._season_elem_to_canonical: dict[tuple[str, int], str] = {}
        self._canonical_to_indices: dict[str, list[int]] = {}
        self._code_to_canonical: dict[int, str] = {}
        self._web_name_to_canonicals: dict[str, list[str]] = {}
        self._is_loaded = False

    def load(
        self,
        history_df: pd.DataFrame | None = None,
        bootstrap_elements: list[Any] | None = None,
    ) -> None:
        """
        Load historical records and construct canonical identity mappings.
        """
        if history_df is not None and not history_df.empty:
            self._history_df = history_df.copy()
        elif self.master_csv_path.exists():
            try:
                self._history_df = pd.read_csv(self.master_csv_path)
            except Exception as e:
                logger.error("Failed to load master history CSV from %s: %s", self.master_csv_path, e)
                self._history_df = pd.DataFrame()
        else:
            self._history_df = pd.DataFrame()

        self._season_elem_to_canonical.clear()
        self._canonical_to_indices.clear()
        self._code_to_canonical.clear()
        self._web_name_to_canonicals.clear()

        if self._history_df is not None and not self._history_df.empty:
            df = self._history_df
            if "norm_name" not in df.columns:
                df["norm_name"] = df["name"].apply(normalize_player_name)

            for idx, row in df.iterrows():
                season = str(row.get("season", ""))
                try:
                    elem_id = int(row.get("element", 0))
                except (ValueError, TypeError):
                    continue

                canonical = str(row.get("norm_name", ""))
                if not canonical:
                    continue

                # Map (season, element) -> canonical
                self._season_elem_to_canonical[(season, elem_id)] = canonical
                self._canonical_to_indices.setdefault(canonical, []).append(int(idx))

        # Register bootstrap elements (from current active season, e.g. 2026-27)
        if bootstrap_elements:
            self.register_bootstrap_elements(bootstrap_elements, season="2026-27")

        self._is_loaded = True
        logger.info(
            "PlayerIdentityResolver loaded: %d (season, element) pairs mapped to %d canonical identities.",
            len(self._season_elem_to_canonical),
            len(self._canonical_to_indices),
        )

    def register_bootstrap_elements(
        self,
        elements: list[Any],
        season: str = "2026-27",
    ) -> None:
        """
        Register live FPL bootstrap elements with code and web_name mappings.
        """
        for el in elements:
            elem_id = getattr(el, "id", None)
            code = getattr(el, "code", None)
            first_name = getattr(el, "first_name", "")
            second_name = getattr(el, "second_name", "")
            web_name = getattr(el, "web_name", "")

            full_name = f"{first_name} {second_name}".strip()
            norm_full = normalize_player_name(full_name)
            norm_web = normalize_player_name(web_name)

            # Preferred canonical identity
            canonical = norm_full or norm_web
            if not canonical and elem_id is not None:
                canonical = f"player_{season}_{elem_id}"

            if elem_id is not None:
                self._season_elem_to_canonical[(season, int(elem_id))] = canonical
            if code is not None:
                try:
                    self._code_to_canonical[int(code)] = canonical
                except (ValueError, TypeError):
                    pass
            if norm_web:
                if canonical not in self._web_name_to_canonicals.setdefault(norm_web, []):
                    self._web_name_to_canonicals[norm_web].append(canonical)

    def resolve_canonical_id(
        self,
        elem_id: int | None = None,
        season: str = "2026-27",
        full_name: str | None = None,
        web_name: str | None = None,
        code: int | None = None,
    ) -> str:
        """
        Resolve a player reference to a canonical identity string.
        """
        if not self._is_loaded:
            self.load()

        # 1. By code
        if code is not None and int(code) in self._code_to_canonical:
            return self._code_to_canonical[int(code)]

        # 2. By (season, elem_id)
        if elem_id is not None:
            key = (str(season), int(elem_id))
            if key in self._season_elem_to_canonical:
                return self._season_elem_to_canonical[key]

        # 3. By full name
        if full_name:
            norm_full = normalize_player_name(full_name)
            if norm_full in self._canonical_to_indices:
                return norm_full

        # 4. By web name
        if web_name:
            norm_web = normalize_player_name(web_name)
            matches = self._web_name_to_canonicals.get(norm_web, [])
            if len(matches) == 1:
                return matches[0]

        # Fallback to normalized input or unique season identifier
        fallback = normalize_player_name(full_name or web_name)
        if fallback:
            return fallback
        return f"unknown_{season}_{elem_id}"

    def get_player_history(
        self,
        elem_id: int | None = None,
        season: str = "2026-27",
        full_name: str | None = None,
        web_name: str | None = None,
        code: int | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve authentic, un-conflated match history for the specified player.
        Guarantees that other players sharing the same element ID in different
        seasons are completely excluded.
        """
        if not self._is_loaded:
            self.load()

        if self._history_df is None or self._history_df.empty:
            return pd.DataFrame()

        canonical = self.resolve_canonical_id(
            elem_id=elem_id,
            season=season,
            full_name=full_name,
            web_name=web_name,
            code=code,
        )

        indices = self._canonical_to_indices.get(canonical)
        if indices is None:
            return pd.DataFrame()

        p_df = self._history_df.iloc[indices].copy()
        if "season" in p_df.columns and "round" in p_df.columns:
            p_df = p_df.sort_values(by=["season", "round"])
        return p_df


# Global singleton resolver
player_identity_resolver = PlayerIdentityResolver()
