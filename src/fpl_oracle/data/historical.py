"""
Historical data ingestion and unification.
Downloads past seasons (vaastav/Fantasy-Premier-League) and merges with
live 2026/27 match data from the official FPL API.
"""

import asyncio
import io
import logging
import urllib.request

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.config import HISTORICAL_DIR

logger = logging.getLogger("fpl_oracle.historical")

SEASONS_TO_FETCH = ["2023-24", "2024-25", "2025-26"]
VAASTAV_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
    "GKP": "GKP",
    "GK": "GKP",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD"
}

STANDARD_COLUMNS = [
    "season",
    "name",
    "element",
    "position",
    "team",
    "round",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
    "value",
    "total_points"
]

class HistoricalDataManager:
    def __init__(self):
        self.output_file = HISTORICAL_DIR / "master_history.csv"

    def ensure_dataset_ready(self, force_refresh: bool = False) -> pd.DataFrame:
        if self.output_file.exists() and not force_refresh:
            logger.info(f"Loading existing historical dataset from {self.output_file}")
            df = pd.read_csv(self.output_file)
            return df

        logger.info("Building unified historical dataset...")
        dfs = []

        # 1. Fetch past seasons
        for season in SEASONS_TO_FETCH:
            season_df = self._fetch_vaastav_season(season)
            if season_df is not None and not season_df.empty:
                dfs.append(season_df)

        # 2. Ingest 2026/27 live season from API
        live_df = self._fetch_live_season_sync()
        if live_df is not None and not live_df.empty:
            dfs.append(live_df)

        if not dfs:
            raise RuntimeError("Failed to build historical dataset: no season data available.")

        master_df = pd.concat(dfs, ignore_index=True)
        master_df = self._clean_and_standardize(master_df)

        master_df.to_csv(self.output_file, index=False)
        logger.info(f"Successfully saved master history ({len(master_df)} rows) to {self.output_file}")
        return master_df

    def _fetch_vaastav_season(self, season: str) -> pd.DataFrame | None:
        cache_path = HISTORICAL_DIR / f"{season}_merged_gw.csv"
        if cache_path.exists():
            try:
                logger.info(f"Loading {season} from local cache...")
                df = pd.read_csv(cache_path, low_memory=False)
                df["season"] = season
                return df
            except Exception as e:
                logger.warning(f"Error reading local {season} cache: {e}")

        url = f"{VAASTAV_BASE_URL}/{season}/gws/merged_gw.csv"
        logger.info(f"Downloading historical season {season} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
                cache_path.write_bytes(raw_bytes)
                df = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                df["season"] = season
                return df
        except Exception as e:
            logger.warning(f"Could not download {season} from vaastav: {e}")
            return None

    def _fetch_live_season_sync(self) -> pd.DataFrame | None:
        """Fetch 2026/27 completed gameweeks from live FPL API."""
        try:
            return asyncio.run(self._fetch_live_season_async())
        except Exception as e:
            logger.error(f"Error fetching live 2026/27 season: {e}")
            return None

    async def _fetch_live_season_async(self) -> pd.DataFrame:
        logger.info("Ingesting 2026/27 live match data from FPL API...")
        bootstrap, _ = await fpl_client.get_bootstrap_static()

        team_id_to_name = {t.id: t.name for t in bootstrap.teams}
        pos_id_to_pos = {et.id: et.singular_name_short for et in bootstrap.element_types}

        # Elements that have played or scored points
        active_elements = [e for e in bootstrap.elements if e.minutes > 0 or e.total_points > 0]
        logger.info(f"Fetching element match histories for {len(active_elements)} active players...")

        records = []
        sem = asyncio.Semaphore(15)

        async def fetch_player(elem):
            async with sem:
                try:
                    summary, _ = await fpl_client.get_element_summary(elem.id)
                    for h in summary.history:
                        rec = {
                            "season": "2026-27",
                            "name": f"{elem.first_name} {elem.second_name}".strip() or elem.web_name,
                            "element": elem.id,
                            "position": pos_id_to_pos.get(elem.element_type, "MID"),
                            "team": team_id_to_name.get(elem.team, str(elem.team)),
                            "round": h.round,
                            "opponent_team": h.opponent_team,
                            "was_home": h.was_home,
                            "kickoff_time": h.kickoff_time,
                            "minutes": h.minutes,
                            "goals_scored": h.goals_scored,
                            "assists": h.assists,
                            "clean_sheets": h.clean_sheets,
                            "goals_conceded": h.goals_conceded,
                            "own_goals": h.own_goals,
                            "penalties_saved": h.penalties_saved,
                            "penalties_missed": h.penalties_missed,
                            "yellow_cards": h.yellow_cards,
                            "red_cards": h.red_cards,
                            "saves": h.saves,
                            "bonus": h.bonus,
                            "bps": h.bps,
                            "influence": h.influence,
                            "creativity": h.creativity,
                            "threat": h.threat,
                            "ict_index": h.ict_index,
                            "starts": h.starts,
                            "expected_goals": h.expected_goals,
                            "expected_assists": h.expected_assists,
                            "expected_goal_involvements": h.expected_goal_involvements,
                            "expected_goals_conceded": h.expected_goals_conceded,
                            "defensive_contribution": h.defensive_contribution or 0,
                            "value": h.value,
                            "total_points": h.total_points
                        }
                        records.append(rec)
                except Exception as ex:
                    logger.debug(f"Failed element summary for {elem.id}: {ex}")

        tasks = [fetch_player(e) for e in active_elements]
        await asyncio.gather(*tasks)

        df = pd.DataFrame(records)
        logger.info(f"Ingested {len(df)} match records for 2026/27.")
        return df

    def _clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize position
        if "position" in df.columns:
            df["position"] = df["position"].map(lambda p: POSITION_MAP.get(p, "MID"))

        # Ensure all standard columns exist
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # Numeric conversions
        numeric_cols = [
            "round", "minutes", "goals_scored", "assists", "clean_sheets",
            "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards", "saves", "bonus", "bps",
            "influence", "creativity", "threat", "ict_index", "starts",
            "expected_goals", "expected_assists", "expected_goal_involvements",
            "expected_goals_conceded", "defensive_contribution", "value", "total_points"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Boolean conversion
        if "was_home" in df.columns:
            df["was_home"] = df["was_home"].astype(bool)

        df = df[STANDARD_COLUMNS]
        # Sort chronologically by season and round
        df = df.sort_values(by=["season", "round", "element"]).reset_index(drop=True)
        return df

historical_manager = HistoricalDataManager()
