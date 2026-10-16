"""
Historical data ingestion and unification.
Downloads past seasons (vaastav/Fantasy-Premier-League) and merges with
live 2026/27 match data from the official FPL API.
"""

import asyncio
import io
import logging
import urllib.request
from typing import Any

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
    "FWD": "FWD",
}

STANDARD_COLUMNS = [
    "season",
    "name",
    "element",
    "position",
    "team",
    "round",
    "fixture",
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
    "total_points",
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

    async def refresh_current_season(self) -> dict:
        """Refresh finalized match rows, atomically; never rebuild/download past seasons."""
        import os
        from datetime import UTC, datetime

        old = pd.read_csv(self.output_file, low_memory=False) if self.output_file.exists() else pd.DataFrame()
        boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
        fixtures, fixture_stale = await fpl_client.get_fixtures(force_refresh=True)
        if stale or fixture_stale:
            raise RuntimeError("History refresh needs fresh official context")
        finalized = {e.id for e in boot.events if e.finished and e.data_checked}
        valid_fixtures = {f.id for f in fixtures if f.event in finalized and f.finished}
        fresh = await self._fetch_live_season_async()
        if fresh.empty:
            raise RuntimeError("No fresh current-season rows; retained old dataset")
        fresh = fresh[fresh["round"].isin(finalized) & fresh["fixture"].isin(valid_fixtures)].copy()
        if fresh.empty:
            return {"status": "awaiting_finalization", "rows": 0}
        from fpl_oracle.utils.season import derive_season

        curr_season = derive_season(format="hyphen")
        past = old[old["season"] != curr_season] if not old.empty else pd.DataFrame()
        current = old[old["season"] == curr_season] if not old.empty else pd.DataFrame()
        # Preserve rows for players whose fetch failed; mark refresh incomplete below.
        if not current.empty and "fixture" not in current:
            current["fixture"] = 0
        if not current.empty:
            replaced = set(zip(fresh["element"], fresh["round"], strict=True))
            current = current[
                [(e, r) not in replaced for e, r in zip(current["element"], current["round"], strict=True)]
            ]
        merged = self._clean_and_standardize(pd.concat([past, current, fresh], ignore_index=True))
        # Past cached corpora may lack fixture identity and contain genuine DGW
        # rows. Never deduplicate those by a null/zero fixture key.
        past_rows = merged[merged["season"] != curr_season]
        current_rows = merged[merged["season"] == curr_season].drop_duplicates(
            ["season", "element", "fixture", "round"], keep="last"
        )
        merged = pd.concat([past_rows, current_rows], ignore_index=True)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_file.with_suffix(".csv.tmp")
        merged.to_csv(tmp, index=False)
        os.replace(tmp, self.output_file)
        import hashlib
        import json

        failed = sorted(getattr(self, "_last_failed_player_ids", []))
        metadata = {
            "status": "partial_refresh" if failed else "refreshed",
            "failed_player_ids": failed,
            "complete": not failed,
            "rows": len(fresh),
            "finalized_gameweeks": sorted(finalized),
            "refreshed_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(self.output_file.read_bytes()).hexdigest(),
        }
        self.output_file.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        from fpl_oracle.ml.predict import projection_engine

        projection_engine.invalidate_cache()
        return metadata

    def get_last_finished_gameweek(self) -> int:
        """Return the maximum finished gameweek round recorded in master_history.csv for current season."""
        if not self.output_file.exists():
            return 0
        try:
            df = pd.read_csv(self.output_file, usecols=["season", "round"])
            from fpl_oracle.utils.season import derive_season

            curr_season = derive_season(format="hyphen")
            curr_df = df[df["season"] == curr_season]
            if curr_df.empty:
                return int(df["round"].max()) if not df.empty else 0
            return int(curr_df["round"].max())
        except Exception:
            return 0

    @staticmethod
    def _build_match_row(
        curr_season: str,
        full_name: str,
        eid: int,
        pos: str,
        team_name: str,
        gw: int,
        f_id: int,
        opp: int,
        home: bool,
        ko_time: str,
        stats: dict[str, Any],
        val: float,
    ) -> dict[str, Any]:
        return {
            "season": curr_season,
            "name": full_name,
            "element": eid,
            "position": pos,
            "team": team_name,
            "round": gw,
            "fixture": float(f_id),
            "opponent_team": opp,
            "was_home": home,
            "kickoff_time": ko_time,
            "minutes": stats.get("minutes", 0),
            "goals_scored": stats.get("goals_scored", 0),
            "assists": stats.get("assists", 0),
            "clean_sheets": stats.get("clean_sheets", 0),
            "goals_conceded": stats.get("goals_conceded", 0),
            "own_goals": stats.get("own_goals", 0),
            "penalties_saved": stats.get("penalties_saved", 0),
            "penalties_missed": stats.get("penalties_missed", 0),
            "yellow_cards": stats.get("yellow_cards", 0),
            "red_cards": stats.get("red_cards", 0),
            "saves": stats.get("saves", 0),
            "bonus": stats.get("bonus", 0),
            "bps": stats.get("bps", 0),
            "influence": float(stats.get("influence", 0.0) or 0.0),
            "creativity": float(stats.get("creativity", 0.0) or 0.0),
            "threat": float(stats.get("threat", 0.0) or 0.0),
            "ict_index": float(stats.get("ict_index", 0.0) or 0.0),
            "starts": stats.get("starts", 0),
            "expected_goals": float(stats.get("expected_goals", 0.0) or 0.0),
            "expected_assists": float(stats.get("expected_assists", 0.0) or 0.0),
            "expected_goal_involvements": float(stats.get("expected_goal_involvements", 0.0) or 0.0),
            "expected_goals_conceded": float(stats.get("expected_goals_conceded", 0.0) or 0.0),
            "defensive_contribution": float(stats.get("defensive_contribution", 0.0) or 0.0),
            "value": val,
            "total_points": stats.get("total_points", 0),
        }

    async def refresh_gameweek_live(self, gw: int, force: bool = False) -> dict[str, Any]:
        """
        Incrementally append new match rows for a finalized gameweek from event/{gw}/live atomically.
        Validates finalization status (finished=True and data_checked=True), schema, and row count.
        Idempotent: replaces existing rows for (season, gw) and never double-appends.
        """
        import hashlib
        import json
        import os
        from datetime import UTC, datetime

        from fpl_oracle.utils.season import derive_season

        boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
        if stale and not force:
            raise RuntimeError("Incremental refresh requires fresh official bootstrap context")

        # Verify gameweek finalization
        ev = next((e for e in boot.events if e.id == gw), None)
        if ev is None:
            raise ValueError(f"Gameweek {gw} not found in official events")
        if not (ev.finished and ev.data_checked) and not force:
            raise RuntimeError(
                f"Gameweek {gw} is not finalised yet (finished={ev.finished}, data_checked={ev.data_checked}). "
                "FPL finalises matches ~09:00 UK the following morning."
            )

        fixtures, fixture_stale = await fpl_client.get_fixtures(event=gw, force_refresh=True)
        if fixture_stale and not force:
            raise RuntimeError("Incremental refresh requires fresh fixture data")

        live_data, live_stale = await fpl_client.get_live_gameweek(gw, force_refresh=True)
        elements = live_data.get("elements", [])
        if not elements:
            raise RuntimeError(f"No element data returned from event/{gw}/live/")

        curr_season = derive_season(format="hyphen")
        team_id_to_name = {t.id: t.name for t in boot.teams}
        pos_id_to_pos = {et.id: et.singular_name_short for et in boot.element_types}
        elem_dict = {e.id: e for e in boot.elements}
        fix_dict = {f.id: f for f in fixtures}
        team_fixtures: dict[int, list[tuple[Any, bool, int]]] = {}
        for f in fixtures:
            team_fixtures.setdefault(f.team_h, []).append((f, True, f.team_a))
            team_fixtures.setdefault(f.team_a, []).append((f, False, f.team_h))

        records: list[dict[str, Any]] = []
        for el in elements:
            eid = int(el.get("id", 0))
            elem = elem_dict.get(eid)
            if not elem:
                continue
            stats = el.get("stats", {})
            explain = el.get("explain", [])

            pos = pos_id_to_pos.get(elem.element_type, "MID")
            team_name = team_id_to_name.get(elem.team, str(elem.team))
            full_name = f"{elem.first_name} {elem.second_name}".strip() or elem.web_name
            val = float(elem.now_cost)

            if explain:
                for exp in explain:
                    fid = int(exp.get("fixture", 0))
                    f_obj = fix_dict.get(fid)
                    if f_obj:
                        was_home = elem.team == f_obj.team_h
                        opp = f_obj.team_a if was_home else f_obj.team_h
                        ko_val = getattr(f_obj, "kickoff_time", None)
                        ko = (
                            ko_val.isoformat()
                            if (ko_val is not None and hasattr(ko_val, "isoformat"))
                            else str(ko_val or "")
                        )
                    else:
                        was_home = False
                        opp = 0
                        ko = ""
                    records.append(
                        self._build_match_row(
                            curr_season, full_name, eid, pos, team_name, gw, fid, opp, was_home, ko, stats, val
                        )
                    )
            else:
                tf = team_fixtures.get(elem.team, [])
                if tf:
                    for f_obj, was_home, opp in tf:
                        ko_val = getattr(f_obj, "kickoff_time", None)
                        ko = (
                            ko_val.isoformat()
                            if (ko_val is not None and hasattr(ko_val, "isoformat"))
                            else str(ko_val or "")
                        )
                        records.append(
                            self._build_match_row(
                                curr_season, full_name, eid, pos, team_name, gw, f_obj.id, opp, was_home, ko, stats, val
                            )
                        )
                else:
                    records.append(
                        self._build_match_row(
                            curr_season, full_name, eid, pos, team_name, gw, 0, 0, False, "", stats, val
                        )
                    )

        fresh_df = pd.DataFrame(records)
        # Validate schema & row counts
        for col in STANDARD_COLUMNS:
            if col not in fresh_df.columns:
                raise ValueError(f"Missing column in incremental refresh: {col}")
        if len(fresh_df) < 500:
            raise ValueError(f"Insufficient rows generated for gameweek {gw}: got {len(fresh_df)}, expected >= 500")

        # Atomic append and idempotency against master_history.csv
        old = pd.read_csv(self.output_file, low_memory=False) if self.output_file.exists() else pd.DataFrame()
        if not old.empty:
            past_rows = old[old["season"] != curr_season].copy()
            curr_rows = old[old["season"] == curr_season].copy()
            # Drop any existing rows for (curr_season, gw) to ensure idempotency (no double-append)
            curr_rows = curr_rows[curr_rows["round"] != gw]
            fresh_cleaned = self._clean_and_standardize(fresh_df)
            curr_merged = pd.concat([curr_rows, fresh_cleaned], ignore_index=True)
            curr_merged = curr_merged.drop_duplicates(subset=["season", "element", "fixture", "round"], keep="last")
            merged = pd.concat([past_rows, curr_merged], ignore_index=True)
        else:
            merged = self._clean_and_standardize(fresh_df)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_file.with_suffix(".csv.tmp")
        merged.to_csv(tmp, index=False)
        os.replace(tmp, self.output_file)

        sha = hashlib.sha256(self.output_file.read_bytes()).hexdigest()
        meta = {
            "status": "refreshed",
            "gameweek": gw,
            "rows_appended": len(fresh_df),
            "total_rows": len(merged),
            "refreshed_at": datetime.now(UTC).isoformat(),
            "sha256": sha,
        }
        self.output_file.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Invalidate projection engine cache
        from fpl_oracle.ml.predict import projection_engine

        projection_engine.invalidate_cache()

        # Reload identity resolver
        from fpl_oracle.data.player_identity import player_identity_resolver

        player_identity_resolver.load(merged)

        return meta

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

        # Include the full registered population. Filtering on realized season
        # minutes/points conditions the training corpus on future participation
        # and drops the zero-point population in low-minutes interval evaluation.
        active_elements = list(bootstrap.elements)
        logger.info(f"Fetching element match histories for {len(active_elements)} active players...")

        from fpl_oracle.utils.season import derive_season

        curr_season = derive_season(format="hyphen")
        records = []
        self._last_failed_player_ids: list[int] = []
        sem = asyncio.Semaphore(15)

        async def fetch_player(elem):
            async with sem:
                try:
                    summary, stale = await fpl_client.get_element_summary(elem.id, force_refresh=True)
                    if stale:
                        raise RuntimeError(f"Stale element history {elem.id}")
                    for h in summary.history:
                        rec = {
                            "season": curr_season,
                            "name": f"{elem.first_name} {elem.second_name}".strip() or elem.web_name,
                            "element": elem.id,
                            "position": pos_id_to_pos.get(elem.element_type, "MID"),
                            "team": team_id_to_name.get(elem.team, str(elem.team)),
                            "round": h.round,
                            "fixture": h.fixture,
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
                            "total_points": h.total_points,
                        }
                        records.append(rec)
                except Exception as ex:
                    self._last_failed_player_ids.append(int(elem.id))
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
            "round",
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
            "total_points",
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
