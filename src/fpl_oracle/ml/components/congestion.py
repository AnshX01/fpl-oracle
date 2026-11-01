"""
Midweek European & Domestic Cup Congestion & Turnaround Fatigue Engine.
Accurately models turnaround hours and rotation risk when Premier League clubs
compete in mid-week UEFA Champions League, Europa League, Conference League,
or Carabao Cup fixtures within 72 hours of weekend Premier League matches.
Zero hardcoded player names in production code.
"""

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_utc_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val.astimezone(UTC)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            return None
    return None


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class CongestionStatus:
    has_midweek: bool
    competition: str
    estimated_turnaround_hours: float
    fatigue_discount: float
    rotation_risk_tier: str  # "LOW", "MODERATE", "ACUTE"


class TurnaroundCongestionEngine:
    """
    Quantifies the impact of 48-72 hour turnarounds on player availability,
    expected minutes, and rotation probability.
    """

    # 2026/27 European Qualification Profile by Club Short Name
    # Newcastle removed from European qualification profile (finished outside European places)
    UCL_CLUBS: frozenset[str] = frozenset({"MCI", "ARS", "LIV", "AVL"})
    UEL_UECL_CLUBS: frozenset[str] = frozenset({"TOT", "CHE", "MUN"})

    # Midweek match calendar for the 2026/27 season (Gameweek -> Midweek Event)
    # Matches preceding or involving the listed Gameweek
    MIDWEEK_EVENTS: dict[int, dict[str, str]] = {
        5: {"UCL": "Matchday 1", "UEL": "Matchday 1"},
        7: {"UCL": "Matchday 2", "UEL": "Matchday 2"},
        9: {"DOMESTIC_CUP": "Carabao Cup Round 4"},
        11: {"UCL": "Matchday 4", "UEL": "Matchday 4"},
        13: {"UCL": "Matchday 5", "UEL": "Matchday 5", "PL_MIDWEEK": "GW13 Midweek Round"},
        14: {"PL_MIDWEEK": "Preceded by GW13 Midweek"},
        15: {"DOMESTIC_CUP": "Carabao Cup Quarter-Finals"},
        16: {"UCL": "Matchday 6", "UEL": "Matchday 6"},
        18: {"PL_MIDWEEK": "Festive Crunch (GW18 Midweek)"},
        19: {"PL_MIDWEEK": "Festive Crunch (GW19 Post-Midweek)"},
        20: {"PL_MIDWEEK": "Festive Crunch (GW20 Midweek)"},
        21: {"UCL": "Matchday 7", "DOMESTIC_CUP": "FA Cup Round 3 / Carabao Cup SF1"},
        22: {"UCL": "Matchday 8", "UEL": "Matchday 7"},
        23: {"UEL": "Matchday 8"},
        25: {"PL_MIDWEEK": "GW25 Midweek Round", "DOMESTIC_CUP": "Carabao Cup SF1"},
        26: {"DOMESTIC_CUP": "FA Cup Round 4"},
        27: {"DOMESTIC_CUP": "Carabao Cup SF2"},
        28: {"PL_MIDWEEK": "GW28 Midweek Round", "DOMESTIC_CUP": "FA Cup Round 5"},
    }

    # Default active clubs for domestic cup rounds in 2026/27
    # Non-European clubs (like Brighton in Carabao Cup Round 4 vs Man City) are actively tracked!
    # Verified rested elements by gameweek for domestic cup rounds (0 minutes played; zero fatigue discount)
    DEFAULT_RESTED_PLAYERS: dict[int, frozenset[int]] = {
        9: frozenset({411, 409, 410, 384, 105, 112}),
    }

    DEFAULT_CUP_ACTIVE_CLUBS: dict[int, frozenset[str]] = {
        9: frozenset({"MCI", "BHA", "ARS", "CHE", "LIV", "TOT", "AVL", "MUN", "NEW"}),
        15: frozenset({"MCI", "BHA", "ARS", "CHE", "LIV", "TOT", "AVL", "NEW"}),
        21: frozenset(
            {
                "MCI",
                "BHA",
                "ARS",
                "CHE",
                "LIV",
                "TOT",
                "AVL",
                "MUN",
                "NEW",
                "BRE",
                "CRY",
                "FUL",
                "BOU",
                "EVE",
                "NFO",
                "IPS",
                "LEE",
                "SUN",
                "HUL",
                "COV",
            }
        ),
        25: frozenset({"MCI", "ARS", "CHE", "LIV"}),
        26: frozenset({"MCI", "ARS", "CHE", "LIV", "TOT", "AVL", "MUN", "NEW"}),
        27: frozenset({"MCI", "ARS", "CHE", "LIV"}),
        28: frozenset({"MCI", "ARS", "CHE", "LIV", "TOT", "AVL"}),
    }

    # Nominal midweek cup kickoffs (UTC)
    CUP_NOMINAL_KICKOFFS: dict[int, datetime] = {
        9: datetime(2026, 10, 28, 19, 45, tzinfo=UTC),
        15: datetime(2026, 12, 16, 20, 0, tzinfo=UTC),
        21: datetime(2027, 1, 13, 20, 0, tzinfo=UTC),
        25: datetime(2027, 2, 3, 20, 0, tzinfo=UTC),
        27: datetime(2027, 2, 24, 20, 0, tzinfo=UTC),
    }

    def __init__(self):
        self.ucl_clubs: set[str] = set(self.UCL_CLUBS)
        self.uel_uecl_clubs: set[str] = set(self.UEL_UECL_CLUBS)
        self.cup_active_clubs: dict[int, set[str]] = {
            gw: set(clubs) for gw, clubs in self.DEFAULT_CUP_ACTIVE_CLUBS.items()
        }
        self.cup_kickoffs: dict[int, datetime] = dict(self.CUP_NOMINAL_KICKOFFS)
        self._registered_cup_matches: dict[int, dict[str, Any]] = {}
        self._rested_player_ids: dict[int, set[int]] = {
            gw: set(ids) for gw, ids in getattr(self, "DEFAULT_RESTED_PLAYERS", {}).items()
        }
        self._fixtures_cache: list[Any] | None = None
        self._team_short_map_cache: dict[int, str] | None = None

    def register_rested_players(
        self,
        gameweek: int,
        element_ids: Any = None,
    ) -> None:
        """Dynamically register rested players (0 midweek minutes) for a gameweek."""
        if gameweek not in self._rested_player_ids:
            self._rested_player_ids[gameweek] = set()
        if element_ids:
            self._rested_player_ids[gameweek].update(int(i) for i in element_ids)

    def register_european_clubs(
        self,
        ucl: Iterable[str] | None = None,
        uel: Iterable[str] | None = None,
    ) -> None:
        """Dynamically update European qualification profile."""
        if ucl is not None:
            self.ucl_clubs = {str(c).strip().upper() for c in ucl}
        if uel is not None:
            self.uel_uecl_clubs = {str(c).strip().upper() for c in uel}

    def register_cup_active_clubs(
        self,
        gameweek: int,
        teams: Iterable[str],
        nominal_kickoff: datetime | str | None = None,
    ) -> None:
        """Dynamically register or update active clubs for a domestic cup round."""
        clean_teams = {str(c).strip().upper() for c in teams}
        if gameweek not in self.cup_active_clubs:
            self.cup_active_clubs[gameweek] = set()
        self.cup_active_clubs[gameweek].update(clean_teams)

        if nominal_kickoff:
            dt = _parse_utc_datetime(nominal_kickoff)
            if dt:
                self.cup_kickoffs[gameweek] = dt

    def set_active_cup_clubs(
        self,
        gameweek: int,
        teams: Iterable[str],
    ) -> None:
        """Overwrites the full active club list for a specific domestic cup round."""
        self.cup_active_clubs[gameweek] = {str(c).strip().upper() for c in teams}

    def register_cup_match(
        self,
        gameweek: int,
        competition: str,
        kickoff_time: str | datetime,
        teams: Iterable[str],
    ) -> None:
        """Register a specific cup match with exact kickoff and participants."""
        dt = _parse_utc_datetime(kickoff_time)
        clean_teams = {str(c).strip().upper() for c in teams}
        self.register_cup_active_clubs(gameweek, clean_teams, nominal_kickoff=dt)
        self._registered_cup_matches[gameweek] = {
            "competition": competition,
            "kickoff_time": dt,
            "teams": clean_teams,
        }

    def _resolve_team_mappings(
        self,
        team_short_map: dict[int, str] | None = None,
    ) -> tuple[dict[int, str], dict[str, int]]:
        if team_short_map:
            t_map = {int(k): str(v).strip().upper() for k, v in team_short_map.items()}
            rev_map = {v: k for k, v in t_map.items()}
            return t_map, rev_map

        if self._team_short_map_cache:
            t_map = self._team_short_map_cache
            return t_map, {v: k for k, v in t_map.items()}

        # 1. Try active data_store
        try:
            from fpl_oracle.data.store import data_store

            b_entry = data_store.get_cache_entry("2026-27:bootstrap-static") or data_store.get_cache_entry(
                "bootstrap-static"
            )
            if b_entry and b_entry[0] and "teams" in b_entry[0]:
                t_map = {int(t["id"]): str(t["short_name"]).strip().upper() for t in b_entry[0]["teams"]}
                self._team_short_map_cache = t_map
                return t_map, {v: k for k, v in t_map.items()}
        except Exception:
            pass

        # 2. Try repo root database directly (e.g. during pytest runs with isolated data dirs)
        try:
            repo_db = Path(__file__).resolve().parents[4] / "data" / "fpl_oracle.db"
            if repo_db.exists():
                conn = sqlite3.connect(str(repo_db))
                cur = conn.cursor()
                cur.execute(
                    "SELECT data_json FROM api_cache WHERE key='2026-27:bootstrap-static' OR key='bootstrap-static'"
                )
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    b_data = json.loads(row[0])
                    if "teams" in b_data:
                        t_map = {int(t["id"]): str(t["short_name"]).strip().upper() for t in b_data["teams"]}
                        self._team_short_map_cache = t_map
                        return t_map, {v: k for k, v in t_map.items()}
        except Exception:
            pass

        return {}, {}

    def _load_fixtures(self, fixtures: list[Any] | None = None) -> list[Any]:
        if fixtures is not None:
            return fixtures
        if self._fixtures_cache is not None:
            return self._fixtures_cache

        # 1. Try active data_store
        try:
            from fpl_oracle.data.store import data_store

            fix_entry = data_store.get_cache_entry("fixtures:all") or data_store.get_cache_entry("2026-27:fixtures:all")
            if fix_entry and fix_entry[0]:
                self._fixtures_cache = fix_entry[0]
                return self._fixtures_cache
        except Exception:
            pass

        # 2. Try repo root database directly (e.g. during pytest runs with isolated data dirs)
        try:
            repo_db = Path(__file__).resolve().parents[4] / "data" / "fpl_oracle.db"
            if repo_db.exists():
                conn = sqlite3.connect(str(repo_db))
                cur = conn.cursor()
                cur.execute("SELECT data_json FROM api_cache WHERE key='fixtures:all' OR key='2026-27:fixtures:all'")
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    self._fixtures_cache = json.loads(row[0])
                    return self._fixtures_cache
        except Exception:
            pass

        return []

    def assess_club_congestion(
        self,
        team_short: str,
        gameweek: int,
        kickoff_time: datetime | str | None = None,
        fixtures: list[Any] | None = None,
        active_cup_clubs: Iterable[str] | None = None,
    ) -> CongestionStatus:
        """
        Assesses whether a club faces acute turnaround fatigue heading into the given gameweek.
        Dynamically computes turnaround hours from exact fixture timestamps.
        """
        team_clean = str(team_short).strip().upper()
        events = self.MIDWEEK_EVENTS.get(gameweek, {})

        # Load fixtures & team mappings
        fix_list = self._load_fixtures(fixtures)
        t_map, rev_map = self._resolve_team_mappings()
        team_id = rev_map.get(team_clean)

        # 1. Determine target kickoff datetime
        target_dt = _parse_utc_datetime(kickoff_time)
        target_fixture = None
        if fix_list and team_id is not None:
            for f in fix_list:
                f_ev = _get_attr(f, "event")
                if f_ev == gameweek:
                    h = _get_attr(f, "team_h")
                    a = _get_attr(f, "team_a")
                    if h == team_id or a == team_id:
                        target_fixture = f
                        break
            if target_dt is None and target_fixture:
                target_dt = _parse_utc_datetime(_get_attr(target_fixture, "kickoff_time"))

        # Fallback nominal kickoff if target_dt is still unknown
        if target_dt is None:
            target_dt = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)

        # 2. Gather candidate previous matches (Premier League, European, Domestic Cup)
        candidates: list[tuple[float, str, datetime]] = []

        # Candidate A: Previous Premier League fixture in schedule
        if fix_list and team_id is not None:
            past_team_fixs: list[tuple[datetime, Any]] = []
            for f in fix_list:
                f_ev = _get_attr(f, "event")
                if f_ev is not None and f_ev >= gameweek:
                    continue
                h = _get_attr(f, "team_h")
                a = _get_attr(f, "team_a")
                if h == team_id or a == team_id:
                    f_ko = _parse_utc_datetime(_get_attr(f, "kickoff_time"))
                    if f_ko and f_ko < target_dt:
                        past_team_fixs.append((f_ko, f))
            if past_team_fixs:
                past_team_fixs.sort(key=lambda x: x[0])
                prev_ko, prev_f = past_team_fixs[-1]
                pl_diff_hours = (target_dt - prev_ko).total_seconds() / 3600.0
                # A compressed Premier League turnaround occurs within <= 84 hours
                if 0 < pl_diff_hours <= 84.0:
                    prev_ev = _get_attr(prev_f, "event", "Prev")
                    candidates.append((pl_diff_hours, f"Premier League (GW{prev_ev})", prev_ko))

        # Candidate B: Midweek European fixtures (UCL / UEL / UECL)
        has_ucl = team_clean in self.ucl_clubs and "UCL" in events
        has_uel = team_clean in self.uel_uecl_clubs and "UEL" in events

        if has_ucl:
            # UCL matches typically Wednesday 20:00 UTC preceding the target kickoff
            days_back = (target_dt.weekday() - 2) % 7
            if days_back == 0:
                days_back = 7
            prev_wed = target_dt.date() - timedelta(days=days_back)
            ucl_ko = datetime(prev_wed.year, prev_wed.month, prev_wed.day, 20, 0, tzinfo=UTC)
            ucl_diff = (target_dt - ucl_ko).total_seconds() / 3600.0
            if 0 < ucl_diff <= 120.0:
                comp_label = f"UEFA Champions League ({events['UCL']})"
                candidates.append((ucl_diff, comp_label, ucl_ko))

        if has_uel:
            # UEL matches typically Thursday 20:00 UTC preceding the target kickoff
            days_back = (target_dt.weekday() - 3) % 7
            if days_back == 0:
                days_back = 7
            prev_thu = target_dt.date() - timedelta(days=days_back)
            uel_ko = datetime(prev_thu.year, prev_thu.month, prev_thu.day, 20, 0, tzinfo=UTC)
            uel_diff = (target_dt - uel_ko).total_seconds() / 3600.0
            if 0 < uel_diff <= 120.0:
                comp_label = f"UEFA Europa / Conference League ({events['UEL']})"
                candidates.append((uel_diff, comp_label, uel_ko))

        # Candidate C: Domestic Cup fixtures (Carabao Cup / FA Cup)
        active_cup_set = (
            {str(c).strip().upper() for c in active_cup_clubs}
            if active_cup_clubs is not None
            else self.cup_active_clubs.get(gameweek, set())
        )
        has_cup = ("DOMESTIC_CUP" in events) and (team_clean in active_cup_set)

        if has_cup:
            cup_ko = self.cup_kickoffs.get(gameweek)
            if cup_ko is None:
                days_back = (target_dt.weekday() - 2) % 7
                if days_back == 0:
                    days_back = 7
                prev_wed = target_dt.date() - timedelta(days=days_back)
                cup_ko = datetime(prev_wed.year, prev_wed.month, prev_wed.day, 19, 45, tzinfo=UTC)
            if cup_ko < target_dt:
                cup_diff = (target_dt - cup_ko).total_seconds() / 3600.0
                if 0 < cup_diff <= 120.0:
                    cup_name = events.get("DOMESTIC_CUP", "Midweek Cup")
                    candidates.append((cup_diff, f"Domestic Cup ({cup_name})", cup_ko))

        # 3. Process candidate results
        if candidates:
            # Sort by turnaround hours (closest preceding match determines fatigue)
            candidates.sort(key=lambda x: x[0])
            best_hours, comp_name, _ = candidates[0]
            hours = round(best_hours, 1)

            # Calculate fatigue multiplier and rotation risk tier
            if hours >= 72.0:
                discount = 1.0
                tier = "LOW"
            elif hours >= 64.0:
                fraction = (72.0 - hours) / 72.0
                discount = round(1.0 - 0.35 * fraction, 3)
                tier = "MODERATE"
            else:
                fraction = (72.0 - hours) / 72.0
                discount = round(1.0 - 0.50 * fraction, 3)
                tier = "ACUTE"

            return CongestionStatus(
                has_midweek=True,
                competition=comp_name,
                estimated_turnaround_hours=hours,
                fatigue_discount=discount,
                rotation_risk_tier=tier,
            )

        # 4. Fallback / legacy nominal handling when fixtures list was not available
        has_pl_midweek = "PL_MIDWEEK" in events
        if not events or not (has_ucl or has_uel or has_cup or has_pl_midweek):
            return CongestionStatus(
                has_midweek=False,
                competition="None",
                estimated_turnaround_hours=144.0,
                fatigue_discount=1.0,
                rotation_risk_tier="LOW",
            )

        # Legacy nominal fallback if event exists but candidate diff couldn't be computed
        if has_ucl:
            comp = f"UEFA Champions League ({events['UCL']})"
            hours = 66.0
        elif has_uel:
            comp = f"UEFA Europa / Conference League ({events['UEL']})"
            hours = 68.0
        elif has_cup:
            comp = f"Domestic Cup ({events.get('DOMESTIC_CUP', 'Cup')})"
            hours = 66.0
        elif has_pl_midweek:
            comp = f"Premier League ({events['PL_MIDWEEK']})"
            hours = 66.0
        else:
            comp = "Midweek Fixture"
            hours = 66.0

        if kickoff_time is not None:
            try:
                dt = _parse_utc_datetime(kickoff_time)
                if dt and dt.weekday() == 5 and dt.hour < 14:
                    hours -= 2.5
            except Exception:
                pass

        if hours >= 72.0:
            discount = 1.0
            tier = "LOW"
        elif hours >= 64.0:
            fraction = (72.0 - hours) / 72.0
            discount = round(1.0 - 0.35 * fraction, 3)
            tier = "MODERATE"
        else:
            fraction = (72.0 - hours) / 72.0
            discount = round(1.0 - 0.50 * fraction, 3)
            tier = "ACUTE"

        return CongestionStatus(
            has_midweek=True,
            competition=comp,
            estimated_turnaround_hours=hours,
            fatigue_discount=discount,
            rotation_risk_tier=tier,
        )

    def compute_player_fatigue_multiplier(
        self,
        position: str,
        congestion: CongestionStatus,
        age: int | None = None,
        squad_depth_rating: float = 1.0,
        midweek_minutes: float | None = None,
    ) -> float:
        """
        Calculates position- and age-specific fatigue multiplier for a player.
        """
        if not congestion.has_midweek or congestion.fatigue_discount >= 1.0:
            return 1.0

        if midweek_minutes is not None and midweek_minutes < 45.0:
            return 1.0  # Fully rested or negligible sub cameo

        pos = str(position).strip().upper()
        if pos == "GKP":
            # Goalkeepers suffer virtually no turnaround physical degradation
            return 1.0

        base_discount = congestion.fatigue_discount

        # Position sensitivity: Full-backs / wing-backs and high-intensity wingers face higher sprint fatigue
        if pos == "DEF":
            pos_weight = 1.10
        elif pos == "FWD":
            pos_weight = 1.05
        else:  # MID
            pos_weight = 1.00

        # Veteran age adjustment (> 30 years old faces steeper recovery curve)
        age_penalty = 0.03 if (age is not None and age >= 31) else 0.0

        raw_mult = 1.0 - (1.0 - base_discount) * pos_weight - age_penalty
        return float(np.clip(round(raw_mult, 3), 0.82, 1.0))

    def apply_congestion_to_projections(
        self,
        projections_df: pd.DataFrame,
        gameweek: int,
        team_short_map: dict[int, str] | None = None,
        fixtures: list[Any] | None = None,
        active_cup_clubs: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """
        Adjusts expected_points, expected_minutes, and p_starts in projection DataFrame.
        Always populates 'turnaround_hours' and 'congestion_fatigue_factor' columns
        so that TurnaroundDragService in LineupOptimizer is active in production.
        """
        if projections_df.empty:
            return projections_df.copy()

        df = projections_df.copy()
        if "team" not in df.columns and "team_short" not in df.columns:
            return df

        resolved_team_map, _ = self._resolve_team_mappings(team_short_map)

        congestion_cache: dict[str, CongestionStatus] = {}
        mults: list[float] = []
        hours_list: list[float] = []

        for _, row in df.iterrows():
            t_id = row.get("team")
            t_short = None
            if resolved_team_map and t_id is not None and pd.notna(t_id):
                t_short = resolved_team_map.get(int(t_id))
            if not t_short and "team_short" in row and pd.notna(row["team_short"]):
                t_short = str(row["team_short"])
            if not t_short:
                mults.append(1.0)
                hours_list.append(144.0)
                continue

            t_short_clean = t_short.strip().upper()
            if t_short_clean not in congestion_cache:
                kickoff = row.get("kickoff_time")
                congestion_cache[t_short_clean] = self.assess_club_congestion(
                    team_short=t_short_clean,
                    gameweek=gameweek,
                    kickoff_time=kickoff,
                    fixtures=fixtures,
                    active_cup_clubs=active_cup_clubs,
                )

            cong = congestion_cache[t_short_clean]
            hours_list.append(cong.estimated_turnaround_hours)

            if not cong.has_midweek:
                mults.append(1.0)
                continue

            elem_id = int(row.get("element", 0)) if pd.notna(row.get("element")) else None
            m_mins = (
                float(row.get("midweek_minutes"))
                if ("midweek_minutes" in row and pd.notna(row.get("midweek_minutes")))
                else None
            )
            if elem_id and elem_id in self._rested_player_ids.get(gameweek, set()):
                m_mins = 0.0

            pos = str(row.get("position", "MID"))
            age = row.get("age")
            p_mult = self.compute_player_fatigue_multiplier(
                pos, cong, age=int(age) if pd.notna(age) else None, midweek_minutes=m_mins
            )
            mults.append(p_mult)

        mult_series = pd.Series(mults, index=df.index)
        hours_series = pd.Series(hours_list, index=df.index)

        if "expected_points" in df.columns:
            df["expected_points"] = (df["expected_points"] * mult_series).round(2)
        if "expected_minutes" in df.columns:
            df["expected_minutes"] = (df["expected_minutes"] * mult_series).round(1)
        if "p_starts" in df.columns:
            df["p_starts"] = (df["p_starts"] * mult_series).round(3)

        df["congestion_fatigue_factor"] = mult_series
        df["turnaround_hours"] = hours_series
        return df


turnaround_congestion_engine = TurnaroundCongestionEngine()
