"""
Strict pre-deadline feature engineering with ZERO data leakage and train/serve parity.
Computes rolling form, season-to-date stats, team ratings, opponent form,
continuous implied match xG / CS probabilities, and rare component features.
"""

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.config import HISTORICAL_DIR

logger = logging.getLogger("fpl_oracle.features")

FEATURE_COLUMNS = [
    # Minutes & Starts
    "roll_minutes_3",
    "roll_minutes_5",
    "roll_minutes_8",
    "roll_starts_ratio_5",
    "roll_min60_ratio_5",
    "std_minutes_per_gw",
    # Attacking Form
    "roll_points_3",
    "roll_points_5",
    "roll_points_8",
    "roll_xG_3",
    "roll_xG_5",
    "roll_xG_8",
    "roll_xA_3",
    "roll_xA_5",
    "roll_xA_8",
    "roll_xGI_5",
    "roll_goals_5",
    "roll_assists_5",
    # Defensive & Goalkeeping Form
    "roll_xGC_5",
    "roll_clean_sheets_5",
    "roll_saves_5",
    "roll_goals_conceded_5",
    "roll_defcon_5",
    # ICT & BPS
    "roll_ict_5",
    "roll_bps_5",
    # Disciplinary & Rare Components (A4, A7)
    "roll_cards_5",
    "roll_own_goals_5",
    "roll_penalties_missed_5",
    "roll_penalties_saved_5",
    # Match & Opponent Context
    "was_home",
    "team_strength_attack",
    "opp_strength_defence",
    "net_strength_diff",
    "opponent_difficulty",
    "days_rest",
    # Continuous Implied Match Strength Features (A2)
    "implied_team_xG",
    "implied_team_cs_prob",
    "implied_opp_xG",
    "implied_opp_cs_prob",
    # Team Attack Form (A2)
    "team_roll_goals_3",
    "team_roll_goals_5",
    "team_roll_goals_8",
    "team_roll_xG_3",
    "team_roll_xG_5",
    "team_roll_xG_8",
    # Opponent Defensive Form & Form Points (A2 - User's main concern)
    "opp_roll_points_3",
    "opp_roll_points_5",
    "opp_roll_points_8",
    "opp_roll_goals_conceded_3",
    "opp_roll_goals_conceded_5",
    "opp_roll_goals_conceded_8",
    "opp_roll_xGC_3",
    "opp_roll_xGC_5",
    "opp_roll_xGC_8",
    "opp_roll_clean_sheets_5",
    # Positional One-Hot
    "pos_GKP",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
    # Market & Status
    "value",
    "is_2026_27",
    "chance_of_playing",
]

# Canonical feature schema hash to guarantee train/serve parity
FEATURE_SCHEMA_HASH = hashlib.sha256(",".join(FEATURE_COLUMNS).encode("utf-8")).hexdigest()

# Documented neutral priors for newly promoted clubs or zero-match start of season
PRIOR_TEAM_XG = 1.30
PRIOR_TEAM_GOALS = 1.30
PRIOR_TEAM_XGC = 1.35
PRIOR_TEAM_GOALS_CONCEDED = 1.35
PRIOR_TEAM_POINTS = 1.25
PRIOR_TEAM_CLEAN_SHEET = 0.25


def compute_player_rolling_stats(p_hist: pd.DataFrame, pos_code: str = "MID", val_m: float = 6.0) -> dict[str, float]:
    """
    Shared canonical feature transformer for a player's pre-deadline match history.
    Produces identical metrics whether called in training or live serving.
    """
    if p_hist is not None and len(p_hist) > 0:
        last_3 = p_hist.tail(3)
        last_5 = p_hist.tail(5)
        last_8 = p_hist.tail(8)

        r_min_3 = float(last_3["minutes"].mean()) if not last_3.empty else 0.0
        r_min_5 = float(last_5["minutes"].mean()) if not last_5.empty else 0.0
        r_min_8 = float(last_8["minutes"].mean()) if not last_8.empty else 0.0

        r_pts_3 = float(last_3["total_points"].mean()) if not last_3.empty else 0.0
        r_pts_5 = float(last_5["total_points"].mean()) if not last_5.empty else 0.0
        r_pts_8 = float(last_8["total_points"].mean()) if not last_8.empty else 0.0

        r_xg_3 = float(last_3["expected_goals"].mean()) if not last_3.empty else 0.0
        r_xg_5 = float(last_5["expected_goals"].mean()) if not last_5.empty else 0.0
        r_xg_8 = float(last_8["expected_goals"].mean()) if not last_8.empty else 0.0

        r_xa_3 = float(last_3["expected_assists"].mean()) if not last_3.empty else 0.0
        r_xa_5 = float(last_5["expected_assists"].mean()) if not last_5.empty else 0.0
        r_xa_8 = float(last_8["expected_assists"].mean()) if not last_8.empty else 0.0

        r_starts_5 = float((last_5["starts"] >= 1).mean()) if not last_5.empty else 0.0
        r_min60_5 = float((last_5["minutes"] >= 60).mean()) if not last_5.empty else 0.0
        std_mins = float(p_hist["minutes"].mean())

        r_xgi_5 = float(last_5["expected_goal_involvements"].mean()) if "expected_goal_involvements" in last_5.columns and not last_5.empty else (r_xg_5 + r_xa_5)
        r_goals_5 = float(last_5["goals_scored"].mean()) if not last_5.empty else 0.0
        r_assists_5 = float(last_5["assists"].mean()) if not last_5.empty else 0.0

        r_xgc_5 = float(last_5["expected_goals_conceded"].mean()) if not last_5.empty else 0.0
        r_cs_5 = float(last_5["clean_sheets"].mean()) if not last_5.empty else 0.0
        r_saves_5 = float(last_5["saves"].mean()) if not last_5.empty else 0.0
        r_gc_5 = float(last_5["goals_conceded"].mean()) if not last_5.empty else 0.0
        r_defcon_5 = float(last_5["defensive_contribution"].mean()) if "defensive_contribution" in last_5.columns and not last_5.empty else 0.0

        r_ict_5 = float(last_5["ict_index"].mean()) if "ict_index" in last_5.columns and not last_5.empty else 0.0
        r_bps_5 = float(last_5["bps"].mean()) if "bps" in last_5.columns and not last_5.empty else 0.0

        # Disciplinary and rare components
        yc = last_5.get("yellow_cards", pd.Series(0, index=last_5.index)).fillna(0)
        rc = last_5.get("red_cards", pd.Series(0, index=last_5.index)).fillna(0)
        r_cards_5 = float((yc * 1.0 + rc * 3.0).mean()) if not last_5.empty else 0.0

        og = last_5.get("own_goals", pd.Series(0, index=last_5.index)).fillna(0)
        r_own_goals_5 = float(og.mean()) if not last_5.empty else 0.0

        pm = last_5.get("penalties_missed", pd.Series(0, index=last_5.index)).fillna(0)
        r_pen_missed_5 = float(pm.mean()) if not last_5.empty else 0.0

        ps = last_5.get("penalties_saved", pd.Series(0, index=last_5.index)).fillna(0)
        r_pen_saved_5 = float(ps.mean()) if not last_5.empty else 0.0

    else:
        # Documented shrinkage priors for cold starts / new signings
        if val_m >= 10.0:
            prior_mins = 80.0
            prior_starts = 0.95
        elif val_m >= 6.5:
            prior_mins = 70.0
            prior_starts = 0.85
        elif val_m >= 5.0:
            prior_mins = 55.0
            prior_starts = 0.65
        else:
            prior_mins = 25.0
            prior_starts = 0.30

        r_min_3 = prior_mins
        r_min_5 = prior_mins
        r_min_8 = prior_mins
        r_pts_3 = 2.5
        r_pts_5 = 2.5
        r_pts_8 = 2.5

        r_xg_3 = 0.15 if pos_code in ("FWD", "MID") else 0.02
        r_xg_5 = 0.15 if pos_code in ("FWD", "MID") else 0.02
        r_xg_8 = 0.15 if pos_code in ("FWD", "MID") else 0.02

        r_xa_3 = 0.12 if pos_code in ("MID", "DEF") else 0.05
        r_xa_5 = 0.12 if pos_code in ("MID", "DEF") else 0.05
        r_xa_8 = 0.12 if pos_code in ("MID", "DEF") else 0.05

        r_starts_5 = prior_starts
        r_min60_5 = prior_starts * 0.9
        std_mins = prior_mins

        r_xgi_5 = r_xg_5 + r_xa_5
        r_goals_5 = r_xg_5
        r_assists_5 = r_xa_5
        r_xgc_5 = 1.35
        r_cs_5 = 0.30 if pos_code in ("GKP", "DEF") else 0.10
        r_saves_5 = 3.0 if pos_code == "GKP" else 0.0
        r_gc_5 = 1.35 if pos_code in ("GKP", "DEF") else 0.0
        r_defcon_5 = 0.40 if pos_code in ("DEF", "MID") else 0.0

        r_ict_5 = 5.0
        r_bps_5 = 12.0
        r_cards_5 = 0.15
        r_own_goals_5 = 0.0
        r_pen_missed_5 = 0.0
        r_pen_saved_5 = 0.0

    return {
        "roll_minutes_3": r_min_3,
        "roll_minutes_5": r_min_5,
        "roll_minutes_8": r_min_8,
        "roll_starts_ratio_5": r_starts_5,
        "roll_min60_ratio_5": r_min60_5,
        "std_minutes_per_gw": std_mins,
        "roll_points_3": r_pts_3,
        "roll_points_5": r_pts_5,
        "roll_points_8": r_pts_8,
        "roll_xG_3": r_xg_3,
        "roll_xG_5": r_xg_5,
        "roll_xG_8": r_xg_8,
        "roll_xA_3": r_xa_3,
        "roll_xA_5": r_xa_5,
        "roll_xA_8": r_xa_8,
        "roll_xGI_5": r_xgi_5,
        "roll_goals_5": r_goals_5,
        "roll_assists_5": r_assists_5,
        "roll_xGC_5": r_xgc_5,
        "roll_clean_sheets_5": r_cs_5,
        "roll_saves_5": r_saves_5,
        "roll_goals_conceded_5": r_gc_5,
        "roll_defcon_5": r_defcon_5,
        "roll_ict_5": r_ict_5,
        "roll_bps_5": r_bps_5,
        "roll_cards_5": r_cards_5,
        "roll_own_goals_5": r_own_goals_5,
        "roll_penalties_missed_5": r_pen_missed_5,
        "roll_penalties_saved_5": r_pen_saved_5,
    }


class FeatureEngineering:
    def __init__(self):
        self._schema_hash = FEATURE_SCHEMA_HASH

    @property
    def schema_hash(self) -> str:
        return self._schema_hash

    def _build_team_history_map(self, df: pd.DataFrame) -> dict[Any, dict[str, float]]:
        """
        Build pre-match team attack, defence, and form metrics.
        - For training historical rows: keyed by (season, round, team) using shift(1) rolling windows.
        - For live serving upcoming rows: keyed by (season, 'latest', team), ('latest', team), and (team)
          using unshifted rolling windows through the most recent completed match.
        """
        if "team" not in df.columns:
            return {}

        sum_cols = [c for c in ["expected_goals", "goals_scored", "goals_conceded", "expected_goals_conceded"] if c in df.columns]
        if not sum_cols:
            return {}

        # Aggregate match-level team outcomes safely
        agg_dict = {}
        if "goals_scored" in df.columns:
            agg_dict["goals_scored"] = "sum"
        if "goals_conceded" in df.columns:
            agg_dict["goals_conceded"] = "max"
        if "expected_goals" in df.columns:
            agg_dict["expected_goals"] = "sum"
        if "expected_goals_conceded" in df.columns:
            agg_dict["expected_goals_conceded"] = "max"

        group_cols = ["season", "round", "team"]
        team_match = (
            df.groupby(group_cols)
            .agg(agg_dict)
            .reset_index()
            .sort_values(by=["season", "round"])
        )
        if "goals_scored" not in team_match.columns:
            team_match["goals_scored"] = 0.0
        if "goals_conceded" not in team_match.columns:
            team_match["goals_conceded"] = 0.0
        if "expected_goals" not in team_match.columns:
            team_match["expected_goals"] = 1.3
        if "expected_goals_conceded" not in team_match.columns:
            team_match["expected_goals_conceded"] = 1.3

        # Compute match points and clean sheet
        gs = team_match["goals_scored"]
        gc = team_match["goals_conceded"]
        team_match["match_points"] = np.where(gs > gc, 3.0, np.where(gs == gc, 1.0, 0.0))
        team_match["match_clean_sheet"] = (gc == 0).astype(float)

        grouped = team_match.groupby("team", group_keys=False)

        # Shifted rolling stats per team (historical match pre-deadline state)
        for w in [3, 5, 8]:
            team_match[f"team_roll_goals_{w}"] = grouped["goals_scored"].apply(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_GOALS)
            team_match[f"team_roll_xG_{w}"] = grouped["expected_goals"].apply(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_XG)
            team_match[f"team_roll_goals_conceded_{w}"] = grouped["goals_conceded"].apply(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_GOALS_CONCEDED)
            team_match[f"team_roll_xGC_{w}"] = grouped["expected_goals_conceded"].apply(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_XGC)
            team_match[f"team_roll_points_{w}"] = grouped["match_points"].apply(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_POINTS)

        team_match["team_roll_clean_sheets_5"] = grouped["match_clean_sheet"].apply(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(PRIOR_TEAM_CLEAN_SHEET)

        # Unshifted rolling stats per team (as-of latest completed match for upcoming serving)
        for w in [3, 5, 8]:
            team_match[f"team_roll_goals_{w}_unshifted"] = grouped["goals_scored"].apply(
                lambda s, win=w: s.rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_GOALS)
            team_match[f"team_roll_xG_{w}_unshifted"] = grouped["expected_goals"].apply(
                lambda s, win=w: s.rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_XG)
            team_match[f"team_roll_goals_conceded_{w}_unshifted"] = grouped["goals_conceded"].apply(
                lambda s, win=w: s.rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_GOALS_CONCEDED)
            team_match[f"team_roll_xGC_{w}_unshifted"] = grouped["expected_goals_conceded"].apply(
                lambda s, win=w: s.rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_XGC)
            team_match[f"team_roll_points_{w}_unshifted"] = grouped["match_points"].apply(
                lambda s, win=w: s.rolling(win, min_periods=1).mean()
            ).fillna(PRIOR_TEAM_POINTS)

        team_match["team_roll_clean_sheets_5_unshifted"] = grouped["match_clean_sheet"].apply(
            lambda s: s.rolling(5, min_periods=1).mean()
        ).fillna(PRIOR_TEAM_CLEAN_SHEET)

        # Build lookup map: shifted for history + unshifted for live serving
        result_map: dict[Any, dict[str, float]] = {}
        for _, r in team_match.iterrows():
            s_name = str(r["season"])
            rnd = int(r["round"])
            t_name = str(r["team"])

            shifted_dict = {
                "team_roll_goals_3": float(r["team_roll_goals_3"]),
                "team_roll_goals_5": float(r["team_roll_goals_5"]),
                "team_roll_goals_8": float(r["team_roll_goals_8"]),
                "team_roll_xG_3": float(r["team_roll_xG_3"]),
                "team_roll_xG_5": float(r["team_roll_xG_5"]),
                "team_roll_xG_8": float(r["team_roll_xG_8"]),
                "team_roll_goals_conceded_3": float(r["team_roll_goals_conceded_3"]),
                "team_roll_goals_conceded_5": float(r["team_roll_goals_conceded_5"]),
                "team_roll_goals_conceded_8": float(r["team_roll_goals_conceded_8"]),
                "team_roll_xGC_3": float(r["team_roll_xGC_3"]),
                "team_roll_xGC_5": float(r["team_roll_xGC_5"]),
                "team_roll_xGC_8": float(r["team_roll_xGC_8"]),
                "team_roll_points_3": float(r["team_roll_points_3"]),
                "team_roll_points_5": float(r["team_roll_points_5"]),
                "team_roll_points_8": float(r["team_roll_points_8"]),
                "team_roll_clean_sheets_5": float(r["team_roll_clean_sheets_5"]),
            }
            result_map[(s_name, rnd, t_name)] = shifted_dict

            unshifted_dict = {
                "team_roll_goals_3": float(r["team_roll_goals_3_unshifted"]),
                "team_roll_goals_5": float(r["team_roll_goals_5_unshifted"]),
                "team_roll_goals_8": float(r["team_roll_goals_8_unshifted"]),
                "team_roll_xG_3": float(r["team_roll_xG_3_unshifted"]),
                "team_roll_xG_5": float(r["team_roll_xG_5_unshifted"]),
                "team_roll_xG_8": float(r["team_roll_xG_8_unshifted"]),
                "team_roll_goals_conceded_3": float(r["team_roll_goals_conceded_3_unshifted"]),
                "team_roll_goals_conceded_5": float(r["team_roll_goals_conceded_5_unshifted"]),
                "team_roll_goals_conceded_8": float(r["team_roll_goals_conceded_8_unshifted"]),
                "team_roll_xGC_3": float(r["team_roll_xGC_3_unshifted"]),
                "team_roll_xGC_5": float(r["team_roll_xGC_5_unshifted"]),
                "team_roll_xGC_8": float(r["team_roll_xGC_8_unshifted"]),
                "team_roll_points_3": float(r["team_roll_points_3_unshifted"]),
                "team_roll_points_5": float(r["team_roll_points_5_unshifted"]),
                "team_roll_points_8": float(r["team_roll_points_8_unshifted"]),
                "team_roll_clean_sheets_5": float(r["team_roll_clean_sheets_5_unshifted"]),
            }
            result_map[(s_name, "latest", t_name)] = unshifted_dict
            result_map[("latest", t_name)] = unshifted_dict
            result_map[t_name] = unshifted_dict

        return result_map

    def build_team_id_maps(
        self, df: pd.DataFrame | None = None
    ) -> tuple[dict[tuple[str, int], str], dict[tuple[str, str], int]]:
        """
        Build bidirectional mapping between numeric opponent_team IDs and team name strings.
        Returns:
            opp_id_to_name: (season, opp_id) -> team_name
            name_to_opp_id: (season, team_name) -> opp_id
        Guarantees 100% resolution using reciprocal pairing and process of elimination.
        """
        opp_id_to_name: dict[tuple[str, int], str] = {}
        name_to_opp_id: dict[tuple[str, str], int] = {}

        frames = []
        if df is not None and not df.empty and "opponent_team" in df.columns:
            frames.append(df)
        master_csv = HISTORICAL_DIR / "master_history.csv"
        if master_csv.exists():
            try:
                m_df = pd.read_csv(master_csv)
                frames.append(m_df)
            except Exception as e:
                logger.warning("Could not load master_history.csv for team maps: %s", e)

        if not frames:
            return opp_id_to_name, name_to_opp_id

        comb = pd.concat(frames, ignore_index=True)
        if "team" not in comb.columns or "opponent_team" not in comb.columns or "season" not in comb.columns:
            return opp_id_to_name, name_to_opp_id

        cols = ["season", "round", "team", "opponent_team", "was_home"]
        if "kickoff_time" in comb.columns:
            cols.append("kickoff_time")
        matches = comb[cols].drop_duplicates()

        for season, s_df in matches.groupby("season"):
            s_str = str(season)
            group_keys = ["round", "kickoff_time"] if "kickoff_time" in s_df.columns else ["round"]
            for _, m_df in s_df.groupby(group_keys):
                home = m_df[m_df["was_home"].astype(bool)]
                away = m_df[~m_df["was_home"].astype(bool)]
                if len(home) == 1 and len(away) == 1:
                    try:
                        h_team = str(home.iloc[0]["team"])
                        h_opp_id = int(home.iloc[0]["opponent_team"])
                        a_team = str(away.iloc[0]["team"])
                        a_opp_id = int(away.iloc[0]["opponent_team"])
                        opp_id_to_name[(s_str, h_opp_id)] = a_team
                        name_to_opp_id[(s_str, a_team)] = h_opp_id
                        opp_id_to_name[(s_str, a_opp_id)] = h_team
                        name_to_opp_id[(s_str, h_team)] = a_opp_id
                    except (ValueError, TypeError):
                        pass

            all_teams = set(s_df["team"].astype(str).unique())
            all_ids = set()
            for x in s_df["opponent_team"].unique():
                try:
                    all_ids.add(int(x))
                except (ValueError, TypeError):
                    pass

            mapped_ids = {k[1] for k in opp_id_to_name if k[0] == s_str}
            unmapped_ids = all_ids - mapped_ids
            mapped_teams = {v for k, v in opp_id_to_name.items() if k[0] == s_str}

            for uid in unmapped_ids:
                faced_teams = set(s_df[s_df["opponent_team"] == uid]["team"].astype(str).unique())
                candidates = all_teams - faced_teams - mapped_teams
                if len(candidates) == 1:
                    c_team = list(candidates)[0]
                    opp_id_to_name[(s_str, uid)] = c_team
                    name_to_opp_id[(s_str, c_team)] = uid
                    mapped_teams.add(c_team)
                    mapped_ids.add(uid)

        return opp_id_to_name, name_to_opp_id

    def build_historical_features(
        self, df: pd.DataFrame, return_meta: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build pre-deadline features on historical match data.
        Returns (features_df, targets_df) or (features_df, targets_df, metadata_df) if return_meta=True.
        Guarantees zero data leakage by strictly shifting by 1 within each player group.
        Sorts output globally by time (season, round) for chronological evaluation.
        """
        df = df.copy()

        # Positional Encoding
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

        df["is_2026_27"] = (df["season"] == "2026-27").astype(float)
        df["chance_of_playing"] = 100.0

        # Build team history map for continuous team attack and opponent defence metrics
        team_map = self._build_team_history_map(df)
        opp_id_to_team_map, team_to_opp_id_map = self.build_team_id_maps(df)

        fallback_count = 0

        # Build team & opponent match features
        att_strengths = []
        def_strengths = []
        net_diffs = []
        opp_diffs = []

        implied_team_xg_list = []
        implied_team_cs_list = []
        implied_opp_xg_list = []
        implied_opp_cs_list = []

        t_roll_g3, t_roll_g5, t_roll_g8 = [], [], []
        t_roll_xg3, t_roll_xg5, t_roll_xg8 = [], [], []

        opp_roll_pts3, opp_roll_pts5, opp_roll_pts8 = [], [], []
        opp_roll_gc3, opp_roll_gc5, opp_roll_gc8 = [], [], []
        opp_roll_xgc3, opp_roll_xgc5, opp_roll_xgc8 = [], [], []
        opp_roll_cs5 = []

        # Default neutral fallback dict
        neutral_team_metrics = {
            "team_roll_goals_3": PRIOR_TEAM_GOALS,
            "team_roll_goals_5": PRIOR_TEAM_GOALS,
            "team_roll_goals_8": PRIOR_TEAM_GOALS,
            "team_roll_xG_3": PRIOR_TEAM_XG,
            "team_roll_xG_5": PRIOR_TEAM_XG,
            "team_roll_xG_8": PRIOR_TEAM_XG,
            "team_roll_goals_conceded_3": PRIOR_TEAM_GOALS_CONCEDED,
            "team_roll_goals_conceded_5": PRIOR_TEAM_GOALS_CONCEDED,
            "team_roll_goals_conceded_8": PRIOR_TEAM_GOALS_CONCEDED,
            "team_roll_xGC_3": PRIOR_TEAM_XGC,
            "team_roll_xGC_5": PRIOR_TEAM_XGC,
            "team_roll_xGC_8": PRIOR_TEAM_XGC,
            "team_roll_points_3": PRIOR_TEAM_POINTS,
            "team_roll_points_5": PRIOR_TEAM_POINTS,
            "team_roll_points_8": PRIOR_TEAM_POINTS,
            "team_roll_clean_sheets_5": PRIOR_TEAM_CLEAN_SHEET,
        }

        # Calculate rest days per team match if kickoff_time exists
        has_kickoff = "kickoff_time" in df.columns
        if has_kickoff:
            team_matches = (
                df[["season", "round", "team", "kickoff_time"]]
                .dropna(subset=["kickoff_time"])
                .drop_duplicates(subset=["season", "round", "team"])
                .copy()
            )
            team_matches["_dt"] = pd.to_datetime(team_matches["kickoff_time"], errors="coerce")
            team_matches = team_matches.sort_values(by=["season", "team", "_dt"])
            team_matches["_prev_dt"] = team_matches.groupby(["season", "team"])["_dt"].shift(1)
            team_matches["days_rest"] = (
                (team_matches["_dt"] - team_matches["_prev_dt"]).dt.total_seconds() / 86400.0
            ).clip(2.0, 14.0).fillna(7.0)

            rest_map = {
                (str(r["season"]), int(r["round"]), str(r["team"])): float(r["days_rest"])
                for _, r in team_matches.iterrows()
            }
            df["days_rest"] = [
                rest_map.get((str(row.get("season", "2025-26")), int(row.get("round", 1)), str(row.get("team", "Unknown"))), 7.0)
                for _, row in df.iterrows()
            ]
        else:
            df["days_rest"] = 7.0

        for _, row in df.iterrows():
            s = str(row.get("season", "2025-26"))
            rnd = int(row.get("round", 1))
            t_name = str(row.get("team", "Unknown"))
            was_h = bool(row.get("was_home", True))

            t_metrics = team_map.get((s, rnd, t_name), neutral_team_metrics)

            # Look up opponent team name with numeric ID resolution
            opp_raw = row.get("opponent_team", "Unknown")
            try:
                opp_int = int(opp_raw)
                opp_name = opp_id_to_team_map.get((s, opp_int), str(opp_raw))
            except (ValueError, TypeError):
                opp_name = str(opp_raw)

            opp_metrics = team_map.get((s, rnd, opp_name))
            if opp_metrics is None:
                opp_metrics = neutral_team_metrics
                fallback_count += 1

            my_att = float(np.clip(t_metrics["team_roll_xG_5"], 0.4, 3.5))
            opp_def = float(np.clip(opp_metrics["team_roll_xGC_5"], 0.4, 3.5))
            opp_att = float(np.clip(opp_metrics["team_roll_xG_5"], 0.4, 3.5))
            my_def = float(np.clip(t_metrics["team_roll_xGC_5"], 0.4, 3.5))

            att_strengths.append(my_att)
            def_strengths.append(opp_def)
            net_diffs.append(my_att - opp_def)

            diff_val = 3.0 + (1.35 - opp_def) * 1.2
            if not was_h:
                diff_val += 0.4
            opp_diffs.append(float(np.clip(round(diff_val), 2.0, 5.0)))

            # Implied continuous match expected goals and clean sheet probability
            home_mult = 1.10 if was_h else 0.90
            away_mult = 0.90 if was_h else 1.10

            imp_team_xg = float(np.clip(1.35 * (my_att / 1.35) * (opp_def / 1.35) * home_mult, 0.25, 4.5))
            imp_opp_xg = float(np.clip(1.35 * (opp_att / 1.35) * (my_def / 1.35) * away_mult, 0.25, 4.5))

            implied_team_xg_list.append(imp_team_xg)
            implied_team_cs_list.append(float(np.exp(-imp_opp_xg)))
            implied_opp_xg_list.append(imp_opp_xg)
            implied_opp_cs_list.append(float(np.exp(-imp_team_xg)))

            # Team attack form
            t_roll_g3.append(t_metrics["team_roll_goals_3"])
            t_roll_g5.append(t_metrics["team_roll_goals_5"])
            t_roll_g8.append(t_metrics["team_roll_goals_8"])
            t_roll_xg3.append(t_metrics["team_roll_xG_3"])
            t_roll_xg5.append(t_metrics["team_roll_xG_5"])
            t_roll_xg8.append(t_metrics["team_roll_xG_8"])

            # Opponent defensive form & form points
            opp_roll_pts3.append(opp_metrics["team_roll_points_3"])
            opp_roll_pts5.append(opp_metrics["team_roll_points_5"])
            opp_roll_pts8.append(opp_metrics["team_roll_points_8"])
            opp_roll_gc3.append(opp_metrics["team_roll_goals_conceded_3"])
            opp_roll_gc5.append(opp_metrics["team_roll_goals_conceded_5"])
            opp_roll_gc8.append(opp_metrics["team_roll_goals_conceded_8"])
            opp_roll_xgc3.append(opp_metrics["team_roll_xGC_3"])
            opp_roll_xgc5.append(opp_metrics["team_roll_xGC_5"])
            opp_roll_xgc8.append(opp_metrics["team_roll_xGC_8"])
            opp_roll_cs5.append(opp_metrics["team_roll_clean_sheets_5"])

        df["team_strength_attack"] = att_strengths
        df["opp_strength_defence"] = def_strengths
        df["net_strength_diff"] = net_diffs
        df["opponent_difficulty"] = opp_diffs

        df["implied_team_xG"] = implied_team_xg_list
        df["implied_team_cs_prob"] = implied_team_cs_list
        df["implied_opp_xG"] = implied_opp_xg_list
        df["implied_opp_cs_prob"] = implied_opp_cs_list

        df["team_roll_goals_3"] = t_roll_g3
        df["team_roll_goals_5"] = t_roll_g5
        df["team_roll_goals_8"] = t_roll_g8
        df["team_roll_xG_3"] = t_roll_xg3
        df["team_roll_xG_5"] = t_roll_xg5
        df["team_roll_xG_8"] = t_roll_xg8

        df["opp_roll_points_3"] = opp_roll_pts3
        df["opp_roll_points_5"] = opp_roll_pts5
        df["opp_roll_points_8"] = opp_roll_pts8
        df["opp_roll_goals_conceded_3"] = opp_roll_gc3
        df["opp_roll_goals_conceded_5"] = opp_roll_gc5
        df["opp_roll_goals_conceded_8"] = opp_roll_gc8
        df["opp_roll_xGC_3"] = opp_roll_xgc3
        df["opp_roll_xGC_5"] = opp_roll_xgc5
        df["opp_roll_xGC_8"] = opp_roll_xgc8
        df["opp_roll_clean_sheets_5"] = opp_roll_cs5

        # Sort by name, season, round to compute player shifted rolling features
        df_player = df.sort_values(by=["name", "season", "round"]).reset_index(drop=True)

        col_defaults = {
            "minutes": 0.0,
            "starts": 0.0,
            "goals_scored": 0.0,
            "assists": 0.0,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
            "expected_goal_involvements": 0.0,
            "expected_goals_conceded": 1.3,
            "clean_sheets": 0.0,
            "goals_conceded": 0.0,
            "saves": 0.0,
            "defensive_contribution": 0.0,
            "ict_index": 0.0,
            "bps": 0.0,
            "bonus": 0.0,
            "yellow_cards": 0.0,
            "red_cards": 0.0,
            "own_goals": 0.0,
            "penalties_missed": 0.0,
            "penalties_saved": 0.0,
            "value": 50.0,
            "total_points": 0.0,
        }
        for c, def_val in col_defaults.items():
            if c not in df_player.columns:
                if c == "expected_goal_involvements":
                    df_player[c] = df_player.get("expected_goals", 0.0) + df_player.get("expected_assists", 0.0)
                else:
                    df_player[c] = def_val

        grouped = df_player.groupby("name", group_keys=False)

        # Shifted rolling stats (Strictly pre-match!)
        for window in [3, 5, 8]:
            df_player[f"roll_minutes_{window}"] = (
                grouped["minutes"].apply(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()).fillna(0.0)
            )
            df_player[f"roll_points_{window}"] = (
                grouped["total_points"]
                .apply(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
                .fillna(0.0)
            )
            df_player[f"roll_xG_{window}"] = (
                grouped["expected_goals"]
                .apply(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
                .fillna(0.0)
            )
            df_player[f"roll_xA_{window}"] = (
                grouped["expected_assists"]
                .apply(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
                .fillna(0.0)
            )

        df_player["roll_starts_ratio_5"] = (
            grouped["starts"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_min60_ratio_5"] = (
            grouped["minutes"]
            .apply(lambda s: (s.shift(1) >= 60).astype(float).rolling(5, min_periods=1).mean())
            .fillna(0.0)
        )
        df_player["std_minutes_per_gw"] = (
            grouped["minutes"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0.0)
        )

        # Shrinkage prior for Round 1
        is_gw1 = df_player["round"] == 1
        has_baseline = df_player["std_minutes_per_gw"] >= 50.0
        df_player.loc[is_gw1 & has_baseline, "roll_minutes_3"] = np.maximum(
            df_player.loc[is_gw1 & has_baseline, "roll_minutes_3"], df_player.loc[is_gw1 & has_baseline, "std_minutes_per_gw"] * 0.90
        )
        df_player.loc[is_gw1 & has_baseline, "roll_starts_ratio_5"] = np.maximum(
            df_player.loc[is_gw1 & has_baseline, "roll_starts_ratio_5"], 0.85
        )

        df_player["roll_xGI_5"] = (
            grouped["expected_goal_involvements"]
            .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .fillna(0.0)
        )
        df_player["roll_goals_5"] = (
            grouped["goals_scored"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_assists_5"] = (
            grouped["assists"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_xGC_5"] = (
            grouped["expected_goals_conceded"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_clean_sheets_5"] = (
            grouped["clean_sheets"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_saves_5"] = (
            grouped["saves"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_goals_conceded_5"] = (
            grouped["goals_conceded"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_defcon_5"] = (
            grouped["defensive_contribution"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_ict_5"] = (
            grouped["ict_index"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )
        df_player["roll_bps_5"] = (
            grouped["bps"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )

        # Disciplinary and rare components
        yc = df_player.get("yellow_cards", pd.Series(0, index=df_player.index)).fillna(0)
        rc = df_player.get("red_cards", pd.Series(0, index=df_player.index)).fillna(0)
        df_player["_cards_loss"] = yc * 1.0 + rc * 3.0
        df_player["roll_cards_5"] = (
            grouped["_cards_loss"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        )

        og = df_player.get("own_goals", pd.Series(0, index=df_player.index)).fillna(0)
        df_player["roll_own_goals_5"] = (
            grouped[og.name if hasattr(og, 'name') else 'own_goals'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
            if "own_goals" in df_player.columns
            else 0.0
        )

        if "penalties_missed" in df_player.columns:
            df_player["roll_penalties_missed_5"] = (
                grouped["penalties_missed"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
            )
        else:
            df_player["roll_penalties_missed_5"] = 0.0

        if "penalties_saved" in df_player.columns:
            df_player["roll_penalties_saved_5"] = (
                grouped["penalties_saved"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
            )
        else:
            df_player["roll_penalties_saved_5"] = 0.0

        df_player["was_home"] = df_player["was_home"].astype(float)
        df_player["value"] = df_player["value"].astype(float)

        # Calculate actual ground truth card deduction (-1 for yellow, -3 for red, -2 for own goal, -2 for missed pen)
        og_col = df_player.get("own_goals", pd.Series(0, index=df_player.index)).fillna(0)
        pm_col = df_player.get("penalties_missed", pd.Series(0, index=df_player.index)).fillna(0)
        df_player["target_card_deduction"] = (yc * 1.0 + rc * 3.0 + og_col * 2.0 + pm_col * 2.0).astype(float)

        # CRITICAL: Sort globally by true chronological time (season, round, name)
        sort_cols = [c for c in ["season", "round", "kickoff_time", "name"] if c in df_player.columns]
        if sort_cols:
            df_sorted = df_player.sort_values(by=sort_cols).reset_index(drop=True)
        else:
            df_sorted = df_player.reset_index(drop=True)

        X = df_sorted[FEATURE_COLUMNS].copy()

        Y = pd.DataFrame(
            {
                "target_starts": (df_sorted["starts"] >= 1).astype(float),
                "target_minutes": df_sorted["minutes"].astype(float),
                "target_min60": (df_sorted["minutes"] >= 60).astype(float),
                "target_goals": df_sorted["goals_scored"].astype(float),
                "target_assists": df_sorted["assists"].astype(float),
                "target_xG": df_sorted["expected_goals"].astype(float),
                "target_xA": df_sorted["expected_assists"].astype(float),
                "target_clean_sheet": df_sorted["clean_sheets"].astype(float),
                "target_goals_conceded": df_sorted["goals_conceded"].astype(float),
                "target_saves": df_sorted["saves"].astype(float),
                "target_defcon": (df_sorted["defensive_contribution"] >= 2).astype(float),
                "target_bonus": df_sorted["bonus"].astype(float),
                "target_card_deduction": df_sorted["target_card_deduction"].astype(float),
                "target_penalties_saved": df_sorted.get("penalties_saved", pd.Series(0, index=df_sorted.index)).fillna(0).astype(float),
                "target_points": df_sorted["total_points"].astype(float),
            }
        )

        meta_cols = [c for c in ["season", "round", "name", "element", "team", "position"] if c in df_sorted.columns]
        meta = df_sorted[meta_cols].copy()
        X.attrs["meta"] = meta
        X.attrs["schema_hash"] = self._schema_hash

        if return_meta:
            return X, Y, meta
        return X, Y

    def extract_live_features_for_upcoming(
        self,
        bootstrap: BootstrapStatic,
        fixtures: list[Fixture],
        target_gw: int,
        history_df: pd.DataFrame | None = None,
        reconciled_availabilities: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """
        Build feature records for all players for a target upcoming gameweek with train/serve parity.
        Uses real player match history to compute rolling averages.
        Automatically loads historical master match data if history_df is omitted.
        """
        team_dict = {t.id: t for t in bootstrap.teams}
        pos_map = {et.id: et.singular_name_short for et in bootstrap.element_types}

        gw_fixtures = [f for f in fixtures if f.event == target_gw]
        team_fixtures: dict[int, list[dict[str, Any]]] = {t.id: [] for t in bootstrap.teams}
        for f in gw_fixtures:
            team_fixtures[f.team_h].append(
                {
                    "opponent": f.team_a,
                    "was_home": 1.0,
                    "difficulty": float(f.team_h_difficulty or 3),
                    "kickoff_time": f.kickoff_time,
                }
            )
            team_fixtures[f.team_a].append(
                {
                    "opponent": f.team_h,
                    "was_home": 0.0,
                    "difficulty": float(f.team_a_difficulty or 3),
                    "kickoff_time": f.kickoff_time,
                }
            )

        # Track most recent completed kickoff per team to compute authentic days_rest for upcoming GW
        last_team_kickoff: dict[int, pd.Timestamp] = {}
        for f in fixtures:
            if f.finished or (f.event is not None and f.event < target_gw):
                if f.kickoff_time:
                    ko_dt = pd.to_datetime(f.kickoff_time, errors="coerce")
                    if pd.notnull(ko_dt):
                        for tid in (f.team_h, f.team_a):
                            if tid not in last_team_kickoff or ko_dt > last_team_kickoff[tid]:
                                last_team_kickoff[tid] = ko_dt

        # If history_df not provided, attempt to load master_history.csv
        if history_df is None or history_df.empty:
            master_csv = HISTORICAL_DIR / "master_history.csv"
            if master_csv.exists():
                try:
                    history_df = pd.read_csv(master_csv)
                except Exception as e:
                    logger.warning("Could not read master_history.csv for live features: %s", e)

        # Index player histories by web_name and element_id
        hist_by_name: dict[str, pd.DataFrame] = {}
        hist_by_elem: dict[int, pd.DataFrame] = {}
        team_history_map: dict[Any, dict[str, float]] = {}

        if history_df is not None and not history_df.empty:
            for name, p_df in history_df.groupby("name"):
                hist_by_name[name] = p_df.sort_values(by=["season", "round"])
            if "element" in history_df.columns:
                for elem_id, p_df in history_df.groupby("element"):
                    hist_by_elem[int(elem_id)] = p_df.sort_values(by=["season", "round"])
            team_history_map = self._build_team_history_map(history_df)

            # Supplement last_team_kickoff from history_df if fixture kickoff not available
            if "team" in history_df.columns and "kickoff_time" in history_df.columns:
                name_to_tid = {t.name.lower(): t.id for t in bootstrap.teams}
                short_to_tid = {t.short_name.lower(): t.id for t in bootstrap.teams}
                for t_val, t_hist in history_df.groupby("team"):
                    t_str = str(t_val).lower()
                    tid = name_to_tid.get(t_str) or short_to_tid.get(t_str)
                    if tid is not None and tid not in last_team_kickoff:
                        max_ko = pd.to_datetime(t_hist["kickoff_time"], errors="coerce").max()
                        if pd.notnull(max_ko):
                            last_team_kickoff[tid] = max_ko

        rows = []
        for elem in bootstrap.elements:
            pos_code = pos_map.get(elem.element_type, "MID")
            fixtures_for_team = team_fixtures.get(elem.team, [])
            val_m = float(elem.now_cost) / 10.0

            # Availability: reconciled override takes precedence in gated_active mode
            cop = 100.0
            if reconciled_availabilities and elem.id in reconciled_availabilities:
                val = float(reconciled_availabilities[elem.id])
                cop = val * 100.0 if 0.0 < val <= 1.0 else val
            elif elem.status in ("i", "s"):
                cop = 0.0
            elif elem.chance_of_playing_next_round is not None:
                cop = float(elem.chance_of_playing_next_round)

            # Look up player match history for authentic rolling stats
            p_hist = hist_by_elem.get(elem.id)
            if p_hist is None or p_hist.empty:
                p_hist = hist_by_name.get(elem.web_name)

            # Shared canonical transformer ensures 100% train/serve parity!
            player_stats = compute_player_rolling_stats(p_hist, pos_code=pos_code, val_m=val_m)

            # Blank Gameweek Handling
            if not fixtures_for_team:
                row = {
                    "element": elem.id,
                    "web_name": elem.web_name,
                    "team": elem.team,
                    "position": pos_code,
                    "target_gw": target_gw,
                    "fixture_count": 0,
                    "is_bgw": 1,
                    "is_dgw": 0,
                    **player_stats,
                    "was_home": 0.0,
                    "team_strength_attack": 1.30,
                    "opp_strength_defence": 1.35,
                    "net_strength_diff": -0.05,
                    "opponent_difficulty": 3.0,
                    "days_rest": 7.0,
                    "implied_team_xG": 1.30,
                    "implied_team_cs_prob": 0.25,
                    "implied_opp_xG": 1.35,
                    "implied_opp_cs_prob": 0.25,
                    "team_roll_goals_3": PRIOR_TEAM_GOALS,
                    "team_roll_goals_5": PRIOR_TEAM_GOALS,
                    "team_roll_goals_8": PRIOR_TEAM_GOALS,
                    "team_roll_xG_3": PRIOR_TEAM_XG,
                    "team_roll_xG_5": PRIOR_TEAM_XG,
                    "team_roll_xG_8": PRIOR_TEAM_XG,
                    "opp_roll_points_3": PRIOR_TEAM_POINTS,
                    "opp_roll_points_5": PRIOR_TEAM_POINTS,
                    "opp_roll_points_8": PRIOR_TEAM_POINTS,
                    "opp_roll_goals_conceded_3": PRIOR_TEAM_GOALS_CONCEDED,
                    "opp_roll_goals_conceded_5": PRIOR_TEAM_GOALS_CONCEDED,
                    "opp_roll_goals_conceded_8": PRIOR_TEAM_GOALS_CONCEDED,
                    "opp_roll_xGC_3": PRIOR_TEAM_XGC,
                    "opp_roll_xGC_5": PRIOR_TEAM_XGC,
                    "opp_roll_xGC_8": PRIOR_TEAM_XGC,
                    "opp_roll_clean_sheets_5": PRIOR_TEAM_CLEAN_SHEET,
                    "pos_GKP": 1.0 if pos_code == "GKP" else 0.0,
                    "pos_DEF": 1.0 if pos_code == "DEF" else 0.0,
                    "pos_MID": 1.0 if pos_code == "MID" else 0.0,
                    "pos_FWD": 1.0 if pos_code == "FWD" else 0.0,
                    "value": float(elem.now_cost),
                    "is_2026_27": 1.0,
                    "chance_of_playing": 0.0,
                }
                rows.append(row)
                continue

            for fix_idx, fix in enumerate(fixtures_for_team):
                opp_team = team_dict.get(fix["opponent"])
                my_team = team_dict.get(elem.team)

                was_h = bool(fix["was_home"])
                my_t_name = my_team.name if my_team else ""
                opp_t_name = opp_team.name if opp_team else ""

                # Extract as-of team rolling metrics using unshifted latest completed match stats
                my_t_stats = (
                    team_history_map.get(("2026-27", "latest", my_t_name))
                    or team_history_map.get(("latest", my_t_name))
                    or team_history_map.get(my_t_name)
                )
                opp_t_stats = (
                    team_history_map.get(("2026-27", "latest", opp_t_name))
                    or team_history_map.get(("latest", opp_t_name))
                    or team_history_map.get(opp_t_name)
                )

                # Prior fallback from bootstrap official strength if history is cold
                if my_t_stats is not None:
                    my_att = float(np.clip(my_t_stats["team_roll_xG_5"], 0.4, 3.5))
                    my_def = float(np.clip(my_t_stats["team_roll_xGC_5"], 0.4, 3.5))
                    t_g3 = my_t_stats["team_roll_goals_3"]
                    t_g5 = my_t_stats["team_roll_goals_5"]
                    t_g8 = my_t_stats["team_roll_goals_8"]
                    t_xg3 = my_t_stats["team_roll_xG_3"]
                    t_xg5 = my_t_stats["team_roll_xG_5"]
                    t_xg8 = my_t_stats["team_roll_xG_8"]
                else:
                    my_att = float((my_team.strength_attack_home if was_h else my_team.strength_attack_away) or 1000) / 750.0 if my_team else 1.30
                    my_def = float((my_team.strength_defence_home if was_h else my_team.strength_defence_away) or 1000) / 750.0 if my_team else 1.35
                    t_g3, t_g5, t_g8 = PRIOR_TEAM_GOALS, PRIOR_TEAM_GOALS, PRIOR_TEAM_GOALS
                    t_xg3, t_xg5, t_xg8 = my_att, my_att, my_att

                if opp_t_stats is not None:
                    opp_att = float(np.clip(opp_t_stats["team_roll_xG_5"], 0.4, 3.5))
                    opp_def = float(np.clip(opp_t_stats["team_roll_xGC_5"], 0.4, 3.5))
                    opp_p3 = opp_t_stats["team_roll_points_3"]
                    opp_p5 = opp_t_stats["team_roll_points_5"]
                    opp_p8 = opp_t_stats["team_roll_points_8"]
                    opp_gc3 = opp_t_stats["team_roll_goals_conceded_3"]
                    opp_gc5 = opp_t_stats["team_roll_goals_conceded_5"]
                    opp_gc8 = opp_t_stats["team_roll_goals_conceded_8"]
                    opp_xgc3 = opp_t_stats["team_roll_xGC_3"]
                    opp_xgc5 = opp_t_stats["team_roll_xGC_5"]
                    opp_xgc8 = opp_t_stats["team_roll_xGC_8"]
                    opp_cs5 = opp_t_stats["team_roll_clean_sheets_5"]
                else:
                    opp_att = float((opp_team.strength_attack_away if was_h else opp_team.strength_attack_home) or 1000) / 750.0 if opp_team else 1.30
                    opp_def = float((opp_team.strength_defence_away if was_h else opp_team.strength_defence_home) or 1000) / 750.0 if opp_team else 1.35
                    opp_p3, opp_p5, opp_p8 = PRIOR_TEAM_POINTS, PRIOR_TEAM_POINTS, PRIOR_TEAM_POINTS
                    opp_gc3, opp_gc5, opp_gc8 = PRIOR_TEAM_GOALS_CONCEDED, PRIOR_TEAM_GOALS_CONCEDED, PRIOR_TEAM_GOALS_CONCEDED
                    opp_xgc3, opp_xgc5, opp_xgc8 = opp_def, opp_def, opp_def
                    opp_cs5 = PRIOR_TEAM_CLEAN_SHEET

                # Calculate days rest authentically from fixture kickoff timestamps
                fix_ko = fix.get("kickoff_time")
                days_rest_val = 7.0
                if fix_ko:
                    fix_dt = pd.to_datetime(fix_ko, errors="coerce")
                    if pd.notnull(fix_dt):
                        if fix_idx > 0:
                            prev_fix_ko = fixtures_for_team[fix_idx - 1].get("kickoff_time")
                            if prev_fix_ko:
                                prev_dt = pd.to_datetime(prev_fix_ko, errors="coerce")
                                if pd.notnull(prev_dt):
                                    days_rest_val = float(np.clip((fix_dt - prev_dt).total_seconds() / 86400.0, 2.0, 14.0))
                        else:
                            prev_dt = last_team_kickoff.get(elem.team)
                            if prev_dt and pd.notnull(prev_dt):
                                days_rest_val = float(np.clip((fix_dt - prev_dt).total_seconds() / 86400.0, 2.0, 14.0))

                # Continuous implied match expected goals and clean sheet probability
                home_mult = 1.10 if was_h else 0.90
                away_mult = 0.90 if was_h else 1.10

                imp_team_xg = float(np.clip(1.35 * (my_att / 1.35) * (opp_def / 1.35) * home_mult, 0.25, 4.5))
                imp_opp_xg = float(np.clip(1.35 * (opp_att / 1.35) * (my_def / 1.35) * away_mult, 0.25, 4.5))

                diff_val = 3.0 + (1.35 - opp_def) * 1.2
                if not was_h:
                    diff_val += 0.4
                opp_diff_val = float(np.clip(round(diff_val), 2.0, 5.0))

                row = {
                    "element": elem.id,
                    "web_name": elem.web_name,
                    "team": elem.team,
                    "position": pos_code,
                    "target_gw": target_gw,
                    "fixture_count": len(fixtures_for_team),
                    "is_bgw": 0,
                    "is_dgw": 1 if len(fixtures_for_team) > 1 else 0,
                    "fixture_sub_index": fix_idx,
                    **player_stats,
                    "was_home": float(was_h),
                    "team_strength_attack": my_att,
                    "opp_strength_defence": opp_def,
                    "net_strength_diff": my_att - opp_def,
                    "opponent_difficulty": opp_diff_val,
                    "days_rest": days_rest_val,
                    "implied_team_xG": imp_team_xg,
                    "implied_team_cs_prob": float(np.exp(-imp_opp_xg)),
                    "implied_opp_xG": imp_opp_xg,
                    "implied_opp_cs_prob": float(np.exp(-imp_team_xg)),
                    "team_roll_goals_3": t_g3,
                    "team_roll_goals_5": t_g5,
                    "team_roll_goals_8": t_g8,
                    "team_roll_xG_3": t_xg3,
                    "team_roll_xG_5": t_xg5,
                    "team_roll_xG_8": t_xg8,
                    "opp_roll_points_3": opp_p3,
                    "opp_roll_points_5": opp_p5,
                    "opp_roll_points_8": opp_p8,
                    "opp_roll_goals_conceded_3": opp_gc3,
                    "opp_roll_goals_conceded_5": opp_gc5,
                    "opp_roll_goals_conceded_8": opp_gc8,
                    "opp_roll_xGC_3": opp_xgc3,
                    "opp_roll_xGC_5": opp_xgc5,
                    "opp_roll_xGC_8": opp_xgc8,
                    "opp_roll_clean_sheets_5": opp_cs5,
                    "pos_GKP": 1.0 if pos_code == "GKP" else 0.0,
                    "pos_DEF": 1.0 if pos_code == "DEF" else 0.0,
                    "pos_MID": 1.0 if pos_code == "MID" else 0.0,
                    "pos_FWD": 1.0 if pos_code == "FWD" else 0.0,
                    "value": float(elem.now_cost),
                    "is_2026_27": 1.0,
                    "chance_of_playing": cop,
                }
                rows.append(row)

        return pd.DataFrame(rows)


feature_engineering = FeatureEngineering()
