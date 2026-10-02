"""
Strict pre-deadline feature engineering with ZERO data leakage.
Computes rolling form, season-to-date stats, team ratings, and contextual match indicators.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.config import RULES, SCORING

logger = logging.getLogger("fpl_oracle.features")

FEATURE_COLUMNS = [
    # Minutes & Starts
    "roll_minutes_3", "roll_minutes_5", "roll_minutes_8",
    "roll_starts_ratio_5", "roll_min60_ratio_5",
    "std_minutes_per_gw",
    # Attacking Form
    "roll_points_3", "roll_points_5", "roll_points_8",
    "roll_xG_3", "roll_xG_5", "roll_xG_8",
    "roll_xA_3", "roll_xA_5", "roll_xA_8",
    "roll_xGI_5", "roll_goals_5", "roll_assists_5",
    # Defensive & Goalkeeping Form
    "roll_xGC_5", "roll_clean_sheets_5", "roll_saves_5", "roll_goals_conceded_5",
    "roll_defcon_5",
    # ICT & BPS
    "roll_ict_5", "roll_bps_5",
    # Match & Opponent Context
    "was_home", "team_strength_attack", "opp_strength_defence", "net_strength_diff",
    "opponent_difficulty", "days_rest",
    # Positional One-Hot
    "pos_GKP", "pos_DEF", "pos_MID", "pos_FWD",
    # Market & Status
    "value", "is_2026_27", "chance_of_playing"
]

class FeatureEngineering:
    def __init__(self):
        pass

    def build_historical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build pre-deadline features on historical match data.
        Returns (features_df, targets_df).
        Guarantees zero data leakage by strictly shifting by 1 within each player group.
        """
        df = df.copy()
        # Sort by name across seasons to properly connect multi-season player history
        df = df.sort_values(by=["name", "season", "round"]).reset_index(drop=True)

        # 1. Positional Encoding
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

        # 2. Season and rule indicator
        df["is_2026_27"] = (df["season"] == "2026-27").astype(float)
        df["chance_of_playing"] = 100.0 # Default for past finished games
        df["days_rest"] = 7.0 # Default rest
        df["opponent_difficulty"] = 3.0 # Neutral default

        # Synthetic team strength from rolling performance
        df["team_strength_attack"] = 3.0
        df["opp_strength_defence"] = 3.0
        df["net_strength_diff"] = 0.0

        # Group by unique player name to compute shifted rolling statistics across seasons
        grouped = df.groupby("name", group_keys=False)

        # Shifted rolling stats (Strictly pre-match!)
        for window in [3, 5, 8]:
            df[f"roll_minutes_{window}"] = grouped["minutes"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)
            df[f"roll_points_{window}"] = grouped["total_points"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)
            df[f"roll_xG_{window}"] = grouped["expected_goals"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)
            df[f"roll_xA_{window}"] = grouped["expected_assists"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)

        df["roll_starts_ratio_5"] = grouped["starts"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_min60_ratio_5"] = grouped["minutes"].apply(lambda s: (s.shift(1) >= 60).astype(float).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["std_minutes_per_gw"] = grouped["minutes"].apply(lambda s: s.shift(1).expanding(min_periods=1).mean()).fillna(0.0)

        # Seasonal Shrinkage Prior for Round 1:
        # At kickoff of a new season, do not penalize regular first-choice starters
        # who were rotated in late-season dead-rubbers (e.g. GW37/38 resting).
        is_gw1 = (df["round"] == 1)
        has_baseline = df["std_minutes_per_gw"] >= 50.0
        df.loc[is_gw1 & has_baseline, "roll_minutes_3"] = np.maximum(
            df.loc[is_gw1 & has_baseline, "roll_minutes_3"],
            df.loc[is_gw1 & has_baseline, "std_minutes_per_gw"] * 0.90
        )
        df.loc[is_gw1 & has_baseline, "roll_starts_ratio_5"] = np.maximum(
            df.loc[is_gw1 & has_baseline, "roll_starts_ratio_5"],
            0.85
        )

        df["roll_xGI_5"] = grouped["expected_goal_involvements"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_goals_5"] = grouped["goals_scored"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_assists_5"] = grouped["assists"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)

        df["roll_xGC_5"] = grouped["expected_goals_conceded"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_clean_sheets_5"] = grouped["clean_sheets"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_saves_5"] = grouped["saves"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_goals_conceded_5"] = grouped["goals_conceded"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_defcon_5"] = grouped["defensive_contribution"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)

        df["roll_ict_5"] = grouped["ict_index"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
        df["roll_bps_5"] = grouped["bps"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)

        df["was_home"] = df["was_home"].astype(float)
        df["value"] = df["value"].astype(float)

        # Build feature DataFrame
        X = df[FEATURE_COLUMNS].copy()

        # Build targets DataFrame
        Y = pd.DataFrame({
            "target_starts": (df["starts"] >= 1).astype(float),
            "target_minutes": df["minutes"].astype(float),
            "target_min60": (df["minutes"] >= 60).astype(float),
            "target_goals": df["goals_scored"].astype(float),
            "target_assists": df["assists"].astype(float),
            "target_xG": df["expected_goals"].astype(float),
            "target_xA": df["expected_assists"].astype(float),
            "target_clean_sheet": df["clean_sheets"].astype(float),
            "target_goals_conceded": df["goals_conceded"].astype(float),
            "target_saves": df["saves"].astype(float),
            "target_defcon": (df["defensive_contribution"] >= 2).astype(float),
            "target_bonus": df["bonus"].astype(float),
            "target_points": df["total_points"].astype(float)
        })

        return X, Y

    def extract_live_features_for_upcoming(
        self,
        bootstrap: BootstrapStatic,
        fixtures: List[Fixture],
        target_gw: int,
        history_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build feature records for all 667 players for a target upcoming gameweek.
        Handles double gameweeks (multiple fixtures for a team) and blank gameweeks.
        """
        # Map team strengths
        team_dict = {t.id: t for t in bootstrap.teams}
        pos_map = {et.id: et.singular_name_short for et in bootstrap.element_types}

        # Filter fixtures for target_gw
        gw_fixtures = [f for f in fixtures if f.event == target_gw]

        # Map team fixtures: team_id -> list of (opponent_id, was_home, difficulty)
        team_fixtures: Dict[int, List[Dict[str, Any]]] = {t.id: [] for t in bootstrap.teams}
        for f in gw_fixtures:
            team_fixtures[f.team_h].append({
                "opponent": f.team_a,
                "was_home": 1.0,
                "difficulty": float(f.team_h_difficulty or 3)
            })
            team_fixtures[f.team_a].append({
                "opponent": f.team_h,
                "was_home": 0.0,
                "difficulty": float(f.team_a_difficulty or 3)
            })

        rows = []
        for elem in bootstrap.elements:
            pos_code = pos_map.get(elem.element_type, "MID")
            fixtures_for_team = team_fixtures.get(elem.team, [])

            # Injury & chance of playing
            cop = 100.0
            if elem.status == "i":
                cop = 0.0
            elif elem.status == "s":
                cop = 0.0
            elif elem.chance_of_playing_next_round is not None:
                cop = float(elem.chance_of_playing_next_round)

            # Rolling stats from element properties as pre-deadline baseline
            form_val = float(elem.form or 0.0)
            ppg_val = float(elem.points_per_game or 0.0)
            ict_val = float(elem.ict_index or 0.0) / max(1, elem.starts or 1)
            bps_val = float(elem.bps or 0.0) / max(1, elem.starts or 1)

            # If blank gameweek (0 fixtures scheduled)
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
                    "roll_minutes_3": 0.0,
                    "roll_minutes_5": 0.0,
                    "roll_minutes_8": 0.0,
                    "roll_starts_ratio_5": 0.0,
                    "roll_min60_ratio_5": 0.0,
                    "std_minutes_per_gw": 0.0,
                    "roll_points_3": 0.0,
                    "roll_points_5": 0.0,
                    "roll_points_8": 0.0,
                    "roll_xG_3": 0.0,
                    "roll_xG_5": 0.0,
                    "roll_xG_8": 0.0,
                    "roll_xA_3": 0.0,
                    "roll_xA_5": 0.0,
                    "roll_xA_8": 0.0,
                    "roll_xGI_5": 0.0,
                    "roll_goals_5": 0.0,
                    "roll_assists_5": 0.0,
                    "roll_xGC_5": 0.0,
                    "roll_clean_sheets_5": 0.0,
                    "roll_saves_5": 0.0,
                    "roll_goals_conceded_5": 0.0,
                    "roll_defcon_5": 0.0,
                    "roll_ict_5": 0.0,
                    "roll_bps_5": 0.0,
                    "was_home": 0.0,
                    "team_strength_attack": 3.0,
                    "opp_strength_defence": 3.0,
                    "net_strength_diff": 0.0,
                    "opponent_difficulty": 3.0,
                    "days_rest": 7.0,
                    "pos_GKP": 1.0 if pos_code == "GKP" else 0.0,
                    "pos_DEF": 1.0 if pos_code == "DEF" else 0.0,
                    "pos_MID": 1.0 if pos_code == "MID" else 0.0,
                    "pos_FWD": 1.0 if pos_code == "FWD" else 0.0,
                    "value": float(elem.now_cost),
                    "is_2026_27": 1.0,
                    "chance_of_playing": 0.0
                }
                rows.append(row)
                continue

            for fix_idx, fix in enumerate(fixtures_for_team):
                opp_team = team_dict.get(fix["opponent"])
                my_team = team_dict.get(elem.team)

                my_att = float((my_team.strength_attack_home if fix["was_home"] else my_team.strength_attack_away) or 1000) / 300.0
                opp_def = float((opp_team.strength_defence_away if fix["was_home"] else opp_team.strength_defence_home) or 1000) / 300.0

                starts_ratio = min(1.0, float(elem.starts or 0) / 5.0) if (elem.starts or 0) > 0 else 0.0
                min60_ratio = 1.0 if (elem.minutes or 0) >= 270 else (float(elem.minutes or 0) / 300.0)

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
                    "roll_minutes_3": float(elem.minutes or 0) / 5.0 * (starts_ratio),
                    "roll_minutes_5": float(elem.minutes or 0) / 5.0 * (starts_ratio),
                    "roll_minutes_8": float(elem.minutes or 0) / 5.0 * (starts_ratio),
                    "roll_starts_ratio_5": starts_ratio,
                    "roll_min60_ratio_5": min60_ratio,
                    "std_minutes_per_gw": float(elem.minutes or 0) / 5.0,
                    "roll_points_3": form_val,
                    "roll_points_5": form_val,
                    "roll_points_8": ppg_val,
                    "roll_xG_3": float(elem.expected_goals or 0.0) / 5.0,
                    "roll_xG_5": float(elem.expected_goals or 0.0) / 5.0,
                    "roll_xG_8": float(elem.expected_goals or 0.0) / 5.0,
                    "roll_xA_3": float(elem.expected_assists or 0.0) / 5.0,
                    "roll_xA_5": float(elem.expected_assists or 0.0) / 5.0,
                    "roll_xA_8": float(elem.expected_assists or 0.0) / 5.0,
                    "roll_xGI_5": float(elem.expected_goal_involvements or 0.0) / 5.0,
                    "roll_goals_5": float(elem.goals_scored or 0) / 5.0,
                    "roll_assists_5": float(elem.assists or 0) / 5.0,
                    "roll_xGC_5": float(elem.expected_goals_conceded or 0.0) / 5.0,
                    "roll_clean_sheets_5": float(elem.clean_sheets or 0) / 5.0,
                    "roll_saves_5": float(elem.saves or 0) / 5.0,
                    "roll_goals_conceded_5": float(elem.goals_conceded or 0) / 5.0,
                    "roll_defcon_5": float(elem.defensive_contribution or 0) / 5.0,
                    "roll_ict_5": ict_val,
                    "roll_bps_5": bps_val,
                    "was_home": fix["was_home"],
                    "team_strength_attack": my_att,
                    "opp_strength_defence": opp_def,
                    "net_strength_diff": my_att - opp_def,
                    "opponent_difficulty": fix["difficulty"],
                    "days_rest": 7.0,
                    "pos_GKP": 1.0 if pos_code == "GKP" else 0.0,
                    "pos_DEF": 1.0 if pos_code == "DEF" else 0.0,
                    "pos_MID": 1.0 if pos_code == "MID" else 0.0,
                    "pos_FWD": 1.0 if pos_code == "FWD" else 0.0,
                    "value": float(elem.now_cost),
                    "is_2026_27": 1.0,
                    "chance_of_playing": cop
                }
                rows.append(row)

        return pd.DataFrame(rows)

feature_engineering = FeatureEngineering()
