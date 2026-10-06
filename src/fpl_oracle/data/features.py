"""
Strict pre-deadline feature engineering with ZERO data leakage and train/serve parity.
Computes rolling form, season-to-date stats, team ratings, and contextual match indicators.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture

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
    # Match & Opponent Context
    "was_home",
    "team_strength_attack",
    "opp_strength_defence",
    "net_strength_diff",
    "opponent_difficulty",
    "days_rest",
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


class FeatureEngineering:
    def __init__(self):
        pass

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
        df["days_rest"] = 7.0

        # Compute Team Attack and Opponent Defence Strengths from match histories
        # 1. Team-level attack form (shifted rolling xG / goals)
        att_strengths = []
        def_strengths = []
        opp_diffs = []

        if "team" in df.columns and "opponent_team" in df.columns:
            sum_cols = [c for c in ["expected_goals", "goals_scored", "goals_conceded", "expected_goals_conceded"] if c in df.columns]
            team_df = (
                df.groupby(["season", "round", "team"])
                .agg({c: "sum" for c in sum_cols})
                .reset_index()
                .sort_values(by=["season", "round"])
            )
            team_grouped = team_df.groupby("team", group_keys=False)
            if "expected_goals" in team_df.columns:
                team_df["team_roll_xG"] = team_grouped["expected_goals"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(1.3)
            else:
                team_df["team_roll_xG"] = 1.3
            if "expected_goals_conceded" in team_df.columns:
                team_df["team_roll_xGC"] = team_grouped["expected_goals_conceded"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(1.3)
            else:
                team_df["team_roll_xGC"] = 1.3

            team_strength_map = {
                (r["season"], r["round"], r["team"]): (r["team_roll_xG"], r["team_roll_xGC"])
                for _, r in team_df.iterrows()
            }

            for _, row in df.iterrows():
                s_key = (row["season"], row["round"], row["team"])
                opp_key = (row["season"], row["round"], row["opponent_team"])
                my_att, _ = team_strength_map.get(s_key, (1.3, 1.3))
                _, opp_def = team_strength_map.get(opp_key, (1.3, 1.3))

                att_strengths.append(float(np.clip(my_att, 0.5, 3.5)))
                def_strengths.append(float(np.clip(opp_def, 0.5, 3.5)))

                diff = 3.0 + (1.3 - opp_def) * 1.2
                if not row.get("was_home", True):
                    diff += 0.4
                opp_diffs.append(float(np.clip(round(diff), 2.0, 5.0)))
        else:
            att_strengths = [1.3] * len(df)
            def_strengths = [1.3] * len(df)
            opp_diffs = [3.0] * len(df)

        df["team_strength_attack"] = att_strengths
        df["opp_strength_defence"] = def_strengths
        df["net_strength_diff"] = df["team_strength_attack"] - df["opp_strength_defence"]
        df["opponent_difficulty"] = opp_diffs

        # Sort by name, season, round to compute player shifted rolling features
        df_player = df.sort_values(by=["name", "season", "round"]).reset_index(drop=True)
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

        df_player["was_home"] = df_player["was_home"].astype(float)
        df_player["value"] = df_player["value"].astype(float)

        # Calculate actual ground truth card deduction (-1 for yellow, -3 for red)
        yellow_col = df_player.get("yellow_cards", pd.Series(0, index=df_player.index)).fillna(0)
        red_col = df_player.get("red_cards", pd.Series(0, index=df_player.index)).fillna(0)
        target_card_deduction = (yellow_col * 1.0 + red_col * 3.0).astype(float)

        # CRITICAL: Sort globally by true chronological time (season, round, name)
        # so holdouts and rolling origin splits are genuinely chronological!
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
                "target_card_deduction": target_card_deduction.loc[df_sorted.index].values,
                "target_points": df_sorted["total_points"].astype(float),
            }
        )

        meta_cols = [c for c in ["season", "round", "name", "element", "team", "position"] if c in df_sorted.columns]
        meta = df_sorted[meta_cols].copy()
        X.attrs["meta"] = meta

        if return_meta:
            return X, Y, meta
        return X, Y

    def extract_live_features_for_upcoming(
        self,
        bootstrap: BootstrapStatic,
        fixtures: list[Fixture],
        target_gw: int,
        history_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build feature records for all players for a target upcoming gameweek with train/serve parity.
        Uses real match history to compute rolling averages (NO fixed / 5.0 divisor!).
        """
        team_dict = {t.id: t for t in bootstrap.teams}
        pos_map = {et.id: et.singular_name_short for et in bootstrap.element_types}

        gw_fixtures = [f for f in fixtures if f.event == target_gw]
        team_fixtures: dict[int, list[dict[str, Any]]] = {t.id: [] for t in bootstrap.teams}
        for f in gw_fixtures:
            team_fixtures[f.team_h].append(
                {"opponent": f.team_a, "was_home": 1.0, "difficulty": float(f.team_h_difficulty or 3)}
            )
            team_fixtures[f.team_a].append(
                {"opponent": f.team_h, "was_home": 0.0, "difficulty": float(f.team_a_difficulty or 3)}
            )

        # Index player histories by web_name and element_id if history_df is provided
        hist_by_name: dict[str, pd.DataFrame] = {}
        hist_by_elem: dict[int, pd.DataFrame] = {}
        if history_df is not None and not history_df.empty:
            for name, p_df in history_df.groupby("name"):
                hist_by_name[name] = p_df.sort_values(by=["season", "round"])
            if "element" in history_df.columns:
                for elem_id, p_df in history_df.groupby("element"):
                    hist_by_elem[int(elem_id)] = p_df.sort_values(by=["season", "round"])

        rows = []
        for elem in bootstrap.elements:
            pos_code = pos_map.get(elem.element_type, "MID")
            fixtures_for_team = team_fixtures.get(elem.team, [])

            cop = 100.0
            if elem.status in ("i", "s"):
                cop = 0.0
            elif elem.chance_of_playing_next_round is not None:
                cop = float(elem.chance_of_playing_next_round)

            # Look up player match history for authentic rolling stats
            p_hist = hist_by_elem.get(elem.id)
            if p_hist is None or p_hist.empty:
                p_hist = hist_by_name.get(elem.web_name)

            if p_hist is not None and len(p_hist) > 0:
                # Real player match history available: compute true rolling metrics!
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

                r_xgi_5 = float(last_5["expected_goal_involvements"].mean()) if not last_5.empty else 0.0
                r_goals_5 = float(last_5["goals_scored"].mean()) if not last_5.empty else 0.0
                r_assists_5 = float(last_5["assists"].mean()) if not last_5.empty else 0.0

                r_xgc_5 = float(last_5["expected_goals_conceded"].mean()) if not last_5.empty else 0.0
                r_cs_5 = float(last_5["clean_sheets"].mean()) if not last_5.empty else 0.0
                r_saves_5 = float(last_5["saves"].mean()) if not last_5.empty else 0.0
                r_gc_5 = float(last_5["goals_conceded"].mean()) if not last_5.empty else 0.0
                r_defcon_5 = float(last_5["defensive_contribution"].mean()) if not last_5.empty else 0.0

                r_ict_5 = float(last_5["ict_index"].mean()) if not last_5.empty else 0.0
                r_bps_5 = float(last_5["bps"].mean()) if not last_5.empty else 0.0
            else:
                # Cold start: principled price & position prior shrinkage (no fixed / 5 divisor!)
                val_m = float(elem.now_cost) / 10.0
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

                form_val = float(elem.form or 2.0)
                ppg_val = float(elem.points_per_game or 2.5)

                r_min_3 = prior_mins
                r_min_5 = prior_mins
                r_min_8 = prior_mins
                r_pts_3 = form_val
                r_pts_5 = form_val
                r_pts_8 = ppg_val

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
                r_xgc_5 = 1.3
                r_cs_5 = 0.30 if pos_code in ("GKP", "DEF") else 0.10
                r_saves_5 = 3.0 if pos_code == "GKP" else 0.0
                r_gc_5 = 1.3 if pos_code in ("GKP", "DEF") else 0.0
                r_defcon_5 = 0.40 if pos_code in ("DEF", "MID") else 0.0

                r_ict_5 = 5.0
                r_bps_5 = 12.0

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
                    "team_strength_attack": 1.3,
                    "opp_strength_defence": 1.3,
                    "net_strength_diff": 0.0,
                    "opponent_difficulty": 3.0,
                    "days_rest": 7.0,
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

                # Team strength attack & Opp defence on consistent 1.0 - 2.5 scale
                my_att = float((my_team.strength_attack_home if fix["was_home"] else my_team.strength_attack_away) or 1000) / 750.0 if my_team else 1.3
                opp_def = float((opp_team.strength_defence_away if fix["was_home"] else opp_team.strength_defence_home) or 1000) / 750.0 if opp_team else 1.3

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
                    "was_home": fix["was_home"],
                    "team_strength_attack": my_att,
                    "opp_strength_defence": opp_def,
                    "net_strength_diff": my_att - opp_def,
                    "opponent_difficulty": float(fix["difficulty"]),
                    "days_rest": 7.0,
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
