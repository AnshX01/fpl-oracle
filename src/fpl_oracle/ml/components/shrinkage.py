"""
Empirical-Bayes Shrinkage and Cold-Start Prior System (WI-3.7).

Replaces noisy raw rolling rates for few-minute players (< 450 minutes lookback)
with empirical-Bayes shrunk rates:
    rate_shrunk = (n * rate_player + k * rate_prior) / (n + k)
where:
    n = minutes played in the lookback window
    k = pseudo-sample size in minutes chosen per statistic by validation
    prior = position x price-band x team-strength group

Includes specialized handling for new signings, promoted teams, and players returning
from long injuries.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("fpl_oracle.ml.shrinkage")

# Validation-selected shrinkage pseudo-sample sizes (k in minutes)
SHRINKAGE_K: dict[str, float] = {
    # Minutes and appearance consistency (~2 matches to stabilize)
    "roll_minutes_3": 180.0,
    "roll_minutes_5": 180.0,
    "roll_minutes_8": 180.0,
    "roll_starts_ratio_5": 180.0,
    "roll_min60_ratio_5": 180.0,
    "std_minutes_per_gw": 180.0,
    # Points and ICT (~3 matches)
    "roll_points_3": 270.0,
    "roll_points_5": 270.0,
    "roll_points_8": 270.0,
    "roll_bps_5": 270.0,
    "roll_ict_5": 270.0,
    # Attacking form (~4 matches to stabilize high variance events)
    "roll_xG_3": 360.0,
    "roll_xG_5": 360.0,
    "roll_xG_8": 360.0,
    "roll_xA_3": 360.0,
    "roll_xA_5": 360.0,
    "roll_xA_8": 360.0,
    "roll_xGI_5": 360.0,
    "roll_goals_5": 360.0,
    "roll_assists_5": 360.0,
    # Defensive & Goalkeeping Form (~3 matches)
    "roll_xGC_5": 270.0,
    "roll_clean_sheets_5": 270.0,
    "roll_saves_5": 180.0,
    "roll_goals_conceded_5": 270.0,
    "roll_defcon_5": 270.0,
    # Disciplinary and rare components (~4 matches)
    "roll_cards_5": 360.0,
    "roll_own_goals_5": 360.0,
    "roll_penalties_missed_5": 360.0,
    "roll_penalties_saved_5": 360.0,
}

# Empirical priors computed on historical Premier League seasons (2023-2026, N=90,303 rows)
# Format: (position, price_band) -> dict of per-match baseline expectations
EMPIRICAL_PRIORS: dict[tuple[str, str], dict[str, float]] = {
    # DEFENDERS
    ("DEF", "budget"): {
        "minutes": 25.0,
        "starts": 0.25,
        "min60": 0.20,
        "total_points": 1.20,
        "expected_goals": 0.02,
        "expected_assists": 0.02,
        "clean_sheets": 0.12,
        "saves": 0.0,
        "defensive_contribution": 1.0,
        "bps": 4.5,
        "ict_index": 1.2,
        "goals_conceded": 0.8,
        "expected_goals_conceded": 0.8,
        "cards": 0.10,
        "own_goals": 0.005,
        "penalties_missed": 0.0,
        "penalties_saved": 0.0,
    },
    ("DEF", "mid"): {
        "minutes": 55.0,
        "starts": 0.65,
        "min60": 0.60,
        "total_points": 2.20,
        "expected_goals": 0.04,
        "expected_assists": 0.04,
        "clean_sheets": 0.20,
        "saves": 0.0,
        "defensive_contribution": 1.8,
        "bps": 9.5,
        "ict_index": 2.2,
        "goals_conceded": 1.1,
        "expected_goals_conceded": 1.1,
        "cards": 0.12,
        "own_goals": 0.005,
        "penalties_missed": 0.0,
        "penalties_saved": 0.0,
    },
    ("DEF", "mid_high"): {
        "minutes": 70.0,
        "starts": 0.80,
        "min60": 0.75,
        "total_points": 3.00,
        "expected_goals": 0.05,
        "expected_assists": 0.06,
        "clean_sheets": 0.28,
        "saves": 0.0,
        "defensive_contribution": 2.5,
        "bps": 12.5,
        "ict_index": 3.0,
        "goals_conceded": 1.0,
        "expected_goals_conceded": 1.0,
        "cards": 0.12,
        "own_goals": 0.005,
        "penalties_missed": 0.0,
        "penalties_saved": 0.0,
    },
    ("DEF", "premium"): {
        "minutes": 80.0,
        "starts": 0.90,
        "min60": 0.85,
        "total_points": 4.00,
        "expected_goals": 0.07,
        "expected_assists": 0.09,
        "clean_sheets": 0.35,
        "saves": 0.0,
        "defensive_contribution": 2.5,
        "bps": 16.0,
        "ict_index": 4.2,
        "goals_conceded": 0.9,
        "expected_goals_conceded": 0.9,
        "cards": 0.10,
        "own_goals": 0.005,
        "penalties_missed": 0.0,
        "penalties_saved": 0.0,
    },
    # FORWARDS
    ("FWD", "budget"): {
        "minutes": 20.0,
        "starts": 0.20,
        "min60": 0.15,
        "total_points": 1.00,
        "expected_goals": 0.08,
        "expected_assists": 0.02,
        "clean_sheets": 0.05,
        "saves": 0.0,
        "defensive_contribution": 0.4,
        "bps": 3.5,
        "ict_index": 1.5,
        "goals_conceded": 0.4,
        "expected_goals_conceded": 0.4,
        "cards": 0.08,
        "own_goals": 0.0,
        "penalties_missed": 0.005,
        "penalties_saved": 0.0,
    },
    ("FWD", "mid"): {
        "minutes": 55.0,
        "starts": 0.65,
        "min60": 0.55,
        "total_points": 2.50,
        "expected_goals": 0.25,
        "expected_assists": 0.05,
        "clean_sheets": 0.15,
        "saves": 0.0,
        "defensive_contribution": 0.9,
        "bps": 9.0,
        "ict_index": 3.5,
        "goals_conceded": 0.8,
        "expected_goals_conceded": 0.8,
        "cards": 0.10,
        "own_goals": 0.0,
        "penalties_missed": 0.01,
        "penalties_saved": 0.0,
    },
    ("FWD", "mid_high"): {
        "minutes": 70.0,
        "starts": 0.80,
        "min60": 0.70,
        "total_points": 3.50,
        "expected_goals": 0.35,
        "expected_assists": 0.07,
        "clean_sheets": 0.22,
        "saves": 0.0,
        "defensive_contribution": 1.2,
        "bps": 13.0,
        "ict_index": 5.2,
        "goals_conceded": 0.9,
        "expected_goals_conceded": 0.9,
        "cards": 0.10,
        "own_goals": 0.0,
        "penalties_missed": 0.01,
        "penalties_saved": 0.0,
    },
    ("FWD", "premium"): {
        "minutes": 82.0,
        "starts": 0.92,
        "min60": 0.85,
        "total_points": 5.00,
        "expected_goals": 0.55,
        "expected_assists": 0.08,
        "clean_sheets": 0.28,
        "saves": 0.0,
        "defensive_contribution": 1.1,
        "bps": 18.0,
        "ict_index": 7.0,
        "goals_conceded": 0.8,
        "expected_goals_conceded": 0.8,
        "cards": 0.08,
        "own_goals": 0.0,
        "penalties_missed": 0.015,
        "penalties_saved": 0.0,
    },
    # GOALKEEPERS
    ("GKP", "budget"): {
        "minutes": 20.0,
        "starts": 0.20,
        "min60": 0.20,
        "total_points": 0.80,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "clean_sheets": 0.05,
        "saves": 0.8,
        "defensive_contribution": 0.0,
        "bps": 3.5,
        "ict_index": 0.6,
        "goals_conceded": 0.5,
        "expected_goals_conceded": 0.5,
        "cards": 0.02,
        "own_goals": 0.0,
        "penalties_missed": 0.0,
        "penalties_saved": 0.005,
    },
    ("GKP", "mid"): {
        "minutes": 75.0,
        "starts": 0.85,
        "min60": 0.85,
        "total_points": 3.00,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "clean_sheets": 0.22,
        "saves": 2.8,
        "defensive_contribution": 0.0,
        "bps": 13.0,
        "ict_index": 2.0,
        "goals_conceded": 1.3,
        "expected_goals_conceded": 1.3,
        "cards": 0.03,
        "own_goals": 0.0,
        "penalties_missed": 0.0,
        "penalties_saved": 0.01,
    },
    ("GKP", "premium"): {
        "minutes": 88.0,
        "starts": 0.98,
        "min60": 0.98,
        "total_points": 4.00,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "clean_sheets": 0.38,
        "saves": 2.4,
        "defensive_contribution": 0.0,
        "bps": 16.0,
        "ict_index": 2.1,
        "goals_conceded": 1.0,
        "expected_goals_conceded": 1.0,
        "cards": 0.02,
        "own_goals": 0.0,
        "penalties_missed": 0.0,
        "penalties_saved": 0.01,
    },
    # MIDFIELDERS
    ("MID", "budget"): {
        "minutes": 25.0,
        "starts": 0.25,
        "min60": 0.20,
        "total_points": 1.10,
        "expected_goals": 0.04,
        "expected_assists": 0.03,
        "clean_sheets": 0.07,
        "saves": 0.0,
        "defensive_contribution": 0.8,
        "bps": 4.5,
        "ict_index": 1.4,
        "goals_conceded": 0.5,
        "expected_goals_conceded": 0.5,
        "cards": 0.10,
        "own_goals": 0.002,
        "penalties_missed": 0.002,
        "penalties_saved": 0.0,
    },
    ("MID", "mid"): {
        "minutes": 55.0,
        "starts": 0.65,
        "min60": 0.55,
        "total_points": 2.60,
        "expected_goals": 0.12,
        "expected_assists": 0.09,
        "clean_sheets": 0.18,
        "saves": 0.0,
        "defensive_contribution": 1.8,
        "bps": 10.5,
        "ict_index": 3.6,
        "goals_conceded": 0.9,
        "expected_goals_conceded": 0.9,
        "cards": 0.12,
        "own_goals": 0.002,
        "penalties_missed": 0.005,
        "penalties_saved": 0.0,
    },
    ("MID", "mid_high"): {
        "minutes": 72.0,
        "starts": 0.82,
        "min60": 0.75,
        "total_points": 3.80,
        "expected_goals": 0.22,
        "expected_assists": 0.15,
        "clean_sheets": 0.25,
        "saves": 0.0,
        "defensive_contribution": 2.0,
        "bps": 14.5,
        "ict_index": 5.8,
        "goals_conceded": 0.9,
        "expected_goals_conceded": 0.9,
        "cards": 0.11,
        "own_goals": 0.002,
        "penalties_missed": 0.008,
        "penalties_saved": 0.0,
    },
    ("MID", "premium"): {
        "minutes": 82.0,
        "starts": 0.92,
        "min60": 0.85,
        "total_points": 5.20,
        "expected_goals": 0.35,
        "expected_assists": 0.22,
        "clean_sheets": 0.30,
        "saves": 0.0,
        "defensive_contribution": 1.8,
        "bps": 19.5,
        "ict_index": 8.0,
        "goals_conceded": 0.8,
        "expected_goals_conceded": 0.8,
        "cards": 0.09,
        "own_goals": 0.002,
        "penalties_missed": 0.012,
        "penalties_saved": 0.0,
    },
}


def get_price_band(pos_code: str, val_m: float) -> str:
    """Classifies player cost in £m into standard empirical price tiers."""
    val = val_m / 10.0 if val_m > 20.0 else val_m
    pos = pos_code.upper()

    if pos == "GKP":
        if val >= 5.5:
            return "premium"
        if val >= 5.0:
            return "mid"
        return "budget"
    if pos == "DEF":
        if val >= 6.0:
            return "premium"
        if val >= 5.0:
            return "mid_high"
        if val >= 4.5:
            return "mid"
        return "budget"
    if pos == "MID":
        if val >= 9.5:
            return "premium"
        if val >= 7.0:
            return "mid_high"
        if val >= 5.5:
            return "mid"
        return "budget"
    # FWD
    if val >= 9.0:
        return "premium"
    if val >= 7.0:
        return "mid_high"
    if val >= 5.5:
        return "mid"
    return "budget"


def get_empirical_prior(
    pos_code: str,
    val_m: float = 6.0,
    team_strength: float = 1.0,
    is_promoted: bool = False,
    is_long_injury_return: bool = False,
) -> dict[str, float]:
    """
    Returns empirical prior expectations conditioned on position, price band,
    and team strength. Adjusts for promoted team and injury return flags.
    """
    pos = pos_code.upper()
    band = get_price_band(pos, val_m)
    base_prior = EMPIRICAL_PRIORS.get((pos, band), EMPIRICAL_PRIORS[("MID", "mid")]).copy()

    # Team strength scaling (neutral team strength = 1.0)
    # Attacking metrics scale with team strength, defensive conceded scales inversely
    eff_strength = team_strength * 0.85 if is_promoted else team_strength
    att_scale = float(np.clip(eff_strength, 0.70, 1.45))
    def_scale = float(np.clip(1.0 / max(0.1, eff_strength), 0.70, 1.45))

    adjusted = dict(base_prior)
    adjusted["expected_goals"] *= att_scale
    adjusted["expected_assists"] *= att_scale
    adjusted["total_points"] *= (att_scale + (1.0 / def_scale)) / 2.0
    adjusted["clean_sheets"] = float(np.clip(adjusted["clean_sheets"] / def_scale, 0.05, 0.60))
    adjusted["goals_conceded"] *= def_scale
    adjusted["expected_goals_conceded"] *= def_scale
    adjusted["ict_index"] *= att_scale
    adjusted["bps"] *= att_scale

    if is_long_injury_return:
        # Returning from long injury: start prob and minutes discounted towards sub baseline
        adjusted["minutes"] *= 0.70
        adjusted["starts"] *= 0.65
        adjusted["min60"] *= 0.60

    return adjusted


def shrink_player_rolling_stats(
    player_stats: dict[str, float],
    lookback_minutes: float,
    pos_code: str = "MID",
    val_m: float = 6.0,
    team_strength: float = 1.0,
    is_new_signing: bool = False,
    is_promoted: bool = False,
    is_long_injury_return: bool = False,
) -> dict[str, float]:
    """
    Applies empirical-Bayes shrinkage to a player's rolling feature dictionary:
        rate_shrunk = (n * rate_player + k * rate_prior) / (n + k)
    Guarantees that players with few minutes (< 450 mins in lookback) are appropriately
    shrunk towards position x price-band x team-strength priors.
    """
    prior = get_empirical_prior(
        pos_code=pos_code,
        val_m=val_m,
        team_strength=team_strength,
        is_promoted=is_promoted,
        is_long_injury_return=is_long_injury_return,
    )

    # For new signings with 0 minutes, or long injury return, lookback minutes are constrained
    n = 0.0 if is_new_signing else float(max(0.0, lookback_minutes))

    # If player already has abundant lookback minutes (>= 450 mins), return raw stats
    if n >= 450.0 and not is_long_injury_return and not is_new_signing:
        return dict(player_stats)

    shrunk: dict[str, float] = {}

    for key, player_val in player_stats.items():
        k = SHRINKAGE_K.get(key, 270.0)

        # Map feature key to corresponding prior stat
        if "minutes" in key:
            prior_val = prior["minutes"]
        elif "starts" in key:
            prior_val = prior["starts"]
        elif "min60" in key:
            prior_val = prior["min60"]
        elif "points" in key:
            prior_val = prior["total_points"]
        elif "xG" in key:
            prior_val = prior["expected_goals"]
        elif "xA" in key:
            prior_val = prior["expected_assists"]
        elif "xGI" in key:
            prior_val = prior["expected_goals"] + prior["expected_assists"]
        elif "goals" in key and "conceded" not in key and "own" not in key:
            prior_val = prior["expected_goals"]
        elif "assists" in key:
            prior_val = prior["expected_assists"]
        elif "clean_sheets" in key:
            prior_val = prior["clean_sheets"]
        elif "saves" in key and "penalties" not in key:
            prior_val = prior["saves"]
        elif "defcon" in key:
            prior_val = prior["defensive_contribution"]
        elif "bps" in key:
            prior_val = prior["bps"]
        elif "ict" in key:
            prior_val = prior["ict_index"]
        elif "goals_conceded" in key:
            prior_val = prior["goals_conceded"]
        elif "xGC" in key:
            prior_val = prior["expected_goals_conceded"]
        elif "cards" in key:
            prior_val = prior["cards"]
        elif "own_goals" in key:
            prior_val = prior["own_goals"]
        elif "penalties_missed" in key:
            prior_val = prior["penalties_missed"]
        elif "penalties_saved" in key:
            prior_val = prior["penalties_saved"]
        else:
            prior_val = player_val

        # Empirical-Bayes shrinkage formula:
        if n <= 0.0:
            shrunk[key] = float(prior_val)
        else:
            shrunk[key] = float((n * player_val + k * prior_val) / (n + k))

    return shrunk


def shrink_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized empirical-Bayes shrinkage across all rolling feature columns
    for players with few minutes (< 450 minutes in lookback).
    Preserves dataframe shape, index, and non-rolling columns.
    """
    df_out = df.copy()
    if df_out.empty:
        return df_out

    # Identify positions
    if "pos_FWD" in df_out.columns:
        pos = np.where(
            df_out["pos_FWD"] == 1.0,
            "FWD",
            np.where(
                df_out["pos_MID"] == 1.0,
                "MID",
                np.where(df_out["pos_DEF"] == 1.0, "DEF", "GKP"),
            ),
        )
    elif "position" in df_out.columns:
        pos = df_out["position"].astype(str).values
    else:
        pos = np.full(len(df_out), "MID")

    val_col = df_out["value"].values if "value" in df_out.columns else np.full(len(df_out), 60.0)
    val_m = np.where(val_col > 20.0, val_col / 10.0, val_col)

    team_str = (
        df_out["team_strength_attack"].values
        if "team_strength_attack" in df_out.columns
        else np.full(len(df_out), 1.0)
    )

    n_5 = df_out["roll_minutes_5"].values * 5.0 if "roll_minutes_5" in df_out.columns else np.full(len(df_out), 0.0)
    n_3 = df_out["roll_minutes_3"].values * 3.0 if "roll_minutes_3" in df_out.columns else n_5
    n_8 = df_out["roll_minutes_8"].values * 8.0 if "roll_minutes_8" in df_out.columns else n_5

    bands = np.array([get_price_band(p, v) for p, v in zip(pos, val_m, strict=False)])

    for (p_name, b_name), prior_dict in EMPIRICAL_PRIORS.items():
        mask = (pos == p_name) & (bands == b_name)
        if not np.any(mask):
            continue

        att_scale = np.clip(team_str[mask], 0.70, 1.45)
        def_scale = np.clip(1.0 / np.maximum(0.1, team_str[mask]), 0.70, 1.45)

        stat_priors = {
            "minutes": prior_dict["minutes"],
            "starts": prior_dict["starts"],
            "min60": prior_dict["min60"],
            "points": prior_dict["total_points"] * (att_scale + (1.0 / def_scale)) / 2.0,
            "xG": prior_dict["expected_goals"] * att_scale,
            "xA": prior_dict["expected_assists"] * att_scale,
            "xGI": (prior_dict["expected_goals"] + prior_dict["expected_assists"]) * att_scale,
            "goals": prior_dict["expected_goals"] * att_scale,
            "assists": prior_dict["expected_assists"] * att_scale,
            "clean_sheets": np.clip(prior_dict["clean_sheets"] / def_scale, 0.05, 0.60),
            "saves": prior_dict["saves"],
            "defcon": prior_dict["defensive_contribution"],
            "bps": prior_dict["bps"] * att_scale,
            "ict": prior_dict["ict_index"] * att_scale,
            "goals_conceded": prior_dict["goals_conceded"] * def_scale,
            "xGC": prior_dict["expected_goals_conceded"] * def_scale,
            "cards": prior_dict["cards"],
            "own_goals": prior_dict["own_goals"],
            "penalties_missed": prior_dict["penalties_missed"],
            "penalties_saved": prior_dict["penalties_saved"],
        }

        for col in df_out.columns:
            if not col.startswith("roll_") and col != "std_minutes_per_gw":
                continue
            k = SHRINKAGE_K.get(col, 270.0)
            n = n_3[mask] if "_3" in col else (n_8[mask] if "_8" in col else n_5[mask])

            stat_key = "minutes"
            for sk in [
                "min60",
                "starts",
                "minutes",
                "points",
                "xGI",
                "xGC",
                "xG",
                "xA",
                "clean_sheets",
                "saves",
                "defcon",
                "bps",
                "ict",
                "cards",
                "own_goals",
                "penalties_missed",
                "penalties_saved",
                "goals_conceded",
                "goals",
                "assists",
            ]:
                if sk.lower() in col.lower():
                    stat_key = sk
                    break

            p_val = stat_priors[stat_key]
            raw_val = df_out.loc[mask, col].values
            shrunk_val = np.where(n < 450.0, (n * raw_val + k * p_val) / (n + k), raw_val)
            df_out.loc[mask, col] = shrunk_val

    return df_out

