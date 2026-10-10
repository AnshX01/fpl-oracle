"""
Contract tests for centralized FPL scoring rules module (WI-1.1).
Verifies:
1. Recomputing total_points from raw event columns matches official FPL total_points exactly.
2. Across historical seasons, scoring logic achieves exact match rate.
3. No raw 10 * GKP goal points or hardcoded goal-rate dictionaries reappear outside rules/scoring.py.
"""

from pathlib import Path
import pandas as pd
import pytest

from fpl_oracle.rules.scoring import (
    APPEARANCE_GE60,
    APPEARANCE_LT60,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    DEFCON_THRESHOLDS,
    GOAL_POINTS,
    GOALS_CONCEDED_PER_MINUS1,
    HIT_COST_POINTS,
    MAX_BANKED_FREE_TRANSFERS,
    MAX_CHIPS_PER_GW,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    RED_CARD_POINTS,
    SAVES_PER_PLUS1,
    SET_1_LAST_GW,
    YELLOW_CARD_POINTS,
    calculate_match_points,
)


def test_scoring_constants_values():
    assert GOAL_POINTS["GKP"] == 6  # Fix D1: Official rule is 6, not 10
    assert GOAL_POINTS["DEF"] == 6
    assert GOAL_POINTS["MID"] == 5
    assert GOAL_POINTS["FWD"] == 4
    assert ASSIST_POINTS == 3
    assert CLEAN_SHEET_POINTS["GKP"] == 4
    assert CLEAN_SHEET_POINTS["DEF"] == 4
    assert CLEAN_SHEET_POINTS["MID"] == 1
    assert CLEAN_SHEET_POINTS["FWD"] == 0
    assert SAVES_PER_PLUS1 == 3
    assert PENALTY_SAVE_POINTS == 5
    assert PENALTY_MISS_POINTS == -2
    assert OWN_GOAL_POINTS == -2
    assert YELLOW_CARD_POINTS == -1
    assert RED_CARD_POINTS == -3
    assert DEFCON_POINTS == 2
    assert DEFCON_THRESHOLDS["DEF"] == 10
    assert DEFCON_THRESHOLDS["MID"] == 12
    assert DEFCON_THRESHOLDS["FWD"] == 12
    assert DEFCON_THRESHOLDS["GKP"] is None
    assert MAX_BANKED_FREE_TRANSFERS == 5
    assert HIT_COST_POINTS == 4
    assert SET_1_LAST_GW == 19


def test_contract_diverse_sample_from_master_history():
    """Verify points calculation on a diverse sample covering all positions and events."""
    csv_path = Path("data/historical/master_history.csv")
    if not csv_path.exists():
        pytest.skip("master_history.csv not found")

    df = pd.read_csv(csv_path)

    # Sample rows covering different events
    gkp_rows = df[(df["position"] == "GKP") & (df["minutes"] > 0) & (df["saves"] >= 3)].head(10)
    def_cs_rows = df[(df["position"] == "DEF") & (df["clean_sheets"] == 1) & (df["minutes"] >= 60)].head(10)
    def_goal_rows = df[(df["position"] == "DEF") & (df["goals_scored"] >= 1)].head(10)
    mid_goal_rows = df[(df["position"] == "MID") & (df["goals_scored"] >= 1) & (df["bonus"] >= 2)].head(10)
    fwd_goal_rows = df[(df["position"] == "FWD") & (df["goals_scored"] >= 1) & (df["assists"] >= 1)].head(10)
    disc_rows = df[(df["yellow_cards"] >= 1) | (df["red_cards"] >= 1) | (df["own_goals"] >= 1)].head(10)
    defcon_rows = df[(df["season"].isin(["2025-26", "2026-27"])) & (df["defensive_contribution"] >= 10)].head(15)

    sample = pd.concat([gkp_rows, def_cs_rows, def_goal_rows, mid_goal_rows, fwd_goal_rows, disc_rows, defcon_rows]).drop_duplicates()
    assert len(sample) >= 50

    for _, row in sample.iterrows():
        recalc = calculate_match_points(
            position=row["position"],
            minutes=int(row["minutes"]),
            goals_scored=int(row["goals_scored"]),
            assists=int(row["assists"]),
            clean_sheet=bool(row["clean_sheets"] > 0),
            goals_conceded=int(row["goals_conceded"]),
            saves=int(row["saves"]),
            penalties_saved=int(row["penalties_saved"]),
            penalties_missed=int(row["penalties_missed"]),
            yellow_cards=int(row["yellow_cards"]),
            red_cards=int(row["red_cards"]),
            own_goals=int(row["own_goals"]),
            defensive_contribution_actions=float(row.get("defensive_contribution", 0)),
            bonus=int(row["bonus"]),
            season=str(row["season"]),
        )
        assert recalc == int(row["total_points"]), (
            f"Mismatch for {row['name']} ({row['position']}) season {row['season']} round {row['round']}: "
            f"expected {row['total_points']}, got {recalc}"
        )


def test_contract_all_stored_2026_27_rows():
    """Verify 100% exact match across all 2026/27 rows in master_history."""
    csv_path = Path("data/historical/master_history.csv")
    if not csv_path.exists():
        pytest.skip("master_history.csv not found")

    df = pd.read_csv(csv_path)
    s26 = df[df["season"] == "2026-27"]
    if len(s26) == 0:
        pytest.skip("No 2026-27 data found")

    mismatches = 0
    for _, row in s26.iterrows():
        recalc = calculate_match_points(
            position=row["position"],
            minutes=int(row["minutes"]),
            goals_scored=int(row["goals_scored"]),
            assists=int(row["assists"]),
            clean_sheet=bool(row["clean_sheets"] > 0),
            goals_conceded=int(row["goals_conceded"]),
            saves=int(row["saves"]),
            penalties_saved=int(row["penalties_saved"]),
            penalties_missed=int(row["penalties_missed"]),
            yellow_cards=int(row["yellow_cards"]),
            red_cards=int(row["red_cards"]),
            own_goals=int(row["own_goals"]),
            defensive_contribution_actions=float(row.get("defensive_contribution", 0)),
            bonus=int(row["bonus"]),
            season="2026-27",
        )
        if recalc != int(row["total_points"]):
            mismatches += 1

    assert mismatches == 0, f"Found {mismatches} mismatches out of {len(s26)} rows in 2026-27"


def test_grep_no_stray_10_multiplier_or_goal_dictionaries():
    """Ensure no other module redefines GKP goal as 10 or hardcodes goal point dictionaries."""
    src_dir = Path("src/fpl_oracle")
    offenders = []

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "scoring.py" and py_file.parent.name == "rules":
            continue
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            # Check for GKP 10.0 or 10 * goals_scored
            if "pos_gkp * 10" in line or "pos == \"GKP\": pts += goals_scored * 10" in line:
                offenders.append(f"{py_file}:{idx}: {line}")
            # Check for hardcoded goal points dictionaries
            if '{"GKP": 10' in line or '{"GKP": 10.0' in line:
                offenders.append(f"{py_file}:{idx}: {line}")

    assert not offenders, f"Found stray GKP 10 / goal dictionary definitions outside rules module:\n" + "\n".join(offenders)
