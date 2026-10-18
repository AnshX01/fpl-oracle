"""
Contract tests for centralized FPL scoring rules module (WI-1.1).
Verifies:
1. The scoring constants match the official 2026/27 rules.
2. No raw 10 * GKP goal points or hardcoded goal-rate dictionaries reappear outside rules/scoring.py.
"""

from pathlib import Path

from fpl_oracle.rules.scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    DEFCON_THRESHOLDS,
    GOAL_POINTS,
    HIT_COST_POINTS,
    MAX_BANKED_FREE_TRANSFERS,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    SET_1_LAST_GW,
)


def test_scoring_constants_values():
    assert GOAL_POINTS["GKP"] == 10  # live bootstrap-static game_config.scoring.goals_scored.GKP
    assert GOAL_POINTS["DEF"] == 6
    assert GOAL_POINTS["MID"] == 5
    assert GOAL_POINTS["FWD"] == 4
    assert ASSIST_POINTS == 3
    assert CLEAN_SHEET_POINTS["GKP"] == 4
    assert CLEAN_SHEET_POINTS["DEF"] == 4
    assert CLEAN_SHEET_POINTS["MID"] == 1
    assert CLEAN_SHEET_POINTS["FWD"] == 0
    assert PENALTY_SAVE_POINTS == 5
    assert PENALTY_MISS_POINTS == -2
    assert OWN_GOAL_POINTS == -2
    assert DEFCON_POINTS == 2
    assert DEFCON_THRESHOLDS["DEF"] == 10
    assert DEFCON_THRESHOLDS["MID"] == 12
    assert DEFCON_THRESHOLDS["FWD"] == 12
    assert DEFCON_THRESHOLDS["GKP"] is None
    assert MAX_BANKED_FREE_TRANSFERS == 5
    assert HIT_COST_POINTS == 4
    assert SET_1_LAST_GW == 19


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
            if "pos_gkp * 10" in line or 'pos == "GKP": pts += goals_scored * 10' in line:
                offenders.append(f"{py_file}:{idx}: {line}")
            # Check for hardcoded goal points dictionaries
            if '{"GKP": 10' in line or '{"GKP": 10.0' in line:
                offenders.append(f"{py_file}:{idx}: {line}")

    assert not offenders, "Found stray GKP 10 / goal dictionary definitions outside rules module:\n" + "\n".join(
        offenders
    )
