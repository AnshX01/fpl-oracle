"""
Unit tests for Feature 4: Midweek European & Domestic Cup Rotation Drag.
Tests minutes and expected points discounting for outfield players with quick turnarounds (<72h),
exemption of goalkeepers, and automated lineup rotation into fresh starters.
Zero hardcoded player names in production code.
"""

import re

import pandas as pd

from fpl_oracle.optimise.lineup import LineupOptimizer
from fpl_oracle.optimise.turnaround import TurnaroundDragService


def test_gkp_exempt_from_turnaround_drag():
    """Goalkeepers are not subject to fatigue rotation drag even with extreme turnaround (<48h)."""
    service = TurnaroundDragService(extreme_discount=0.80)
    df = pd.DataFrame([
        {
            "element": 1,
            "position": "GKP",
            "expected_minutes": 90.0,
            "expected_points": 4.5,
            "turnaround_hours": 42.0,
            "midweek_minutes": 90.0,
        }
    ])
    res = service.apply_turnaround_drag(df)

    assert res.loc[0, "expected_minutes"] == 90.0
    assert res.loc[0, "expected_points"] == 4.5
    assert not bool(res.loc[0, "midweek_fatigue_risk"])


def test_outfield_extreme_crunch_discount():
    """Outfielder with 90 midweek mins and 42h turnaround receives extreme discount (0.80)."""
    service = TurnaroundDragService(extreme_discount=0.80)
    df = pd.DataFrame([
        {
            "element": 10,
            "position": "MID",
            "expected_minutes": 85.0,
            "expected_points": 6.0,
            "turnaround_hours": 42.0,
            "midweek_minutes": 90.0,
        }
    ])
    res = service.apply_turnaround_drag(df)

    assert res.loc[0, "expected_minutes"] == 68.0  # 85 * 0.80 = 68.0
    assert res.loc[0, "expected_points"] == 4.8  # 6.0 * (68/85) = 4.8
    assert bool(res.loc[0, "midweek_fatigue_risk"])


def test_outfield_sub_minutes_exempt():
    """Outfielder playing < 45 mins in midweek does not incur fatigue drag."""
    service = TurnaroundDragService()
    df = pd.DataFrame([
        {
            "element": 11,
            "position": "FWD",
            "expected_minutes": 80.0,
            "expected_points": 5.5,
            "turnaround_hours": 42.0,
            "midweek_minutes": 20.0,
        }
    ])
    res = service.apply_turnaround_drag(df)

    assert res.loc[0, "expected_minutes"] == 80.0
    assert res.loc[0, "expected_points"] == 5.5
    assert not bool(res.loc[0, "midweek_fatigue_risk"])


def test_lineup_optimizer_reacts_to_midweek_fatigue():
    """Lineup optimizer benches fatigued asset in favor of a fresh starter."""
    # Build 15-player squad in 3-4-3 formation where only 4 MIDs can start:
    # Top 3 MIDs: elements 8, 9, 10 (7.0, 6.5, 6.0 xP)
    # 4th MID spot contested between:
    # - Element 11 (MID_Tired): 5.0 xP, 85 mins, but played 90m UCL with 40h turnaround -> drops to 4.0 xP
    # - Element 12 (MID_Fresh): 4.8 xP, 90 mins, fresh (no midweek)
    rows = [
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "expected_points": 4.5, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 2, "web_name": "GKP_2", "position": "GKP", "expected_points": 1.0, "chance_of_playing": 100.0, "expected_minutes": 0.0},
        # 5 DEFs (top 3 start: 5.0, 4.5, 4.0)
        {"element": 3, "web_name": "DEF_1", "position": "DEF", "expected_points": 5.0, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 4, "web_name": "DEF_2", "position": "DEF", "expected_points": 4.5, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 5, "web_name": "DEF_3", "position": "DEF", "expected_points": 4.0, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 6, "web_name": "DEF_4", "position": "DEF", "expected_points": 2.0, "chance_of_playing": 100.0, "expected_minutes": 10.0},
        {"element": 7, "web_name": "DEF_5", "position": "DEF", "expected_points": 1.5, "chance_of_playing": 100.0, "expected_minutes": 0.0},
        # 5 MIDs
        {"element": 8, "web_name": "MID_1", "position": "MID", "expected_points": 7.0, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 9, "web_name": "MID_2", "position": "MID", "expected_points": 6.5, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 10, "web_name": "MID_3", "position": "MID", "expected_points": 6.0, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 11, "web_name": "MID_Tired", "position": "MID", "expected_points": 5.0, "chance_of_playing": 100.0, "expected_minutes": 85.0},
        {"element": 12, "web_name": "MID_Fresh", "position": "MID", "expected_points": 4.8, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        # 3 FWDs (all 3 start with high xP)
        {"element": 13, "web_name": "FWD_1", "position": "FWD", "expected_points": 7.5, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 14, "web_name": "FWD_2", "position": "FWD", "expected_points": 7.0, "chance_of_playing": 100.0, "expected_minutes": 90.0},
        {"element": 15, "web_name": "FWD_3", "position": "FWD", "expected_points": 6.5, "chance_of_playing": 100.0, "expected_minutes": 90.0},
    ]
    df = pd.DataFrame(rows)

    opt = LineupOptimizer()
    # Without fatigue, element 11 starts over element 12 (5.0 xP vs 4.8 xP in 3-4-3)
    res_normal = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    normal_starters = [int(p["element"]) for p in res_normal["starters"].to_dict(orient="records")]
    assert 11 in normal_starters
    assert 12 not in normal_starters

    # With midweek fatigue data on element 11 (40h turnaround after 90m UCL tie):
    midweek_info = {11: {"turnaround_hours": 40.0, "midweek_minutes": 90.0}}
    res_fatigued = opt.select_lineup_and_captain(df, midweek_data=midweek_info, simulate_captaincy=False)
    fatigued_starters = [int(p["element"]) for p in res_fatigued["starters"].to_dict(orient="records")]

    # Fresh element 12 now starts; fatigued element 11 is rotated to bench
    assert 12 in fatigued_starters
    assert 11 not in fatigued_starters


def test_no_hardcoded_player_names_in_turnaround_module():
    """Enforces zero hardcoded player names in src/fpl_oracle/optimise/turnaround.py."""
    with open("src/fpl_oracle/optimise/turnaround.py", encoding="utf-8") as f:
        content = f.read().lower()

    for forbidden in ["haaland", "salah", "palmer", "saka", "watkins", "gabriel", "son"]:
        assert not re.search(
            rf"\b{forbidden}\b", content
        ), f"Forbidden hardcoded player name '{forbidden}' found in turnaround.py"
