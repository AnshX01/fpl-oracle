"""
Unit tests for Feature 3: Cameo Risk Penalty & Smart 1st Sub Guarantee (Hall of Fame lineup heuristic).
"""

import pandas as pd

from fpl_oracle.optimise.lineup import LineupOptimizer


def make_squad(mid_a_cop: float = 100.0, mid_a_mins: float = 90.0):
    rows = [
        # GKP (2)
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "expected_points": 4.5, "chance_of_playing": 100.0, "team": 1},
        {"element": 2, "web_name": "GKP_2", "position": "GKP", "expected_points": 1.5, "chance_of_playing": 100.0, "team": 2},
        # DEF (5)
        {"element": 3, "web_name": "DEF_1", "position": "DEF", "expected_points": 5.0, "chance_of_playing": 100.0, "team": 3},
        {"element": 4, "web_name": "DEF_2", "position": "DEF", "expected_points": 5.0, "chance_of_playing": 100.0, "team": 4},
        {"element": 5, "web_name": "DEF_3", "position": "DEF", "expected_points": 5.0, "chance_of_playing": 100.0, "team": 5},
        {"element": 6, "web_name": "DEF_4", "position": "DEF", "expected_points": 2.0, "chance_of_playing": 100.0, "team": 6},
        {"element": 7, "web_name": "DEF_5", "position": "DEF", "expected_points": 1.5, "chance_of_playing": 100.0, "team": 7},
        # MID (5)
        # MID_8: 6.0 xP, nailed
        {"element": 8, "web_name": "MID_1", "position": "MID", "expected_points": 6.0, "chance_of_playing": 100.0, "team": 8},
        {"element": 9, "web_name": "MID_2", "position": "MID", "expected_points": 6.0, "chance_of_playing": 100.0, "team": 9},
        {"element": 10, "web_name": "MID_3", "position": "MID", "expected_points": 6.0, "chance_of_playing": 100.0, "team": 10},
        # MID_11: 4.5 xP, doubtful/cameo risk
        {
            "element": 11,
            "web_name": "MID_CAMEO",
            "position": "MID",
            "expected_points": 4.5,
            "chance_of_playing": mid_a_cop,
            "expected_minutes": mid_a_mins,
            "team": 11,
        },
        # MID_12: 4.3 xP, 100% fit and nailed
        {
            "element": 12,
            "web_name": "MID_FIT",
            "position": "MID",
            "expected_points": 4.3,
            "chance_of_playing": 100.0,
            "expected_minutes": 90.0,
            "team": 12,
        },
        # FWD (3)
        {"element": 13, "web_name": "FWD_1", "position": "FWD", "expected_points": 7.0, "chance_of_playing": 100.0, "team": 13},
        {"element": 14, "web_name": "FWD_2", "position": "FWD", "expected_points": 7.0, "chance_of_playing": 100.0, "team": 14},
        {"element": 15, "web_name": "FWD_3", "position": "FWD", "expected_points": 7.0, "chance_of_playing": 100.0, "team": 15},
    ]
    return pd.DataFrame(rows)


def test_cameo_risk_penalty_benches_doubtful_player():
    """
    Test that a 50% flagged player with slightly higher xP (4.5 vs 4.3)
    is benched under cameo risk penalty to prevent 1-point cameo blocking.
    """
    # Without penalty (cameo_risk_penalty=0.0): MID_CAMEO (4.5) starts over MID_FIT (4.3)
    df = make_squad(mid_a_cop=50.0)
    opt = LineupOptimizer(cameo_risk_penalty=0.0)
    res_no_pen = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    starters_no_pen = set(res_no_pen["starters"]["element"])
    assert 11 in starters_no_pen
    assert 12 not in starters_no_pen

    # With cameo risk penalty (cameo_risk_penalty=0.75):
    # MID_CAMEO effective score is 4.5 - (0.5 * 0.75) = 4.125 < 4.3.
    # MID_FIT starts instead!
    opt_pen = LineupOptimizer(cameo_risk_penalty=0.75)
    res_pen = opt_pen.select_lineup_and_captain(df, simulate_captaincy=False)
    starters_pen = set(res_pen["starters"]["element"])
    assert 12 in starters_pen
    assert 11 not in starters_pen


def test_smart_first_sub_guarantees_fit_asset():
    """
    Test that when ordering the bench, a 100% fit player with guaranteed minutes
    is selected ahead of an uncertain/rotation risk bench player even if the latter
    has higher raw xP.
    """
    df = make_squad(mid_a_cop=100.0)
    # Give FWD_3 low xP so formation is 3-5-2 with all 5 MIDs starting
    df.loc[df["element"] == 15, "expected_points"] = 1.0
    # Modify DEF_4 (elem 6) to have 70% chance of playing with 3.2 xP
    # and DEF_5 (elem 7) to have 100% chance of playing with 3.0 xP
    df.loc[df["element"] == 6, "expected_points"] = 3.2
    df.loc[df["element"] == 6, "chance_of_playing"] = 70.0
    df.loc[df["element"] == 7, "expected_points"] = 3.0
    df.loc[df["element"] == 7, "chance_of_playing"] = 100.0

    opt = LineupOptimizer()
    res = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    bench_elements = list(res["bench"]["element"])
    # GKP is at index 0. Outfield subs are index 1, 2, 3.
    # Sub 1 (index 1) should be DEF_5 (element 7) due to 100% appearance certainty
    assert bench_elements[0] == 2  # backup GKP
    assert bench_elements[1] == 7  # 100% fit DEF_5 (3.0 xP)
    assert bench_elements[2] == 6  # 70% fit DEF_4 (3.2 xP)
    assert bench_elements[3] == 15  # FWD_3 (1.0 xP)


def test_low_minutes_cameo_penalty():
    """
    Test that a player with low expected minutes (e.g. 25 min late cameo)
    incurs additional minutes cameo penalty.
    """
    # 100% chance of playing, but only 25 expected minutes
    df = make_squad(mid_a_cop=100.0, mid_a_mins=25.0)
    opt = LineupOptimizer(cameo_risk_penalty=0.75)
    res = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    starters = set(res["starters"]["element"])
    # 25 min player should be benched in favor of 90 min fit player
    assert 12 in starters
    assert 11 not in starters
