import pandas as pd
import pytest

from fpl_oracle.optimise.lineup import LineupOptimizer


@pytest.fixture
def risk_test_squad():
    """
    Construct a 15-player squad where:
    - 1 GKP starter, 1 GKP bench
    - 3 DEF starters (locked)
    - 2 MID starters (locked)
    - 1 FWD starter (locked)
    - Total locked starters: 1 + 3 + 2 + 1 = 7 players.
    - 3 flex spots filled by high xP players (8.0 xP).
    - Total locked + flex = 10 starters.
    - Exactly 1 slot remains contested between:
      1) Player 11 (MID, Doubtful high-variance star):
         expected_points = 5.0, chance_of_playing = 75.0, p10 = 2.0, p90 = 10.0
      2) Player 12 (DEF, Secure starter):
         expected_points = 4.8, chance_of_playing = 100.0, p10 = 3.8, p90 = 5.8
    - Players 13, 14, 15 have very low xP (1.0 xP) and are clearly on the bench.
    """
    rows = [
        # GKPs (2)
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "expected_points": 5.0, "p10": 3.0, "p90": 7.0, "chance_of_playing": 100.0},
        {"element": 2, "web_name": "GKP_2", "position": "GKP", "expected_points": 1.0, "p10": 0.0, "p90": 2.0, "chance_of_playing": 100.0},

        # DEFs (5)
        {"element": 3, "web_name": "DEF_1", "position": "DEF", "expected_points": 6.0, "p10": 4.0, "p90": 8.0, "chance_of_playing": 100.0},
        {"element": 4, "web_name": "DEF_2", "position": "DEF", "expected_points": 6.0, "p10": 4.0, "p90": 8.0, "chance_of_playing": 100.0},
        {"element": 5, "web_name": "DEF_3", "position": "DEF", "expected_points": 6.0, "p10": 4.0, "p90": 8.0, "chance_of_playing": 100.0},
        {"element": 6, "web_name": "DEF_SECURE", "position": "DEF", "expected_points": 4.8, "p10": 3.8, "p90": 5.8, "chance_of_playing": 100.0},
        {"element": 7, "web_name": "DEF_BENCH", "position": "DEF", "expected_points": 1.0, "p10": 0.0, "p90": 2.0, "chance_of_playing": 100.0},

        # MIDs (5)
        {"element": 8, "web_name": "MID_1", "position": "MID", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
        {"element": 9, "web_name": "MID_2", "position": "MID", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
        {"element": 10, "web_name": "MID_3", "position": "MID", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
        {"element": 11, "web_name": "MID_DOUBTFUL", "position": "MID", "expected_points": 5.0, "p10": 2.0, "p90": 10.0, "chance_of_playing": 75.0},
        {"element": 12, "web_name": "MID_BENCH", "position": "MID", "expected_points": 1.0, "p10": 0.0, "p90": 2.0, "chance_of_playing": 100.0},

        # FWDs (3)
        {"element": 13, "web_name": "FWD_1", "position": "FWD", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
        {"element": 14, "web_name": "FWD_2", "position": "FWD", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
        {"element": 15, "web_name": "FWD_3", "position": "FWD", "expected_points": 7.0, "p10": 5.0, "p90": 9.0, "chance_of_playing": 100.0},
    ]
    return pd.DataFrame(rows)


def test_flag_off_preserves_pure_expected_points(risk_test_squad):
    """
    F-011: When risk_aware_lineup flag is False (default), all risk modes (conservative,
    balanced, aggressive) use pure raw expected points for starter selection.
    """
    optimizer = LineupOptimizer()

    for mode in ["balanced", "conservative", "aggressive"]:
        res = optimizer.select_lineup_and_captain(risk_test_squad, risk_preference=mode, risk_aware_lineup=False)
        starters = set(res["starters"]["element"])
        assert 11 in starters, f"With flag OFF, MID_DOUBTFUL (5.0 xP) must start over DEF_SECURE (4.8 xP) in {mode} mode"
        assert 6 not in starters, f"With flag OFF, DEF_SECURE must be benched in {mode} mode"


def test_conservative_mode_benches_doubtful_starter_while_aggressive_starts(risk_test_squad):
    """
    F-011: When risk_aware_lineup is enabled, a 75%-flagged high-variance player (MID_DOUBTFUL) must be
    benched in favor of a secure starter (DEF_SECURE) in conservative mode. In aggressive mode, the high-upside
    MID_DOUBTFUL must be started. Balanced mode must remain identical to raw expected points.
    """
    optimizer = LineupOptimizer()

    # 1. Balanced mode: raw expected points dictates that MID_DOUBTFUL (5.0) beats DEF_SECURE (4.8)
    res_balanced = optimizer.select_lineup_and_captain(
        risk_test_squad, risk_preference="balanced", risk_aware_lineup=True
    )
    balanced_starters = set(res_balanced["starters"]["element"])
    assert 11 in balanced_starters, "MID_DOUBTFUL should start in balanced mode (5.0 > 4.8)"
    assert 6 not in balanced_starters, "DEF_SECURE should be benched in balanced mode"

    # 2. Aggressive mode: MID_DOUBTFUL has high ceiling (p90=10.0) -> must start
    res_aggressive = optimizer.select_lineup_and_captain(
        risk_test_squad, risk_preference="aggressive", risk_aware_lineup=True
    )
    aggressive_starters = set(res_aggressive["starters"]["element"])
    assert 11 in aggressive_starters, "MID_DOUBTFUL should start in aggressive mode (seeking upside)"

    # 3. Conservative mode: MID_DOUBTFUL carries 75% flag and wide spread -> must be benched for DEF_SECURE
    res_conservative = optimizer.select_lineup_and_captain(
        risk_test_squad, risk_preference="conservative", risk_aware_lineup=True
    )
    conservative_starters = set(res_conservative["starters"]["element"])
    assert 11 not in conservative_starters, "MID_DOUBTFUL must be BENCHED in conservative mode!"
    assert 6 in conservative_starters, "DEF_SECURE must be STARTED in conservative mode!"
