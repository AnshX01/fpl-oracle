"""
Unit tests for Feature 1: Top-10k Effective Ownership (EO) & Armband Downside Shield.
Tests downside shielding of consensus captains against marginal differential noise,
hurdle clearance, and risk-mode adaptability.
Zero hardcoded player names in production code.
"""

import pandas as pd

from fpl_oracle.optimise.lineup import LineupOptimizer


def _make_squad(
    cand1_eo: float = 160.0,
    cand1_xp: float = 6.0,
    cand2_eo: float = 20.0,
    cand2_xp: float = 6.3,
) -> pd.DataFrame:
    """Helper creating a 15-player squad with 2 competing captain candidates."""
    rows = [
        # 2 GKPs
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "expected_points": 4.5, "chance_of_playing": 100.0, "top10k_eo": 10.0},
        {"element": 2, "web_name": "GKP_2", "position": "GKP", "expected_points": 1.0, "chance_of_playing": 100.0, "top10k_eo": 2.0},
        # 5 DEFs
        {"element": 3, "web_name": "DEF_1", "position": "DEF", "expected_points": 4.5, "chance_of_playing": 100.0, "top10k_eo": 25.0},
        {"element": 4, "web_name": "DEF_2", "position": "DEF", "expected_points": 4.0, "chance_of_playing": 100.0, "top10k_eo": 15.0},
        {"element": 5, "web_name": "DEF_3", "position": "DEF", "expected_points": 4.0, "chance_of_playing": 100.0, "top10k_eo": 10.0},
        {"element": 6, "web_name": "DEF_4", "position": "DEF", "expected_points": 2.0, "chance_of_playing": 100.0, "top10k_eo": 5.0},
        {"element": 7, "web_name": "DEF_5", "position": "DEF", "expected_points": 1.5, "chance_of_playing": 100.0, "top10k_eo": 2.0},
        # 5 MIDs
        {"element": 8, "web_name": "MID_1", "position": "MID", "expected_points": 4.5, "chance_of_playing": 100.0, "top10k_eo": 30.0},
        {"element": 9, "web_name": "MID_2", "position": "MID", "expected_points": 4.0, "chance_of_playing": 100.0, "top10k_eo": 15.0},
        {"element": 10, "web_name": "MID_Consensus", "position": "MID", "expected_points": cand1_xp, "chance_of_playing": 100.0, "top10k_eo": cand1_eo},
        {"element": 11, "web_name": "MID_Differential", "position": "MID", "expected_points": cand2_xp, "chance_of_playing": 100.0, "top10k_eo": cand2_eo},
        {"element": 12, "web_name": "MID_Bench", "position": "MID", "expected_points": 1.5, "chance_of_playing": 100.0, "top10k_eo": 2.0},
        # 3 FWDs
        {"element": 13, "web_name": "FWD_1", "position": "FWD", "expected_points": 4.0, "chance_of_playing": 100.0, "top10k_eo": 20.0},
        {"element": 14, "web_name": "FWD_2", "position": "FWD", "expected_points": 3.0, "chance_of_playing": 100.0, "top10k_eo": 10.0},
        {"element": 15, "web_name": "FWD_3", "position": "FWD", "expected_points": 1.5, "chance_of_playing": 100.0, "top10k_eo": 1.0},
    ]
    df = pd.DataFrame(rows)
    df["p10"] = df["expected_points"] * 0.5
    df["p90"] = df["expected_points"] * 1.6
    return df


def test_consensus_captain_shielded_against_marginal_differential():
    """Consensus candidate (element 10, 160% EO, 6.0 xP) beats marginal differential (6.3 xP, +0.3 delta < 1.2 hurdle)."""
    squad = _make_squad(cand1_eo=160.0, cand1_xp=6.0, cand2_eo=20.0, cand2_xp=6.3)
    opt = LineupOptimizer(eo_hedge_weight=0.80, differential_captain_hurdle=1.20)
    res = opt.select_lineup_and_captain(squad, risk_preference="balanced", simulate_captaincy=False)

    assert res["captain"]["element"] == 10
    assert res["captain_eo_shield"]["shield_applied"] is True
    assert res["captain_eo_shield"]["consensus_element"] == 10


def test_differential_clears_hurdle():
    """Differential candidate (element 11, 7.5 xP, +1.5 delta >= 1.2 hurdle) clears the shield and gets the armband."""
    squad = _make_squad(cand1_eo=160.0, cand1_xp=6.0, cand2_eo=20.0, cand2_xp=7.5)
    opt = LineupOptimizer(eo_hedge_weight=0.80, differential_captain_hurdle=1.20)
    res = opt.select_lineup_and_captain(squad, risk_preference="balanced", simulate_captaincy=False)

    assert res["captain"]["element"] == 11
    assert res["captain_eo_shield"]["shield_applied"] is False


def test_aggressive_mode_lifts_eo_shield():
    """In aggressive/chasing mode, marginal differential (6.3 xP vs 6.0 xP) takes the armband without penalty."""
    squad = _make_squad(cand1_eo=160.0, cand1_xp=6.0, cand2_eo=20.0, cand2_xp=6.3)
    opt = LineupOptimizer(eo_hedge_weight=0.80, differential_captain_hurdle=1.20)
    res = opt.select_lineup_and_captain(squad, risk_preference="aggressive", simulate_captaincy=False)

    assert res["captain"]["element"] == 11
    assert "lifted" in res["captain_eo_shield"]["reason"].lower()


def test_selected_by_percent_fallback():
    """When top10k_eo is not explicitly provided, selected_by_percent provides high-EO inference."""
    squad = _make_squad(cand1_eo=0.0, cand1_xp=6.0, cand2_eo=0.0, cand2_xp=6.3)
    squad = squad.drop(columns=["top10k_eo"])
    squad["selected_by_percent"] = 5.0
    squad.loc[squad["element"] == 10, "selected_by_percent"] = 65.0  # High ownership -> estimated EO > 100%

    opt = LineupOptimizer(eo_hedge_weight=0.80, differential_captain_hurdle=1.20)
    res = opt.select_lineup_and_captain(squad, risk_preference="balanced", simulate_captaincy=False)

    assert res["captain"]["element"] == 10
    assert res["captain_eo_shield"]["shield_applied"] is True
