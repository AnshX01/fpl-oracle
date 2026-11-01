import pandas as pd

from fpl_oracle.league.strategy import LeagueStrategyAdvisor
from fpl_oracle.optimise.lineup import LineupOptimizer


def test_lineup_prefers_higher_capital_within_noise_band():
    """
    When a premium talisman (e.g. Cole Palmer £9.8m, 3.70 xP) and a budget defender
    (e.g. James Justin £4.5m, 3.72 xP) are separated by <= 0.05 xP (pure estimator noise),
    the lineup optimizer must break the tie in favor of active team capital on pitch,
    starting the talisman in a 3-5-2 rather than benching him in a 4-4-2.
    """
    rows = [
        # GKPs (2)
        {"element": 1, "web_name": "GKP_1", "position": "GKP", "expected_points": 3.6, "now_cost": 45, "team": 1},
        {"element": 2, "web_name": "GKP_2", "position": "GKP", "expected_points": 3.5, "now_cost": 45, "team": 2},
        # DEFs (5)
        {"element": 3, "web_name": "Konsa", "position": "DEF", "expected_points": 4.56, "now_cost": 47, "team": 3},
        {"element": 4, "web_name": "Silva", "position": "DEF", "expected_points": 4.51, "now_cost": 50, "team": 4},
        {"element": 5, "web_name": "VanHecke", "position": "DEF", "expected_points": 4.04, "now_cost": 50, "team": 5},
        {"element": 6, "web_name": "Justin", "position": "DEF", "expected_points": 3.72, "now_cost": 45, "team": 6},
        {"element": 7, "web_name": "vanEwijk", "position": "DEF", "expected_points": 2.90, "now_cost": 39, "team": 7},
        # MIDs (5)
        {"element": 8, "web_name": "Mbeumo", "position": "MID", "expected_points": 5.61, "now_cost": 79, "team": 8},
        {"element": 9, "web_name": "Rogers", "position": "MID", "expected_points": 5.48, "now_cost": 81, "team": 9},
        {"element": 10, "web_name": "Gross", "position": "MID", "expected_points": 4.21, "now_cost": 62, "team": 2},
        {"element": 11, "web_name": "Rudoni", "position": "MID", "expected_points": 4.19, "now_cost": 49, "team": 7},
        {"element": 12, "web_name": "Palmer", "position": "MID", "expected_points": 3.70, "now_cost": 98, "team": 9},
        # FWDs (3)
        {"element": 13, "web_name": "Isak", "position": "FWD", "expected_points": 6.03, "now_cost": 92, "team": 10},
        {"element": 14, "web_name": "Haaland", "position": "FWD", "expected_points": 5.79, "now_cost": 158, "team": 11},
        {"element": 15, "web_name": "DCL", "position": "FWD", "expected_points": 3.53, "now_cost": 61, "team": 6},
    ]
    df = pd.DataFrame(rows)
    df["chance_of_playing"] = 100.0
    df["p10"] = df["expected_points"] * 0.4
    df["p90"] = df["expected_points"] * 1.6

    optimizer = LineupOptimizer()
    res = optimizer.select_lineup_and_captain(df, risk_preference="balanced", simulate_captaincy=False)

    starters = set(res["starters"]["element"])
    # Element 12 is Palmer (£9.8m), Element 6 is Justin (£4.5m)
    assert 12 in starters, "Palmer (£9.8m) must start in the XI when within 0.02 xP of a budget defender"
    assert 6 not in starters, "Justin (£4.5m) should be on the bench, yielding to premium capital"
    assert res["formation"] == "3-5-2", f"Expected 3-5-2 formation with 5 midfielders, got {res['formation']}"


def test_defending_lead_posture_when_rank_1():
    """A manager in 1st place (even with a 5-point lead) must be classified as DEFENDING_LEAD."""
    advisor = LeagueStrategyAdvisor()
    posture = advisor.determine_risk_posture(points_gap=5.0, user_rank=1, current_gw=10)
    assert posture == "DEFENDING_LEAD", f"Expected DEFENDING_LEAD for Rank 1 with 5pt lead, got {posture}"


def test_captain_shadowing_defending_lead():
    """When defending a lead, captain shadowing should shield against rival/template captain."""
    advisor = LeagueStrategyAdvisor()
    guidance = advisor.evaluate_captain_shadowing(
        top_captain_id=13,  # Isak (6.03 xP)
        starters_ids=[1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14],  # 14 is Haaland
        rival_captain_id=14,  # Haaland
        posture="DEFENDING_LEAD",
    )
    assert guidance["action"] == "SHADOW_RIVAL_CAPTAIN"
    assert guidance["captain_id"] == 14
    assert guidance["should_switch"] is True
