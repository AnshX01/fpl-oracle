"""
Unit tests for 2026/27 Chip Strategy Engine, Set boundaries, and Banking rules.
"""

from fpl_oracle.api.models import ChipHistoryItem, ManagerHistory
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.config import RULES


def test_chip_rules_config_2026_27():
    """Verify that chip configuration matches official 2026/27 rules."""
    assert RULES["chips"]["total_chips_in_season"] == 8
    assert RULES["chips"]["max_chips_per_event"] == 1
    assert RULES["chips"]["assistant_manager_enabled"] is False
    assert RULES["transfers"]["max_banked_free_transfers"] == 5
    assert RULES["chips"]["set_1"]["deadline_event"] == 19
    assert RULES["chips"]["set_2"]["start_event"] == 20

def test_remaining_chips_set_boundaries():
    """Verify that played chips are allocated correctly to Set 1 (GW 1-19) vs Set 2 (GW 20-38)."""
    # Manager played Wildcard in GW3, Triple Captain in GW15
    hist = ManagerHistory(
        current=[],
        past=[],
        chips=[
            ChipHistoryItem(name="wildcard", event=3, time="2026-09-01T12:00:00Z"),
            ChipHistoryItem(name="3xc", event=15, time="2026-11-20T12:00:00Z")
        ]
    )

    status = chip_planner.get_remaining_chips(hist)

    assert "wildcard" in status["set_1_used"]
    assert "3xc" in status["set_1_used"]
    assert "freehit" in status["set_1_remaining"]
    assert "bboost" in status["set_1_remaining"]

    # Set 2 chips should all remain untouched since GW < 20
    assert len(status["set_2_remaining"]) == 4
    assert "wildcard" in status["set_2_remaining"]
    assert "3xc" in status["set_2_remaining"]

def test_joint_chip_assignment_no_conflicts():
    """Verify that joint beam search assigns each chip to a distinct gameweek (1 chip per GW rule)."""
    import pandas as pd
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame([
        {"element": i, "web_name": f"P{i}", "team": (i % 20) + 1, "position": positions[i-1], "value": 70, "expected_points": 5.0}
        for i in range(1, 16)
    ])
    projections = {gw: squad_df for gw in range(6, 20)}
    assign, total_gain, alts = chip_planner.optimize_joint_assignment(
        available_gws=list(range(6, 20)),
        remaining_chips=["wildcard", "freehit", "3xc", "bboost"],
        current_squad_df=squad_df,
        horizon_projections=projections,
        dgw_gws=[12],
        bgw_gws=[10],
        budget=1000.0
    )

    # 4 distinct chips
    assert len(assign) == 4
    # All assigned GWs must be strictly unique (max 1 chip per gameweek)
    assert len(set(assign.values())) == 4
    # All assigned GWs must be in range [6, 19]
    for gw in assign.values():
        assert 6 <= gw <= 19
    assert total_gain > 0.0
    assert len(alts) > 0

def test_chip_set_1_expiry_opportunity_cost():
    """Verify that Set 1 expiry warning includes non-zero opportunity cost for unplayed chips."""
    import pandas as pd
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame([
        {"element": i, "web_name": f"P{i}", "team": (i % 20) + 1, "position": positions[i-1], "value": 70, "expected_points": 5.0}
        for i in range(1, 16)
    ])
    projections = {gw: squad_df for gw in range(16, 20)}
    res = chip_planner.generate_chip_strategy(
        current_gw=16,
        current_squad_df=squad_df,
        horizon_projections=projections,
        fixtures=[],
        bootstrap=type("MockBoot", (), {"teams": []})(),
        manager_history=None
    )

    assert res["set_1_deadline_warning"] is not None
    assert "opportunity cost" in res["set_1_deadline_warning"].lower()
    assert res["total_set_1_opportunity_cost"] > 0
    # Every chip in plan table has trigger conditions and confidence
    for row in res["chip_plan_table"]:
        assert "trigger_conditions" in row
        assert "confidence" in row
        assert "alternative_gw" in row

