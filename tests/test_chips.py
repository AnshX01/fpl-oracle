"""
Unit tests for 2026/27 Chip Strategy Engine, Set boundaries, and Banking rules (C1-C4).
"""

import pandas as pd

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
    hist = ManagerHistory(
        current=[],
        past=[],
        chips=[
            ChipHistoryItem(name="wildcard", event=3, time="2026-09-01T12:00:00Z"),
            ChipHistoryItem(name="3xc", event=15, time="2026-11-20T12:00:00Z"),
        ],
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
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame(
        [
            {
                "element": i,
                "web_name": f"P{i}",
                "team": (i % 20) + 1,
                "position": positions[i - 1],
                "value": 70,
                "expected_points": 5.0,
            }
            for i in range(1, 16)
        ]
    )
    projections = {gw: squad_df for gw in range(6, 20)}
    assign, total_gain, alts = chip_planner.optimize_joint_assignment(
        available_gws=list(range(6, 20)),
        remaining_chips=["wildcard", "freehit", "3xc", "bboost"],
        current_squad_df=squad_df,
        horizon_projections=projections,
        dgw_gws=[12],
        bgw_gws=[10],
        budget=1000.0,
    )

    # 4 distinct chips
    assert len(assign) == 4
    # All assigned GWs must be strictly unique (max 1 chip per gameweek)
    assert len(set(assign.values())) == 4
    for gw in assign.values():
        assert 6 <= gw <= 19
    assert total_gain > 0.0
    assert len(alts) > 0


def test_chip_set_1_expiry_opportunity_cost():
    """Verify that Set 1 expiry warning includes non-zero opportunity cost for unplayed chips."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame(
        [
            {
                "element": i,
                "web_name": f"P{i}",
                "team": (i % 20) + 1,
                "position": positions[i - 1],
                "value": 70,
                "expected_points": 5.0,
            }
            for i in range(1, 16)
        ]
    )
    projections = {gw: squad_df for gw in range(16, 20)}
    res = chip_planner.generate_chip_strategy(
        current_gw=16,
        current_squad_df=squad_df,
        horizon_projections=projections,
        fixtures=[],
        bootstrap=type("MockBoot", (), {"teams": []})(),
        manager_history=None,
    )

    assert res["set_1_deadline_warning"] is not None
    assert "opportunity cost" in res["set_1_deadline_warning"].lower()
    assert res["total_set_1_opportunity_cost"] > 0
    for row in res["chip_plan_table"]:
        assert "trigger_conditions" in row
        assert "confidence" in row
        assert "alternative_gw" in row


def test_chip_calendar_benefit_vs_baseline():
    """C1: Verifies multi-GW chip calendar computes expected benefit vs no-chip baseline and labels low-confidence beyond horizon."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame(
        [
            {
                "element": i,
                "web_name": f"P{i}",
                "team": (i % 20) + 1,
                "position": positions[i - 1],
                "value": 70,
                "expected_points": 5.0,
            }
            for i in range(1, 16)
        ]
    )
    projections = {gw: squad_df for gw in range(5, 20)}
    res = chip_planner.generate_chip_strategy(
        current_gw=5,
        current_squad_df=squad_df,
        horizon_projections=projections,
        fixtures=[],
        bootstrap=type("MockBoot", (), {"teams": []})(),
        manager_history=None,
    )

    plan_table = res["chip_plan_table"]
    assert len(plan_table) > 0
    for row in plan_table:
        assert "expected_gain" in row
        assert "baseline_no_chip_xp" in row
        assert "with_chip_xp" in row
        assert row["with_chip_xp"] == round(row["baseline_no_chip_xp"] + row["expected_gain"], 1)
        assert "confidence" in row
        if row["recommended_gw"] > (5 + 4):
            assert "Beyond 5-GW Horizon" in row["confidence"]


def test_joint_chip_and_transfer_planning():
    """C2: Verifies transfer planner cross-awareness for upcoming chips (e.g. upcoming Wildcard)."""
    # Case 1: Wildcard scheduled next week -> penalty against hits
    schedule_wc_next = {6: "wildcard"}
    advice_wc = chip_planner.get_joint_transfer_advice(schedule_wc_next, current_gw=5)
    assert "UPCOMING WILDCARD" in advice_wc["cross_awareness_alert"]
    assert advice_wc["hit_penalty_multiplier"] > 1.0

    # Case 2: Free Hit active this week -> temporary squad notice
    schedule_fh_now = {5: "freehit"}
    advice_fh = chip_planner.get_joint_transfer_advice(schedule_fh_now, current_gw=5)
    assert "FREE HIT ACTIVE" in advice_fh["cross_awareness_alert"]


def test_strict_recommendation_threshold_and_set1_cutoff():
    """C3: Verifies strict threshold gating and Set 1 GW19 hard cutoff."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame(
        [
            {
                "element": i,
                "web_name": f"P{i}",
                "team": (i % 20) + 1,
                "position": positions[i - 1],
                "value": 70,
                "expected_points": 5.0,
            }
            for i in range(1, 16)
        ]
    )
    projections = {gw: squad_df for gw in range(21, 38)}

    # When current_gw > 19: Set 1 is strictly expired
    res_gw21 = chip_planner.generate_chip_strategy(
        current_gw=21,
        current_squad_df=squad_df,
        horizon_projections=projections,
        fixtures=[],
        bootstrap=type("MockBoot", (), {"teams": []})(),
        manager_history=None,
    )

    assert "SET 1 EXPIRED" in res_gw21["set_1_deadline_warning"]
    set1_rows = [r for r in res_gw21["chip_plan_table"] if r["set"] == 1]
    assert len(set1_rows) == 0


def test_rival_chips_tracking_context():
    """C4: Verifies tracking rivals' remaining chips, used percentages, and tactical context."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_df = pd.DataFrame(
        [
            {
                "element": i,
                "web_name": f"P{i}",
                "team": (i % 20) + 1,
                "position": positions[i - 1],
                "value": 70,
                "expected_points": 5.0,
            }
            for i in range(1, 16)
        ]
    )
    projections = {gw: squad_df for gw in range(5, 10)}

    # Mock rival squads with chip history
    rivals_analysis = {
        "rival_squads": [
            {"entry_id": 1, "chips_used": ["bboost"]},
            {"entry_id": 2, "chips_used": ["bboost", "3xc"]},
            {"entry_id": 3, "chips_used": []},
            {"entry_id": 4, "chips_used": ["wildcard"]},
        ]
    }

    res = chip_planner.generate_chip_strategy(
        current_gw=5,
        current_squad_df=squad_df,
        horizon_projections=projections,
        fixtures=[],
        bootstrap=type("MockBoot", (), {"teams": []})(),
        manager_history=None,
        rivals_analysis=rivals_analysis,
    )

    rival_summary = res["rival_chips_summary"]
    assert "bboost" in rival_summary
    assert rival_summary["bboost"]["rivals_used_count"] == 2
    assert rival_summary["bboost"]["rivals_used_pct"] == 50.0
    assert rival_summary["bboost"]["rivals_remaining_count"] == 2
