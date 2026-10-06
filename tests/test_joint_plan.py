"""
Unit and Integration Tests for Joint Chip & Transfer Planning (F11).
Verifies stateful joint trajectory search, chip deployment vs hold comparison,
chip retention opportunity costs, and Set 1 / Set 2 rule enforcement.
"""

import pandas as pd
import pytest

from fpl_oracle.api.models import ChipHistoryItem, ManagerHistory
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.optimise.transfers import transfer_optimizer


@pytest.fixture
def sample_squad_and_pool():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_rows = []
    for i in range(1, 16):
        pos = positions[i - 1]
        xp = 5.0 if i <= 11 else 2.0
        squad_rows.append(
            {
                "element": i,
                "web_name": f"Squad_{i}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 55,
                "selling_price": 55,
                "purchase_price": 55,
                "expected_points": xp,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    # Pool with 30 players
    pool_rows = list(squad_rows)
    for i in range(16, 35):
        pos = positions[(i - 16) % 15]
        pool_rows.append(
            {
                "element": i,
                "web_name": f"Pool_{i}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 60,
                "selling_price": 60,
                "purchase_price": 60,
                "expected_points": 5.5,
            }
        )
    pool_df = pd.DataFrame(pool_rows)

    horizon_projections = {gw: pool_df.copy() for gw in range(6, 11)}
    return squad_df, pool_df, horizon_projections


def test_joint_plan_hold_when_chips_suboptimal(sample_squad_and_pool):
    """When chip gains do not exceed opportunity costs, joint optimizer holds chips."""
    squad_df, pool_df, horizon_projections = sample_squad_and_pool

    res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_projections,
        current_gw=6,
        target_gw=6,
        available_chips=["3xc", "bboost"],
        chip_retention_values={"3xc": 12.0, "bboost": 15.0},  # High retention costs (e.g. saving for DGW)
    )

    assert res["recommended_chip"] is None
    assert res["chip_action"] == "HOLD"
    assert "chip_comparison_table" in res
    assert len(res["chip_comparison_table"]) >= 3  # HOLD, 3xc, bboost

    # Verify HOLD candidate is marked recommended
    hold_entry = next(c for c in res["chip_comparison_table"] if c["chip"] == "HOLD")
    assert hold_entry["is_recommended"] is True
    assert hold_entry["net_gain_vs_hold"] == 0.0


def test_joint_plan_recommends_tc_on_massive_captain_xp(sample_squad_and_pool):
    """When an elite captain xP exceeds the retention threshold, joint optimizer deploys TC."""
    squad_df, pool_df, horizon_projections = sample_squad_and_pool

    # Give player 15 (Haaland surrogate) massive expected points
    squad_df.loc[squad_df["element"] == 15, "expected_points"] = 16.0
    for gw in horizon_projections:
        horizon_projections[gw].loc[horizon_projections[gw]["element"] == 15, "expected_points"] = 16.0

    res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_projections,
        current_gw=6,
        target_gw=6,
        available_chips=["3xc"],
        chip_retention_values={"3xc": 6.0},  # Standard retention cost
    )

    assert res["recommended_chip"] == "3xc"
    assert res["chip_action"] == "DEPLOY"

    tc_entry = next(c for c in res["chip_comparison_table"] if c["chip"] == "3xc")
    assert tc_entry["is_recommended"] is True
    assert tc_entry["gross_gain_vs_hold"] >= 15.0
    assert tc_entry["opportunity_cost"] == 6.0
    assert tc_entry["net_gain_vs_hold"] > 0.0
    assert res["recommended_plan"]["chip_applied"] == "3xc"


def test_joint_plan_opportunity_cost_quantification(sample_squad_and_pool):
    """Verify that every evaluated chip explicitly quantifies gross gain, retention cost, and net gain."""
    squad_df, pool_df, horizon_projections = sample_squad_and_pool

    res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=2,
        horizon_projections=horizon_projections,
        current_gw=6,
        target_gw=6,
        available_chips=["3xc", "bboost"],
    )

    for entry in res["chip_comparison_table"]:
        assert "chip" in entry
        assert "action" in entry
        assert "gross_gain_vs_hold" in entry
        assert "opportunity_cost" in entry
        assert "net_gain_vs_hold" in entry
        assert "effective_trajectory_score" in entry
        assert "reason" in entry


def test_chip_planner_evaluate_joint_plan_respects_sets(sample_squad_and_pool):
    """Verify that chip_planner.evaluate_joint_plan filters chips based on Set 1 vs Set 2 remaining chips."""
    squad_df, pool_df, horizon_projections = sample_squad_and_pool

    # History with 3xc used in Set 1 (GW 3)
    history = ManagerHistory(
        current=[],
        past=[],
        chips=[
            ChipHistoryItem(name="3xc", event=3, time="2026-09-01T12:00:00Z"),
        ],
    )

    res = chip_planner.evaluate_joint_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_projections,
        current_gw=6,
        target_gw=6,
        manager_history=history,
    )

    chip_codes = [c["chip"] for c in res["chip_comparison_table"]]
    # 3xc should not be available in Set 1 since it was already played in GW3
    assert "3xc" not in chip_codes
    assert "HOLD" in chip_codes
