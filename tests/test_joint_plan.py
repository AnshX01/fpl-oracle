"""
Unit and Integration Tests for Joint Chip & Transfer Planning (F11).
Verifies stateful joint trajectory search, chip deployment vs hold comparison,
chip retention opportunity costs, and Set 1 / Set 2 rule enforcement.
"""

import pandas as pd
import pytest

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
    )

    assert res["recommended_chip"] == "3xc"
    assert res["chip_action"] == "DEPLOY"

    tc_entry = next(c for c in res["chip_comparison_table"] if c["chip"] == "3xc")
    assert tc_entry["is_recommended"] is True
    assert tc_entry["gross_gain_vs_hold"] >= 15.0
    assert tc_entry["net_gain_vs_hold"] > 0.0
    assert res["recommended_plan"]["chip_applied"] == "3xc"


def test_chip_preservation_frontier_exposes_future_value_gap(sample_squad_and_pool):
    squad, pool, projections = sample_squad_and_pool
    result = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=10,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=6,
        available_chips=["3xc", "bboost"],
    )
    assert {row["chip"] for row in result["resource_frontier"]} == {"3xc", "bboost"}
    for row in result["resource_frontier"]:
        assert all(step["chip"] != row["chip"] for step in row["retained_trajectory"])
        assert row["tail_value_break_even_at_horizon_end"] >= row["discounted_horizon_cost_to_preserve"] >= 0
    assert result["decision_scope"]["season_tail_value_estimated"] is False


def test_freehit_not_recommended_on_zero_gain_normal_week():
    from tests.test_sequential_search import fixture_data

    frame, pools, maps = fixture_data()

    # A zero-benefit Free Hit is never recommended by the fallback search.
    result = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=frame,
        player_pool_df=frame,
        bank=0,
        free_transfers=1,
        horizon_projections={18: pools[18]},
        target_gw=18,
        current_gw=17,
        available_chips=["freehit"],
        horizon_len=1,
        locked_in_ids=set(frame.element),
    )
    assert result["recommended_chip"] is None
    assert result["best_candidate"]["chip"] == "HOLD"
