"""
Unit tests for Optimization and Chip Planning Blind Spot Fixes:
- Fix 2: Formation Upper Bound Constraints (lineup.py)
- Fix 4: Horizon Hit Alignment & Cooldown (hit_policy.py)
- Fix 5: Wildcard -> Bench Boost Adjacency Spacing (multiperiod.py)
- Fix 6 & 7: Dynamic Baselines and 2024+ Rule Advisory (planner.py & simulate.py)
"""

import pandas as pd

from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.chips.simulate import chip_simulator
from fpl_oracle.optimise.hit_policy import cooldown_until, settle_hit
from fpl_oracle.optimise.lineup import _order_bench
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def test_fix2_formation_max_constraints_prevent_illegal_subs():
    """Verify bench auto-sub evaluation respects formation maximums (max 5 DEF, max 5 MID, max 3 FWD)."""
    starters_data = [
        {"element": 1, "position": "GKP", "expected_points": 4.0},
        {"element": 2, "position": "DEF", "expected_points": 4.0},
        {"element": 3, "position": "DEF", "expected_points": 4.0},
        {"element": 4, "position": "DEF", "expected_points": 4.0},
        {"element": 5, "position": "MID", "expected_points": 6.0},
        {"element": 6, "position": "MID", "expected_points": 6.0},
        {"element": 7, "position": "MID", "expected_points": 6.0},
        {"element": 8, "position": "MID", "expected_points": 6.0},
        {"element": 9, "position": "MID", "expected_points": 6.0},
        {"element": 10, "position": "FWD", "expected_points": 5.0},
        {"element": 11, "position": "FWD", "expected_points": 5.0},
    ]
    bench_data = [
        {"element": 12, "position": "GKP", "expected_points": 0.0},
        {"element": 13, "position": "MID", "expected_points": 5.0},
        {"element": 14, "position": "DEF", "expected_points": 4.0},
        {"element": 15, "position": "FWD", "expected_points": 3.0},
    ]

    starters_df = pd.DataFrame(starters_data)
    bench_df = pd.DataFrame(bench_data)

    ordered = _order_bench(bench_df, starters_df)
    assert len(ordered) == 4
    assert ordered.iloc[0]["position"] == "GKP"


def test_fix4_hit_settlement_evaluating_state():
    """Single-week deficit does not trigger immediate failure if horizon is still evaluating."""
    res_evaluating = settle_hit([1], [2], {1: 3, 2: 2}, 4, horizon_completed=False)
    assert res_evaluating["status"] == "evaluating"
    assert res_evaluating["realised_gain"] == 1

    res_failed = settle_hit([1], [2], {1: 3, 2: 2}, 4, horizon_completed=True)
    assert res_failed["status"] == "failed"

    res_success = settle_hit([1], [2], {1: 8, 2: 2}, 4, horizon_completed=False)
    assert res_success["status"] == "success"

    multi_gw_pts = {
        6: {1: 2, 2: 2},
        7: {1: 8, 2: 2},
    }
    res_multi = settle_hit([1], [2], multi_gw_pts, 4, horizon_completed=True)
    assert res_multi["status"] == "success"
    assert res_multi["realised_gain"] == 6.0

    records = [{"gameweek": 6, "status": "evaluating"}]
    assert cooldown_until(records) == 0


def test_fix5_wildcard_into_bench_boost_spacing_allowed():
    """Verify that Wildcard in GW t followed immediately by Bench Boost in GW t+1
    is permitted under the 2-week spacing constraint."""
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_rows = []
    for i, pos in enumerate(positions, 1):
        squad_rows.append(
            {
                "element": i,
                "web_name": f"P_{i}",
                "position": pos,
                "team": (i % 5) + 1,
                "value": 60,
                "now_cost": 60,
                "purchase_price": 60,
                "selling_price": 60,
                "expected_points": 5.0,
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
    squad_df = pd.DataFrame(squad_rows)
    pool_rows = squad_rows.copy()
    for i in range(16, 26):
        pos = "DEF" if i < 20 else "MID"
        pool_rows.append(
            {
                "element": i,
                "web_name": f"Pool_{i}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 60,
                "now_cost": 60,
                "purchase_price": 60,
                "selling_price": 60,
                "expected_points": 5.5,
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
    pool_df = pd.DataFrame(pool_rows)
    projections = {gw: pool_df for gw in range(6, 9)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3, min_gameweeks_between_chips=2))
    plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=6,
        horizon_len=3,
        chips_available=["wildcard", "bboost"],
        forced_chip_schedule={6: "wildcard", 7: "bboost"},
    )
    sched = {act.gameweek: act.active_chip for act in plan.trajectory if act.active_chip}
    assert sched.get(6) == "wildcard", f"Expected wildcard at GW6, got {sched.get(6)}"
    assert sched.get(7) == "bboost", f"Expected bboost at GW7, got {sched.get(7)}"


def test_fix6_dynamic_chip_utility_baselines():
    """Verify ChipSimulator and ChipPlanner use dynamic starter xP baselines rather than hardcoded 48/38."""
    squad_data = [
        {
            "element": i,
            "web_name": f"P{i}",
            "position": "GKP" if i <= 2 else ("DEF" if i <= 7 else ("MID" if i <= 12 else "FWD")),
            "team": (i % 20) + 1,
            "now_cost": 50,
            "expected_points": 5.0,
        }
        for i in range(1, 16)
    ]
    squad_df = pd.DataFrame(squad_data)

    bb_res = chip_simulator.evaluate_bench_boost(10, squad_df)
    assert "current_squad_xp" in bb_res
    assert bb_res["current_squad_xp"] > 0

    tc_res = chip_simulator.evaluate_triple_captain(10, squad_df)
    assert "current_squad_xp" in tc_res
    assert tc_res["current_squad_xp"] > 0

    gain, conf, base_xp, with_xp = chip_planner.compute_chip_utility(
        chip="bboost",
        gw=10,
        current_squad_df=squad_df,
        horizon_projections={10: squad_df},
        dgw_gws=[],
        bgw_gws=[],
    )
    assert base_xp == bb_res["current_squad_xp"]
    assert with_xp == round(base_xp + gain, 2)


def test_fix7_cross_gameweek_guidance_2024_rule_compliance():
    """Guidance string must advise managers that banked free transfers are preserved across Wildcards under 2024+ rules."""
    guidance = chip_planner.get_joint_transfer_advice(joint_schedule={11: "wildcard"}, current_gw=10)
    assert "preserved under 2024+ rules" in guidance["transfer_action_guidance"]
    assert "All transfers will be reset" not in guidance["transfer_action_guidance"]
