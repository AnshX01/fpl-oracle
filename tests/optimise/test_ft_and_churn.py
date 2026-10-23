import pandas as pd
import pytest

from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def make_test_squad_and_pool():
    rows = []
    # 2 GKP
    for i in [1, 2]:
        rows.append({
            "element": i,
            "web_name": f"GKP_{i}",
            "position": "GKP",
            "team": i,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
        })
    # 5 DEF
    for i in range(3, 8):
        rows.append({
            "element": i,
            "web_name": f"DEF_{i}",
            "position": "DEF",
            "team": i,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
        })
    # 5 MID: 8, 9, 10, 11, 12
    # Element 8 has 5.0 xP
    for i in [8, 9, 10, 11, 12]:
        rows.append({
            "element": i,
            "web_name": f"MID_{i}",
            "position": "MID",
            "team": (i % 7) + 1,
            "value": 65,
            "purchase_price": 65,
            "selling_price": 65,
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
        })
    # 3 FWD: 13, 14, 15
    for i in [13, 14, 15]:
        rows.append({
            "element": i,
            "web_name": f"FWD_{i}",
            "position": "FWD",
            "team": (i % 7) + 1,
            "value": 75,
            "purchase_price": 75,
            "selling_price": 75,
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
        })

    squad_df = pd.DataFrame(rows).copy()

    # Target in pool: MID_TARGET (element 20), same price 65, 5.3 xP (+0.3 xP marginal gain)
    marginal_target = {
        "element": 20,
        "web_name": "MID_MARGINAL",
        "position": "MID",
        "team": 9,
        "value": 65,
        "purchase_price": 65,
        "selling_price": 65,
        "expected_points": 5.3,  # only +0.3 xP over MID_8
        "chance_of_playing": 100.0,
    }

    # Big target in pool: MID_STAR (element 21), same price 65, 8.5 xP (+3.5 xP big gain)
    big_target = {
        "element": 21,
        "web_name": "MID_STAR",
        "position": "MID",
        "team": 10,
        "value": 65,
        "purchase_price": 65,
        "selling_price": 65,
        "expected_points": 8.5,  # +3.5 xP over MID_8
        "chance_of_playing": 100.0,
    }

    pool_df = pd.concat([squad_df, pd.DataFrame([marginal_target, big_target])], ignore_index=True)
    horizon = {1: pool_df.copy(), 2: pool_df.copy()}
    return squad_df, pool_df, horizon


def test_ft_option_value_forces_roll_on_marginal_gain():
    """
    Test that when an incoming transfer offers only a marginal gain (+0.3 xP),
    the FT Option Value rule preserves the free transfer and recommends Roll.
    """
    squad_df, pool_df, horizon = make_test_squad_and_pool()
    # Filter out big_target so only marginal_target is available
    pool_marginal = pool_df[pool_df["element"] != 21].copy()
    horizon_marginal = {1: pool_marginal.copy(), 2: pool_marginal.copy()}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, ft_option_value=1.80))
    res = opt.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_marginal,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon_marginal,
        current_gw=1,
        target_gw=1,
        horizon_len=2,
        chips_available=[],
    )

    rec = res["recommended_plan"]
    # With FT option value = 1.80, a +0.3 xP gain is rejected in favor of rolling
    assert rec["plan_type"] == "ROLL_TRANSFER"
    assert "Roll" in rec["recommendation_summary"]


def test_ft_option_value_allows_high_conviction_transfer():
    """
    Test that when a transfer provides massive expected gain (+3.5 xP),
    it easily clears the FT Option Value hurdle and is executed.
    """
    squad_df, pool_df, horizon = make_test_squad_and_pool()

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, ft_option_value=1.80))
    res = opt.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon,
        current_gw=1,
        target_gw=1,
        horizon_len=2,
        chips_available=[],
    )

    rec = res["recommended_plan"]
    assert rec["transfers_count"] == 1
    assert any(t["element"] == 21 for t in rec["transfers_in"])


def test_anti_churn_penalty_applied_on_lateral_transfers():
    """
    Test that lateral transfers between same-position, same-cost healthy players
    incur anti-churn friction penalty.
    """
    squad_df, pool_df, horizon = make_test_squad_and_pool()
    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, anti_churn_penalty=0.50))
    res = opt.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon,
        current_gw=1,
        target_gw=1,
        horizon_len=2,
        stability_threshold=0.0,
        chips_available=[],
    )

    # Find the candidate plan that buys MID_STAR (21)
    cand_21 = next((p for p in res["candidate_plans"] if any(t["element"] == 21 for t in p["transfers_in"])), None)
    assert cand_21 is not None
    # Swapping MID_8 (65) for MID_21 (65) is in the same price tier and same position
    assert cand_21.get("anti_churn_penalty", 0.0) == 0.50
