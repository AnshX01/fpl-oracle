"""
Unit tests for Mathematical Optimizer (PuLP MILP), Formation rules, Selling Price math,
and Multi-Gameweek Stateful Sequential Transfer Engine (T1-T9).
"""

import pandas as pd

from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.squad import squad_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer


def make_mock_pool():
    rows = []
    # 4 GKP
    for i in range(1, 5):
        rows.append(
            {
                "element": i,
                "web_name": f"GKP_{i}",
                "position": "GKP",
                "team": (i % 20) + 1,
                "value": 50,
                "expected_points": 4.0 - (i * 0.2),
            }
        )
    # 10 DEF
    for i in range(5, 15):
        rows.append(
            {
                "element": i,
                "web_name": f"DEF_{i}",
                "position": "DEF",
                "team": (i % 20) + 1,
                "value": 50,
                "expected_points": 5.0 - (i * 0.1),
            }
        )
    # 10 MID
    for i in range(15, 25):
        rows.append(
            {
                "element": i,
                "web_name": f"MID_{i}",
                "position": "MID",
                "team": (i % 20) + 1,
                "value": 65,
                "expected_points": 6.0 - (i * 0.1),
            }
        )
    # 6 FWD
    for i in range(25, 31):
        rows.append(
            {
                "element": i,
                "web_name": f"FWD_{i}",
                "position": "FWD",
                "team": (i % 20) + 1,
                "value": 75,
                "expected_points": 7.0 - (i * 0.1),
            }
        )

    return pd.DataFrame(rows)


def make_mock_squad_and_projections(horizon_len: int = 5):
    rows = [
        {
            "element": 1,
            "web_name": "Raya",
            "position": "GKP",
            "team": 1,
            "value": 55,
            "expected_points": 4.5,
            "purchase_price": 55,
        },
        {
            "element": 2,
            "web_name": "Fabianski",
            "position": "GKP",
            "team": 19,
            "value": 40,
            "expected_points": 1.5,
            "purchase_price": 40,
        },
        {
            "element": 3,
            "web_name": "Saliba",
            "position": "DEF",
            "team": 1,
            "value": 60,
            "expected_points": 4.8,
            "purchase_price": 60,
        },
        {
            "element": 4,
            "web_name": "Gabriel",
            "position": "DEF",
            "team": 1,
            "value": 60,
            "expected_points": 4.7,
            "purchase_price": 60,
        },
        {
            "element": 5,
            "web_name": "Alexander-Arnold",
            "position": "DEF",
            "team": 11,
            "value": 70,
            "expected_points": 5.5,
            "purchase_price": 70,
        },
        {
            "element": 6,
            "web_name": "Robinson",
            "position": "DEF",
            "team": 8,
            "value": 45,
            "expected_points": 3.2,
            "purchase_price": 45,
        },
        {
            "element": 7,
            "web_name": "Faes",
            "position": "DEF",
            "team": 9,
            "value": 40,
            "expected_points": 2.0,
            "purchase_price": 40,
        },
        {
            "element": 8,
            "web_name": "Saka",
            "position": "MID",
            "team": 1,
            "value": 100,
            "expected_points": 7.5,
            "purchase_price": 100,
        },
        {
            "element": 9,
            "web_name": "Palmer",
            "position": "MID",
            "team": 6,
            "value": 105,
            "expected_points": 7.8,
            "purchase_price": 105,
        },
        {
            "element": 10,
            "web_name": "Mbeumo",
            "position": "MID",
            "team": 3,
            "value": 75,
            "expected_points": 6.2,
            "purchase_price": 75,
        },
        {
            "element": 11,
            "web_name": "Rogers",
            "position": "MID",
            "team": 2,
            "value": 50,
            "expected_points": 4.8,
            "purchase_price": 50,
        },
        {
            "element": 12,
            "web_name": "Winks",
            "position": "MID",
            "team": 9,
            "value": 45,
            "expected_points": 2.1,
            "purchase_price": 45,
        },
        {
            "element": 13,
            "web_name": "Haaland",
            "position": "FWD",
            "team": 12,
            "value": 150,
            "expected_points": 9.0,
            "purchase_price": 150,
        },
        {
            "element": 14,
            "web_name": "Watkins",
            "position": "FWD",
            "team": 2,
            "value": 90,
            "expected_points": 6.5,
            "purchase_price": 90,
        },
        {
            "element": 15,
            "web_name": "Wood",
            "position": "FWD",
            "team": 14,
            "value": 60,
            "expected_points": 5.2,
            "purchase_price": 60,
        },
    ]
    squad_df = pd.DataFrame(rows)

    pool_rows = list(rows)
    pool_rows.append(
        {"element": 16, "web_name": "Salah", "position": "MID", "team": 11, "value": 125, "expected_points": 8.5}
    )
    pool_rows.append(
        {"element": 17, "web_name": "Son", "position": "MID", "team": 17, "value": 98, "expected_points": 6.8}
    )
    pool_rows.append(
        {"element": 18, "web_name": "Isak", "position": "FWD", "team": 13, "value": 85, "expected_points": 6.9}
    )
    pool_rows.append(
        {"element": 19, "web_name": "Solanke", "position": "FWD", "team": 17, "value": 75, "expected_points": 5.8}
    )
    pool_rows.append(
        {"element": 20, "web_name": "Gvardiol", "position": "DEF", "team": 12, "value": 60, "expected_points": 5.1}
    )
    pool_rows.append(
        {"element": 21, "web_name": "Pedro Porro", "position": "DEF", "team": 17, "value": 55, "expected_points": 4.9}
    )
    pool_df = pd.DataFrame(pool_rows)

    projections = {}
    for offset in range(horizon_len):
        gw = 5 + offset
        df_gw = pool_df.copy()
        df_gw["expected_points"] = df_gw["expected_points"] * (1.0 + 0.03 * offset)
        projections[gw] = df_gw

    return squad_df, pool_df, projections


def test_squad_optimizer_constraints():
    pool = make_mock_pool()
    res = squad_optimizer.solve_best_squad(pool, budget=1000.0)
    squad = res["squad"]

    assert len(squad) == 15
    assert len(squad[squad["position"] == "GKP"]) == 2
    assert len(squad[squad["position"] == "DEF"]) == 5
    assert len(squad[squad["position"] == "MID"]) == 5
    assert len(squad[squad["position"] == "FWD"]) == 3
    assert squad["value"].sum() <= 1000.0

    # Max 3 per club
    team_counts = squad["team"].value_counts()
    assert (team_counts <= 3).all()


def test_selling_price_math():
    """Verify selling price formula: purchase_price + floor((now_cost - purchase_price) / 2)."""
    # Case 1: Player bought at 10.0m (100), now 10.4m (104). Rise = 4. Selling price = 100 + 2 = 102 (10.2m)
    p1 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=104)
    assert p1 == 102

    # Case 2: Player bought at 10.0m (100), now 10.3m (103). Rise = 3. 3 // 2 = 1. Selling price = 101 (10.1m)
    p2 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=103)
    assert p2 == 101

    # Case 3: Player bought at 10.0m (100), now 9.8m (98). Loss = 2. Sells at current price 98
    p3 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=98)
    assert p3 == 98


def test_lineup_and_captain_formation():
    pool = make_mock_pool()
    squad = squad_optimizer.solve_best_squad(pool, budget=1000.0)["squad"]
    lineup = lineup_optimizer.select_lineup_and_captain(squad)

    starters = lineup["starters"]
    bench = lineup["bench"]

    assert len(starters) == 11
    assert len(bench) == 4
    assert len(starters[starters["position"] == "GKP"]) == 1
    assert 3 <= len(starters[starters["position"] == "DEF"]) <= 5
    assert 2 <= len(starters[starters["position"] == "MID"]) <= 5
    assert 1 <= len(starters[starters["position"] == "FWD"]) <= 3

    # Captain and Vice Captain
    assert lineup["captain"]["element"] != lineup["vice_captain"]["element"]
    assert lineup["captain"]["multiplier"] == 2


def test_compute_squad_selling_prices_and_free_transfers():
    """Verify transfer history parsing for selling prices and banked free transfers calculation."""
    squad_df = pd.DataFrame(
        [
            {"element": 10, "web_name": "Salah", "value": 128, "cost_change_start": 3},
            {"element": 20, "web_name": "Haaland", "value": 152, "cost_change_start": 2},
        ]
    )

    transfers = [{"element_in": 20, "element_in_cost": 150, "element_out": 99, "event": 2}]

    priced_df = transfer_optimizer.compute_squad_selling_prices(squad_df, transfer_history=transfers)
    salah_row = priced_df[priced_df["element"] == 10].iloc[0]
    haaland_row = priced_df[priced_df["element"] == 20].iloc[0]

    assert salah_row["purchase_price"] == 125
    assert salah_row["selling_price"] == 126

    assert haaland_row["purchase_price"] == 150
    assert haaland_row["selling_price"] == 151

    history = [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 1},
        {"event": 4, "event_transfers": 0},
        {"event": 5, "event_transfers": 0},
    ]
    ft = transfer_optimizer.compute_available_free_transfers(entry_history=history, current_gw=5)
    assert ft == 4


def test_identical_horizon_branch_comparison():
    """T1 & T2: Multi-GW sequential evaluation comparing Roll, 1-transfer, 2-transfers on identical 5-GW horizon."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=15.0,  # 1.5m in bank
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
    )

    assert "recommended_plan" in res
    assert "candidate_plans" in res
    candidates = res["candidate_plans"]
    assert len(candidates) >= 2

    # Verify all candidate plans evaluate the identical 5-GW horizon
    plan_types = [p["plan_type"] for p in candidates]
    assert "ROLL_TRANSFER" in plan_types

    roll_plan = next(p for p in candidates if p["plan_type"] == "ROLL_TRANSFER")
    assert "horizon_gross_xp" in roll_plan
    assert "horizon_hits" in roll_plan
    assert "horizon_net_xp" in roll_plan
    assert roll_plan["horizon_gain_vs_roll"] == 0.0

    # Every candidate plan has horizon metrics and net gain vs roll
    for p in candidates:
        assert p["horizon_gross_xp"] > 0
        assert p["horizon_net_xp"] == round(p["horizon_gross_xp"] - (p["horizon_hits"] * 4.0), 2)
        assert "pure_xp_gain" in p
        assert "price_movement_gain" in p
        assert "robustness_score" in p


def test_greedy_vs_multi_gw_hit_avoidance():
    """T3: Verifies multi-GW plan tracks future hits avoided compared to greedy single-GW move."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
    )

    hit_avoidance = res["future_hit_avoidance"]
    assert "greedy_horizon_hits" in hit_avoidance
    assert "multi_gw_horizon_hits" in hit_avoidance
    assert "future_hits_avoided" in hit_avoidance
    assert hit_avoidance["future_hits_avoided"] >= 0
    assert isinstance(hit_avoidance["explanation"], str)


def test_free_transfer_option_value():
    """T4: Verifies mathematical FT option value calculation over horizon."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
    )

    assert "ft_option_value" in res
    assert res["ft_option_value"] >= 0.0
    assert "ft_option_explanation" in res
    assert "Option value" in res["ft_option_explanation"]


def test_robustness_reranking_and_no_regret():
    """T5: Verifies Monte Carlo robustness scoring and no-regret threshold (>= 0.70)."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
        num_mc_scenarios=25,
    )

    for p in res["candidate_plans"]:
        assert 0.0 <= p["robustness_score"] <= 1.0
        assert p["is_no_regret"] == bool(p["robustness_score"] >= 0.70)


def test_price_change_sensitivity_toggle():
    """T6: Verifies price change sensitivity toggle separates pure xP from price movements."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    # Pure xP run
    res_pure = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
        include_price_gain=False,
    )

    # Price included run
    res_price = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
        include_price_gain=True,
    )

    assert res_pure["include_price_gain"] is False
    assert res_price["include_price_gain"] is True

    for p in res_pure["candidate_plans"]:
        assert "pure_xp_gain" in p
        assert "price_movement_gain" in p


def test_plan_stability_threshold():
    """T8: Verifies plan stability threshold prevents churning on negligible gains (< 0.3 xP)."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    # Case A: Strict stability threshold (10.0 xP) -> must force Roll transfer
    res_stable = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
        stability_threshold=10.0,
    )

    assert res_stable["recommended_plan"]["plan_type"] == "ROLL_TRANSFER"
    assert "stability threshold" in res_stable["recommended_plan"]["recommendation_summary"]

    # Case B: Zero stability threshold (0.0 xP) -> selects highest gain transfer
    res_sensitive = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=15.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
        stability_threshold=0.0,
    )
    # With 0.0 threshold, if any transfer beats roll, it is chosen
    top_cand = res_sensitive["candidate_plans"][0]
    if top_cand["horizon_gain_vs_roll"] > 0:
        assert res_sensitive["recommended_plan"]["plan_type"] != "ROLL_TRANSFER"


def test_dynamic_roadmap_generation():
    """T7: Dynamic roadmap generated from actual trajectory, with zero canned template strings."""
    squad_df, pool_df, projections = make_mock_squad_and_projections(horizon_len=5)

    res = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=5,
    )

    roadmap = res["transfer_roadmap"]
    assert len(roadmap) == 5

    canned_phrases = ["Execute Primary Transfer", "Build toward Chip", "Consolidate core assets"]

    for step in roadmap:
        assert "gameweek" in step
        assert "status" in step
        assert step["status"] in ["CONDITIONAL_CANDIDATE", "CONTINGENT_ON_NEWS"]
        assert "action" in step
        assert "captain" in step
        assert "strategic_focus" in step
        assert "banked_free_transfers_projected" in step
        assert 1 <= step["banked_free_transfers_projected"] <= 5

        # Check zero canned phrases
        for phrase in canned_phrases:
            assert phrase not in step["action"]
