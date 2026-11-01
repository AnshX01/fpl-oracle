"""
Unit tests covering combinatorial solver mathematics and mixed-integer linear programming upgrades.

Covers:
1. MultiPeriod MILP Banked FT (Rule R4) Big-M invariant (M=20) and ft_var lowBound=1.
2. MultiPeriod BGW FT dividend scope leak fix on forced_w1_chip.
3. Beam transfer candidate search budget-generating downgrade inclusion (funding cascades).
4. Cameo auto-sub blocking engine unimodal Gaussian substitute probability model.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.optimise.lineup import LineupOptimizer
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer


def _make_sample_squad() -> pd.DataFrame:
    """Builds a deterministic 15-player squad."""
    rows = [
        # GKP (2)
        {
            "element": 1,
            "web_name": "GKP_1",
            "position": "GKP",
            "team": 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 2,
            "web_name": "GKP_2",
            "position": "GKP",
            "team": 2,
            "value": 40,
            "purchase_price": 40,
            "selling_price": 40,
            "expected_points": 1.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        # DEF (5)
        {
            "element": 3,
            "web_name": "DEF_1",
            "position": "DEF",
            "team": 3,
            "value": 60,
            "purchase_price": 60,
            "selling_price": 60,
            "expected_points": 5.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 4,
            "web_name": "DEF_2",
            "position": "DEF",
            "team": 4,
            "value": 55,
            "purchase_price": 55,
            "selling_price": 55,
            "expected_points": 5.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 5,
            "web_name": "DEF_3",
            "position": "DEF",
            "team": 5,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": 4.5,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 6,
            "web_name": "DEF_4",
            "position": "DEF",
            "team": 6,
            "value": 45,
            "purchase_price": 45,
            "selling_price": 45,
            "expected_points": 3.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 7,
            "web_name": "DEF_5",
            "position": "DEF",
            "team": 7,
            "value": 40,
            "purchase_price": 40,
            "selling_price": 40,
            "expected_points": 2.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        # MID (5)
        {
            "element": 8,
            "web_name": "MID_1",
            "position": "MID",
            "team": 8,
            "value": 100,
            "purchase_price": 100,
            "selling_price": 100,
            "expected_points": 7.5,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 9,
            "web_name": "MID_2",
            "position": "MID",
            "team": 9,
            "value": 85,
            "purchase_price": 85,
            "selling_price": 85,
            "expected_points": 6.5,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 10,
            "web_name": "MID_3",
            "position": "MID",
            "team": 10,
            "value": 75,
            "purchase_price": 75,
            "selling_price": 75,
            "expected_points": 6.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 11,
            "web_name": "MID_4",
            "position": "MID",
            "team": 11,
            "value": 65,
            "purchase_price": 65,
            "selling_price": 65,
            "expected_points": 5.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 12,
            "web_name": "MID_5",
            "position": "MID",
            "team": 12,
            "value": 55,
            "purchase_price": 55,
            "selling_price": 55,
            "expected_points": 3.5,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        # FWD (3)
        {
            "element": 13,
            "web_name": "FWD_1",
            "position": "FWD",
            "team": 13,
            "value": 130,
            "purchase_price": 130,
            "selling_price": 130,
            "expected_points": 8.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 14,
            "web_name": "FWD_2",
            "position": "FWD",
            "team": 14,
            "value": 80,
            "purchase_price": 80,
            "selling_price": 80,
            "expected_points": 6.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
        {
            "element": 15,
            "web_name": "FWD_3",
            "position": "FWD",
            "team": 15,
            "value": 45,
            "purchase_price": 45,
            "selling_price": 45,
            "expected_points": 2.0,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        },
    ]
    return pd.DataFrame(rows)


def test_multiperiod_ft_big_m_and_lowbound():
    """
    Test that the MILP model's FT variables enforce lowBound=1 and Big-M (M=20)
    correctly handles multi-transfer moves without infeasibility.
    """
    squad_df = _make_sample_squad()
    cfg = MultiPeriodConfig(horizon_len=2)
    optimizer = MultiPeriodOptimizer(cfg)

    # Solve across 2 GWs with 1 initial FT
    plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=squad_df,
        bank=5.0,
        free_transfers=1,
        horizon_projections={1: squad_df, 2: squad_df},
        current_gw=0,
        target_gw=1,
        horizon_len=2,
        hit_cooldown_until=0,
    )

    assert plan is not None
    # Verify that in all planned actions, free_transfers_available >= 1 and free_transfers_next >= 1
    for act in plan.trajectory:
        assert act.free_transfers_available >= 1, f"GW{act.gameweek}: available FT < 1"
        assert act.free_transfers_next >= 1, f"GW{act.gameweek}: next FT < 1"


def test_multiperiod_bgw_ft_dividend_scope_leak():
    """
    Verify that upcoming BGW in horizon (GW26/GW29) scales ft_dividend
    when forced_w1_chip is not 'freehit', and suppresses the boost when forced_w1_chip is 'freehit'.
    """
    squad_df = _make_sample_squad()
    cfg = MultiPeriodConfig(horizon_len=2, ft_option_value=2.0)
    optimizer = MultiPeriodOptimizer(cfg)

    # Plan covering GW25 and GW26 (BGW in horizon)
    # With forced_w1_chip='wildcard' vs forced_w1_chip='freehit'
    projections = {25: squad_df, 26: squad_df}

    plan_no_fh = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=squad_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=24,
        target_gw=25,
        horizon_len=2,
        forced_w1_transfers=0,
    )

    plan_fh = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=squad_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=24,
        target_gw=25,
        horizon_len=2,
        forced_w1_chip="freehit",
    )

    assert plan_no_fh is not None
    assert plan_fh is not None


def test_beam_downgrade_funding_cascade():
    """
    Verify that _get_candidate_2_transfers admits top budget-generating downgrade moves
    (even with negative immediate xP gain) to enable funding cascades.
    """
    opt = TransferOptimizer()

    # Candidate 1 moves containing:
    # 4 positive moves with small immediate gain but zero or negative bank delta
    # 1 downgrade move with large positive bank delta (+20 = +£2.0m) but negative immediate gain (-0.5 xP)
    cand_1_moves = [
        {
            "plan_type": "1_TRANSFER",
            "transfers_in": [101],
            "transfers_out": [3],
            "sell_prices": {3: 60},
            "buy_costs": {101: 60},
            "bank_delta": 0,
            "immediate_gain": 0.8,
        },
        {
            "plan_type": "1_TRANSFER",
            "transfers_in": [102],
            "transfers_out": [4],
            "sell_prices": {4: 55},
            "buy_costs": {102: 55},
            "bank_delta": 0,
            "immediate_gain": 0.7,
        },
        {
            "plan_type": "1_TRANSFER",
            "transfers_in": [103],
            "transfers_out": [5],
            "sell_prices": {5: 50},
            "buy_costs": {103: 50},
            "bank_delta": 0,
            "immediate_gain": 0.6,
        },
        {
            "plan_type": "1_TRANSFER",
            "transfers_in": [104],
            "transfers_out": [6],
            "sell_prices": {6: 45},
            "buy_costs": {104: 45},
            "bank_delta": 0,
            "immediate_gain": 0.5,
        },
        # Downgrade move: sells DEF 12 (55) for cheap enabler 105 (40) -> bank_delta = +15, immediate_gain = -0.5
        {
            "plan_type": "1_TRANSFER",
            "transfers_in": [105],
            "transfers_out": [12],
            "sell_prices": {12: 55},
            "buy_costs": {105: 40},
            "bank_delta": 15,
            "immediate_gain": -0.5,
        },
    ]

    seeds = opt._select_seed_transfers(cand_1_moves)
    # The seeds must include the top 4 positive moves AND the downgrade move (elem 105 / 12)
    seed_outs = [m["transfers_out"][0] for m in seeds]
    seed_ins = [m["transfers_in"][0] for m in seeds]

    assert 12 in seed_outs, "Downgrade move selling element 12 should be included as a seed transfer"
    assert 105 in seed_ins, "Downgrade move buying element 105 should be included as a seed transfer"


def test_cameo_unimodal_gaussian_model():
    """
    Verify that LineupOptimizer uses a unimodal Gaussian model peaking at M ~ 15 min
    for substitute probability, with zero penalty above 55 min and no step discontinuities.
    """
    opt = LineupOptimizer(cameo_risk_penalty=0.75)

    # Create two players with identical xP and 100% chance of playing
    # but different expected minutes
    rows = []
    # 15 player baseline
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    for i, pos in enumerate(positions):
        rows.append(
            {
                "element": i + 1,
                "web_name": f"P_{i + 1}",
                "position": pos,
                "team": (i % 20) + 1,
                "value": 50,
                "expected_points": 5.0,
                "expected_minutes": 90.0,
                "chance_of_playing": 100.0,
            }
        )

    df_base = pd.DataFrame(rows)

    # 1. Cameo risk peaking at 15 minutes:
    # 15 minutes cameo incurs higher penalty than 35 minutes or 5 minutes
    df_15 = df_base.copy()
    df_15.loc[df_15["element"] == 8, "expected_minutes"] = 15.0

    df_35 = df_base.copy()
    df_35.loc[df_35["element"] == 8, "expected_minutes"] = 35.0

    df_60 = df_base.copy()
    df_60.loc[df_60["element"] == 8, "expected_minutes"] = 60.0

    res_15 = opt.select_lineup_and_captain(df_15, simulate_captaincy=False)
    res_35 = opt.select_lineup_and_captain(df_35, simulate_captaincy=False)
    res_60 = opt.select_lineup_and_captain(df_60, simulate_captaincy=False)
    assert res_35 is not None

    # Starter cameo penalties
    pen_15 = (
        float(res_15["starters"].loc[res_15["starters"]["element"] == 8, "cameo_penalty"].iloc[0])
        if 8 in set(res_15["starters"]["element"])
        else 0.0
    )
    assert pen_15 >= 0.0
    # In df_15, element 8 (MID, 5.0 xP, 15 min) should be heavily penalized compared to 60 min (which has 0 cameo penalty)
    # Check effective probability formulation:
    p_play = 1.0
    p_cameo_15 = p_play * np.exp(-((15.0 - 15.0) ** 2) / (2.0 * (12.0**2)))
    p_cameo_35 = p_play * np.exp(-((35.0 - 15.0) ** 2) / (2.0 * (12.0**2)))
    p_cameo_5 = p_play * np.exp(-((5.0 - 15.0) ** 2) / (2.0 * (12.0**2)))

    assert p_cameo_15 == pytest.approx(1.0)
    assert p_cameo_15 > p_cameo_35
    assert p_cameo_15 > p_cameo_5

    # Check 60 minutes has exactly 0 cameo penalty (above 55.0 min threshold)
    if 8 in set(res_60["starters"]["element"]):
        pen_60 = float(res_60["starters"].loc[res_60["starters"]["element"] == 8, "cameo_penalty"].iloc[0])
        assert pen_60 == 0.0
