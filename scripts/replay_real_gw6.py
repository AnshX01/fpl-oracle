"""
Replay real GW6 deadline state for manager 1085141 (Ansh Saxena, Red Djed Redemption)
in IceMakers League (ID: 435956) using multiperiod MILP optimizer.
"""

import asyncio
import logging

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import ManagerStateService
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer

logging.basicConfig(level=logging.WARNING)


async def run_gw6_replay():
    print("=" * 80)
    print("REAL GW6 REPLAY: Manager 1085141 (IceMakers League ID: 435956)")
    print("=" * 80)

    # 1. Fetch bootstrap and fixtures for projections
    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elements_map = {e.id: e for e in bootstrap.elements}
    teams_map = {t.id: t.name for t in bootstrap.teams}

    # 2. Historical data to determine exact purchase price entering GW6
    # Transfers prior to GW6:
    # GW1 initial: Kinsky(496: 45), Verbruggen(109: 45), Calafiori(8: 55), Maguire(418: 50),
    #              White(10: 55), Diop(259: 40), Mbeumo(427: 80), Szoboszlai(368: 70),
    #              Gross(124: 55), Haaland(411: 155), DCL(346: 60)
    # GW3 in: Ajer(87: 45)
    # GW4 in: Palmer(154: 97), Joao Pedro(165: 77)
    # GW5 in: Belloumi(286: 51)
    purchase_prices = {
        496: 45,  # Kinsky
        109: 45,  # Verbruggen
        8: 55,  # Calafiori
        418: 50,  # Maguire
        87: 45,  # Ajer
        10: 55,  # White
        259: 40,  # Diop
        427: 80,  # Mbeumo
        154: 97,  # Palmer
        368: 70,  # Szoboszlai
        286: 51,  # Belloumi
        124: 55,  # Gross
        411: 155,  # Haaland
        346: 60,  # Calvert-Lewin
        165: 77,  # Joao Pedro
    }

    # GW5 values (entering GW6 deadline)
    # Calafiori rose to 58 (sell: 55 + (3//2) = 56)
    # Maguire dropped to 49 (sell: 49)
    # Gross rose to 57 (sell: 55 + (2//2) = 56)
    # Haaland rose to 156 (sell: 155 + (1//2) = 155)
    # Joao Pedro rose to 78 (sell: 77 + (1//2) = 77)
    # Mbeumo dropped to 79 (sell: 79)
    # All others at par
    squad_eids = [496, 8, 418, 87, 427, 154, 368, 286, 124, 411, 346, 109, 165, 10, 259]
    gw5_market_costs = {
        496: 45,
        109: 45,
        8: 58,
        418: 49,
        87: 45,
        10: 55,
        259: 40,
        427: 79,
        154: 97,
        368: 70,
        286: 51,
        124: 57,
        411: 156,
        346: 60,
        165: 78,
    }

    # 3. Generate multi-gameweek projections (GW6 to GW13, H=8)
    projs = projection_engine.predict_multi_gameweeks(start_gw=6, horizon=8, bootstrap=bootstrap, fixtures=fixtures)
    gw6_pool = projs[6].copy()

    # Reconstruct exact squad dataframe entering GW6
    squad_rows = []
    print("\n--- SQUAD ENTERING GW6 DEADLINE ---")
    print(f"{'Player Name':<28} | {'Pos':<4} | {'Club':<14} | {'Buy':>5} | {'Curr':>5} | {'Sell':>5} | {'GW6 xP':>6}")
    print("-" * 80)
    for eid in squad_eids:
        elem = elements_map[eid]
        pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[elem.element_type]
        team_str = teams_map[elem.team]
        buy_p = purchase_prices[eid]
        now_c = gw5_market_costs[eid]
        sell_p = ManagerStateService.calculate_selling_price(buy_p, now_c)

        # Match with GW6 projection row
        p_row = gw6_pool[gw6_pool["element"] == eid]
        xp = float(p_row["expected_points"].iloc[0]) if not p_row.empty else 0.0

        squad_rows.append(
            {
                "element": eid,
                "web_name": elem.web_name,
                "name": f"{elem.first_name} {elem.second_name}",
                "position": pos_str,
                "team": elem.team,
                "value": now_c,
                "purchase_price": buy_p,
                "selling_price": sell_p,
                "expected_points": xp,
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
        print(
            f"{elem.web_name:<28} | {pos_str:<4} | {team_str:<14} | {buy_p / 10:>4.1f}m | {now_c / 10:>4.1f}m | {sell_p / 10:>4.1f}m | {xp:>6.2f}"
        )

    squad_df = pd.DataFrame(squad_rows)
    bank_tenths = 18.0  # £1.8m in bank
    free_transfers = 1
    chips_available = {"wildcard", "freehit", "3xc", "bboost"}

    tot_sell = squad_df["selling_price"].sum() / 10.0
    print("-" * 80)
    print(
        f"Total Squad Selling Value: £{tot_sell:.1f}m | Bank: £{bank_tenths / 10:.1f}m | Liquid Team Value: £{(tot_sell + bank_tenths / 10):.1f}m"
    )
    print(f"Available Free Transfers: {free_transfers} | Active Chips Available: {sorted(chips_available)}")

    # 4. Run MultiPeriodOptimizer
    optimizer = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=8, solver_name="highs", mip_gap=0.005))

    # A. Roll Plan (0 transfers)
    roll_plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=8,
        forced_w1_transfers=0,
    )

    # B. 1-Transfer Plan (force 1 transfer)
    plan_1 = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=8,
        forced_w1_transfers=1,
    )

    # C. Unconstrained Optimal Plan
    opt_plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=8,
        chips_available=chips_available,
    )

    # D. Force Wildcard Plan
    wc_plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=8,
        chips_available=chips_available,
        forced_w1_chip="wildcard",
    )

    # Helper to resolve player names
    def format_moves(act):
        outs = [
            f"{elements_map[t['element']].web_name} (£{t.get('selling_price', elements_map[t['element']].now_cost) / 10:.1f}m)"
            for t in act.transfers_out
        ]
        ins = [
            f"{elements_map[t['element']].web_name} (£{t.get('cost', elements_map[t['element']].now_cost) / 10:.1f}m)"
            for t in act.transfers_in
        ]
        return ", ".join(outs) + " -> " + ", ".join(ins) if outs or ins else "None (Roll)"

    print("\n" + "=" * 80)
    print("MULTI-PERIOD OPTIMIZER RESULTS (H=8, Gamma=0.95)")
    print("=" * 80)

    plans = [
        ("Roll Transfer (0 FT)", roll_plan),
        ("Best 1-Transfer (1 FT, 0 Hits)", plan_1),
        ("Unconstrained Optimal Plan", opt_plan),
        ("Forced Wildcard (Set 1)", wc_plan),
    ]

    for name, p in plans:
        act = p.recommended_action
        j_val = p.total_objective
        h_net = p.horizon_net_xp
        gain_vs_roll = round(j_val - roll_plan.total_objective, 2)
        imm_gain = round(act.net_xp - roll_plan.recommended_action.net_xp, 2)
        chip_used = act.active_chip or "None"
        moves_str = format_moves(act)
        print(f"\n[{name}]")
        print(f"  Plan Type:       {p.plan_type}")
        print(f"  Chip Decision:   {chip_used}")
        print(f"  Transfers:       {act.transfers_count} (Hits: {act.hits})")
        print(f"  Moves:           {moves_str}")
        print(f"  GW6 Lineup xP:   {act.lineup_xp} (Captain: {elements_map[act.captain].web_name})")
        print(f"  GW6 Net xP:      {act.net_xp} (Immediate Delta vs Roll: {imm_gain:+.2f} xP)")
        print(f"  Horizon Net xP:  {h_net:.2f}")
        print(f"  J(plan):         {j_val:.2f} (Delta J vs Roll: {gain_vs_roll:+.2f})")
        print(f"  Bank End:        £{act.bank_end / 10:.1f}m | FT Next: {act.free_transfers_next}")
        print(f"  Solve Time:      {p.solve_time_sec:.2f}s (Status: {p.solver_status})")

    # Evaluation using evaluate_transfer_options
    print("\n" + "=" * 80)
    print("FULL DECISION CARD / COMPARISON SET EVALUATION")
    print("=" * 80)
    eval_res = optimizer.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        chips_available=chips_available,
        risk_preference="balanced",
        mini_league_mode="chase_deficit",
        mini_league_deficit=26.0,
    )

    rec = eval_res["recommended_plan"]
    print(f"RECOMMENDED PLAN TYPE: {rec['plan_type']}")
    print(f"CHIP RECOMMENDATION:   {rec.get('chip') or 'NO CHIP (Preserve Wildcard)'}")
    print(f"TRANSFERS COUNT:       {rec['transfers_count']} (Hits: {rec['hits']})")
    print(f"TRANSFERS IN:          {[elements_map[t['element']].web_name for t in rec['transfers_in']]}")
    print(f"TRANSFERS OUT:         {[elements_map[t['element']].web_name for t in rec['transfers_out']]}")
    print(f"HIT VERDICT:           {eval_res['hit_verdict']}")
    print(f"J(plan) vs J(roll):    +{rec['horizon_gain_vs_roll']:.2f} xP gain over rolling")
    print(f"ROBUSTNESS WIN RATE:   {rec.get('robustness_score', 0.0) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("ACTUAL USER TRANSFER PERFORMED IN GW6:")
    print("  OUT: Bryan Mbeumo (£7.9m) -> IN: Bukayo Saka (£9.6m) [Cost: £9.6m, Bank left: £0.1m, Hits: 0]")
    print("=" * 80)


asyncio.run(run_gw6_replay())
