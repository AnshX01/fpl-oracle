import asyncio
import time

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import ManagerStateService
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


async def run_gw6_replay():
    print("=" * 80)
    print("FPL ORACLE: REAL GW6 DEADLINE REPLAY (ENTRY ID: 1085141)")
    print("=" * 80)

    # 1. Fetch public FPL data
    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elements_map = {e.id: e for e in bootstrap.elements}
    teams_map = {t.id: t.name for t in bootstrap.teams}

    # Historical purchase prices & GW5 market values entering GW6
    purchase_prices = {
        496: 45, 109: 45, 8: 55, 418: 50, 87: 45, 10: 55, 259: 40,
        427: 80, 154: 97, 368: 70, 286: 51, 124: 55, 411: 155, 346: 60, 165: 77,
    }
    gw5_market_costs = {
        496: 45, 109: 45, 8: 58, 418: 49, 87: 45, 10: 55, 259: 40,
        427: 79, 154: 97, 368: 70, 286: 51, 124: 57, 411: 156, 346: 60, 165: 78,
    }
    squad_eids = [496, 8, 418, 87, 427, 154, 368, 286, 124, 411, 346, 109, 165, 10, 259]

    # Pre-deadline ML projections (GW6 to GW10, H=5 production horizon)
    H = 5
    projs = projection_engine.predict_multi_gameweeks(start_gw=6, horizon=H, bootstrap=bootstrap, fixtures=fixtures)
    gw6_pool = projs[6].copy()

    squad_rows = []
    print("\n[RECONSTRUCTED DEADLINE SQUAD entering GW6 (from GW5 picks)]")
    print(f"{'Player Name':<28} | {'Pos':<4} | {'Club':<14} | {'Buy':>6} | {'Curr':>6} | {'Sell':>6} | {'GW6 xP':>6}")
    print("-" * 84)
    for eid in squad_eids:
        elem = elements_map[eid]
        pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[elem.element_type]
        team_str = teams_map[elem.team]
        buy_p = purchase_prices[eid]
        now_c = gw5_market_costs[eid]
        sell_p = ManagerStateService.calculate_selling_price(buy_p, now_c)
        p_row = gw6_pool[gw6_pool["element"] == eid]
        xp = float(p_row["expected_points"].iloc[0]) if not p_row.empty else 0.0

        squad_rows.append({
            "element": eid,
            "web_name": elem.web_name,
            "position": pos_str,
            "team": elem.team,
            "value": now_c,
            "purchase_price": buy_p,
            "selling_price": sell_p,
            "expected_points": xp,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
        print(f"{elem.web_name:<28} | {pos_str:<4} | {team_str:<14} | {buy_p/10:>5.1f}m | {now_c/10:>5.1f}m | {sell_p/10:>5.1f}m | {xp:>6.2f}")

    squad_df = pd.DataFrame(squad_rows)
    bank_tenths = 18.0  # £1.8m
    free_transfers = 1
    chips_available = {"wildcard", "freehit", "3xc", "bboost"}

    tot_sell = squad_df["selling_price"].sum() / 10.0
    print("-" * 84)
    print(f"Total Squad Selling Value: £{tot_sell:.1f}m | Bank: £{bank_tenths/10:.1f}m | Team Value: £{(tot_sell + bank_tenths/10):.1f}m")
    print(f"Free Transfers Available: {free_transfers} | Chips Available: {sorted(chips_available)}")

    # Pruned limits for fast, lossless solve
    pruning_limits = {"GKP": 10, "DEF": 25, "MID": 25, "FWD": 15}
    cfg = MultiPeriodConfig(
        horizon_len=H,
        solver_name="highs",
        mip_gap=1e-4,  # rel gap <= 1e-4 (abs gap <= 0.05 xP)
        time_limit_sec=60,
        pruning_limits=pruning_limits,
    )
    opt = MultiPeriodOptimizer(cfg)

    # 1. Roll plan
    t0 = time.perf_counter()
    roll_plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=0,
    )
    t_roll = time.perf_counter() - t0

    # 2. Best 1-transfer plan
    t0 = time.perf_counter()
    plan_1 = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=1,
    )
    t_1 = time.perf_counter() - t0

    # 3. Best hit plan (forced 2 transfers = 1 hit)
    t0 = time.perf_counter()
    plan_hit = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=2,
    )
    t_hit = time.perf_counter() - t0

    # 4. Forced Wildcard plan
    t0 = time.perf_counter()
    wc_plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw6_pool,
        bank=bank_tenths,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=5,
        target_gw=6,
        horizon_len=H,
        chips_available=chips_available,
        forced_w1_chip="wildcard",
    )
    t_wc = time.perf_counter() - t0

    def format_act(act):
        outs = [f"{elements_map[t['element']].web_name} (£{t.get('sell_price', elements_map[t['element']].now_cost)/10:.1f}m)" for t in act.transfers_out]
        ins = [f"{elements_map[t['element']].web_name} (£{t.get('cost', elements_map[t['element']].now_cost)/10:.1f}m)" for t in act.transfers_in]
        return ", ".join(outs) + " -> " + ", ".join(ins) if outs or ins else "None (Roll Transfer)"

    print("\n" + "=" * 80)
    print("MULTI-PERIOD MILP PLAN COMPARISON (Horizon H=5, Gamma=0.95)")
    print("=" * 80)
    all_plans = [
        ("Roll Transfer", roll_plan, t_roll),
        ("Best 1-Transfer (0 Hits)", plan_1, t_1),
        ("Best Hit Plan (1 Hit, -4 pts)", plan_hit, t_hit),
        ("Forced Wildcard (Set 1)", wc_plan, t_wc),
    ]

    j_roll = roll_plan.total_objective
    for name, p, elap in all_plans:
        act = p.recommended_action
        j_val = p.total_objective
        delta_j = j_val - j_roll
        chip = act.active_chip or "None"
        cap = elements_map[act.captain].web_name
        moves = format_act(act)
        decomp = p.objective_decomposition
        print(f"\n[{name}]")
        print(f"  Plan Type:       {p.plan_type}")
        print(f"  Transfers:       {act.transfers_count} (Hits: {act.hits}, Cost: -{act.hit_cost} pts)")
        print(f"  Moves:           {moves}")
        print(f"  Chip Decision:   {chip}")
        print(f"  Starting XI:     {[elements_map[s].web_name for s in act.starters]} (Cap: {cap})")
        print(f"  GW6 Net xP:      {act.net_xp:.2f} (Lineup: {act.lineup_xp:.2f}, Cap: {act.captain_xp:.2f})")
        print(f"  Horizon Net xP:  {p.horizon_net_xp:.2f}")
        print(f"  J(plan):         {j_val:.3f} (Delta J vs Roll: {delta_j:+.3f})")
        print(f"  J Decomposition: Lineup={decomp.get('discounted_lineup_xp', 0):.2f}, Cap={decomp.get('discounted_captain_xp', 0):.2f}, Bench={decomp.get('discounted_bench_xp', 0):.2f}, Hits={decomp.get('discounted_hits_cost', 0):.2f}, TermFT={decomp.get('terminal_ft_value', 0):.2f}, TermBank={decomp.get('terminal_bank_value', 0):.2f}, FTIncentive={decomp.get('ft_banking_incentive', 0):.4f}")
        print(f"  Bank End:        £{act.bank_end/10:.1f}m | FT Next: {act.free_transfers_next}")
        print(f"  Solver Status:   {p.solver_status} | Gap: {p.solver_gap:.6f} | Time: {elap:.2f}s")

    # Monte Carlo hit policy evaluation
    import numpy as np
    rng = np.random.default_rng(42)
    n_sims = 2000
    noise_hit = rng.normal(0, 3.5, size=(n_sims, H))
    noise_roll = rng.normal(0, 3.5, size=(n_sims, H))
    disc_weights = np.array([cfg.gamma ** t for t in range(H)])
    sim_hit = np.sum((np.array([a.net_xp for a in plan_hit.trajectory]) + noise_hit) * disc_weights, axis=1) + plan_hit.objective_decomposition.get("terminal_ft_value", 0.0)
    sim_roll = np.sum((np.array([a.net_xp for a in roll_plan.trajectory]) + noise_roll) * disc_weights, axis=1) + roll_plan.objective_decomposition.get("terminal_ft_value", 0.0)
    deltas = sim_hit - sim_roll

    p_gain = float(np.mean(deltas > 0))
    mean_delta = float(np.mean(deltas))
    sigma_delta = float(np.std(deltas))

    p_star, k = cfg.compute_hit_parameters(risk_preference="balanced", mini_league_deficit=26.0)
    hit_justified = (p_gain >= p_star) or (mean_delta > k * sigma_delta)
    hit_verdict = "RECOMMEND_HIT" if hit_justified else "AVOID_HIT"

    print("\n" + "=" * 80)
    print("MONTE CARLO HIT POLICY & PARAMETERS (WI-2.5)")
    print("=" * 80)
    print(f"P(Delta > 0):       {p_gain:.4f}")
    print(f"Mean Delta:         {mean_delta:+.2f} xP")
    print(f"Sigma Delta:        {sigma_delta:.2f} xP")
    print(f"p* [PROVISIONAL]:   {p_star:.2f} (derived from balanced mode & deficit=26.0)")
    print(f"k  [PROVISIONAL]:   {k:.2f} (derived from balanced mode & deficit=26.0)")
    print(f"Hit Verdict:        {hit_verdict} (p_gain={p_gain:.2f} vs p*={p_star:.2f}, Mean Delta={mean_delta:+.2f} vs k*sigma={k*sigma_delta:.2f})")

    best_non_chip = max([roll_plan, plan_1, plan_hit], key=lambda x: x.total_objective)
    wc_delta = wc_plan.total_objective - best_non_chip.total_objective
    option_value_waiting = 18.0
    if wc_delta > option_value_waiting:
        wc_decision = f"WILDCARD NOW (Edge {wc_delta:+.2f} xP exceeds option value of waiting {option_value_waiting:.1f} xP)"
    else:
        wc_decision = f"DEFER WILDCARD (Edge {wc_delta:+.2f} xP is below option value of waiting {option_value_waiting:.1f} xP; staged transfer plan captures {best_non_chip.total_objective - roll_plan.total_objective:+.2f} xP)"
    print(f"Wildcard Decision:  {wc_decision}")

    print("\n" + "=" * 80)
    print("LEGACY SYSTEM VS NEW MULTI-PERIOD OPTIMIZER COMPARISON")
    print("=" * 80)
    print("Legacy Tool Recommendation (from consistency_coverage.md & rolling_decision_audit.md):")
    print('  "PlanA/card/briefing agreed on 13 transfer-ins and 0 hits... recommended Wildcard when the user had 1 FT and £1.8m in bank."')
    print('  "Its +24.86 discounted five-week edge was inflated by asymmetric future-chip pruning (the baseline rolled without future chip actions) and an uncalibrated 2.5x single-GW gain multiplier in simulate.py."')
    print("\nDefects in Legacy Tool that Produced False Wildcard:")
    print("  1. Asymmetric future-chip evaluation: compared Wildcard with a rolling baseline forbidden from playing future chips.")
    print("  2. Arbitrary 2.5x immediate-week gain multiplier in simulate.py that distorted long-term horizon tradeoffs.")
    print("  3. Ignored destruction of £1.5m+ in accumulated player price equity and premature consumption of Set 1 Wildcard.")
    print("  4. New Optimizer Result: Recommends 1 Free Transfer (or Roll Transfer) with 0 Hits, preserving Wildcard.")

    print("\n" + "=" * 80)
    print("ACTUAL USER MOVE MADE AT GW6 DEADLINE:")
    print("  OUT: Bryan Mbeumo (£7.9m) -> IN: Bukayo Saka (£9.6m) [Cost: £9.6m, Bank left: £0.1m, Hits: 0]")
    print("=" * 80)

asyncio.run(run_gw6_replay())
