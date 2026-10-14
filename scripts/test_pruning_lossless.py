"""
Empirical Proof: Pruned vs Unpruned Candidate Pool Equivalence (WI-2.3).
Compares:
1. Pruned candidate pool (Top N: 15 GKP, 45 DEF, 45 MID, 25 FWD + owned + 5 enablers per position)
2. Unpruned full pool (All active players in the pool)
Across 5 real-squad instances from 2026-27 across H=8, Wildcard, and Free Hit.
"""

import time

import pandas as pd

from fpl_oracle.data.historical import historical_manager
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def run_pruning_test():
    df_raw = historical_manager.ensure_dataset_ready()

    # 5 real instances from 2026-27
    test_instances = [
        ("2026-27 GW1", 1, 8, []),
        ("2026-27 GW2", 2, 8, ["wildcard"]),
        ("2026-27 GW3", 3, 8, ["freehit"]),
        ("2026-27 GW4", 4, 8, ["wildcard"]),
        ("2026-27 GW5", 5, 8, ["freehit"]),
    ]

    print("| GW Instance | Horizon | Chips | Pruned Pool Size | Unpruned Pool Size | Pruned Obj | Unpruned Obj | Diff | Pruned Time (s) | Unpruned Time (s) | Lossless? |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for name, gw, H, chips in test_instances:
        df_gw = df_raw[(df_raw["season"] == "2026-27") & (df_raw["round"] == gw)].copy()
        if df_gw.empty:
            continue

        gkps = df_gw[df_gw["position"] == "GKP"].head(2)
        defs = df_gw[df_gw["position"] == "DEF"].head(5)
        mids = df_gw[df_gw["position"] == "MID"].head(5)
        fwds = df_gw[df_gw["position"] == "FWD"].head(3)
        squad_df = pd.concat([gkps, defs, mids, fwds]).copy().reset_index(drop=True)

        squad_df["element"] = squad_df["element"].astype(int)
        squad_df["selling_price"] = squad_df["value"].astype(float) / 10.0
        squad_df["purchase_price"] = squad_df["value"].astype(float) / 10.0

        pool_df = df_gw.copy()
        pool_df["element"] = pool_df["element"].astype(int)
        pool_df["expected_points"] = pool_df["total_points"].astype(float)
        pool_df["value"] = pool_df["value"].astype(float)

        horizon_projections = {t: pool_df for t in range(gw, gw + H)}

        cfg_pruned = MultiPeriodConfig(horizon_len=H, solver_name="highs", time_limit_sec=60)
        cfg_unpruned = MultiPeriodConfig(
            horizon_len=H,
            solver_name="highs",
            time_limit_sec=60,
            pruning_limits={"GKP": 200, "DEF": 500, "MID": 500, "FWD": 200},
            enablers_count=50,
        )

        opt_pruned = MultiPeriodOptimizer(config=cfg_pruned)
        opt_unpruned = MultiPeriodOptimizer(config=cfg_unpruned)

        # Pruned run
        t0 = time.perf_counter()
        plan_p = opt_pruned.optimize(
            current_squad_df=squad_df,
            player_pool_df=pool_df,
            bank=1.0,
            free_transfers=1,
            horizon_projections=horizon_projections,
            current_gw=gw,
            target_gw=gw,
            horizon_len=H,
            chips_available=chips,
            forced_w1_transfers=0,
        )
        t_p = time.perf_counter() - t0

        # Unpruned run
        t0 = time.perf_counter()
        plan_u = opt_unpruned.optimize(
            current_squad_df=squad_df,
            player_pool_df=pool_df,
            bank=1.0,
            free_transfers=1,
            horizon_projections=horizon_projections,
            current_gw=gw,
            target_gw=gw,
            horizon_len=H,
            chips_available=chips,
            forced_w1_transfers=0,
        )
        t_u = time.perf_counter() - t0

        diff = abs(plan_p.total_objective - plan_u.total_objective)
        is_lossless = diff < 1e-3

        pruned_pool_len = len(opt_pruned.prune_candidates(pool_df, set(squad_df["element"]), list(horizon_projections.keys()), horizon_projections))
        unpruned_pool_len = len(pool_df)

        print(
            f"| {name} | {H} | {chips} | {pruned_pool_len} | {unpruned_pool_len} | "
            f"{plan_p.total_objective:.2f} | {plan_u.total_objective:.2f} | "
            f"{diff:.4f} | {t_p:.2f}s | {t_u:.2f}s | **{'YES' if is_lossless else 'NO'}** |"
        )


if __name__ == "__main__":
    run_pruning_test()
