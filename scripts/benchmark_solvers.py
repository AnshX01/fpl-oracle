"""
Empirical Benchmark: HiGHS vs CBC Solver Performance Investigation (WI-2.4).
Measures:
1. Problem construction / build time (Python & solver bridge)
2. Pure solver execution time (MIP branch-and-bound)
3. Total wall-clock time
4. Solution objective equivalence and optimality
Across 5 instances:
- H=3, No Chips
- H=5, No Chips
- H=8, No Chips
- H=5, Wildcard
- H=8, All Chips
"""

import time

import pandas as pd

from fpl_oracle.data.historical import historical_manager
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def run_benchmarks():
    df_raw = historical_manager.ensure_dataset_ready()
    s26_gw1 = df_raw[(df_raw["season"] == "2025-26") & (df_raw["round"] == 1)].copy()

    # Create a realistic 15-player squad (2 GKP, 5 DEF, 5 MID, 3 FWD) with valid team distribution
    gkps = s26_gw1[s26_gw1["position"] == "GKP"].head(2)
    defs = s26_gw1[s26_gw1["position"] == "DEF"].head(5)
    mids = s26_gw1[s26_gw1["position"] == "MID"].head(5)
    fwds = s26_gw1[s26_gw1["position"] == "FWD"].head(3)
    squad_df = pd.concat([gkps, defs, mids, fwds]).copy().reset_index(drop=True)

    squad_df["element"] = squad_df["element"].astype(int)
    squad_df["selling_price"] = squad_df["value"].astype(float) / 10.0
    squad_df["purchase_price"] = squad_df["value"].astype(float) / 10.0

    pool_df = s26_gw1.copy()
    pool_df["element"] = pool_df["element"].astype(int)
    pool_df["expected_points"] = pool_df["total_points"].astype(float)
    pool_df["value"] = pool_df["value"].astype(float)

    instances = [
        ("H=3, No Chips", 3, []),
        ("H=5, No Chips", 5, []),
        ("H=8, No Chips", 8, []),
        ("H=5, Wildcard", 5, ["wildcard"]),
        ("H=8, All Chips", 8, ["wildcard", "freehit", "bboost", "3xc"]),
    ]

    results = []

    print("| Instance | Horizon | Chips | Solver | Build Time (s) | Solve Time (s) | Total Time (s) | Objective | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for name, H, chips in instances:
        horizon_projections = {t: pool_df for t in range(1, 1 + H)}

        for solver_name in ["cbc", "highs"]:
            cfg = MultiPeriodConfig(horizon_len=H, solver_name=solver_name)
            opt = MultiPeriodOptimizer(config=cfg)

            t_start = time.perf_counter()
            plan = opt.optimize(
                current_squad_df=squad_df,
                player_pool_df=pool_df,
                bank=1.5,
                free_transfers=1,
                horizon_projections=horizon_projections,
                current_gw=1,
                target_gw=1,
                chips_available=chips,
                forced_w1_transfers=0,
            )
            t_total = time.perf_counter() - t_start

            # Solve time recorded inside plan
            t_solve = plan.solve_time_sec
            t_build = max(0.0, round(t_total - t_solve, 3))

            print(f"| {name} | {H} | {chips} | **{solver_name.upper()}** | {t_build:.3f} | {t_solve:.3f} | {t_total:.3f} | {plan.total_objective:.2f} | {plan.solver_status} |")
            results.append({
                "name": name,
                "H": H,
                "chips": chips,
                "solver": solver_name,
                "build_time": t_build,
                "solve_time": t_solve,
                "total_time": round(t_total, 3),
                "objective": plan.total_objective,
            })

if __name__ == "__main__":
    run_benchmarks()
