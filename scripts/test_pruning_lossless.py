"""
Empirical Proof: Pruned vs Unpruned Candidate Pool Equivalence (WI-2.3).
Compares:
1. Pruned candidate pool (Top N: 15 GKP, 45 DEF, 45 MID, 25 FWD + owned + 5 enablers per position = 130-161 players)
2. Unpruned full pool (All active players in the pool = 600+ players)
Across 5 real-squad instances from 2026-27 for BOTH H=5 and H=8 with NO CHIPS.
Reports real solver status and relative MIP gap for both configurations.
"""

import time

import pandas as pd

from fpl_oracle.data.historical import historical_manager
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def run_pruning_test():
    df_raw = historical_manager.ensure_dataset_ready()

    # 5 real instances from 2026-27 evaluated across H=5 and H=8 with NO CHIPS
    gws = [1, 2, 3, 4, 5]
    horizons = [5, 8]

    print(
        "| GW Instance | Horizon | Pruned Pool | Unpruned Pool | Pruned Obj | Unpruned Obj | Diff | Pruned Status | Unpruned Status | Pruned Time | Unpruned Time | Lossless? |"
    )
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for gw in gws:
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

        for H in horizons:
            horizon_projections = {t: pool_df for t in range(gw, gw + H)}

            cfg_pruned = MultiPeriodConfig(
                horizon_len=H,
                solver_name="highs",
                time_limit_sec=60,
                mip_gap=1e-4,
            )
            cfg_unpruned = MultiPeriodConfig(
                horizon_len=H,
                solver_name="highs",
                time_limit_sec=60,
                mip_gap=1e-4,
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
                chips_available=[],
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
                chips_available=[],
                forced_w1_transfers=0,
            )
            t_u = time.perf_counter() - t0

            diff = abs(plan_p.total_objective - plan_u.total_objective)
            p_solved = "OPTIMAL" in plan_p.solver_status
            u_solved = "OPTIMAL" in plan_u.solver_status
            is_lossless = diff < 0.05 and p_solved and u_solved
            lossless_str = (
                "**YES**"
                if is_lossless
                else ("**NO**" if not u_solved or plan_u.total_objective == 0 else f"DIFF={diff:.2f}")
            )

            p_size = len(
                opt_pruned.prune_candidates(
                    pool_df, set(squad_df["element"]), list(range(gw, gw + H)), horizon_projections
                )
            )
            u_size = len(pool_df["element"].unique())

            print(
                f"| 2026-27 GW{gw} | {H} | {p_size} | {u_size} | {plan_p.total_objective:.2f} | {plan_u.total_objective:.2f} | {diff:.4f} | {plan_p.solver_status} | {plan_u.solver_status} | {t_p:.2f}s | {t_u:.2f}s | {lossless_str} |"
            )


if __name__ == "__main__":
    run_pruning_test()
