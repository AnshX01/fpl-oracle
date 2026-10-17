"""
D6: Forward Holdout Statistics (2026-27 GW1-6 Active Players)
Computes per-gameweek MAE and Spearman gain vs best baseline,
gameweek-block bootstrap 95% confidence intervals, and documents
the sample size caveat.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import ScoringEnsemble
from fpl_oracle.ml.eval import model_evaluator


def compute_gw_block_bootstrap_gain_cis(
    df_active: pd.DataFrame, n_boot: int = 5000, random_state: int = 42
) -> dict[str, list[float]]:
    """Computes gameweek-block bootstrap 95% CIs for MAE gain and Spearman gain."""
    rng = np.random.default_rng(random_state)
    rounds = np.sort(df_active["round"].unique())
    n_blocks = len(rounds)

    boot_mae_gains = []
    boot_sp_gains = []

    for _ in range(n_boot):
        sample_rounds = rng.choice(rounds, size=n_blocks, replace=True)
        blocks = [df_active[df_active["round"] == r] for r in sample_rounds]
        sample_df = pd.concat(blocks, ignore_index=True)

        y = sample_df["actual_points"].values
        ml_p = sample_df["ml_xp"].values
        b_rec = sample_df["b_recent"].values
        b_sea = sample_df["b_season"].values
        b_fix = sample_df["b_fixture"].values

        ml_mae = float(mean_absolute_error(y, ml_p))
        rec_mae = float(mean_absolute_error(y, b_rec))
        sea_mae = float(mean_absolute_error(y, b_sea))
        fix_mae = float(mean_absolute_error(y, b_fix))
        best_base_mae = min(rec_mae, sea_mae, fix_mae)
        mae_gain = best_base_mae - ml_mae
        boot_mae_gains.append(mae_gain)

        ml_sp = float(spearmanr(ml_p, y)[0])
        rec_sp = float(spearmanr(b_rec, y)[0])
        sea_sp = float(spearmanr(b_sea, y)[0])
        fix_sp = float(spearmanr(b_fix, y)[0])
        best_base_sp = max(rec_sp, sea_sp, fix_sp)
        sp_gain = ml_sp - best_base_sp
        boot_sp_gains.append(sp_gain)

    return {
        "mae_gain_ci_95": [
            float(np.round(np.percentile(boot_mae_gains, 2.5), 4)),
            float(np.round(np.percentile(boot_mae_gains, 97.5), 4)),
        ],
        "spearman_gain_ci_95": [
            float(np.round(np.percentile(boot_sp_gains, 2.5), 4)),
            float(np.round(np.percentile(boot_sp_gains, 97.5), 4)),
        ],
    }


def run_d6_forward_holdout_eval():
    print("=" * 80)
    print("D6: 2026-27 GW1-6 FORWARD HOLDOUT PER-GAMEWEEK PERFORMANCE & BOOTSTRAP CI")
    print("=" * 80)

    # 1. Load data
    df = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    seasons = meta["season"]

    train_mask = seasons.isin(["2023-24", "2024-25"])
    cal_mask = seasons == "2025-26"
    eval_mask = seasons == "2026-27"

    X_train, Y_train = X[train_mask].copy(), Y[train_mask].copy()
    X_cal, Y_cal = X[cal_mask].copy(), Y[cal_mask].copy()
    X_eval, Y_eval = X[eval_mask].copy(), Y[eval_mask].copy()
    meta_eval = meta[eval_mask].copy()

    # 2. Fit component models strictly on pre-2026 data
    print("Fitting candidate models on pre-2026 data (zero leakage)...")
    models = {
        "minutes_model": MinutesModel(),
        "attacking_model": AttackingModel(),
        "defending_model": DefendingModel(),
        "defcon_model": DefConModel(),
        "bonus_model": BonusModel(),
        "cards_saves_model": CardsSavesModel(),
    }
    for m in models.values():
        m.fit(X_train, Y_train)

    comps_cal = {}
    for m in models.values():
        comps_cal.update(m.predict(X_cal))

    scoring_ensemble = ScoringEnsemble(blend_weights=(0.72, 0.04, 0.24))
    scoring_ensemble.fit_stacking_weights(comps_cal, X_cal, Y_cal)
    scoring_ensemble.calibrate(comps_cal, X_cal, Y_cal)

    # Predict on 2026-27
    comps_eval = {}
    for m in models.values():
        comps_eval.update(m.predict(X_eval))

    preds_df = scoring_ensemble.aggregate_components(comps_eval, X_eval)

    # Compute baseline projections
    b_recent = model_evaluator.compute_baseline_projections(X_eval)
    b_season = model_evaluator.compute_season_avg_baseline(X_eval)
    b_fixture = model_evaluator.compute_fixture_adjusted_baseline(X_eval)

    df_eval = pd.DataFrame(
        {
            "element": meta_eval["element"].values,
            "round": meta_eval["round"].values,
            "position": meta_eval["position"].values,
            "actual_minutes": Y_eval["target_minutes"].values,
            "actual_points": Y_eval["target_points"].values,
            "ml_xp": preds_df["expected_points"].values,
            "b_recent": b_recent,
            "b_season": b_season,
            "b_fixture": b_fixture,
        }
    )

    df_active = df_eval[df_eval["actual_minutes"] > 0].copy()
    total_active = len(df_active)
    rounds = sorted(df_active["round"].unique())

    print(f"\nTotal 2026-27 Active Observations: {total_active} across rounds {rounds}")

    per_gw_results: list[dict[str, Any]] = []

    print("\n--- PER-GAMEWEEK PERFORMANCE BREAKDOWN (ACTIVE PLAYERS) ---")
    print(
        f"{'GW':<4} | {'N':<5} | {'ML MAE':<7} | {'Best Base MAE':<13} | {'MAE Gain':<9} | {'ML RMSE':<8} | {'ML Spear':<8} | {'Best Base Spear':<15} | {'Spear Gain'}"
    )
    print("-" * 105)

    for r in rounds:
        sub = df_active[df_active["round"] == r]
        n_gw = len(sub)
        y = sub["actual_points"].values
        ml_p = sub["ml_xp"].values
        b_rec_p = sub["b_recent"].values
        b_sea_p = sub["b_season"].values
        b_fix_p = sub["b_fixture"].values

        ml_mae = float(np.round(mean_absolute_error(y, ml_p), 3))
        b_rec_mae = float(np.round(mean_absolute_error(y, b_rec_p), 3))
        b_sea_mae = float(np.round(mean_absolute_error(y, b_sea_p), 3))
        b_fix_mae = float(np.round(mean_absolute_error(y, b_fix_p), 3))
        best_base_mae = min(b_rec_mae, b_sea_mae, b_fix_mae)
        best_base_mae_name = (
            "Recent" if best_base_mae == b_rec_mae else ("Season" if best_base_mae == b_sea_mae else "Fixture")
        )
        mae_gain = float(np.round(best_base_mae - ml_mae, 3))

        ml_rmse = float(np.round(root_mean_squared_error(y, ml_p), 3))

        ml_sp = float(np.round(spearmanr(ml_p, y)[0], 3))
        b_rec_sp = float(np.round(spearmanr(b_rec_p, y)[0], 3))
        b_sea_sp = float(np.round(spearmanr(b_sea_p, y)[0], 3))
        b_fix_sp = float(np.round(spearmanr(b_fix_p, y)[0], 3))
        best_base_sp = max(b_rec_sp, b_sea_sp, b_fix_sp)
        best_base_sp_name = (
            "Recent" if best_base_sp == b_rec_sp else ("Season" if best_base_sp == b_sea_sp else "Fixture")
        )
        sp_gain = float(np.round(ml_sp - best_base_sp, 3))

        print(
            f"GW{r:<2} | {n_gw:<5} | {ml_mae:<7.3f} | {best_base_mae:<5.3f} ({best_base_mae_name:<7}) | {mae_gain:+7.3f}   | {ml_rmse:<8.3f} | {ml_sp:<8.3f} | {best_base_sp:<5.3f} ({best_base_sp_name:<7})   | {sp_gain:+8.3f}"
        )

        per_gw_results.append(
            {
                "gameweek": int(r),
                "sample_count": n_gw,
                "ml_mae": ml_mae,
                "best_baseline_mae": best_base_mae,
                "best_baseline_mae_type": best_base_mae_name,
                "mae_gain": mae_gain,
                "ml_rmse": ml_rmse,
                "ml_spearman": ml_sp,
                "best_baseline_spearman": best_base_sp,
                "best_baseline_spearman_type": best_base_sp_name,
                "spearman_gain": sp_gain,
                "all_baselines_mae": {
                    "recent": b_rec_mae,
                    "season": b_sea_mae,
                    "fixture": b_fix_mae,
                },
                "all_baselines_spearman": {
                    "recent": b_rec_sp,
                    "season": b_sea_sp,
                    "fixture": b_fix_sp,
                },
            }
        )

    # Pooled overall statistics
    y_tot = df_active["actual_points"].values
    ml_tot = df_active["ml_xp"].values
    b_rec_tot = df_active["b_recent"].values
    b_sea_tot = df_active["b_season"].values
    b_fix_tot = df_active["b_fixture"].values

    ml_mae_tot = float(np.round(mean_absolute_error(y_tot, ml_tot), 3))
    rec_mae_tot = float(np.round(mean_absolute_error(y_tot, b_rec_tot), 3))
    sea_mae_tot = float(np.round(mean_absolute_error(y_tot, b_sea_tot), 3))
    fix_mae_tot = float(np.round(mean_absolute_error(y_tot, b_fix_tot), 3))
    best_mae_tot = min(rec_mae_tot, sea_mae_tot, fix_mae_tot)
    best_mae_name_tot = (
        "Recent" if best_mae_tot == rec_mae_tot else ("Season" if best_mae_tot == sea_mae_tot else "Fixture")
    )
    mae_gain_tot = float(np.round(best_mae_tot - ml_mae_tot, 3))

    ml_rmse_tot = float(np.round(root_mean_squared_error(y_tot, ml_tot), 3))

    ml_sp_tot = float(np.round(spearmanr(ml_tot, y_tot)[0], 3))
    rec_sp_tot = float(np.round(spearmanr(b_rec_tot, y_tot)[0], 3))
    sea_sp_tot = float(np.round(spearmanr(b_sea_tot, y_tot)[0], 3))
    fix_sp_tot = float(np.round(spearmanr(b_fix_tot, y_tot)[0], 3))
    best_sp_tot = max(rec_sp_tot, sea_sp_tot, fix_sp_tot)
    best_sp_name_tot = "Recent" if best_sp_tot == rec_sp_tot else ("Season" if best_sp_tot == sea_sp_tot else "Fixture")
    sp_gain_tot = float(np.round(ml_sp_tot - best_sp_tot, 3))

    print("-" * 105)
    print(
        f"TOT | {total_active:<5} | {ml_mae_tot:<7.3f} | {best_mae_tot:<5.3f} ({best_mae_name_tot:<7}) | {mae_gain_tot:+7.3f}   | {ml_rmse_tot:<8.3f} | {ml_sp_tot:<8.3f} | {best_sp_tot:<5.3f} ({best_sp_name_tot:<7})   | {sp_gain_tot:+8.3f}"
    )

    # Gameweek-block bootstrap CIs
    print("\n--- GAMEWEEK-BLOCK BOOTSTRAP 95% CONFIDENCE INTERVALS (5,000 ITERATIONS) ---")
    boot_res = compute_gw_block_bootstrap_gain_cis(df_active, n_boot=5000)
    print(
        f"Pooled MAE Gain vs Best Baseline:      {mae_gain_tot:+.3f} pts | 95% Block Bootstrap CI: [{boot_res['mae_gain_ci_95'][0]:+.4f}, {boot_res['mae_gain_ci_95'][1]:+.4f}]"
    )
    print(
        f"Pooled Spearman Gain vs Best Baseline: {sp_gain_tot:+.3f}     | 95% Block Bootstrap CI: [{boot_res['spearman_gain_ci_95'][0]:+.4f}, {boot_res['spearman_gain_ci_95'][1]:+.4f}]"
    )

    # Sample size caveat
    sample_size_caveat = (
        "SAMPLE SIZE CAVEAT: The 2026-27 forward holdout comprises 6 early-season gameweeks (GW1-6) "
        "encompassing N=1,853 active player observations (mean ~308 active players/GW). While observation "
        "counts are statistically substantial for overall point estimates, temporal clustering at the gameweek "
        "level yields an effective sample size governed by B=6 gameweek blocks. Gameweek-level systemic shocks "
        "(e.g., gameweek fixture variance, tactical anomalies, referee cards) induce intra-gameweek correlation. "
        "Consequently, gameweek-block bootstrap confidence intervals are reported to account for block-level dependence."
    )
    print(f"\n{sample_size_caveat}")

    payload = {
        "evaluation_horizon": "2026-27 Gameweeks 1 through 6",
        "population": "Active players (actual_minutes > 0)",
        "total_active_observations": total_active,
        "gameweek_count": len(rounds),
        "per_gameweek_results": per_gw_results,
        "pooled_overall_metrics": {
            "sample_count": total_active,
            "ml_mae": ml_mae_tot,
            "best_baseline_mae": best_mae_tot,
            "best_baseline_mae_type": best_mae_name_tot,
            "mae_gain_vs_best_baseline": mae_gain_tot,
            "ml_rmse": ml_rmse_tot,
            "ml_spearman": ml_sp_tot,
            "best_baseline_spearman": best_sp_tot,
            "best_baseline_spearman_type": best_sp_name_tot,
            "spearman_gain_vs_best_baseline": sp_gain_tot,
        },
        "block_bootstrap_95_ci": boot_res,
        "sample_size_caveat": sample_size_caveat,
    }

    out_file = Path("reports/d6_holdout_stats.json")
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved machine artifact to {out_file}")
    print("\nD6 FORWARD HOLDOUT STATS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_d6_forward_holdout_eval()
