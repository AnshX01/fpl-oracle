import json
from pathlib import Path

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
from fpl_oracle.ml.quantiles import calc_pinball_loss


def run_c5_active_holdout_eval():
    print("=" * 80)
    print("C5: 2026-27 GW1-6 FORWARD HOLDOUT ON ACTIVE PLAYERS (MINUTES > 0)")
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

    # Calibrate on 2025-26
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

    # Baselines on all eval rows
    b_recent = model_evaluator.compute_baseline_projections(X_eval)
    b_season = model_evaluator.compute_season_avg_baseline(X_eval)
    b_fixture = model_evaluator.compute_fixture_adjusted_baseline(X_eval)

    # Attach all to a unified dataframe
    df_eval = pd.DataFrame(
        {
            "element": meta_eval["element"].values,
            "round": meta_eval["round"].values,
            "position": meta_eval["position"].values,
            "actual_minutes": Y_eval["target_minutes"].values,
            "actual_points": Y_eval["target_points"].values,
            "ml_xp": preds_df["expected_points"].values,
            "p10": preds_df["p10"].values,
            "p90": preds_df["p90"].values,
            "b_recent": b_recent,
            "b_season": b_season,
            "b_fixture": b_fixture,
        }
    )

    # Filter to ACTIVE players only (minutes > 0)
    df_active = df_eval[df_eval["actual_minutes"] > 0].copy()
    print(f"Total 2026-27 holdout rows: {len(df_eval)}")
    print(f"Total Active player observations (minutes > 0): {len(df_active)}")
    print(f"Position breakdown of active players: {df_active['position'].value_counts().to_dict()}")

    positions = ["GKP", "DEF", "MID", "FWD", "ALL"]
    report_rows = []

    for pos in positions:
        sub = df_active if pos == "ALL" else df_active[df_active["position"] == pos]
        n = len(sub)
        y = sub["actual_points"].values
        ml_p = sub["ml_xp"].values
        b_rec_p = sub["b_recent"].values
        b_sea_p = sub["b_season"].values
        b_fix_p = sub["b_fixture"].values
        p10 = sub["p10"].values
        p90 = sub["p90"].values

        # MAEs
        ml_mae = float(np.round(mean_absolute_error(y, ml_p), 3))
        b_rec_mae = float(np.round(mean_absolute_error(y, b_rec_p), 3))
        b_sea_mae = float(np.round(mean_absolute_error(y, b_sea_p), 3))
        b_fix_mae = float(np.round(mean_absolute_error(y, b_fix_p), 3))

        # RMSEs
        ml_rmse = float(np.round(root_mean_squared_error(y, ml_p), 3))
        b_rec_rmse = float(np.round(root_mean_squared_error(y, b_rec_p), 3))
        b_sea_rmse = float(np.round(root_mean_squared_error(y, b_sea_p), 3))
        b_fix_rmse = float(np.round(root_mean_squared_error(y, b_fix_p), 3))

        # Spearmans
        ml_sp = float(np.round(spearmanr(ml_p, y)[0], 3))
        b_rec_sp = float(np.round(spearmanr(b_rec_p, y)[0], 3))
        b_sea_sp = float(np.round(spearmanr(b_sea_p, y)[0], 3))
        b_fix_sp = float(np.round(spearmanr(b_fix_p, y)[0], 3))

        # 80% Coverage & Losses
        cov = float(np.round(np.mean((y >= p10) & (y <= p90)) * 100.0, 2))
        width = float(np.round(np.mean(p90 - p10), 2))
        pb10 = float(np.round(calc_pinball_loss(y, p10, 0.10), 4))
        pb90 = float(np.round(calc_pinball_loss(y, p90, 0.90), 4))

        report_rows.append(
            {
                "position": pos,
                "sample_count": n,
                "ml_mae": ml_mae,
                "b_recent_mae": b_rec_mae,
                "b_season_mae": b_sea_mae,
                "b_fixture_mae": b_fix_mae,
                "ml_rmse": ml_rmse,
                "b_recent_rmse": b_rec_rmse,
                "b_season_rmse": b_sea_rmse,
                "b_fixture_rmse": b_fix_rmse,
                "ml_spearman": ml_sp,
                "b_recent_spearman": b_rec_sp,
                "b_season_spearman": b_sea_sp,
                "b_fixture_spearman": b_fix_sp,
                "coverage_80_pct": cov,
                "mean_interval_width": width,
                "pinball_loss_p10": pb10,
                "pinball_loss_p90": pb90,
            }
        )

    out_file = Path("reports/c5_active_holdout_metrics.json")
    out_file.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable metrics to {out_file}")

    # Format Markdown Table
    print("\n" + "=" * 120)
    print("2026-27 GW1-6 ACTIVE PLAYERS (MINUTES > 0) EVALUATION TABLE")
    print("=" * 120)
    header = f"{'Pos':<6} {'N':<6} | {'ML MAE':<7} {'Rec MAE':<8} {'Sea MAE':<8} {'Fix MAE':<8} | {'ML RMSE':<8} {'Rec RMSE':<9} {'Sea RMSE':<9} {'Fix RMSE':<9} | {'ML Sp':<6} {'Rec Sp':<7} {'Sea Sp':<7} {'Fix Sp':<7} | {'Cov 80%':<8} {'Width':<6}"
    print(header)
    print("-" * 120)
    for r in report_rows:
        line = (
            f"{r['position']:<6} {r['sample_count']:<6} | "
            f"{r['ml_mae']:<7.3f} {r['b_recent_mae']:<8.3f} {r['b_season_mae']:<8.3f} {r['b_fixture_mae']:<8.3f} | "
            f"{r['ml_rmse']:<8.3f} {r['b_recent_rmse']:<9.3f} {r['b_season_rmse']:<9.3f} {r['b_fixture_rmse']:<9.3f} | "
            f"{r['ml_spearman']:<6.3f} {r['b_recent_spearman']:<7.3f} {r['b_season_spearman']:<7.3f} {r['b_fixture_spearman']:<7.3f} | "
            f"{r['coverage_80_pct']:<7.2f}% {r['mean_interval_width']:<6.2f}"
        )
        print(line)
    print("-" * 120)


if __name__ == "__main__":
    run_c5_active_holdout_eval()
