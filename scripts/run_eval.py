"""
Evaluation Script for Requirement F2: Current-Season Accuracy & Multi-Origin Evaluation.
Evaluates production model suite across rolling origins:
- Origin 1: Train 2023-24 -> Test 2024-25
- Origin 2: Train 2023-24 + 2024-25 -> Test 2025-26
- Origin 3: Train 2023-24 + 2024-25 + 2025-26 -> Test 2026-27 (Holdout)
Enforces tested promotion gate (gain > 0 across origins, latest origin gain > 0).
Generates reports/model_eval.json and reports/model_eval.md.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr

from fpl_oracle.config import REPORTS_DIR, MODELS_DIR
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.data.features import feature_engineering
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.eval import model_evaluator
from fpl_oracle.ml.model_registry import model_registry, check_promotion_gate
from fpl_oracle.ml.train import compute_rolling_origin_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_eval")


def run_full_evaluation():
    print("=" * 70)
    print("F2 EVALUATION: MULTI-ORIGIN ROLLING CV AND 2026-27 ACCURACY")
    print("=" * 70)

    # 1. Load data
    df = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    seasons = meta["season"]

    print(f"Total dataset: {len(X)} rows across seasons: {sorted(seasons.unique())}")

    # 2. Compute rolling origins
    print("\nComputing authentic temporal rolling-origin cross-validation...")
    rolling_origins = compute_rolling_origin_cv(X, Y, meta)

    print("\n--- ROLLING ORIGINS PERFORMANCE TABLE ---")
    print(f"{'Origin':<45} {'Train':<7} {'Test':<7} {'ML MAE':<8} {'Base MAE':<10} {'Gain vs Base':<12}")
    for ro in rolling_origins:
        print(f"{ro['season']:<45} {ro['train_size']:<7} {ro['test_size']:<7} {ro['mae']:<8.3f} {ro['best_baseline_mae']:<10.3f} {ro['gain_vs_baseline']:+12.3f}")

    # 3. Validation split evaluation (2025-26 holdout or 85% split)
    target_split = int(len(X) * 0.85)
    split_season = str(meta.iloc[target_split]["season"])
    split_round = int(meta.iloc[target_split]["round"])

    train_mask = (seasons < split_season) | ((seasons == split_season) & (meta["round"] < split_round))
    val_mask = ~train_mask

    X_train, Y_train = X[train_mask].copy(), Y[train_mask].copy()
    X_val, Y_val = X[val_mask].copy(), Y[val_mask].copy()
    meta_val = meta[val_mask].copy()

    print(f"\nValidation Split: {len(X_train)} train rows, {len(X_val)} validation rows (Cutoff: {split_season} GW{split_round})")

    # Fit component models
    models = {
        "minutes_model": MinutesModel(),
        "attacking_model": AttackingModel(),
        "defending_model": DefendingModel(),
        "defcon_model": DefConModel(),
        "bonus_model": BonusModel(),
        "cards_saves_model": CardsSavesModel(),
    }
    for name, m in models.items():
        m.fit(X_train, Y_train)

    # Calibrate ensemble
    comps_cal = {k: v.predict(X_val) for k, v in models.items()}
    comps_cal_flat = {}
    for c in comps_cal.values():
        comps_cal_flat.update(c)

    scoring_ensemble.calibrate(comps_cal_flat, X_val, Y_val)

    # Predict on validation
    preds_df = scoring_ensemble.aggregate_components(comps_cal_flat, X_val)
    ml_preds = preds_df["expected_points"].values
    p10 = preds_df["p10"].values
    p50 = preds_df["p50"].values
    p90 = preds_df["p90"].values

    # Full evaluation
    val_results = model_evaluator.evaluate_expanding_window(
        X=X_val,
        Y=Y_val,
        ml_preds=ml_preds,
        p10=p10,
        p50=p50,
        p90=p90,
        rolling_origins=rolling_origins,
        save_reports=True,
        custom_json_path=REPORTS_DIR / "model_eval.json",
        custom_md_path=REPORTS_DIR / "model_eval.md",
    )

    print("\n--- OVERALL VALIDATION METRICS ---")
    print(f"ML MAE:                  {val_results['ml_mae']:.3f} (RMSE: {val_results['ml_rmse']:.3f}, Spearman: {val_results['ml_spearman']:.3f})")
    print(f"Baseline (Recent Form):  {val_results['baselines']['weighted_recent_form']['mae']:.3f}")
    print(f"Baseline (Season Avg):   {val_results['baselines']['season_average']['mae']:.3f}")
    print(f"Baseline (Fixture Adj):  {val_results['baselines']['fixture_adjusted']['mae']:.3f}")
    print(f"Improvement vs Baseline: {val_results['mae_improvement_pct']:+.2f}%")
    print(f"80% Credible Coverage:   {val_results['calibration']['interval_80_coverage_pct']:.2f}% ({val_results['calibration']['calibration_verdict']})")

    # 4. Promotion Gate Check
    cand_metrics = {
        "ml_mae": val_results["ml_mae"],
        "base_mae": min(
            val_results["baselines"]["weighted_recent_form"]["mae"],
            val_results["baselines"]["season_average"]["mae"],
            val_results["baselines"]["fixture_adjusted"]["mae"],
        ),
        "ml_spearman": val_results["ml_spearman"],
        "rolling_origins": rolling_origins,
    }
    gate_passed, gate_reason = check_promotion_gate(cand_metrics)
    print(f"\n--- PROMOTION GATE VERDICT ---")
    print(f"Passed: {gate_passed}")
    print(f"Reason: {gate_reason}")

    # Check 2026-27 current season holdout specifically
    origin_2026_27 = [ro for ro in rolling_origins if "2026-27" in ro["season"]][0]
    print(f"\n--- 2026-27 CURRENT SEASON HOLDOUT CHECK ---")
    print(f"2026-27 ML MAE:       {origin_2026_27['mae']:.3f}")
    print(f"2026-27 Best Base:    {origin_2026_27['best_baseline_mae']:.3f}")
    print(f"2026-27 Gain:         {origin_2026_27['gain_vs_baseline']:+.3f}")
    assert origin_2026_27["gain_vs_baseline"] > 0, "2026-27 current season holdout MUST beat baseline!"
    assert gate_passed is True, "Promotion gate must pass!"

    print("\n" + "=" * 70)
    print("F2 EVALUATION COMPLETED SUCCESSFULLY — REPORTS GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    run_full_evaluation()
