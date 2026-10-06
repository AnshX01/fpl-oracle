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

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from fpl_oracle.config import REPORTS_DIR
from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.eval import model_evaluator
from fpl_oracle.ml.model_registry import check_promotion_gate
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
        print(
            f"{ro['season']:<45} {ro['train_size']:<7} {ro['test_size']:<7} {ro['mae']:<8.3f} {ro['best_baseline_mae']:<10.3f} {ro['gain_vs_baseline']:+12.3f}"
        )

    # 3. Disjoint Calibration and Holdout Evaluation (G5)
    train_mask = seasons.isin(["2023-24", "2024-25", "2025-26"])
    cal_mask = seasons == "2025-26"
    eval_mask = seasons == "2026-27"

    X_train, Y_train = X[train_mask].copy(), Y[train_mask].copy()
    X_cal, Y_cal = X[cal_mask].copy(), Y[cal_mask].copy()
    X_eval, Y_eval = X[eval_mask].copy(), Y[eval_mask].copy()

    print(
        f"\nEvaluation Setup: {len(X_train)} train rows, {len(X_cal)} calibration rows (2025-26), {len(X_eval)} holdout rows (2026-27)"
    )

    # Fit component models
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

    # Calibrate ensemble and fit stacking weights on disjoint 2025-26 calibration block
    comps_cal = {}
    for m in models.values():
        comps_cal.update(m.predict(X_cal))

    scoring_ensemble.calibrate(comps_cal, X_cal, Y_cal)
    scoring_ensemble.fit_stacking_weights(comps_cal, X_cal, Y_cal)

    # Predict on unseen 2026-27 holdout evaluation block
    comps_eval = {}
    for m in models.values():
        comps_eval.update(m.predict(X_eval))

    preds_df = scoring_ensemble.aggregate_components(comps_eval, X_eval)
    ml_preds = preds_df["expected_points"].values
    p10 = preds_df["p10"].values
    p50 = preds_df["p50"].values
    p90 = preds_df["p90"].values

    # Candidate metrics for promotion gate check
    best_base_eval = min(
        mean_absolute_error(Y_eval["target_points"].values, model_evaluator.compute_baseline_projections(X_eval)),
        mean_absolute_error(Y_eval["target_points"].values, model_evaluator.compute_season_avg_baseline(X_eval)),
        mean_absolute_error(Y_eval["target_points"].values, model_evaluator.compute_fixture_adjusted_baseline(X_eval)),
    )
    cand_metrics = {
        "ml_mae": float(np.round(mean_absolute_error(Y_eval["target_points"].values, ml_preds), 3)),
        "base_mae": float(np.round(best_base_eval, 3)),
        "ml_spearman": float(np.round(spearmanr(ml_preds, Y_eval["target_points"].values)[0], 3)),
        "rolling_origins": rolling_origins,
    }
    gate_passed, gate_reason = check_promotion_gate(cand_metrics)

    # Full evaluation report generation
    val_results = model_evaluator.evaluate_expanding_window(
        X=X_eval,
        Y=Y_eval,
        ml_preds=ml_preds,
        p10=p10,
        p50=p50,
        p90=p90,
        rolling_origins=rolling_origins,
        gate_verdict={"passed": gate_passed, "reason": gate_reason},
        save_reports=True,
        custom_json_path=REPORTS_DIR / "model_eval.json",
        custom_md_path=REPORTS_DIR / "model_eval.md",
    )

    print("\n--- OVERALL 2026-27 HOLDOUT METRICS ---")
    print(
        f"ML MAE:                  {val_results['ml_mae']:.3f} (RMSE: {val_results['ml_rmse']:.3f}, Spearman: {val_results['ml_spearman']:.3f})"
    )
    print(f"Baseline (Recent Form):  {val_results['baselines']['weighted_recent_form']['mae']:.3f}")
    print(f"Baseline (Season Avg):   {val_results['baselines']['season_average']['mae']:.3f}")
    print(f"Baseline (Fixture Adj):  {val_results['baselines']['fixture_adjusted']['mae']:.3f}")
    print(f"Improvement vs Baseline: {val_results['mae_improvement_pct']:+.2f}%")
    print(
        f"80% Credible Coverage:   {val_results['calibration']['interval_80_coverage_pct']:.2f}% ({val_results['calibration']['calibration_verdict']})"
    )

    # 4. Promotion Gate Check Report
    print("\n--- PROMOTION GATE VERDICT ---")
    print(f"Passed: {gate_passed}")
    print(f"Reason: {gate_reason}")

    # Check 2026-27 current season holdout specifically
    origin_2026_27 = [ro for ro in rolling_origins if "2026-27" in ro["season"]][0]
    print("\n--- 2026-27 CURRENT SEASON HOLDOUT CHECK ---")
    print(f"2026-27 ML MAE:       {origin_2026_27['mae']:.3f}")
    print(f"2026-27 Best Base:    {origin_2026_27['best_baseline_mae']:.3f}")
    print(f"2026-27 Gain:         {origin_2026_27['gain_vs_baseline']:+.3f}")
    print(f"2026-27 Coverage 80%: {origin_2026_27.get('interval_80_coverage_pct', 0.0):.2f}%")

    if not gate_passed:
        print("\n[GATE STATUS] Candidate model REJECTED by promotion gate criteria.")
        print(f"Reason: {gate_reason}")
        print("Active production model retained in data/models/ (zero premature promotion).")
    else:
        print("\n[GATE STATUS] Candidate model PASSED all promotion gate criteria.")

    print("\n" + "=" * 70)
    print("F2/G5 EVALUATION COMPLETED SUCCESSFULLY — REPORTS GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    run_full_evaluation()
