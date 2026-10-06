"""
Diagnostic script for 2026-27 holdout performance (Requirement F2 step a).
Analyzes ML vs Baselines across gameweeks, positions, minutes buckets, and player segments.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
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
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.eval import model_evaluator


def run_diagnosis():
    print("=" * 70)
    print("F2 DIAGNOSIS: 2026-27 CURRENT SEASON PERFORMANCE BREAKDOWN")
    print("=" * 70)

    # 1. Load data
    df = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)

    # Train mask: 2023-24, 2024-25, 2025-26
    train_mask = meta["season"].isin(["2023-24", "2024-25", "2025-26"])
    test_mask = meta["season"] == "2026-27"

    X_tr, Y_tr = X[train_mask].copy(), Y[train_mask].copy()
    X_te, Y_te = X[test_mask].copy(), Y[test_mask].copy()
    meta_te = meta[test_mask].copy()

    print(f"Training rows: {len(X_tr)} (seasons 2023-24 through 2025-26)")
    print(f"Testing rows:  {len(X_te)} (season 2026-27, rounds {meta_te['round'].min()} to {meta_te['round'].max()})")

    # 2. Fit models on train
    print("\nFitting component models on historical seasons...")
    models = {
        "minutes_model": MinutesModel(),
        "attacking_model": AttackingModel(),
        "defending_model": DefendingModel(),
        "defcon_model": DefConModel(),
        "bonus_model": BonusModel(),
        "cards_saves_model": CardsSavesModel(),
    }
    for m in models.values():
        m.fit(X_tr, Y_tr)

    # 3. Predict on 2026-27
    print("Predicting on 2026-27...")
    comps = {
        **models["minutes_model"].predict(X_te),
        **models["attacking_model"].predict(X_te),
        **models["defending_model"].predict(X_te),
        **models["defcon_model"].predict(X_te),
        **models["bonus_model"].predict(X_te),
        **models["cards_saves_model"].predict(X_te),
    }
    preds_df = scoring_ensemble.aggregate_components(comps, X_te)
    ml_preds = preds_df["expected_points"].values
    actual = Y_te["target_points"].values

    # Baselines
    base_recent = model_evaluator.compute_baseline_projections(X_te)
    base_season = model_evaluator.compute_season_avg_baseline(X_te)
    base_fixture = model_evaluator.compute_fixture_adjusted_baseline(X_te)

    # 4. Overall metrics
    ml_mae = mean_absolute_error(actual, ml_preds)
    b_recent_mae = mean_absolute_error(actual, base_recent)
    b_season_mae = mean_absolute_error(actual, base_season)
    b_fixture_mae = mean_absolute_error(actual, base_fixture)
    best_base_mae = min(b_recent_mae, b_season_mae, b_fixture_mae)
    best_base_name = (
        "recent_form"
        if best_base_mae == b_recent_mae
        else ("season_avg" if best_base_mae == b_season_mae else "fixture_adj")
    )

    print("\n--- OVERALL COMPARISON (2026-27 Holdout) ---")
    print(
        f"ML MAE:                  {ml_mae:.4f}  (RMSE: {root_mean_squared_error(actual, ml_preds):.4f}, Spearman: {spearmanr(ml_preds, actual)[0]:.4f})"
    )
    print(f"Baseline (Recent Form):  {b_recent_mae:.4f}")
    print(f"Baseline (Season Avg):   {b_season_mae:.4f}")
    print(f"Baseline (Fixture Adj):  {b_fixture_mae:.4f}")
    print(f"Best Baseline:           {best_base_mae:.4f} ({best_base_name})")
    print(f"Gain vs Best Baseline:   {best_base_mae - ml_mae:+.4f} (Negative = ML is WORSE)")

    # 5. Breakdown by Gameweek
    print("\n--- BREAKDOWN BY GAMEWEEK ---")
    print(
        f"{'GW':<6} {'N':<6} {'Actual Mean':<12} {'ML Pred Mean':<14} {'ML MAE':<10} {'Best Base MAE':<14} {'Gain':<10}"
    )
    for gw in sorted(meta_te["round"].unique()):
        gw_mask = meta_te["round"].values == gw
        act_gw = actual[gw_mask]
        ml_gw = ml_preds[gw_mask]
        b_gw = base_season[gw_mask]  # season avg is usually best baseline
        b_mae_gw = mean_absolute_error(act_gw, b_gw)
        ml_mae_gw = mean_absolute_error(act_gw, ml_gw)
        print(
            f"GW{gw:<4} {gw_mask.sum():<6} {act_gw.mean():<12.3f} {ml_gw.mean():<14.3f} {ml_mae_gw:<10.3f} {b_mae_gw:<14.3f} {b_mae_gw - ml_mae_gw:+10.3f}"
        )

    # 6. Breakdown by Position
    print("\n--- BREAKDOWN BY POSITION ---")
    print(
        f"{'Pos':<6} {'N':<6} {'Actual Mean':<12} {'ML Pred Mean':<14} {'ML MAE':<10} {'Best Base MAE':<14} {'Gain':<10}"
    )
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_mask = X_te[f"pos_{pos}"].values > 0
        if pos_mask.sum() > 0:
            act_p = actual[pos_mask]
            ml_p = ml_preds[pos_mask]
            b_p = base_season[pos_mask]
            b_mae_p = mean_absolute_error(act_p, b_p)
            ml_mae_p = mean_absolute_error(act_p, ml_p)
            print(
                f"{pos:<6} {pos_mask.sum():<6} {act_p.mean():<12.3f} {ml_p.mean():<14.3f} {ml_mae_p:<10.3f} {b_mae_p:<14.3f} {b_mae_p - ml_mae_p:+10.3f}"
            )

    # 7. Breakdown by Minutes Played Bucket (Outcome actual minutes)
    print("\n--- BREAKDOWN BY MINUTES BUCKET ---")
    print(
        f"{'Bucket':<14} {'N':<6} {'Actual Mean':<12} {'ML Pred Mean':<14} {'ML MAE':<10} {'Best Base MAE':<14} {'Gain':<10}"
    )
    mins_col = Y_te["target_minutes"].values if "target_minutes" in Y_te.columns else np.zeros_like(actual)
    buckets = [
        ("0 mins", mins_col == 0),
        ("1-59 mins", (mins_col > 0) & (mins_col < 60)),
        ("60-90 mins", mins_col >= 60),
    ]
    for b_name, b_mask in buckets:
        if b_mask.sum() > 0:
            act_b = actual[b_mask]
            ml_b = ml_preds[b_mask]
            b_b = base_season[b_mask]
            b_mae_b = mean_absolute_error(act_b, b_b)
            ml_mae_b = mean_absolute_error(act_b, ml_b)
            print(
                f"{b_name:<14} {b_mask.sum():<6} {act_b.mean():<12.3f} {ml_b.mean():<14.3f} {ml_mae_b:<10.3f} {b_mae_b:<14.3f} {b_mae_b - ml_mae_b:+10.3f}"
            )

    # 8. Component Error Analysis
    print("\n--- COMPONENT LEVEL PREDICTIONS ON 2026-27 ---")
    for comp_name, comp_vals in comps.items():
        if isinstance(comp_vals, np.ndarray) and comp_vals.ndim == 1:
            print(
                f"  {comp_name:<28}: mean={comp_vals.mean():.3f}, std={comp_vals.std():.3f}, min={comp_vals.min():.3f}, max={comp_vals.max():.3f}"
            )

    # 9. Top-N Ranked Players (Picks Evaluation)
    print("\n--- TOP-N RANKED PLAYERS (Decision Quality) ---")
    for top_n in [15, 30, 50]:
        # Top N by ML pred vs Top N by Baseline
        top_ml_idx = np.argsort(ml_preds)[-top_n:]
        top_base_idx = np.argsort(base_season)[-top_n:]

        mae_top_ml = mean_absolute_error(actual[top_ml_idx], ml_preds[top_ml_idx])
        mae_top_base = mean_absolute_error(actual[top_base_idx], base_season[top_base_idx])
        pts_top_ml = actual[top_ml_idx].sum()
        pts_top_base = actual[top_base_idx].sum()

        print(
            f"Top {top_n:<3}: ML Actual Points={pts_top_ml:.1f} (MAE {mae_top_ml:.3f}) vs Baseline Actual Points={pts_top_base:.1f} (MAE {mae_top_base:.3f})"
        )

    print("\n" + "=" * 70)
    print("END DIAGNOSIS")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnosis()
