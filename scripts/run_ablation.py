"""
True Retraining Feature Ablation Runner (Requirement F3).
Retrains the component model suite from scratch without each feature group
on an identical training split, evaluates on identical test holdout rows,
and calculates bootstrap confidence intervals (1,000 resamples).
Generates single authoritative reports/ablation.json artifact.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation")


FEATURE_GROUPS_TO_ABLATE = {
    "fixture_difficulty_and_context": [
        "was_home",
        "team_strength_attack",
        "opp_strength_defence",
        "net_strength_diff",
        "opponent_difficulty",
        "days_rest",
        "implied_team_xG",
        "implied_team_cs_prob",
    ],
    "team_form_attack_defense": [
        "team_roll_goals_for_5",
        "team_roll_goals_against_5",
        "opp_roll_goals_for_5",
        "opp_roll_goals_against_5",
        "team_form_attack_ratio_5",
        "opp_form_defence_ratio_5",
    ],
    "player_underlying_metrics": [
        "roll_xG_3",
        "roll_xG_5",
        "roll_xG_8",
        "roll_xA_3",
        "roll_xA_5",
        "roll_xA_8",
        "roll_xGI_5",
        "roll_xGC_5",
    ],
    "minutes_and_starts": [
        "roll_minutes_3",
        "roll_minutes_5",
        "roll_minutes_8",
        "roll_starts_ratio_5",
        "roll_min60_ratio_5",
        "std_minutes_per_gw",
    ],
    "disciplinary_and_rare": [
        "roll_cards_5",
        "roll_own_goals_5",
        "roll_penalties_missed_5",
        "roll_penalties_saved_5",
    ],
}


def fit_and_evaluate_models(
    X_tr: pd.DataFrame, Y_tr: pd.DataFrame, X_te: pd.DataFrame, Y_te: pd.DataFrame
) -> tuple[np.ndarray, float, float, float]:
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

    comps = {
        **models["minutes_model"].predict(X_te),
        **models["attacking_model"].predict(X_te),
        **models["defending_model"].predict(X_te),
        **models["defcon_model"].predict(X_te),
        **models["bonus_model"].predict(X_te),
        **models["cards_saves_model"].predict(X_te),
    }
    preds_df = scoring_ensemble.aggregate_components(comps, X_te)
    preds = preds_df["expected_points"].values
    actual = Y_te["target_points"].values

    mae = float(np.round(mean_absolute_error(actual, preds), 4))
    rmse = float(np.round(root_mean_squared_error(actual, preds), 4))
    sp, _ = spearmanr(preds, actual)
    sp = float(np.round(sp, 4))
    return preds, mae, rmse, sp


def compute_bootstrap_ci(
    actual: np.ndarray, preds_full: np.ndarray, preds_ablated: np.ndarray, n_boot: int = 1000
) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval on MAE delta (ablated_mae - full_mae)."""
    rng = np.random.default_rng(42)
    n = len(actual)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mae_f = mean_absolute_error(actual[idx], preds_full[idx])
        mae_a = mean_absolute_error(actual[idx], preds_ablated[idx])
        deltas.append(mae_a - mae_f)
    ci_lower = float(np.round(np.percentile(deltas, 2.5), 4))
    ci_upper = float(np.round(np.percentile(deltas, 97.5), 4))
    return ci_lower, ci_upper


def run_ablation_experiment():
    print("=" * 70)
    print("F3 TRUE RETRAINING FEATURE ABLATION EXPERIMENT")
    print("=" * 70)

    # 1. Load data
    df = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    seasons = meta["season"]

    # Split: Train 2023-24 + 2024-25, Test 2025-26 (Disjoint temporal holdout)
    train_mask = seasons.isin(["2023-24", "2024-25"])
    test_mask = seasons == "2025-26"

    X_train_full = X[train_mask].copy()
    Y_train = Y[train_mask].copy()
    X_test_full = X[test_mask].copy()
    Y_test = Y[test_mask].copy()
    actual_test = Y_test["target_points"].values

    print(f"Dataset: Train {len(X_train_full)} rows (2023-24, 2024-25) -> Test {len(X_test_full)} rows (2025-26)")

    # 2. Fit Full Model
    print("\n[1/6] Training FULL model with all features...")
    preds_full, full_mae, full_rmse, full_sp = fit_and_evaluate_models(X_train_full, Y_train, X_test_full, Y_test)
    print(f"FULL Model: MAE={full_mae:.4f}, RMSE={full_rmse:.4f}, Spearman={full_sp:.4f}")

    # 3. Retrain without each group
    group_results = []
    idx = 2
    for grp_name, drop_cols in FEATURE_GROUPS_TO_ABLATE.items():
        existing_drop = [c for c in drop_cols if c in X_train_full.columns]
        if not existing_drop:
            continue

        print(f"\n[{idx}/6] Retraining WITHOUT group '{grp_name}' (dropping {len(existing_drop)} features)...")
        X_tr_abl = X_train_full.drop(columns=existing_drop)
        X_te_abl = X_test_full.drop(columns=existing_drop)

        preds_abl, abl_mae, abl_rmse, abl_sp = fit_and_evaluate_models(X_tr_abl, Y_train, X_te_abl, Y_test)
        mae_delta = float(np.round(abl_mae - full_mae, 4))
        mae_gain_pct = float(np.round((mae_delta / abl_mae) * 100.0, 2)) if abl_mae > 0 else 0.0

        # Bootstrap CI
        ci_low, ci_high = compute_bootstrap_ci(actual_test, preds_full, preds_abl, n_boot=1000)

        print(f"  Ablated MAE: {abl_mae:.4f} (Delta vs Full: {mae_delta:+.4f}, Gain: {mae_gain_pct:+.2f}%)")
        print(f"  95% Bootstrap CI on Delta: [{ci_low:+.4f}, {ci_high:+.4f}]")

        group_results.append(
            {
                "group_name": grp_name,
                "features_removed": existing_drop,
                "feature_count_removed": len(existing_drop),
                "ablated_mae": abl_mae,
                "ablated_rmse": abl_rmse,
                "ablated_spearman": abl_sp,
                "mae_delta_vs_full": mae_delta,
                "mae_gain_pct": mae_gain_pct,
                "ci_95": [ci_low, ci_high],
            }
        )
        idx += 1

    # 4. Save ablation artifact
    ablation_payload = {
        "evaluation_type": "true_retraining_ablation",
        "description": "Models retrained from scratch without each feature group on identical temporal split",
        "timestamp": datetime.now(UTC).isoformat(),
        "train_size": len(X_train_full),
        "test_size": len(X_test_full),
        "full_model": {
            "mae": full_mae,
            "rmse": full_rmse,
            "spearman": full_sp,
            "feature_count": len(X_train_full.columns),
        },
        "groups": group_results,
    }

    out_path = REPORTS_DIR / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ablation_payload, f, indent=2)

    print(f"\nSaved single authoritative ablation artifact: {out_path}")
    print("=" * 70)
    print("F3 ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation_experiment()
