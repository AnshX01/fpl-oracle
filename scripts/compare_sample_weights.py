"""
Temporal holdout comparison of sample weights (1x, 2x, 3x, 5x) for F-009.
Evaluates the impact of weighting recent 2026/27 observations on:
1. Effective Sample Size (Kish's ESS)
2. Out-of-time holdout performance for DefCon and Bonus component models
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from fpl_oracle.data.features import feature_engineering  # noqa: E402
from fpl_oracle.data.historical import historical_manager  # noqa: E402


def run_weight_comparison() -> pd.DataFrame:
    print("=" * 80)
    print("F-009: Comparing Sample Weights (1x, 2x, 3x, 5x) on Temporal Holdout")
    print("=" * 80)

    # 1. Load data and extract features
    df = historical_manager.ensure_dataset_ready()
    print(f"Total raw records: {len(df)}")
    rows_2026 = int((df["season"] == "2026-27").sum())
    print(f"2026/27 records: {rows_2026} ({rows_2026 / len(df) * 100:.2f}%)")

    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    print(f"Engineered feature set: {X.shape[0]} rows, {X.shape[1]} columns.")

    # 2. Define temporal holdout:
    # Use rolling origin 3: Train on 2023-24, 2024-25, 2025-26 + early 2026-27 (round < 4)
    # Test on late 2026-27 (round >= 4)
    seasons = meta["season"]
    rounds = meta["round"]

    # In 2026-27 data:
    mask_2026 = seasons == "2026-27"
    max_round_2026 = int(rounds[mask_2026].max()) if mask_2026.sum() > 0 else 0
    split_round = max(1, max_round_2026 // 2)

    train_mask = (seasons < "2026-27") | ((seasons == "2026-27") & (rounds <= split_round))
    test_mask = (seasons == "2026-27") & (rounds > split_round)

    if test_mask.sum() == 0:
        # Fallback to standard 85/15 temporal boundary if 2026-27 has few rounds
        target_split = int(len(X) * 0.85)
        split_s = str(meta.iloc[target_split]["season"])
        split_r = int(meta.iloc[target_split]["round"])
        train_mask = (meta["season"] < split_s) | ((meta["season"] == split_s) & (meta["round"] < split_r))
        test_mask = ~train_mask

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    print(f"Train split size: {len(X_train)} (2026/27 rows in train: {(X_train['is_2026_27'] > 0).sum()})")
    print(f"Holdout split size: {len(X_test)} (2026/27 rows in test: {(X_test['is_2026_27'] > 0).sum()})")
    print("-" * 80)

    weights_to_test = [1.0, 2.0, 3.0, 5.0]
    results = []

    for mult in weights_to_test:
        w_train = np.where(X_train["is_2026_27"] > 0, mult, 1.0)
        sum_w = float(np.sum(w_train))
        sum_w_sq = float(np.sum(w_train ** 2))
        kish_ess = float((sum_w ** 2) / sum_w_sq)
        ess_loss = float(len(X_train) - kish_ess)

        # Evaluate DefCon Model (Classifier)
        y_defcon_tr = Y_train["target_defcon"].values
        y_defcon_te = Y_test["target_defcon"].values

        clf = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        clf.fit(X_train, y_defcon_tr, sample_weight=w_train)
        probs_defcon = clf.predict_proba(X_test)[:, 1]

        defcon_brier = float(brier_score_loss(y_defcon_te, probs_defcon))
        defcon_logloss = float(log_loss(y_defcon_te, probs_defcon))
        defcon_auc = float(roc_auc_score(y_defcon_te, probs_defcon)) if len(np.unique(y_defcon_te)) > 1 else 0.5

        # Evaluate Bonus Model (Regressor)
        y_bonus_tr = Y_train["target_bonus"].values
        y_bonus_te = Y_test["target_bonus"].values

        reg = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4
        )
        reg.fit(X_train, y_bonus_tr, sample_weight=w_train)
        preds_bonus = np.clip(reg.predict(X_test), 0.0, 3.0)

        bonus_mae = float(mean_absolute_error(y_bonus_te, preds_bonus))
        bonus_rmse = float(root_mean_squared_error(y_bonus_te, preds_bonus))
        bonus_sp, _ = spearmanr(preds_bonus, y_bonus_te)
        bonus_sp = float(bonus_sp)

        row = {
            "Weight Multiplier": f"{mult:.1f}x",
            "Kish ESS": round(kish_ess, 1),
            "ESS Reduction": round(ess_loss, 1),
            "DefCon LogLoss": round(defcon_logloss, 4),
            "DefCon Brier": round(defcon_brier, 4),
            "DefCon AUC": round(defcon_auc, 4),
            "Bonus MAE": round(bonus_mae, 4),
            "Bonus RMSE": round(bonus_rmse, 4),
            "Bonus Spearman": round(bonus_sp, 4),
        }
        results.append(row)
        print(
            f"Weight {mult:.1f}x -> Kish ESS: {kish_ess:,.1f} (-{ess_loss:,.1f}) | "
            f"DefCon LogLoss: {defcon_logloss:.4f} | Bonus MAE: {bonus_mae:.4f}"
        )

    print("=" * 80)
    summary_df = pd.DataFrame(results)
    print("\nSummary Results Table:")
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Conclusion: Per F-009 instructions, weights are NOT changed yet in hotfix phase.")
    print("Optimal weighting policy will be tuned via out-of-fold temporal CV in Backtest Harness (WI-0.1).")
    return summary_df


if __name__ == "__main__":
    run_weight_comparison()
