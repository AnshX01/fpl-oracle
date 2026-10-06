"""
Full ML training and cross-validation pipeline.
Run with: python -m fpl_oracle.ml.train or python run.py train.
"""

import logging
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
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fpl_oracle.train")


def compute_rolling_origin_cv(
    X: pd.DataFrame, Y: pd.DataFrame, meta: pd.DataFrame
) -> list[dict[str, Any]]:
    """
    True temporal rolling-origin cross-validation:
    Origin 1: Train 2023-24 -> Test 2024-25
    Origin 2: Train 2023-24 + 2024-25 -> Test 2025-26
    Origin 3: Train 2023-24 + 2024-25 + 2025-26 -> Test 2026-27
    """
    origins_config = [
        ("2024-25", ["2023-24"], ["2024-25"]),
        ("2025-26", ["2023-24", "2024-25"], ["2025-26"]),
        ("2026-27", ["2023-24", "2024-25", "2025-26"], ["2026-27"]),
    ]

    results = []
    seasons = meta["season"]

    for test_label, train_seasons, test_seasons in origins_config:
        train_mask = seasons.isin(train_seasons)
        test_mask = seasons.isin(test_seasons)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_tr, Y_tr = X[train_mask], Y[train_mask]
        X_te, Y_te = X[test_mask], Y[test_mask]

        logger.info(
            f"Evaluating rolling origin: Train {train_seasons} ({len(X_tr)}) -> Test {test_seasons} ({len(X_te)})..."
        )

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

        mins_p = models["minutes_model"].predict(X_te)
        att_p = models["attacking_model"].predict(X_te)
        def_p = models["defending_model"].predict(X_te)
        defcon_p = models["defcon_model"].predict(X_te)
        bonus_p = models["bonus_model"].predict(X_te)
        cards_p = models["cards_saves_model"].predict(X_te)

        comps = {**mins_p, **att_p, **def_p, **defcon_p, **bonus_p, **cards_p}
        preds_df = scoring_ensemble.aggregate_components(comps, X_te)
        preds = preds_df["expected_points"].values
        actual = Y_te["target_points"].values

        mae = float(np.round(mean_absolute_error(actual, preds), 3))
        rmse = float(np.round(root_mean_squared_error(actual, preds), 3))
        sp, _ = spearmanr(preds, actual)
        sp = float(np.round(sp, 3))

        results.append(
            {
                "season": f"Holdout {test_label} (trained on {', '.join(train_seasons)})",
                "train_size": int(len(X_tr)),
                "test_size": int(len(X_te)),
                "mae": mae,
                "rmse": rmse,
                "spearman": sp,
            }
        )

    return results


def train_all_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== Starting FPL Oracle ML Training Pipeline ===")

    # 1. Load historical master dataset
    df = historical_manager.ensure_dataset_ready()
    logger.info(f"Loaded master history with {len(df)} total match records.")

    # 2. Build pre-deadline features
    logger.info("Engineering zero-leakage pre-deadline features...")
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    logger.info(f"Engineered {X.shape[1]} features across {len(X)} rows.")

    # 3. Rolling-Origin Cross-Validation across historical seasons
    logger.info("Running authentic rolling-origin cross-validation...")
    rolling_origins = compute_rolling_origin_cv(X, Y, meta)

    # 4. Out-of-time holdout validation split (strictly latest gameweeks / season)
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]

    logger.info(f"Training split: {len(X_train)} samples. Holdout validation split: {len(X_val)} samples.")

    # 5. Train candidate component models on train split
    logger.info("Training candidate models on training split...")
    candidate_models = {
        "minutes_model": MinutesModel(),
        "attacking_model": AttackingModel(),
        "defending_model": DefendingModel(),
        "defcon_model": DefConModel(),
        "bonus_model": BonusModel(),
        "cards_saves_model": CardsSavesModel(),
    }
    for name, m in candidate_models.items():
        logger.info(f"Fitting candidate {name}...")
        m.fit(X_train, Y_train)

    # 6. Evaluate candidate models on holdout validation split with rolling origin metrics
    logger.info("Evaluating candidate models on holdout validation split...")
    cand_metrics = model_registry.evaluate_model_suite(
        candidate_models, X_val, Y_val, rolling_origins=rolling_origins
    )
    logger.info(
        f"Candidate Metrics: MAE={cand_metrics['ml_mae']}, Spearman={cand_metrics['ml_spearman']}, Baseline MAE={cand_metrics['base_mae']}"
    )

    # Check active production model metrics on same validation split
    active_metrics = model_registry.evaluate_production_weights(X_val, Y_val, rolling_origins=rolling_origins)
    if active_metrics:
        logger.info(
            f"Active Production Model Metrics: MAE={active_metrics['ml_mae']}, Spearman={active_metrics['ml_spearman']}"
        )

    # 7. Verify and promote or engage automated rollback
    promote_res = model_registry.verify_and_promote(
        candidate_models=candidate_models,
        candidate_metrics=cand_metrics,
        active_metrics=active_metrics,
        tolerance=0.05,
        notes="Automated retrain pipeline with true rolling origins",
    )

    if not promote_res.get("promoted", True):
        logger.warning(f"[ModelRollback] Retrain rejected: {promote_res.get('reason')}")
        logger.info("Retaining existing production models. Aborting full fit.")
        return X, Y

    # 8. Final fit on full dataset so production models have latest signals
    logger.info("Fitting final production models on full historical dataset...")
    projection_engine.train(X, Y)

    logger.info("=== ML Training Pipeline Completed Successfully! ===")
    return X, Y


if __name__ == "__main__":
    train_all_models()
