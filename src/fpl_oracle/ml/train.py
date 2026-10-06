"""
Full ML training and cross-validation pipeline.
Run with: python -m fpl_oracle.ml.train or python run.py train.
"""

import logging

import pandas as pd

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.predict import projection_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fpl_oracle.train")


def train_all_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== Starting FPL Oracle ML Training Pipeline ===")

    # 1. Load historical master dataset
    df = historical_manager.ensure_dataset_ready()
    logger.info(f"Loaded master history with {len(df)} total match records.")

    # 2. Build pre-deadline features
    logger.info("Engineering zero-leakage pre-deadline features...")
    X, Y = feature_engineering.build_historical_features(df)
    logger.info(f"Engineered {X.shape[1]} features across {len(X)} rows.")

    # 3. Time-aware train / validation split
    # Reserve last 15% chronologically for out-of-time evaluation
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    Y_train, Y_val = Y.iloc[:split_idx], Y.iloc[split_idx:]

    logger.info(f"Training split: {len(X_train)} samples. Validation split: {len(X_val)} samples.")

    from fpl_oracle.ml.components.attacking import AttackingModel
    from fpl_oracle.ml.components.bonus import BonusModel
    from fpl_oracle.ml.components.cards_saves import CardsSavesModel
    from fpl_oracle.ml.components.defcon import DefConModel
    from fpl_oracle.ml.components.defending import DefendingModel
    from fpl_oracle.ml.components.minutes import MinutesModel
    from fpl_oracle.ml.model_registry import model_registry

    # 4. Train candidate component models on train split
    logger.info("Training candidate models on training split...")
    candidate_models = {
        "minutes_model": MinutesModel(),
        "attacking_model": AttackingModel(),
        "defending_model": DefendingModel(),
        "defcon_model": DefConModel(),
        "bonus_model": BonusModel(),
        "cards_saves_model": CardsSavesModel(),
    }
    candidate_models["minutes_model"].fit(X_train, Y_train)
    candidate_models["attacking_model"].fit(X_train, Y_train)
    candidate_models["defending_model"].fit(X_train, Y_train)
    candidate_models["defcon_model"].fit(X_train, Y_train)
    candidate_models["bonus_model"].fit(X_train, Y_train)
    candidate_models["cards_saves_model"].fit(X_train, Y_train)

    # 5. Evaluate candidate models on holdout validation split
    logger.info("Evaluating candidate models on holdout validation split...")
    cand_metrics = model_registry.evaluate_model_suite(candidate_models, X_val, Y_val)
    logger.info(
        f"Candidate Metrics: MAE={cand_metrics['ml_mae']}, Spearman={cand_metrics['ml_spearman']}, Baseline MAE={cand_metrics['base_mae']}"
    )

    # Check active production model metrics on same validation split
    active_metrics = model_registry.evaluate_production_weights(X_val, Y_val)
    if active_metrics:
        logger.info(
            f"Active Production Model Metrics: MAE={active_metrics['ml_mae']}, Spearman={active_metrics['ml_spearman']}"
        )

    # 6. Verify and promote or engage automated rollback
    promote_res = model_registry.verify_and_promote(
        candidate_models=candidate_models,
        candidate_metrics=cand_metrics,
        active_metrics=active_metrics,
        tolerance=0.05,
        notes="Automated retrain pipeline",
    )

    if not promote_res.get("promoted", True):
        logger.warning(f"[ModelRollback] Retrain rejected: {promote_res.get('reason')}")
        logger.info("Retaining existing production models. Aborting full fit.")
        return X, Y

    # 7. Final fit on full dataset so production models have latest signals
    logger.info("Fitting final production models on full historical dataset...")
    projection_engine.train(X, Y)

    logger.info("=== ML Training Pipeline Completed Successfully! ===")
    return X, Y


if __name__ == "__main__":
    train_all_models()
