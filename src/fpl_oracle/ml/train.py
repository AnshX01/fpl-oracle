"""
Full ML training and cross-validation pipeline.
Run with: python -m fpl_oracle.ml.train or python run.py train.
"""

import logging

import pandas as pd

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.eval import model_evaluator
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

    # 4. Train component models on train split
    projection_engine.train(X_train, Y_train)

    # 5. Evaluate on holdout validation split
    logger.info("Generating predictions on holdout validation split...")
    mins_p = projection_engine.minutes_model.predict(X_val)
    att_p = projection_engine.attacking_model.predict(X_val)
    def_p = projection_engine.defending_model.predict(X_val)
    defcon_p = projection_engine.defcon_model.predict(X_val)
    bonus_p = projection_engine.bonus_model.predict(X_val)
    cards_p = projection_engine.cards_saves_model.predict(X_val)

    from fpl_oracle.ml.ensemble import scoring_ensemble
    components = {**mins_p, **att_p, **def_p, **defcon_p, **bonus_p, **cards_p}
    val_preds_df = scoring_ensemble.aggregate_components(components, X_val)
    ml_preds = val_preds_df["expected_points"].values

    # Run evaluation and generate reports/model_eval.md
    metrics = model_evaluator.evaluate_expanding_window(X_val, Y_val, ml_preds)
    logger.info(f"Validation MAE: ML={metrics['ml_mae']} vs Baseline={metrics['base_mae']} ({metrics['mae_improvement_pct']}% improvement)")
    logger.info(f"Validation Spearman Correlation: ML={metrics['ml_spearman']} vs Baseline={metrics['base_spearman']}")

    # 6. Final fit on full dataset so models have latest 2026/27 signals
    logger.info("Fitting final production models on full historical dataset...")
    projection_engine.train(X, Y)

    logger.info("=== ML Training Pipeline Completed Successfully! ===")
    return X, Y

if __name__ == "__main__":
    train_all_models()
