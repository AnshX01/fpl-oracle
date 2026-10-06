"""
ML Projection Engine.
Coordinates feature generation, component model inference, and scoring aggregation
to deliver single and multi-gameweek player point distributions.
"""

import logging
from datetime import UTC
from typing import Any

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.config import MODELS_DIR
from fpl_oracle.data.features import FEATURE_COLUMNS, feature_engineering
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import scoring_ensemble

logger = logging.getLogger("fpl_oracle.predict")


class ProjectionEngine:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.minutes_model = MinutesModel()
        self.attacking_model = AttackingModel()
        self.defending_model = DefendingModel()
        self.defcon_model = DefConModel()
        self.bonus_model = BonusModel()
        self.cards_saves_model = CardsSavesModel()
        self.is_loaded = False
        self._cache: dict[tuple[int, int, int], pd.DataFrame] = {}

    def load_or_train(self, X: pd.DataFrame | None = None, Y: pd.DataFrame | None = None):
        """Load trained model weights from disk or train if missing."""
        weights = [
            (self.minutes_model, self.models_dir / "minutes_model.pkl"),
            (self.attacking_model, self.models_dir / "attacking_model.pkl"),
            (self.defending_model, self.models_dir / "defending_model.pkl"),
            (self.defcon_model, self.models_dir / "defcon_model.pkl"),
            (self.bonus_model, self.models_dir / "bonus_model.pkl"),
            (self.cards_saves_model, self.models_dir / "cards_saves_model.pkl"),
        ]

        all_exist = all(p.exists() for _, p in weights)
        if all_exist:
            logger.info("Loading pre-trained component models from disk...")
            try:
                for model, path in weights:
                    loaded = model.load(path)
                    model.__dict__.update(loaded.__dict__)
                self.is_loaded = True
                return
            except Exception as e:
                logger.warning(f"Error loading models from disk: {e}. Retraining...")

        # If models don't exist, we must train them
        if X is not None and Y is not None:
            self.train(X, Y)
        else:
            from fpl_oracle.ml.train import train_all_models

            train_all_models()
            self.load_or_train()

    def train(self, X: pd.DataFrame, Y: pd.DataFrame):
        logger.info(f"Training ML component models on {len(X)} records...")
        self.minutes_model.fit(X, Y)
        self.minutes_model.save(self.models_dir / "minutes_model.pkl")

        self.attacking_model.fit(X, Y)
        self.attacking_model.save(self.models_dir / "attacking_model.pkl")

        self.defending_model.fit(X, Y)
        self.defending_model.save(self.models_dir / "defending_model.pkl")

        self.defcon_model.fit(X, Y)
        self.defcon_model.save(self.models_dir / "defcon_model.pkl")

        self.bonus_model.fit(X, Y)
        self.bonus_model.save(self.models_dir / "bonus_model.pkl")

        self.cards_saves_model.fit(X, Y)
        self.cards_saves_model.save(self.models_dir / "cards_saves_model.pkl")

        self.is_loaded = True
        logger.info("All component models successfully trained and persisted.")

    def predict_gameweek(
        self,
        target_gw: int,
        bootstrap: BootstrapStatic,
        fixtures: list[Fixture],
        reconciled_availabilities: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """
        Generate expected points for all players for a specific gameweek.
        Handles double gameweeks and blank gameweeks.
        """
        if not self.is_loaded:
            self.load_or_train()

        cache_key = (target_gw, len(bootstrap.elements), len(fixtures))
        if reconciled_availabilities is None and cache_key in self._cache:
            return self._cache[cache_key].copy()

        features_df = feature_engineering.extract_live_features_for_upcoming(
            bootstrap=bootstrap,
            fixtures=fixtures,
            target_gw=target_gw,
            reconciled_availabilities=reconciled_availabilities,
        )

        if features_df.empty:
            return pd.DataFrame()

        # Isolate model features
        X = features_df[FEATURE_COLUMNS].copy()

        # Component predictions
        mins_pred = self.minutes_model.predict(X)
        att_pred = self.attacking_model.predict(X)
        def_pred = self.defending_model.predict(X)
        defcon_pred = self.defcon_model.predict(X)
        bonus_pred = self.bonus_model.predict(X)
        cards_pred = self.cards_saves_model.predict(X)

        components = {**mins_pred, **att_pred, **def_pred, **defcon_pred, **bonus_pred, **cards_pred}

        # Aggregate through scoring ensemble (passes chance_of_playing for availability scaling)
        X_agg = X.copy()
        if "chance_of_playing" in features_df.columns:
            X_agg["chance_of_playing"] = features_df["chance_of_playing"].values
        res_df = scoring_ensemble.aggregate_components(components, X_agg)

        # Merge metadata
        meta_cols = [
            "element",
            "web_name",
            "team",
            "position",
            "value",
            "target_gw",
            "is_bgw",
            "is_dgw",
            "opponent_difficulty",
            "chance_of_playing",
        ]
        for col in meta_cols:
            res_df[col] = features_df[col].values

        # If a player has a Double Gameweek (2 fixtures in same GW), sum the expectations
        # Group by element
        dgw_grouped = res_df.groupby("element", as_index=False).agg(
            {
                "web_name": "first",
                "team": "first",
                "position": "first",
                "value": "first",
                "target_gw": "first",
                "is_bgw": "first",
                "is_dgw": "max",
                "opponent_difficulty": "mean",
                "chance_of_playing": "first",
                "expected_points": "sum",
                "p10": "sum",
                "p50": "sum",
                "p90": "sum",
                "variance": "sum",
                "exp_appearance": "sum",
                "exp_goals_pts": "sum",
                "exp_assists_pts": "sum",
                "exp_cs_pts": "sum",
                "exp_defcon_pts": "sum",
                "exp_bonus_pts": "sum",
                "p_starts": "max",
                "p_min60": "max",
            }
        )

        # Blank gameweek zeroing
        dgw_grouped.loc[dgw_grouped["is_bgw"] == 1, ["expected_points", "p10", "p50", "p90", "variance"]] = 0.0

        if reconciled_availabilities is None:
            self._cache[cache_key] = dgw_grouped.copy()

        return dgw_grouped

    def predict_multi_gameweeks(
        self,
        start_gw: int,
        horizon: int,
        bootstrap: BootstrapStatic,
        fixtures: list[Fixture],
        reconciled_availabilities: dict[int, float] | None = None,
    ) -> dict[int, pd.DataFrame]:
        """
        Generate projections across an N-gameweek horizon.
        """
        multi_projections = {}
        for gw in range(start_gw, start_gw + horizon):
            if gw > 38:
                break
            gw_df = self.predict_gameweek(
                gw, bootstrap, fixtures, reconciled_availabilities=reconciled_availabilities
            )
            multi_projections[gw] = gw_df
        return multi_projections

    def get_model_status(self) -> dict[str, Any]:
        """Return diagnostic metrics on trained model files and loaded state."""
        weights = [
            "minutes_model.pkl",
            "attacking_model.pkl",
            "defending_model.pkl",
            "defcon_model.pkl",
            "bonus_model.pkl",
            "cards_saves_model.pkl",
        ]
        existing = [w for w in weights if (self.models_dir / w).exists()]
        mtimes = [(self.models_dir / w).stat().st_mtime for w in existing]
        latest_mtime = max(mtimes) if mtimes else None

        from datetime import datetime

        trained_iso = datetime.fromtimestamp(latest_mtime, tz=UTC).isoformat() if latest_mtime else None

        return {
            "version": "v1.0.0-lgbm-2026/27",
            "is_loaded": self.is_loaded,
            "components_ready": f"{len(existing)}/{len(weights)}",
            "all_components_present": len(existing) == len(weights),
            "last_trained_timestamp": trained_iso,
        }


projection_engine = ProjectionEngine()
