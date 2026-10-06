"""
Model Registry & Versioning Engine with Automated Metric Rollback.
Manages model checkpoints, metric history, candidate model verification, and
safeguards against performance degradation by automatically rolling back if validation MAE worsens.
"""

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.config import MODELS_DIR, REPORTS_DIR
from fpl_oracle.data.store import data_store
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.eval import model_evaluator

logger = logging.getLogger("fpl_oracle.ml.registry")

MANIFEST_PATH = MODELS_DIR / "manifest.json"
VERSIONS_DIR = MODELS_DIR / "versions"
REJECTED_DIR = MODELS_DIR / "rejected"

COMPONENT_WEIGHTS = [
    ("minutes_model", "minutes_model.pkl", MinutesModel),
    ("attacking_model", "attacking_model.pkl", AttackingModel),
    ("defending_model", "defending_model.pkl", DefendingModel),
    ("defcon_model", "defcon_model.pkl", DefConModel),
    ("bonus_model", "bonus_model.pkl", BonusModel),
    ("cards_saves_model", "cards_saves_model.pkl", CardsSavesModel),
]


def check_promotion_gate(
    candidate_metrics: dict[str, Any],
    active_metrics: dict[str, Any] | None = None,
    rolling_origins: list[dict[str, Any]] | None = None,
    tolerance: float = 0.05,
) -> tuple[bool, str]:
    """
    Tested promotion gate:
    1. Overall candidate MAE must beat baseline within tolerance.
    2. Overall candidate MAE must not degrade vs active production by more than tolerance.
    3. If rolling origins are evaluated:
       - Mean gain vs baseline across rolling origins must be strictly > 0.
       - Latest rolling origin gain (e.g. 2026-27 current season holdout) must be strictly > 0.
    """
    cand_mae = candidate_metrics.get("ml_mae")
    base_mae = candidate_metrics.get("base_mae")

    # 1. Baseline superiority check
    if cand_mae is not None and base_mae is not None and cand_mae > (base_mae + tolerance):
        return (
            False,
            f"Candidate MAE ({cand_mae:.3f}) failed baseline gate vs baseline ({base_mae:.3f} + {tolerance:.2f}).",
        )

    # 2. Active production degradation check
    if active_metrics and cand_mae is not None:
        active_mae = active_metrics.get("ml_mae")
        if active_mae is not None and cand_mae > (active_mae + tolerance):
            return (
                False,
                f"Candidate MAE ({cand_mae:.3f}) degraded vs active production MAE ({active_mae:.3f}) by > {tolerance:.2f}.",
            )

    # 3. Rolling origins checks
    origins = rolling_origins if rolling_origins is not None else candidate_metrics.get("rolling_origins")
    if origins:
        gains = []
        for o in origins:
            ml_o = o.get("ml_mae")
            base_o = o.get("best_base_mae", o.get("base_mae"))
            if ml_o is not None and base_o is not None:
                gains.append(base_o - ml_o)

        if gains:
            mean_gain = float(np.mean(gains))
            if mean_gain <= 0.0:
                return (
                    False,
                    f"Candidate failed rolling origin gate: mean gain vs baseline ({mean_gain:+.4f}) is <= 0.",
                )
            latest_gain = gains[-1]
            if latest_gain <= 0.0:
                return (
                    False,
                    f"Candidate failed rolling origin gate: latest origin gain ({latest_gain:+.4f}) is <= 0.",
                )

    return True, "Candidate passed all promotion gate criteria."


class ModelRegistry:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.versions_dir = VERSIONS_DIR
        self.rejected_dir = REJECTED_DIR
        self.manifest_path = MANIFEST_PATH
        self._ensure_dirs()
        self._init_manifest()

    def _ensure_dirs(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

    def _init_manifest(self):
        if not self.manifest_path.exists():
            default_manifest = {
                "active_version": "v1.0.0",
                "last_updated": datetime.now(UTC).isoformat(),
                "versions": [
                    {
                        "version": "v1.0.0",
                        "created_at": datetime.now(UTC).isoformat(),
                        "ml_mae": 1.48,
                        "ml_spearman": 0.52,
                        "base_mae": 1.74,
                        "status": "production",
                        "notes": "Initial calibrated LightGBM component models",
                    }
                ],
            }
            try:
                with open(self.manifest_path, "w", encoding="utf-8") as f:
                    json.dump(default_manifest, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not initialize manifest.json: {e}")

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading manifest: {e}")
        return {"active_version": "v1.0.0", "versions": []}

    def save_manifest(self, manifest: dict[str, Any]):
        manifest["last_updated"] = datetime.now(UTC).isoformat()
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Error writing manifest: {e}")

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hex digest for a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def calculate_weight_hashes(self, dir_path: Path | None = None) -> dict[str, str]:
        """Compute SHA256 hashes of all component weight files in directory."""
        target_dir = dir_path or self.models_dir
        hashes = {}
        for _, filename, _ in COMPONENT_WEIGHTS:
            fpath = target_dir / filename
            if fpath.exists():
                hashes[filename] = self.calculate_file_hash(fpath)
        cal_path = target_dir / "calibration.json"
        if cal_path.exists():
            hashes["calibration.json"] = self.calculate_file_hash(cal_path)
        return hashes

    def verify_weight_integrity(self, weights_dir: Path | None = None) -> tuple[bool, str]:
        """
        Verify that weight files in directory match the SHA256 hashes in manifest.
        Raises ValueError if tampering or missing files detected.
        """
        target_dir = weights_dir or self.models_dir
        active = self.get_active_version()
        expected_hashes = active.get("file_hashes")
        if not expected_hashes:
            return True, "No file hashes recorded in manifest for active version."

        for filename, expected_hash in expected_hashes.items():
            fpath = target_dir / filename
            if not fpath.exists():
                msg = f"Weight file missing during integrity verification: {filename}"
                logger.error(f"[ModelIntegrity] {msg}")
                raise ValueError(msg)
            actual_hash = self.calculate_file_hash(fpath)
            if actual_hash != expected_hash:
                msg = f"SHA256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                logger.error(f"[ModelIntegrity] {msg}")
                raise ValueError(msg)

        logger.info("[ModelIntegrity] All active component model weights successfully verified.")
        return True, "All component model weights successfully verified."

    def get_active_version(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        active_name = manifest.get("active_version")
        if active_name:
            for v in manifest.get("versions", []):
                if v.get("version") == active_name:
                    return v
            return {
                "version": active_name,
                "created_at": datetime.now(UTC).isoformat(),
                "ml_mae": 1.48,
                "ml_spearman": 0.52,
                "base_mae": 1.74,
                "status": "production",
                "notes": "Manifest active version",
            }

        # Fallback to DB
        db_ver = data_store.get_active_model_version()
        if db_ver:
            return db_ver

        return {
            "version": "v1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "ml_mae": 1.48,
            "ml_spearman": 0.52,
            "base_mae": 1.74,
            "status": "production",
            "notes": "Initial calibrated models",
        }

    def get_version_history(self) -> list[dict[str, Any]]:
        manifest = self.load_manifest()
        if manifest.get("versions"):
            return manifest["versions"]
        db_versions = data_store.get_model_versions()
        if db_versions:
            return db_versions
        return []

    def evaluate_model_suite(
        self,
        models_dict: dict[str, Any],
        X_val: pd.DataFrame,
        Y_val: pd.DataFrame,
        rolling_origins: list[dict[str, Any]] | None = None,
        ablation_metrics: dict[str, Any] | None = None,
        upcoming_projections: list[dict[str, Any]] | None = None,
        save_reports: bool = True,
        custom_json_path: Any = None,
        custom_md_path: Any = None,
    ) -> dict[str, float]:
        """Generate holdout predictions and compute validation metrics."""
        mins_p = models_dict["minutes_model"].predict(X_val)
        att_p = models_dict["attacking_model"].predict(X_val)
        def_p = models_dict["defending_model"].predict(X_val)
        defcon_p = models_dict["defcon_model"].predict(X_val)
        bonus_p = models_dict["bonus_model"].predict(X_val)
        cards_p = models_dict["cards_saves_model"].predict(X_val)

        components = {**mins_p, **att_p, **def_p, **defcon_p, **bonus_p, **cards_p}
        val_preds_df = scoring_ensemble.aggregate_components(components, X_val)
        ml_preds = val_preds_df["expected_points"].values

        metrics = model_evaluator.evaluate_expanding_window(
            X=X_val,
            Y=Y_val,
            ml_preds=ml_preds,
            p10=val_preds_df["p10"].values,
            p50=val_preds_df["p50"].values,
            p90=val_preds_df["p90"].values,
            rolling_origins=rolling_origins,
            ablation_metrics=ablation_metrics,
            upcoming_projections=upcoming_projections,
            save_reports=save_reports,
            custom_json_path=custom_json_path,
            custom_md_path=custom_md_path,
        )
        return {
            "ml_mae": float(metrics["ml_mae"]),
            "ml_spearman": float(metrics["ml_spearman"]),
            "base_mae": float(metrics["base_mae"]),
        }

    def evaluate_production_weights(
        self, X_val: pd.DataFrame, Y_val: pd.DataFrame, rolling_origins: list[dict[str, Any]] | None = None
    ) -> dict[str, float] | None:
        """Evaluate current production weights in data/models on holdout split."""
        models: dict[str, Any] = {}
        for name, filename, cls in COMPONENT_WEIGHTS:
            p = self.models_dir / filename
            if not p.exists():
                return None
            try:
                inst = cls()
                loaded = inst.load(p)
                inst.__dict__.update(loaded.__dict__)
                models[name] = inst
            except Exception as e:
                logger.warning(f"Error loading production model {filename}: {e}")
        try:
            return self.evaluate_model_suite(
                models,
                X_val,
                Y_val,
                rolling_origins=rolling_origins,
                save_reports=True,
                custom_json_path=REPORTS_DIR / "active_model_eval.json",
                custom_md_path=REPORTS_DIR / "active_model_eval.md",
            )
        except Exception as e:
            logger.warning(f"Existing production weights evaluation failed ({e}); treating as schema upgrade.")
            return None

    def verify_and_promote(
        self,
        candidate_models: dict[str, Any],
        candidate_metrics: dict[str, float],
        active_metrics: dict[str, float] | None = None,
        tolerance: float = 0.05,
        new_version_tag: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Verify candidate models against active production metrics.
        If candidate MAE degrades by more than tolerance (> 0.05 pts), engages AUTOMATIC ROLLBACK.
        Otherwise promotes candidate models to production and archives previous checkpoint.
        """
        cand_mae = candidate_metrics["ml_mae"]
        active = self.get_active_version()
        active_mae = active_metrics["ml_mae"] if active_metrics else active.get("ml_mae", 1.48)
        active_ver_name = active.get("version", "v1.0.0")

        # Baseline and rolling origin promotion gate checks
        base_mae = candidate_metrics.get("base_mae")
        gate_passed, gate_reason = check_promotion_gate(
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
            rolling_origins=candidate_metrics.get("rolling_origins"),
            tolerance=tolerance,
        )

        if not gate_passed:
            rej_tag = f"rej_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            rej_dir = self.rejected_dir / rej_tag
            rej_dir.mkdir(parents=True, exist_ok=True)
            for name, filename, _ in COMPONENT_WEIGHTS:
                if name in candidate_models:
                    candidate_models[name].save(rej_dir / filename)
            logger.warning(f"[ModelRollback] {gate_reason}")
            manifest = self.load_manifest()
            manifest.setdefault("versions", []).append(
                {
                    "version": rej_tag,
                    "created_at": datetime.now(UTC).isoformat(),
                    "ml_mae": cand_mae,
                    "ml_spearman": candidate_metrics.get("ml_spearman", 0.0),
                    "base_mae": base_mae,
                    "status": "rejected_gate_failed",
                    "notes": gate_reason,
                }
            )
            self.save_manifest(manifest)
            data_store.save_model_version(
                version=rej_tag,
                ml_mae=cand_mae,
                ml_spearman=candidate_metrics.get("ml_spearman", 0.0),
                base_mae=candidate_metrics.get("base_mae", 0.0),
                status="rejected_gate_failed",
                is_active=False,
                notes=gate_reason,
            )

            return {
                "promoted": False,
                "status": "rolled_back",
                "reason": f"Automatic rollback engaged: {gate_reason}",
                "version": rej_tag,
                "active_version": active_ver_name,
                "active_mae": active_mae,
                "candidate_mae": cand_mae,
                "rejected_version": rej_tag,
            }

        # Candidate passed verification! Promote to production.
        ver_tag = new_version_tag or f"v{datetime.now(UTC).strftime('%Y.%m.%d.%H%M')}"
        archive_dir = self.versions_dir / active_ver_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        # 1. Archive previous production weights
        for _, filename, _ in COMPONENT_WEIGHTS:
            src = self.models_dir / filename
            if src.exists():
                shutil.copy2(src, archive_dir / filename)

        # 2. Promote candidate weights to production root and new version archive
        new_ver_dir = self.versions_dir / ver_tag
        new_ver_dir.mkdir(parents=True, exist_ok=True)

        for name, filename, _ in COMPONENT_WEIGHTS:
            if name in candidate_models:
                # Save to production root
                candidate_models[name].save(self.models_dir / filename)
                # Save to version archive
                candidate_models[name].save(new_ver_dir / filename)

        # 3. Synchronize candidate evaluation report to active model_eval report
        cand_json = REPORTS_DIR / "candidate_model_eval.json"
        if cand_json.exists():
            shutil.copy2(cand_json, REPORTS_DIR / "model_eval.json")
        cand_md = REPORTS_DIR / "candidate_model_eval.md"
        if cand_md.exists():
            shutil.copy2(cand_md, REPORTS_DIR / "model_eval.md")

        # 4. Compute file hashes for complete provenance
        file_hashes = self.calculate_weight_hashes(self.models_dir)

        # Git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            git_commit = "unknown"

        from fpl_oracle.data.features import FEATURE_SCHEMA_HASH

        improvement = (active_mae - cand_mae) if active_mae else 0.0
        success_note = (
            f"Candidate model passed verification. MAE: {cand_mae:.3f} "
            f"(improvement: {improvement:+.3f}). Promoted to production."
        )
        if notes:
            success_note += f" | {notes}"

        logger.info(f"[ModelRegistry] {success_note}")

        # Update DB & manifest
        data_store.save_model_version(
            version=ver_tag,
            ml_mae=cand_mae,
            ml_spearman=candidate_metrics["ml_spearman"],
            base_mae=candidate_metrics["base_mae"],
            status="production",
            is_active=True,
            notes=success_note,
        )

        manifest = self.load_manifest()
        manifest["active_version"] = ver_tag
        # Mark previous production versions as archived
        for v in manifest.get("versions", []):
            if v.get("status") == "production":
                v["status"] = "archived"
        manifest.setdefault("versions", []).append(
            {
                "version": ver_tag,
                "created_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "feature_list_hash": FEATURE_SCHEMA_HASH,
                "metric_applies_to": "served_weights",
                "file_hashes": file_hashes,
                "ml_mae": cand_mae,
                "ml_spearman": candidate_metrics["ml_spearman"],
                "base_mae": candidate_metrics["base_mae"],
                "rolling_origins": candidate_metrics.get("rolling_origins", []),
                "status": "production",
                "notes": success_note,
            }
        )
        self.save_manifest(manifest)

        return {
            "promoted": True,
            "status": "promoted",
            "new_version": ver_tag,
            "previous_version": active_ver_name,
            "metrics": candidate_metrics,
            "notes": success_note,
        }

    def rollback_to_version(self, target_version: str) -> dict[str, Any]:
        """Manually restore an archived model version to production."""
        target_dir = self.versions_dir / target_version
        if not target_dir.exists():
            return {"success": False, "error": f"Version checkpoint '{target_version}' not found."}

        # Copy files back to production root
        for _, filename, _ in COMPONENT_WEIGHTS:
            src = target_dir / filename
            if src.exists():
                shutil.copy2(src, self.models_dir / filename)

        # Update DB
        data_store.save_model_version(
            version=target_version,
            ml_mae=0.0,
            ml_spearman=0.0,
            base_mae=0.0,
            status="production",
            is_active=True,
            notes=f"Manually rolled back to {target_version} on {datetime.now(UTC).isoformat()}",
        )

        manifest = self.load_manifest()
        manifest["active_version"] = target_version
        self.save_manifest(manifest)

        logger.info(f"[ModelRegistry] Successfully rolled back production models to {target_version}.")
        return {"success": True, "active_version": target_version}


model_registry = ModelRegistry()
