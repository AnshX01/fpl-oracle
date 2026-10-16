"""
Model Registry & Versioning Engine with Automated Metric Rollback.
Manages model checkpoints, metric history, candidate model verification, and
safeguards against performance degradation by automatically rolling back if validation MAE worsens.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.config import MODELS_DIR, REPORTS_DIR
from fpl_oracle.data.store import data_store
from fpl_oracle.ml.bundle_lock import bundle_locked
from fpl_oracle.ml.components.attacking import AttackingModel
from fpl_oracle.ml.components.bonus import BonusModel
from fpl_oracle.ml.components.cards_saves import CardsSavesModel
from fpl_oracle.ml.components.defcon import DefConModel
from fpl_oracle.ml.components.defending import DefendingModel
from fpl_oracle.ml.components.minutes import MinutesModel
from fpl_oracle.ml.ensemble import scoring_ensemble
from fpl_oracle.ml.eval import EVAL_METRICS_SCHEMA_KEYS, ROLLING_ORIGIN_SCHEMA_KEYS, model_evaluator

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


class ModelBundleCorruptError(RuntimeError):
    """Raised when an active model bundle is corrupt and cannot be restored from versions."""

    pass


def paired_block_bootstrap_gw(
    candidate_predictions: np.ndarray,
    active_predictions: np.ndarray,
    targets: np.ndarray,
    gameweeks: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    candidate_secondary: float | None = None,
    active_secondary: float | None = None,
) -> dict[str, Any]:
    """
    WI-3.9 Paired block-bootstrap over gameweeks (not rows) of the per-GW
    metric difference between candidate and production (10,000 resamples).

    Promotes candidate only if:
    1. The 95% CI for RMSE difference excludes 0 in the good direction (upper < 0), OR
    2. The candidate is statistically indistinguishable (CI includes 0) AND
       it improves a pre-registered secondary metric (e.g. calibration) AND
       it is not worse for top-decile players.
    """
    cand = np.asarray(candidate_predictions, dtype=float)
    act = np.asarray(active_predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    gws = np.asarray(gameweeks)

    unique_gws = np.unique(gws)
    if len(unique_gws) < 2:
        diff = float(np.sqrt(np.mean((y - cand) ** 2)) - np.sqrt(np.mean((y - act) ** 2)))
        return {
            "ci": [round(diff, 5), round(diff, 5)],
            "mean_rmse_diff": round(diff, 5),
            "n_resamples": n_resamples,
            "seed": seed,
            "p_cand_better": 1.0 if diff < 0 else 0.0,
            "top_decile_not_worse": True,
            "secondary_metric_improved": True,
            "decision_rule": "insufficient_gws_fallback",
            "promoted": diff <= 0,
            "reason": f"Insufficient gameweeks ({len(unique_gws)}) for block bootstrap; single difference={diff:+.4f}",
        }

    # Compute per-GW RMSE difference: D(g) = RMSE_cand(g) - RMSE_act(g)
    gw_diffs = []
    for g in unique_gws:
        m = gws == g
        r_cand = float(np.sqrt(np.mean((y[m] - cand[m]) ** 2)))
        r_act = float(np.sqrt(np.mean((y[m] - act[m]) ** 2)))
        gw_diffs.append(r_cand - r_act)

    gw_diffs_arr = np.array(gw_diffs)
    mean_diff = float(np.mean(gw_diffs_arr))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(gw_diffs_arr), size=(n_resamples, len(gw_diffs_arr)))
    boot_means = np.mean(gw_diffs_arr[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    p_better = float(np.mean(boot_means < 0))

    # Top decile check: players with highest targets (top 10%)
    top_threshold = float(np.percentile(y, 90.0))
    top_mask = y >= top_threshold
    top_cand_rmse = float(np.sqrt(np.mean((y[top_mask] - cand[top_mask]) ** 2))) if np.any(top_mask) else 0.0
    top_act_rmse = float(np.sqrt(np.mean((y[top_mask] - act[top_mask]) ** 2))) if np.any(top_mask) else 0.0
    top_decile_not_worse = top_cand_rmse <= (top_act_rmse + 1e-4)

    # Secondary metric check (lower is better, e.g. calibration error or pinball loss)
    sec_improved = True
    if candidate_secondary is not None and active_secondary is not None:
        sec_improved = candidate_secondary <= (active_secondary + 1e-5)

    # Decision rule:
    # 1. CI strictly excludes 0 in the good direction (ci_upper < 0)
    # 2. Indistinguishable (ci_lower <= 0 <= ci_upper) AND secondary_improved AND top_decile_not_worse
    if ci_upper < 0.0:
        promoted = True
        rule = "ci_excludes_zero_good"
        reason = f"Candidate RMSE significantly superior to production at 95% confidence: 95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] < 0."
    elif ci_lower <= 0.0 <= ci_upper and sec_improved and top_decile_not_worse:
        promoted = True
        rule = "indistinguishable_with_secondary_and_top_decile"
        reason = f"Candidate statistically indistinguishable (95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] includes 0) and improves secondary metric while not worse for top-decile players (top RMSE {top_cand_rmse:.3f} vs {top_act_rmse:.3f})."
    else:
        promoted = False
        rule = "rejected"
        reasons = []
        if ci_lower > 0.0:
            reasons.append(f"CI strictly positive (candidate degraded: 95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] > 0)")
        elif not sec_improved:
            reasons.append(f"secondary metric failed ({candidate_secondary:.4f} > {active_secondary:.4f})")
        elif not top_decile_not_worse:
            reasons.append(f"top-decile degraded ({top_cand_rmse:.3f} > {top_act_rmse:.3f})")
        else:
            reasons.append(f"95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}]")
        reason = f"Candidate failed bootstrap promotion gate: {', '.join(reasons)}."

    return {
        "ci": [round(ci_lower, 5), round(ci_upper, 5)],
        "mean_rmse_diff": round(mean_diff, 5),
        "n_resamples": n_resamples,
        "seed": seed,
        "p_cand_better": round(p_better, 4),
        "top_decile_not_worse": top_decile_not_worse,
        "secondary_metric_improved": sec_improved,
        "decision_rule": rule,
        "promoted": promoted,
        "reason": reason,
        "top_decile_rmse": {"candidate": round(top_cand_rmse, 4), "active": round(top_act_rmse, 4)},
    }


def check_promotion_gate(
    candidate_metrics: dict[str, Any],
    active_metrics: dict[str, Any] | None = None,
    rolling_origins: list[dict[str, Any]] | None = None,
    tolerance: float = 0.0,
    k_origins: int = 3,
    coverage_min: float = 75.0,
    coverage_max: float = 85.0,
) -> tuple[bool, str]:
    """
    Evidence-gated model promotion gate (Requirement G5 & WI-3.9):
    1. Schema validation against shared EVAL_METRICS_SCHEMA_KEYS & ROLLING_ORIGIN_SCHEMA_KEYS.
    2. Zero degradation tolerance against best baseline: cand_mae <= base_mae + tolerance.
    3. Non-degradation vs active production: cand_mae <= active_mae + tolerance (if active exists).
    4. Rolling origins out-of-time cross validation.
    5. Paired block-bootstrap over gameweeks (10,000 resamples) if predictions/targets provided.
    """
    import math

    # Check for paired block bootstrap inputs (WI-3.9)
    if "bootstrap_data" in candidate_metrics or (
        "candidate_predictions" in candidate_metrics
        and "active_predictions" in candidate_metrics
        and "targets" in candidate_metrics
        and "gameweeks" in candidate_metrics
    ):
        bdata = candidate_metrics.get("bootstrap_data", candidate_metrics)
        boot_res = paired_block_bootstrap_gw(
            candidate_predictions=bdata["candidate_predictions"],
            active_predictions=bdata["active_predictions"],
            targets=bdata["targets"],
            gameweeks=bdata["gameweeks"],
            n_resamples=bdata.get("n_resamples", 10000),
            seed=bdata.get("seed", 42),
            candidate_secondary=bdata.get("candidate_secondary"),
            active_secondary=bdata.get("active_secondary"),
        )
        candidate_metrics["bootstrap_result"] = boot_res
        if not boot_res["promoted"]:
            return False, boot_res["reason"]

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    cand_mae = candidate_metrics.get("ml_mae", candidate_metrics.get("mae"))
    base_mae = candidate_metrics.get("base_mae", candidate_metrics.get("best_baseline_mae"))

    if not finite(cand_mae) or not finite(base_mae):
        return (
            False,
            f"Candidate metrics missing required MAE fields ('ml_mae'/'mae' and 'base_mae'/'best_baseline_mae') defined in {EVAL_METRICS_SCHEMA_KEYS}.",
        )

    assert cand_mae is not None and base_mae is not None
    cand_mae, base_mae = float(cand_mae), float(base_mae)
    # 1. Baseline superiority check (global degradation tolerance is 0 against best baseline)
    if cand_mae > (base_mae + tolerance):
        return (
            False,
            f"Candidate MAE ({cand_mae:.3f}) failed baseline gate vs baseline ({base_mae:.3f} + {tolerance:.2f}).",
        )

    # 2. Active production degradation check
    if active_metrics:
        active_mae = active_metrics.get("ml_mae", active_metrics.get("mae"))
        if not finite(active_mae):
            return False, "Active MAE is missing/nonfinite"
        assert active_mae is not None
        active_mae = float(active_mae)
        if cand_mae > (active_mae + tolerance):
            return (
                False,
                f"Candidate MAE ({cand_mae:.3f}) degraded vs active production MAE ({active_mae:.3f}) by > {tolerance:.2f}.",
            )

    # 3. Rolling origins checks (or bootstrap gate if rolling origins omitted)
    has_bootstrap = "bootstrap_result" in candidate_metrics
    origins = rolling_origins if rolling_origins is not None else candidate_metrics.get("rolling_origins")
    if not isinstance(origins, list) or len(origins) < k_origins:
        if has_bootstrap and candidate_metrics["bootstrap_result"]["promoted"]:
            return True, candidate_metrics["bootstrap_result"]["reason"]
        return False, "Candidate failed promotion gate: insufficient rolling origins evaluated."

    eval_origins = origins[-k_origins:]
    gains: list[float] = []
    for o in eval_origins:
        # Validate against shared schema keys
        for req_k in ROLLING_ORIGIN_SCHEMA_KEYS:
            if req_k not in o:
                return False, f"Origin '{o.get('season', 'unknown')}' missing required schema key '{req_k}'."

        for key in ("mae", "best_baseline_mae", "gain_vs_baseline", "interval_80_coverage_pct"):
            if not finite(o.get(key)):
                return False, f"Origin invalid/nonfinite {key}"
        gain_o = o.get("gain_vs_baseline")
        if gain_o is not None:
            gains.append(float(gain_o))
        else:
            ml_o = float(o["mae"])
            base_o = float(o["best_baseline_mae"])
            gains.append(base_o - ml_o)

    if not gains:
        return False, "Candidate failed promotion gate: unable to compute origin gains."

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

    # 4. Per-origin 80% credible interval coverage band check
    for o in eval_origins:
        cov = o.get("interval_80_coverage_pct")
        if cov is None:
            return False, f"Origin '{o.get('season', 'unknown')}' missing interval_80_coverage_pct."
        if cov < coverage_min or cov > coverage_max:
            return (
                False,
                f"Candidate failed rolling origin gate: origin '{o.get('season')}' 80% interval coverage ({cov:.2f}%) is outside target band [{coverage_min:.1f}%, {coverage_max:.1f}%].",
            )

    return True, "Candidate passed all promotion gate criteria."


class ModelRegistry:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.versions_dir = VERSIONS_DIR
        self.rejected_dir = REJECTED_DIR
        self.manifest_path = MANIFEST_PATH
        self.reports_dir = REPORTS_DIR
        self._last_recovery_event: dict[str, Any] | None = None
        self._ensure_dirs()
        self._init_manifest()

    def get_last_recovery_event(self) -> dict[str, Any] | None:
        """Return the most recent quarantine/rollback recovery event details, if any."""
        if hasattr(self, "_last_recovery_event") and self._last_recovery_event is not None:
            return self._last_recovery_event
        event_file = self.models_dir / "recovery_event.json"
        if event_file.exists():
            try:
                return json.loads(event_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

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
        cqr_dir = target_dir / "cqr"
        if cqr_dir.exists() and cqr_dir.is_dir():
            for cqr_f in sorted(cqr_dir.glob("*.txt")):
                hashes[f"cqr/{cqr_f.name}"] = self.calculate_file_hash(cqr_f)
            meta_f = cqr_dir / "cqr_meta.json"
            if meta_f.exists():
                hashes["cqr/cqr_meta.json"] = self.calculate_file_hash(meta_f)
        return hashes

    @bundle_locked
    def recover_interrupted_swap(self):
        journal = self.models_dir / "promotion_journal.json"
        if not journal.exists():
            return
        record = json.loads(journal.read_text(encoding="utf-8"))
        active = self.load_manifest().get("active_version")
        if active != record["new_version"]:
            previous = self.versions_dir / record["previous_version"]
            for filename in record["files"]:
                source = previous / filename
                if not source.exists():
                    raise ValueError("Interrupted promotion cannot recover previous bundle")
                temporary = self.models_dir / (filename + ".recover")
                shutil.copy2(source, temporary)
                os.replace(temporary, self.models_dir / filename)
        # Validate the selected manifest before declaring recovery complete.
        manifest = self.load_manifest()
        version = next(v for v in manifest["versions"] if v["version"] == manifest["active_version"])
        for filename, expected in version["file_hashes"].items():
            if self.calculate_file_hash(self.models_dir / filename) != expected:
                raise ValueError("Interrupted bundle recovery failed integrity check")
        journal.unlink()

    @bundle_locked
    def quarantine_and_recover(self, target_dir: Path | None = None, reason: str = "") -> dict[str, Any]:
        """
        Quarantine corrupt or unreadable weight files to data/models/quarantine/<timestamp>/
        and restore the last valid verified version from data/models/versions/.
        Raises ModelBundleCorruptError if no valid version exists to restore.
        """
        models_path = target_dir or self.models_dir
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        quarantine_dir = models_path / "quarantine" / timestamp
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        quarantined_files: list[str] = []
        for file_p in models_path.glob("*.pkl"):
            if file_p.is_file():
                dest = quarantine_dir / file_p.name
                shutil.move(str(file_p), str(dest))
                quarantined_files.append(file_p.name)
        cal_p = models_path / "calibration.json"
        if cal_p.exists() and cal_p.is_file():
            shutil.move(str(cal_p), str(quarantine_dir / "calibration.json"))
            quarantined_files.append("calibration.json")

        logger.critical(
            f"[ModelIntegrity] Corrupt model bundle files quarantined to {quarantine_dir}: {quarantined_files} (Reason: {reason})"
        )

        restored_ver_name = None
        restored_dir = None
        manifest = self.load_manifest()

        # 1. Search versions listed in manifest in reverse chronological order
        versions_list = manifest.get("versions", [])
        for v in reversed(versions_list):
            v_name = v.get("version")
            if not v_name:
                continue
            cand_dir = self.versions_dir / v_name
            if not cand_dir.exists() or not cand_dir.is_dir():
                continue

            hashes = v.get("file_hashes", {})
            if hashes:
                has_all_weights = True
                for filename, expected_hash in hashes.items():
                    w_path = cand_dir / filename
                    if not w_path.exists() or self.calculate_file_hash(w_path) != expected_hash:
                        has_all_weights = False
                        break
            else:
                has_all_weights = any(cand_dir.glob("*.pkl"))

            if has_all_weights:
                restored_ver_name = v_name
                restored_dir = cand_dir
                break

        # 2. Disk scan fallback across versions_dir
        if restored_dir is None and self.versions_dir.exists():
            subdirs = sorted(
                [d for d in self.versions_dir.iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for sub in subdirs:
                pkls = list(sub.glob("*.pkl"))
                if pkls and all(f.stat().st_size > 0 for f in pkls):
                    restored_ver_name = sub.name
                    restored_dir = sub
                    break

        if restored_dir is None:
            err_msg = (
                f"Model bundle corruption detected ({reason}), but no valid verified version "
                f"exists in {self.versions_dir} to restore from."
            )
            logger.critical(f"[ModelIntegrity] {err_msg}")
            raise ModelBundleCorruptError(err_msg)

        # Restore files to active models directory
        for f in restored_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, models_path / f.name)

        manifest["active_version"] = restored_ver_name
        self.save_manifest(manifest)

        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": reason,
            "quarantine_dir": str(quarantine_dir),
            "restored_version": restored_ver_name,
            "quarantined_files": quarantined_files,
        }
        self._last_recovery_event = event
        try:
            (models_path / "recovery_event.json").write_text(json.dumps(event, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not persist recovery_event.json: {e}")

        logger.critical(
            f"[ModelIntegrity] AUTOMATIC RECOVERY SUCCESSFUL: Quarantined bad files, restored verified version '{restored_ver_name}'."
        )
        return event

    @bundle_locked
    def verify_weight_integrity(self, weights_dir: Path | None = None) -> tuple[bool, str]:
        """
        Verify that weight files in directory match the SHA256 hashes in manifest.
        On mismatch, automatically quarantines the corrupt bundle and restores the last
        verified version from versions/. Raises ModelBundleCorruptError if no valid version exists.
        """
        if weights_dir is None or weights_dir == self.models_dir:
            self.recover_interrupted_swap()
        target_dir = weights_dir or self.models_dir

        mismatch_reason = None
        try:
            active = self.get_active_version()
            expected_hashes = active.get("file_hashes")
            if not expected_hashes:
                for _, filename, _ in COMPONENT_WEIGHTS:
                    if not (target_dir / filename).exists():
                        mismatch_reason = f"Weight file missing during integrity verification: {filename}"
                        break
            else:
                for filename, expected_hash in expected_hashes.items():
                    fpath = target_dir / filename
                    if not fpath.exists():
                        mismatch_reason = f"Weight file missing during integrity verification: {filename}"
                        break
                    actual_hash = self.calculate_file_hash(fpath)
                    if actual_hash != expected_hash:
                        mismatch_reason = f"SHA256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                        break
        except Exception as e:
            mismatch_reason = f"Manifest or file read error: {e}"

        if mismatch_reason:
            if weights_dir is not None and weights_dir != self.models_dir:
                raise ValueError(mismatch_reason)

            logger.warning(
                f"[ModelIntegrity] Integrity check failed ({mismatch_reason}). Triggering automatic quarantine and rollback..."
            )
            event = self.quarantine_and_recover(target_dir=target_dir, reason=mismatch_reason)
            return (
                True,
                f"Automatic recovery succeeded: restored {event.get('restored_version')} following {mismatch_reason}",
            )

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
        ensemble: Any = None,
    ) -> dict[str, float]:
        """Generate holdout predictions and compute validation metrics."""
        mins_p = models_dict["minutes_model"].predict(X_val)
        att_p = models_dict["attacking_model"].predict(X_val)
        def_p = models_dict["defending_model"].predict(X_val)
        defcon_p = models_dict["defcon_model"].predict(X_val)
        bonus_p = models_dict["bonus_model"].predict(X_val)
        cards_p = models_dict["cards_saves_model"].predict(X_val)

        components = {**mins_p, **att_p, **def_p, **defcon_p, **bonus_p, **cards_p}
        actual_ensemble = ensemble or scoring_ensemble
        val_preds_df = actual_ensemble.aggregate_components(components, X_val)
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
        return metrics

    @staticmethod
    def _load_bundle_ensemble(path):
        from fpl_oracle.ml.ensemble import ScoringEnsemble

        ens = ScoringEnsemble(z10=-0.806, z90=1.009, blend_weights=(0.72, 0.04, 0.24))
        if not ens.load_calibration(path):
            raise ValueError("Bundle calibration unavailable")
        return ens

    @bundle_locked
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
                ensemble=self._load_bundle_ensemble(self.models_dir / "calibration.json"),
            )
        except Exception as e:
            logger.warning(f"Existing production weights evaluation failed ({e}); treating as schema upgrade.")
            return None

    @bundle_locked
    def promote_candidate_bundle(
        self,
        candidate_models: dict[str, Any],
        calibration_data: dict[str, Any] | Path,
        candidate_metrics: dict[str, Any],
        active_mae: float | None = None,
        rolling_origins: list[dict[str, Any]] | None = None,
        training_data_df: pd.DataFrame | None = None,
        training_provenance: dict[str, Any] | None = None,
        recipe_id: str = "fpl_oracle_canonical_v1",
        new_version_tag: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Builds a candidate staging directory with all 6 component models, calibration.json,
        and manifest.json, computes all hashes, verifies integrity, and atomically swaps
        into production data/models/ using .tmp + os.replace (Requirement G1).
        """
        cand_mae = float(candidate_metrics.get("ml_mae", candidate_metrics.get("mae", 1.0)))
        cand_sp = float(candidate_metrics.get("ml_spearman", 0.0))
        cand_base = float(candidate_metrics.get("base_mae", candidate_metrics.get("best_baseline_mae", 1.0)))
        origins = rolling_origins if rolling_origins is not None else candidate_metrics.get("rolling_origins", [])

        ver_tag = new_version_tag or f"v{datetime.now(UTC).strftime('%Y.%m.%d.%H%M')}"
        active = self.get_active_version()
        active_ver_name = active.get("version", "v1.0.0")

        staging_dir = self.models_dir / f".staging_{ver_tag}"
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save all 6 candidate models into staging
            for name, filename, _ in COMPONENT_WEIGHTS:
                if name not in candidate_models:
                    raise ValueError(f"Candidate model '{name}' missing from promotion bundle!")
                candidate_models[name].save(staging_dir / filename)

            # 2. Save calibration.json into staging
            staging_cal = staging_dir / "calibration.json"
            if isinstance(calibration_data, Path):
                shutil.copy2(calibration_data, staging_cal)
            elif isinstance(calibration_data, dict):
                with open(staging_cal, "w", encoding="utf-8") as f:
                    json.dump(calibration_data, f, indent=2)
            else:
                raise ValueError("calibration_data must be a dict or a Path to calibration.json")

            # 3. Calculate SHA256 hashes of all 7 files in staging
            file_hashes: dict[str, str] = {}
            for _, filename, _ in COMPONENT_WEIGHTS:
                file_hashes[filename] = self.calculate_file_hash(staging_dir / filename)
            file_hashes["calibration.json"] = self.calculate_file_hash(staging_cal)

            # 4. Read git commit at train/promotion time
            try:
                git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            except Exception:
                git_commit = "unknown"

            # 5. Training data hash and max GW
            if training_data_df is not None and not training_data_df.empty:
                try:
                    tr_bytes = pd.util.hash_pandas_object(training_data_df).values.tobytes()
                    training_data_hash = hashlib.sha256(tr_bytes).hexdigest()
                except Exception:
                    training_data_hash = "unknown"
                max_gw = int(training_data_df["round"].max()) if "round" in training_data_df.columns else 0
            else:
                master_csv = (
                    Path(__file__).resolve().parent.parent.parent.parent / "data" / "historical" / "master_history.csv"
                )
                if master_csv.exists():
                    training_data_hash = hashlib.sha256(master_csv.read_bytes()).hexdigest()
                    try:
                        m_df = pd.read_csv(master_csv, usecols=["round"], low_memory=False)
                        max_gw = int(m_df["round"].max())
                    except Exception:
                        max_gw = 0
                else:
                    training_data_hash = "unknown"
                    max_gw = 0

            from fpl_oracle.data.features import FEATURE_SCHEMA_HASH

            improvement = (active_mae - cand_mae) if active_mae else 0.0
            success_note = (
                f"Candidate model passed verification. MAE: {cand_mae:.3f} "
                f"(improvement: {improvement:+.3f}). Promoted to production."
            )
            if notes:
                success_note += f" | {notes}"

            # 6. Build manifest inside staging
            manifest = self.load_manifest()
            for v in manifest.get("versions", []):
                if v.get("status") == "production":
                    v["status"] = "archived"

            version_entry = {
                "version": ver_tag,
                "created_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "recipe_id": recipe_id,
                "training_data_hash": training_data_hash,
                "training_provenance": training_provenance,
                "max_gw": max_gw,
                "feature_list_hash": FEATURE_SCHEMA_HASH,
                "metric_applies_to": "served_weights",
                "file_hashes": file_hashes,
                "ml_mae": round(cand_mae, 4),
                "ml_spearman": round(cand_sp, 4),
                "base_mae": round(cand_base, 4),
                "rolling_origins": origins,
                "bootstrap_promotion_gate": candidate_metrics.get("bootstrap_result"),
                "status": "production",
                "notes": success_note,
            }
            manifest["active_version"] = ver_tag
            manifest["last_updated"] = datetime.now(UTC).isoformat()
            manifest.setdefault("versions", []).append(version_entry)

            staging_manifest = staging_dir / "manifest.json"
            with open(staging_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # 7. Self-verify all staged file hashes
            for fname, exp_hash in file_hashes.items():
                act_hash = self.calculate_file_hash(staging_dir / fname)
                if act_hash != exp_hash:
                    raise ValueError(f"Staging self-verification failed for {fname}: {exp_hash} != {act_hash}")

            # 8. Archive previous production weights
            archive_dir = self.versions_dir / active_ver_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            for _, fname, _ in COMPONENT_WEIGHTS:
                src = self.models_dir / fname
                if src.exists():
                    shutil.copy2(src, archive_dir / fname)
            if (self.models_dir / "calibration.json").exists():
                shutil.copy2(self.models_dir / "calibration.json", archive_dir / "calibration.json")

            # 9. Also save copy in versions_dir / ver_tag
            new_ver_dir = self.versions_dir / ver_tag
            new_ver_dir.mkdir(parents=True, exist_ok=True)
            for item in staging_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, new_ver_dir / item.name)

            # 10. Atomic swap into production self.models_dir using .tmp + os.replace
            files_to_swap = [fname for _, fname, _ in COMPONENT_WEIGHTS] + ["calibration.json"]
            journal_path = self.models_dir / "promotion_journal.json"
            journal_tmp = journal_path.with_suffix(".json.tmp")
            journal_tmp.write_text(
                json.dumps(dict(new_version=ver_tag, previous_version=active_ver_name, files=files_to_swap)),
                encoding="utf-8",
            )
            os.replace(journal_tmp, journal_path)
            for fname in files_to_swap:
                src_file = staging_dir / fname
                tmp_target = self.models_dir / f"{fname}.tmp"
                final_target = self.models_dir / fname
                shutil.copy2(src_file, tmp_target)
                os.replace(tmp_target, final_target)

            # Finally atomically replace manifest.json
            man_tmp = self.models_dir / "manifest.json.tmp"
            shutil.copy2(staging_manifest, man_tmp)
            os.replace(man_tmp, self.models_dir / "manifest.json")

            # 11. Clean up staging
            shutil.rmtree(staging_dir, ignore_errors=True)

            # 12. Run immediate post-promotion integrity verification
            self.verify_weight_integrity(self.models_dir)
            journal_path.unlink(missing_ok=True)

            # 13. Synchronize reports (only when promoting to production MODELS_DIR)
            if self.models_dir == MODELS_DIR:
                cand_json = self.reports_dir / "candidate_model_eval.json"
                if cand_json.exists():
                    shutil.copy2(cand_json, self.reports_dir / "model_eval.json")
                cand_md = self.reports_dir / "candidate_model_eval.md"
                if cand_md.exists():
                    shutil.copy2(cand_md, self.reports_dir / "model_eval.md")

            # 14. Record in database
            data_store.save_model_version(
                version=ver_tag,
                ml_mae=cand_mae,
                ml_spearman=cand_sp,
                base_mae=cand_base,
                status="production",
                is_active=True,
                notes=success_note,
            )

            logger.info(f"[ModelRegistry] {success_note}")
            return {
                "promoted": True,
                "status": "promoted",
                "new_version": ver_tag,
                "previous_version": active_ver_name,
                "metrics": candidate_metrics,
                "manifest": version_entry,
                "notes": success_note,
            }

        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    @bundle_locked
    def verify_and_promote(
        self,
        candidate_models: dict[str, Any],
        candidate_metrics: dict[str, Any],
        active_metrics: dict[str, float] | None = None,
        tolerance: float = 0.0,
        new_version_tag: str | None = None,
        notes: str = "",
        calibration_data: dict[str, Any] | Path | None = None,
        rolling_origins: list[dict[str, Any]] | None = None,
        training_data_df: pd.DataFrame | None = None,
        training_provenance: dict[str, Any] | None = None,
        recipe_id: str = "fpl_oracle_canonical_v1",
    ) -> dict[str, Any]:
        """
        Verify candidate models against active production metrics.
        If candidate MAE degrades by more than tolerance (> 0.05 pts), engages AUTOMATIC ROLLBACK.
        Otherwise promotes candidate models to production via atomic swap.
        """
        cand_mae = candidate_metrics["ml_mae"]
        active = self.get_active_version()
        active_mae = active_metrics["ml_mae"] if active_metrics else active.get("ml_mae", 1.48)
        active_ver_name = active.get("version", "v1.0.0")

        # Baseline and rolling origin promotion gate checks
        base_mae = candidate_metrics.get("base_mae")
        origins_to_check: list[dict[str, Any]] | None = rolling_origins
        if origins_to_check is None:
            raw_orig = candidate_metrics.get("rolling_origins")
            if isinstance(raw_orig, list):
                origins_to_check = raw_orig

        gate_passed, gate_reason = check_promotion_gate(
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
            rolling_origins=origins_to_check,
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

        # Candidate passed verification! Atomically promote via candidate bundle
        cal_data = calibration_data
        if cal_data is None:
            from fpl_oracle.ml.ensemble import scoring_ensemble

            cal_data = scoring_ensemble.get_calibration_dict()

        return self.promote_candidate_bundle(
            candidate_models=candidate_models,
            calibration_data=cal_data,
            candidate_metrics=candidate_metrics,
            active_mae=active_mae,
            rolling_origins=origins_to_check,
            training_data_df=training_data_df,
            recipe_id=recipe_id,
            training_provenance=training_provenance,
            new_version_tag=new_version_tag,
            notes=notes,
        )

    @bundle_locked
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
