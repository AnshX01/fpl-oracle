"""
Forward Holdout Logging & Scoring Module (Requirement G6).

Guarantees honest forward evaluation of the FPL Oracle ML model on unseen future gameweeks:
1. Pre-deadline freeze: Before each gameweek deadline, serializes served predictions
   to data/holdout/gw<N>_predictions.json with manifest hash, model version, and UTC timestamp.
2. Post-gameweek scoring: Once the gameweek concludes and official points finalize,
   scores frozen predictions against actual points and logs metrics to reports/holdout_log.json.
3. Honesty guarantee: GW1-5 is acknowledged as contaminated by prior inspection.
   Forward scoring commences strictly with GW6 onwards.
"""

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import game_state_manager
from fpl_oracle.config import DATA_DIR, HOLDOUT_DIR, REPORTS_DIR
from fpl_oracle.ml.model_registry import model_registry

logger = logging.getLogger("fpl_oracle.ml.holdout")

HOLDOUT_LOG_PATH = REPORTS_DIR / "holdout_log.json"


def ensure_holdout_log_initialized(next_gw: int = 6) -> dict[str, Any]:
    """Ensure reports/holdout_log.json exists with honest status accounting."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if HOLDOUT_LOG_PATH.exists():
        try:
            with open(HOLDOUT_LOG_PATH, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "evaluated_gameweeks" in data:
                    return data
        except Exception as e:
            logger.warning(f"Error reading existing holdout log ({e}); reinitializing.")

    initial_log = {
        "status": "AWAITING_FORWARD_GAMEWEEKS",
        "evaluated_gameweeks": [],
        "note": (
            "No forward holdout result exists yet. GW1-5 data is contaminated by prior model inspection; "
            "honest forward scoring commences once GW6 concludes."
        ),
        "next_scheduled_freeze_gw": next_gw,
        "last_updated": datetime.now(UTC).isoformat(),
    }
    with open(HOLDOUT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(initial_log, f, indent=2)
    return initial_log


def compute_manifest_sha256() -> str:
    """Compute current SHA256 of production manifest.json."""
    man_path = DATA_DIR / "models" / "manifest.json"
    if not man_path.exists():
        return "unmanifested"
    sha = hashlib.sha256()
    with open(man_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


async def freeze_predictions(
    target_gw: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    custom_holdout_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Freeze served predictions for upcoming gameweek to data/holdout/gw<N>_predictions.json.
    """
    out_dir = custom_holdout_dir or HOLDOUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_gw is None:
        _, next_gw = await fpl_client.get_current_and_next_gw()
        if next_gw is None:
            raise ValueError("No official upcoming GW")
        target_gw = next_gw

    target_file = out_dir / f"gw{target_gw}_predictions.json"
    if target_file.exists():
        logger.info(f"Predictions already frozen for GW{target_gw} at {target_file}. Skipping freeze.")
        with open(target_file, encoding="utf-8") as f:
            existing = json.load(f)
        if (
            existing.get("schema_version") != 2
            or existing.get("source_kind") != "official_forward"
            or existing.get("gameweek") != target_gw
        ):
            raise ValueError("Existing freeze is legacy/invalid; retained unchanged, not valid forward evidence")
        frozen = datetime.fromisoformat(existing["frozen_at"].replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(existing["deadline_utc"].replace("Z", "+00:00"))
        if frozen.tzinfo is None or deadline.tzinfo is None or frozen >= deadline or not existing.get("predictions"):
            raise ValueError("Existing freeze is not valid pre-deadline evidence")
        return existing

    # Strict target event deadline, never a clamped seconds counter or default GW.
    import subprocess

    from fpl_oracle.server.analysis import analysis_service

    bootstrap, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
    fixtures, fix_stale = await fpl_client.get_fixtures(force_refresh=True)
    event = next((e for e in bootstrap.events if e.id == target_gw), None)
    if event is None or not event.deadline_time or stale or fix_stale:
        raise ValueError("Cannot freeze without fresh target event and official deadline")
    deadline = datetime.fromisoformat(event.deadline_time.replace("Z", "+00:00"))
    if deadline.tzinfo is None or datetime.now(UTC) >= deadline:
        raise ValueError("Late/unknown deadline: forward freeze forbidden")
    active_version = model_registry.get_active_version()
    manifest_hash = compute_manifest_sha256()
    if manifest_hash == "unmanifested":
        raise ValueError("Missing manifest")
    model_ver = active_version.get("version")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    horizon = await analysis_service.projections(target_gw, 5, bootstrap, fixtures)
    preds_df = horizon.get(target_gw)
    if preds_df is None:
        raise ValueError("Missing target projection snapshot")
    snapshot = {
        "snapshot_id": preds_df.attrs.get("snapshot_id"),
        "inputs": (
            json.loads(preds_df.attrs["snapshot_inputs_json"])
            if "snapshot_inputs_json" in preds_df.attrs
            else preds_df.attrs.get("snapshot_inputs")
        ),
    }
    if not snapshot["snapshot_id"] or not snapshot["inputs"]:
        raise ValueError("Projection snapshot provenance missing")

    if preds_df.empty:
        raise ValueError(f"Unable to generate predictions for GW{target_gw} (empty DataFrame returned).")

    # 4. Serialize player projections
    player_projections = []
    for _, row in preds_df.iterrows():
        player_projections.append(
            {
                "element": int(row["element"]),
                "web_name": str(row.get("web_name", "")),
                "team": str(row.get("team", "")),
                "position": str(row.get("position", "")),
                "expected_points": round(float(row["expected_points"]), 3),
                "p10": round(float(row.get("p10", 0.0)), 3),
                "p50": round(float(row.get("p50", 0.0)), 3),
                "p90": round(float(row.get("p90", 0.0)), 3),
                "opponent_difficulty": float(row.get("opponent_difficulty", 3.0)),
                "chance_of_playing": float(row.get("chance_of_playing", 1.0)),
            }
        )

    # Inference can finish after the deadline even if it began before it.
    if datetime.now(UTC) >= deadline or manifest_hash != compute_manifest_sha256():
        raise ValueError("Deadline passed or model changed during inference; freeze forbidden")
    freeze_payload = {
        "gameweek": target_gw,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model_version": model_ver,
        "git_commit": git_commit,
        "manifest_hash": manifest_hash,
        "schema_version": 2,
        "source_kind": "official_forward",
        "deadline_utc": deadline.isoformat(),
        "snapshot_id": snapshot["snapshot_id"],
        "snapshot_inputs": snapshot["inputs"],
        "player_count": len(player_projections),
        "predictions": player_projections,
    }

    if not dry_run:
        if datetime.now(UTC) >= deadline:
            raise ValueError("Deadline passed before write; freeze forbidden")
        tmp_file = out_dir / f"gw{target_gw}_predictions.json.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(freeze_payload, f, indent=2)
        os.replace(tmp_file, target_file)
        logger.info(
            f"Successfully frozen {len(player_projections)} player predictions for GW{target_gw} to {target_file}"
        )
    else:
        logger.info(f"[Dry Run] Generated freeze payload for GW{target_gw} ({len(player_projections)} players).")

    return freeze_payload


class FinalizedActuals(dict):
    """Only official retrieval constructs this marker. Plain maps are research-only."""

    def __init__(self, gw, points, fetched_at):
        super().__init__(points)
        self.gameweek = gw
        self.fetched_at = fetched_at
        self.source_kind = "official_finalized_live"


async def fetch_finalized_actuals(gw: int):
    bootstrap, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
    fixtures, fix_stale = await fpl_client.get_fixtures(event=gw, force_refresh=True)
    event = next((e for e in bootstrap.events if e.id == gw), None)
    if (
        stale
        or fix_stale
        or event is None
        or not event.finished
        or not event.data_checked
        or not fixtures
        or not all(f.finished for f in fixtures)
    ):
        return None
    live, live_stale = await fpl_client.get_live_gameweek(gw, force_refresh=True)
    if live_stale or not isinstance(live.get("elements"), list):
        return None
    return FinalizedActuals(
        gw, {int(e["id"]): float(e["stats"]["total_points"]) for e in live["elements"]}, datetime.now(UTC).isoformat()
    )


def score_frozen_predictions(
    gw: int,
    actual_points_map: dict[int, float] | None = None,
    dry_run: bool = False,
    custom_holdout_dir: Path | None = None,
    custom_log_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Score frozen predictions for a finished gameweek against actual outcomes.
    Appends metrics to reports/holdout_log.json.
    """
    holdout_dir = custom_holdout_dir or HOLDOUT_DIR
    log_path = custom_log_path or HOLDOUT_LOG_PATH
    freeze_path = holdout_dir / f"gw{gw}_predictions.json"

    if not freeze_path.exists():
        logger.warning(f"No frozen predictions found for GW{gw} at {freeze_path}. Cannot score.")
        return None

    with open(freeze_path, encoding="utf-8") as f:
        freeze_data = json.load(f)

    preds_list = freeze_data.get("predictions", [])
    if not preds_list:
        logger.warning(f"Frozen predictions for GW{gw} contain 0 records.")
        return None

    # Live finalized actuals must be provided by the async official retrieval path.
    # Cached master CSV is not evidence that a GW is complete.
    if actual_points_map is None:
        return None
    if freeze_data.get("source_kind") != "official_forward" or freeze_data.get("schema_version") != 2:
        if not dry_run:
            raise ValueError("Legacy/synthetic freeze cannot enter measured forward log")
    if freeze_data.get("source_kind") == "official_forward":
        frozen = datetime.fromisoformat(freeze_data["frozen_at"].replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(freeze_data["deadline_utc"].replace("Z", "+00:00"))
        if frozen.tzinfo is None or deadline.tzinfo is None or frozen >= deadline:
            raise ValueError("Invalid pre-deadline provenance")

    if not dry_run and (not isinstance(actual_points_map, FinalizedActuals) or actual_points_map.gameweek != gw):
        raise ValueError("Measured forward scoring requires finalized official live actuals")
    if not actual_points_map:
        logger.warning(f"Actual points for GW{gw} not yet available in dataset.")
        return None

    # Match predictions to actual points
    y_true = []
    y_pred = []
    p10_list = []
    p90_list = []

    for p in preds_list:
        elem = p["element"]
        if elem in actual_points_map:
            y_pred.append(p["expected_points"])
            y_true.append(actual_points_map[elem])
            p10_list.append(p.get("p10", p["expected_points"] - 1.5))
            p90_list.append(p.get("p90", p["expected_points"] + 1.5))

    if len(y_true) != len(preds_list) or len(y_true) < 10:
        logger.warning(f"Insufficient matched observations for GW{gw} ({len(y_true)} matches).")
        return None

    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    p10_arr = np.array(p10_list, dtype=float)
    p90_arr = np.array(p90_list, dtype=float)

    mae = float(np.round(mean_absolute_error(y_t, y_p), 3))
    rmse = float(np.round(root_mean_squared_error(y_t, y_p), 3))
    sp, _ = spearmanr(y_p, y_t)
    sp_val = float(np.round(sp, 3)) if not np.isnan(sp) else 0.0

    in_interval = (y_t >= p10_arr) & (y_t <= p90_arr)
    cov_pct = float(np.round(float(in_interval.mean()) * 100.0, 2))

    score_record = {
        "gameweek": gw,
        "scored_at": datetime.now(UTC).isoformat(),
        "frozen_at": freeze_data.get("frozen_at"),
        "model_version": freeze_data.get("model_version"),
        "git_commit": freeze_data.get("git_commit"),
        "manifest_hash": freeze_data.get("manifest_hash"),
        "actual_source": getattr(actual_points_map, "source_kind", "research_only"),
        "actuals_fetched_at": getattr(actual_points_map, "fetched_at", None),
        "sample_count": len(y_true),
        "ml_mae": mae,
        "ml_rmse": rmse,
        "ml_spearman": sp_val,
        "coverage_80_pct": cov_pct,
    }

    if not dry_run:
        if log_path.exists():
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
        else:
            log_data = {"status": "AWAITING_FORWARD_GAMEWEEKS", "evaluated_gameweeks": []}
        existing_gw = [idx for idx, eg in enumerate(log_data["evaluated_gameweeks"]) if eg["gameweek"] == gw]
        if existing_gw:
            log_data["evaluated_gameweeks"][existing_gw[0]] = score_record
        else:
            log_data["evaluated_gameweeks"].append(score_record)

        log_data["status"] = "ACTIVE_FORWARD_EVALUATION"
        log_data["last_updated"] = datetime.now(UTC).isoformat()
        log_data["next_scheduled_freeze_gw"] = gw + 1

        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_log = log_path.with_suffix(".json.tmp")
        tmp_log.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
        os.replace(tmp_log, log_path)

        logger.info(f"Scored GW{gw} forward holdout: MAE={mae}, RMSE={rmse}, Spearman={sp_val}, Cov={cov_pct}%")

    return score_record


async def run_holdout_cycle(dry_run: bool = False) -> dict[str, Any]:
    """Execute full forward holdout cycle: check upcoming freeze and evaluate finished GWs."""
    print("=" * 70)
    print("FORWARD HOLDOUT LOGGING & HONEST EVALUATION (Requirement G6)")
    print("=" * 70)

    log_data = ensure_holdout_log_initialized()
    eval_gws = log_data.get("evaluated_gameweeks", [])

    print(f"Current Forward Holdout Status: {log_data.get('status')}")
    print(f"Evaluated Gameweeks:            {len(eval_gws)}")
    print(f"Policy Note:                    {log_data.get('note')}")

    # Check game state
    game_state = await game_state_manager.get_game_state()
    curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

    if next_gw is None:
        raise ValueError("No official upcoming gameweek for freeze cycle")
    target_freeze_gw = next_gw
    print(f"\nCurrent Season Gameweek Phase:   {game_state.phase.value}")
    print(f"Next Gameweek:                  GW{target_freeze_gw}")
    print(f"Seconds to Deadline:            {game_state.seconds_to_deadline}s")

    freeze_file = HOLDOUT_DIR / f"gw{target_freeze_gw}_predictions.json"
    if freeze_file.exists():
        with open(freeze_file, encoding="utf-8") as f:
            fdata = json.load(f)
        print(f"[STATUS] GW{target_freeze_gw} predictions already frozen at {fdata.get('frozen_at')}.")
        print(f"         Model version: {fdata.get('model_version')}, Players: {fdata.get('player_count')}")
    else:
        print(f"[STATUS] GW{target_freeze_gw} predictions NOT frozen yet.")
        if not dry_run:
            print(f"Executing pre-deadline freeze for GW{target_freeze_gw}...")
            await freeze_predictions(target_gw=target_freeze_gw, dry_run=False)
        else:
            await freeze_predictions(target_gw=target_freeze_gw, dry_run=True)
            print(f"[Dry Run] Validated pre-deadline payload for GW{target_freeze_gw}; not written.")

    # Check for scored gameweeks
    if eval_gws:
        print("\n--- FORWARD HOLDOUT PERFORMANCE TABLE ---")
        print(f"{'GW':<5} {'Scored At':<25} {'Samples':<8} {'ML MAE':<8} {'RMSE':<8} {'Spearman':<10} {'Coverage 80%'}")
        for eg in eval_gws:
            print(
                f"GW{eg['gameweek']:<3} {eg['scored_at'][:19]:<25} {eg['sample_count']:<8} "
                f"{eg['ml_mae']:<8.3f} {eg['ml_rmse']:<8.3f} {eg['ml_spearman']:<10.3f} {eg['coverage_80_pct']:.2f}%"
            )
    else:
        print("\n[REPORT] No forward holdout evaluations recorded yet.")
        print("         GW1-5 is contaminated by prior inspection. Forward scoring will commence when GW6 concludes.")

    print("\n" + "=" * 70)
    print("FORWARD HOLDOUT AUDIT COMPLETED")
    print("=" * 70)

    return log_data
