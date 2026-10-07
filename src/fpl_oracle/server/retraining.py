"""Durable finalized-GW candidate training, with bounded retries and no serving refit."""

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.config import DATA_DIR
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.model_registry import model_registry

STATE_PATH = DATA_DIR / "retraining_state.json"
_lock = asyncio.Lock()


def _save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, STATE_PATH)


async def retrain_finalized_gameweeks():
    if _lock.locked():
        return "Candidate training already running"
    async with _lock:
        if STATE_PATH.exists():
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if state.get("schema_version") != 1:
                raise ValueError("Unknown retraining state schema; no training run")
        else:
            state = {"schema_version": 1, "gameweeks": {}}
        boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
        if stale:
            raise RuntimeError("Finalization requires fresh official event data")
        finalized = sorted(e.id for e in boot.events if e.finished and e.data_checked)
        pending = [
            g
            for g in finalized
            if state["gameweeks"].get(f"2026-27:{g}", {}).get("status") not in ("promoted", "rejected")
        ]
        if not pending:
            return "No new finalized gameweek"
        # One refresh can contain several previously completed GWs; one candidate covers them.
        key = f"2026-27:{pending[-1]}"
        previous = state["gameweeks"].get(key, {})
        retry_at = previous.get("retry_after")
        if retry_at and datetime.now(UTC) < datetime.fromisoformat(retry_at):
            return "Previous training failed; waiting for bounded retry"
        state["gameweeks"][key] = {
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "covered_gameweeks": pending,
            "attempt": previous.get("attempt", 0) + 1,
        }
        _save(state)
        try:
            metadata = await historical_manager.refresh_current_season()
            if not metadata.get("complete") or not set(pending).issubset(metadata.get("finalized_gameweeks", [])):
                raise RuntimeError("History refresh incomplete; candidate not trained")
            from fpl_oracle.ml.train import train_all_models

            X, _ = await asyncio.to_thread(train_all_models)
            result = X.attrs.get("training_outcome")
            if not result or "promoted" not in result:
                raise RuntimeError("Training returned no verified promotion/rejection outcome")
            record = {
                "status": "promoted" if result["promoted"] else "rejected",
                "completed_at": datetime.now(UTC).isoformat(),
                "history": metadata,
                "history_sha256": hashlib.sha256(historical_manager.output_file.read_bytes()).hexdigest(),
                "outcome": result,
                "active_version": model_registry.get_active_version().get("version"),
            }
            for g in pending:
                state["gameweeks"][f"2026-27:{g}"] = record
            _save(state)
            from fpl_oracle.server.analysis import analysis_service

            if result["promoted"]:
                from fpl_oracle.ml.predict import projection_engine

                projection_engine.is_loaded = False
            await analysis_service.invalidate()
            return f"Finalized GW{pending[-1]} candidate {record['status']}; active {record['active_version']}"
        except Exception as error:
            state["gameweeks"][key].update(
                status="failed", error=str(error), retry_after=(datetime.now(UTC) + timedelta(hours=1)).isoformat()
            )
            _save(state)
            raise
