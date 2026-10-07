"""Shared snapshot and worker cache. No secret/user data is placed in public reports."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.news.analyse import news_analyzer


class AnalysisService:
    def __init__(self):
        self._projection_lock = asyncio.Lock()
        self._news_lock = asyncio.Lock()
        self._news_key = None
        self._news_at = 0.0
        self._snapshots = {}
        self._plan_tasks = {}
        self._plans = {}

    async def projections(self, start_gw, horizon, bootstrap, fixtures):
        horizon = max(horizon, 8)
        async with self._projection_lock:
            body = bootstrap.model_dump(mode="json")
            import time

            news_key = (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), start_gw)
            async with self._news_lock:
                if self._news_key != news_key or time.monotonic() - self._news_at >= 300:
                    await news_analyzer.get_player_news_signals(bootstrap, target_gw=start_gw)
                    self._news_key = news_key
                    self._news_at = time.monotonic()
            inputs = {
                g: news_analyzer.get_reconciled_inputs(bootstrap, g)
                for g in range(start_gw, min(39, start_gw + horizon))
            }
            key_body = {
                "bootstrap": body,
                "fixtures": [f.model_dump(mode="json") for f in fixtures],
                "inputs": {
                    g: {
                        eid: {
                            k: v
                            for k, v in r.model_dump(mode="json").items()
                            if k not in ("reconciliation_reason", "rejected_signals")
                        }
                        for eid, r in rs.items()
                    }
                    for g, rs in inputs.items()
                },
                "model": model_registry.get_active_version(),
                "start": start_gw,
                "horizon": horizon,
            }
            from fpl_oracle.config import HISTORICAL_DIR

            hist = HISTORICAL_DIR / "master_history.csv"
            key_body["history"] = hashlib.sha256(hist.read_bytes()).hexdigest() if hist.exists() else None
            key = hashlib.sha256(json.dumps(key_body, sort_keys=True).encode()).hexdigest()
            if key not in self._snapshots:
                result = await asyncio.to_thread(
                    projection_engine.predict_multi_gameweeks,
                    start_gw,
                    horizon,
                    bootstrap,
                    fixtures,
                    reconciled_inputs_by_gw=inputs,
                )
                self._snapshots = {
                    key: {
                        "snapshot_id": key,
                        "created_at": datetime.now(UTC).isoformat(),
                        "inputs": key_body,
                        "projections": result,
                    }
                }
            result = {}
            import copy

            snapshot = self._snapshots[key]
            for g, df in snapshot["projections"].items():
                frame = df.copy(deep=True)
                frame.attrs["snapshot_id"] = key
                frame.attrs["snapshot_inputs"] = copy.deepcopy(snapshot["inputs"])
                frame.attrs["snapshot_created_at"] = snapshot["created_at"]
                result[g] = frame
            return result

    async def joint_plan(self, **kwargs):
        from fpl_oracle.optimise.transfers import transfer_optimizer

        for name in ("locked_in_ids", "locked_out_ids", "excluded_team_ids"):
            kwargs.setdefault(name, set())

        def normalize(value):
            if isinstance(value, pd.DataFrame):
                return value.to_json(orient="split", double_precision=10)
            if isinstance(value, dict):
                return {str(k): normalize(v) for k, v in value.items()}
            if isinstance(value, set):
                return [normalize(v) for v in sorted(value)]
            if isinstance(value, (list, tuple)):
                return [normalize(v) for v in value]
            return value

        key = hashlib.sha256(json.dumps(normalize(kwargs), sort_keys=True, default=str).encode()).hexdigest()
        if key in self._plans:
            import copy

            return copy.deepcopy(self._plans[key])
        task = self._plan_tasks.get(key)
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(transfer_optimizer.evaluate_joint_transfer_and_chip_plan, **kwargs)
            )
            self._plan_tasks[key] = task
        try:
            result = await asyncio.shield(task)
            self._plans = {key: result}
            import copy

            return copy.deepcopy(result)
        finally:
            if task.done():
                self._plan_tasks.pop(key, None)


analysis_service = AnalysisService()
