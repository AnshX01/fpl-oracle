"""Shared snapshot and worker cache. No secret/user data is placed in public reports."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import numpy as np
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
        self._news_signals = []
        self._snapshots = {}
        self._plan_tasks = {}
        self._plans = {}
        self._chip_tasks = {}
        self._chip_plans = {}
        self._league_context_cache = {}

    async def invalidate(self):
        """Wait for existing inference to finish, then drop dependent served caches."""
        async with self._projection_lock:
            projection_engine._cache.clear()
            self._snapshots.clear()
            self._plans.clear()
            self._chip_plans.clear()
            self._league_context_cache.clear()
            self._news_at = 0.0
            self._news_key = None

    async def projections(self, start_gw, horizon, bootstrap, fixtures):
        horizon = max(horizon, 8)
        async with self._projection_lock:
            body = bootstrap.model_dump(mode="json")
            import time

            news_key = (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), start_gw)
            async with self._news_lock:
                if self._news_key != news_key or time.monotonic() - self._news_at >= 300:
                    self._news_signals = await news_analyzer.get_player_news_signals(bootstrap, target_gw=start_gw)
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
                current_history = hashlib.sha256(hist.read_bytes()).hexdigest() if hist.exists() else None
                if current_history != key_body["history"] or model_registry.get_active_version() != key_body["model"]:
                    projection_engine._cache.clear()
                    raise RuntimeError("Model or history changed during inference; snapshot not published")
                self._snapshots = {
                    key: {
                        "snapshot_id": key,
                        "created_at": datetime.now(UTC).isoformat(),
                        "inputs": key_body,
                        "projections": result,
                    }
                }
            result = {}
            snapshot = self._snapshots[key]
            serialized_inputs = json.dumps(snapshot["inputs"], sort_keys=True, default=str)
            for g, df in snapshot["projections"].items():
                frame = df.copy(deep=True)
                frame.attrs["snapshot_id"] = key
                frame.attrs["snapshot_inputs_json"] = serialized_inputs
                frame.attrs["snapshot_created_at"] = snapshot["created_at"]
                result[g] = frame
            return result

    async def league_context(self, target_gw):
        import time

        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store
        from fpl_oracle.league.rivals import rival_analyzer
        from fpl_oracle.league.standings import league_standings_manager

        profile = data_store.get_profile()
        if not profile.manager_id or not profile.target_league_id:
            return None
        key = (profile.manager_id, profile.target_league_id, target_gw)
        cached = self._league_context_cache.get(key)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]
        standings = await league_standings_manager.get_league_standings(profile.target_league_id)
        if standings.get("is_stale") or standings.get("coverage", {}).get("partial"):
            return None
        rows = standings.get("standings", [])
        user = next((r for r in rows if r.get("entry") == profile.manager_id), None)
        if user is None:
            return None
        boot, stale = await fpl_client.get_bootstrap_static()
        if stale:
            return None
        result = await rival_analyzer.analyze_rivals(rows, profile.manager_id, target_gw, boot)
        if result.get("is_stale"):
            return None
        context = dict(
            user_points=user["total"],
            rivals=[
                dict(points=r["total_points"], elements=[p["element"] for p in r["squad"]])
                for r in result.get("rival_squads", [])
            ],
        )
        self._league_context_cache = {key: (time.monotonic(), context)}
        return context

    async def news_signals(self, bootstrap, target_gw):
        # Advice surfaces use the same news evidence that built their projections.
        key = (
            hashlib.sha256(json.dumps(bootstrap.model_dump(mode="json"), sort_keys=True).encode()).hexdigest(),
            target_gw,
        )
        if key == self._news_key:
            import copy

            return copy.deepcopy(self._news_signals)
        return await news_analyzer.get_player_news_signals(bootstrap, target_gw=target_gw)

    async def chip_strategy(self, **kwargs):
        import copy

        from fpl_oracle.chips.planner import chip_planner

        squad = kwargs["current_squad_df"]
        for frame in kwargs["horizon_projections"].values():
            if (
                frame.empty
                or "element" not in frame
                or "expected_points" not in frame
                or not set(squad["element"]).issubset(set(frame["element"]))
                or not np.isfinite(pd.to_numeric(frame["expected_points"], errors="coerce")).all()
            ):
                from fastapi import HTTPException

                raise HTTPException(status_code=409, detail="Projection snapshot unavailable; no chip advice generated")
        prices = [
            {
                "element": int(r["element"]),
                "selling_price": float(r.get("selling_price", r["value"])),
                "value": float(r["value"]),
            }
            for r in squad.sort_values("element").to_dict("records")
        ]
        key = hashlib.sha256(
            json.dumps(
                {
                    "squad": sorted(squad["element"].tolist()),
                    "prices": prices,
                    "snapshots": {
                        g: df.attrs.get("snapshot_id") or df.to_json()
                        for g, df in kwargs["horizon_projections"].items()
                    },
                    "current_gw": kwargs["current_gw"],
                    "history": kwargs.get("manager_history"),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if key in self._chip_plans:
            return copy.deepcopy(self._chip_plans[key])
        if key not in self._chip_tasks:
            self._chip_tasks[key] = asyncio.create_task(
                asyncio.to_thread(chip_planner.generate_chip_strategy, **kwargs)
            )
        task = self._chip_tasks[key]
        try:
            result = await asyncio.shield(task)
            self._chip_plans[key] = result
            if len(self._chip_plans) > 8:
                self._chip_plans.pop(next(iter(self._chip_plans)))
            return copy.deepcopy(result)
        finally:
            if task.done():
                self._chip_tasks.pop(key, None)

    def bind_chip_schedule(self, calendar, joint):
        """Public chip recommendations follow the selected resource trajectory."""
        result = dict(calendar)
        result["independent_calendar_context"] = calendar.get("chip_plan_table", [])
        rows = []
        for step in joint["recommended_plan"].get("trajectory", []):
            if step.get("chip"):
                rows.append(
                    {
                        "chip": step["chip"],
                        "code": step["chip"],
                        "recommended_gw": step["gameweek"],
                        "expected_gain": None,
                        "reasoning": "Selected legal joint transfer/chip trajectory; isolated chip gain is not estimated.",
                    }
                )
        result["chip_plan_table"] = rows
        result["joint_schedule"] = {r["recommended_gw"]: r["code"] for r in rows}
        result["set_1_deadline_warning"] = (
            "Unused Set 1 chips expire at the GW19 deadline. This roadmap is conditional; unknown later option value is not a verified points loss."
        )
        result["recommended_chip"] = joint["recommended_chip"]
        result["chip_comparison_table"] = joint["chip_comparison_table"]
        return result

    async def rival_scenarios(self, joint, projections, target_gw):
        from fpl_oracle.league.scenarios import compare_plan_scenarios

        context = await self.league_context(target_gw)
        maps = {
            g: {int(r["element"]): r for r in frame.to_dict("records")}
            for g, frame in projections.items()
            if g in joint["decision_scope"]["horizon_gameweeks"]
        }
        states = [
            dict(first_chip=c["chip_code"], history=c["plan"]["trajectory"]) for c in joint["chip_comparison_table"]
        ]
        return await asyncio.to_thread(compare_plan_scenarios, states, context, maps)

    async def expiry_sensitivity(self):
        from fpl_oracle.api.read_context import read_context

        token = read_context.set({})
        try:
            return await self._expiry_sensitivity()
        finally:
            read_context.reset(token)

    async def _expiry_sensitivity(self):
        """On-demand model stress through chip expiry. Never blocks core advice."""
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store
        from fpl_oracle.domain.manager_state import manager_state_service
        from fpl_oracle.ml.holdout import compute_manifest_sha256
        from fpl_oracle.optimise.transfers import transfer_optimizer
        from fpl_oracle.server.advice_job import profile_key

        model_before = model_registry.get_active_version()
        manifest_before = compute_manifest_sha256()
        owner_before = profile_key(data_store.get_profile())
        state = await manager_state_service.get_current_state()
        start = state.target_gw
        if start > 19:
            return dict(status="unavailable", reason="First-set expiry research applies before GW19")
        boot, stale = await fpl_client.get_bootstrap_static()
        fixtures, fixture_stale = await fpl_client.get_fixtures()
        if stale or fixture_stale or state.is_stale:
            return dict(status="unavailable", reason="Fresh sources required for expiry sensitivity")
        projections = await self.projections(start, 20 - start, boot, fixtures)
        kwargs = dict(
            current_squad_df=state.to_squad_dataframe(),
            player_pool_df=projections[start],
            bank=state.bank_tenths,
            free_transfers=state.free_transfers,
            horizon_projections=projections,
            current_gw=state.current_gw,
            target_gw=start,
            available_chips=state.chips_remaining_set_1,
            chips_by_set={1: state.chips_remaining_set_1, 2: state.chips_remaining_set_2},
            chips_already_used=[c["name"] for c in state.chips_used if c["event"] <= 19],
            rival_context=await self.league_context(start),
            previous_chip=next((c["name"] for c in state.chips_used if c["event"] == start - 1), None),
            num_mc_scenarios=0,
        )
        short, extended = await asyncio.to_thread(
            lambda: (
                transfer_optimizer.evaluate_joint_transfer_and_chip_plan(**kwargs, horizon_len=min(8, 20 - start)),
                transfer_optimizer.evaluate_joint_transfer_and_chip_plan(**kwargs, horizon_len=20 - start),
            )
        )

        if model_registry.get_active_version() != model_before or compute_manifest_sha256() != manifest_before:
            raise ValueError("Model changed during expiry research; result not published")
        if profile_key(data_store.get_profile()) != owner_before:
            raise ValueError("Profile changed during expiry research; result not published")
        from fpl_oracle.api.cache import cache_manager
        from fpl_oracle.api.read_context import read_context

        for key, (value, _, _) in (read_context.get() or {}).items():
            live = cache_manager.get_with_meta(key)
            if live is None or live[0] != value:
                raise ValueError("Sources changed during expiry research; result not published")

        def summarize(plan):
            return dict(
                gameweeks=plan["decision_scope"]["horizon_gameweeks"],
                first_chip=plan["recommended_chip"],
                discounted_edge=plan["best_candidate"]["net_gain_vs_hold"],
                trajectory=[
                    dict(gameweek=r["gameweek"], chip=r.get("chip"), transfers_count=r["transfers_count"])
                    for r in plan["recommended_plan"]["trajectory"]
                ],
                options=[
                    dict(first_chip=c["chip_code"], discounted_edge=c["net_gain_vs_hold"])
                    for c in plan["chip_comparison_table"]
                ],
                retained_resource_frontier=plan.get("resource_frontier"),
            )

        return dict(
            status="model_expiry_sensitivity",
            calibrated=False,
            promotion_allowed=False,
            snapshot_id=projections[start].attrs.get("snapshot_id"),
            short=summarize(short),
            extended=summarize(extended),
            first_chip_changes=short["recommended_chip"] != extended["recommended_chip"],
            assumptions="Current price/minutes/form extrapolation. Unknown postponements, price moves and future news not known. Pruned search, not global optimum.",
            unresolved="No deadline-grounded out-of-time tail calibration. Rival choices are stress assumptions, not predictions. Agreement between horizons does not prove optimality.",
        )

    async def joint_plan(self, **kwargs):
        from fpl_oracle.data.store import data_store
        from fpl_oracle.optimise.transfers import transfer_optimizer

        if "previous_chip" not in kwargs:
            from fpl_oracle.domain.manager_state import manager_state_service

            state = await manager_state_service.get_current_state()
            kwargs["previous_chip"] = next(
                (c["name"] for c in state.chips_used if c["event"] == kwargs["target_gw"] - 1), None
            )
        if "rival_context" not in kwargs:
            try:
                kwargs["rival_context"] = await self.league_context(kwargs["target_gw"])
            except Exception:
                kwargs["rival_context"] = None
        kwargs.setdefault(
            "risk_preference", getattr(data_store.get_profile(), "risk_preference", "balanced") or "balanced"
        )
        for name in ("locked_in_ids", "locked_out_ids", "excluded_team_ids"):
            kwargs.setdefault(name, set())

        # Search all supported loaded weeks, with no fabricated season-tail forecast.
        if "horizon_len" not in kwargs:
            start = kwargs["target_gw"]
            consecutive = 0
            while start + consecutive in kwargs["horizon_projections"] and consecutive < 8:
                consecutive += 1
            kwargs["horizon_len"] = consecutive
        kwargs.setdefault("evaluate_rival_scenarios", False)
        # Different surfaces attach different projection columns and numeric
        # dtypes to the same owned squad. Canonicalize before cache-keying so
        # they share one CPU plan instead of several identical concurrent beams.
        target = kwargs["horizon_projections"].get(kwargs["target_gw"], pd.DataFrame())
        owned_ids = set(kwargs["current_squad_df"].get("element", []))
        if (
            target.empty
            or "expected_points" not in target
            or "element" not in target
            or not owned_ids.issubset(set(target["element"]))
            or not np.isfinite(pd.to_numeric(target["expected_points"], errors="coerce")).all()
        ):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=409,
                detail="Projection snapshot unavailable or missing configured players; no current advice generated",
            )
        owned = kwargs["current_squad_df"]
        identity_columns = [c for c in ("element", "purchase_price", "selling_price", "price_provenance") if c in owned]
        squad = owned[identity_columns].copy()
        projection_columns = [
            c for c in target.columns if c not in ("element", "selling_price", "purchase_price", "price_provenance")
        ]
        by_element = target.set_index("element")
        for column in projection_columns:
            squad[column] = squad["element"].map(by_element[column])
        kwargs["current_squad_df"] = squad.sort_values("element").reset_index(drop=True)
        kwargs["bank"] = int(kwargs["bank"])
        kwargs["free_transfers"] = int(kwargs["free_transfers"])

        def normalize(value):
            if isinstance(value, pd.DataFrame):
                return value.reindex(sorted(value.columns), axis=1).to_json(orient="split", double_precision=10)
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
            self._plans[key] = result
            if len(self._plans) > 8:
                self._plans.pop(next(iter(self._plans)))
            import copy

            return copy.deepcopy(result)
        finally:
            if task.done():
                self._plan_tasks.pop(key, None)


analysis_service = AnalysisService()
