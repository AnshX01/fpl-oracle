"""Coalesced background advice publication, independent of browser request lifetime."""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime

from fpl_oracle.api.read_context import read_context


class AdvicePublisher:
    def __init__(self):
        self.current = None
        self.task = None
        self.expiry_task = None
        self.expiry_cache = {}

    async def invalidate(self):
        for task in (self.task, self.expiry_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.current = None
        self.task = None
        self.expiry_task = None
        self.expiry_cache.clear()

    def start(self, profile_key):
        if self.task and not self.task.done():
            if self.current["profile_key"] != profile_key:
                return dict(status="busy", reason="Another profile snapshot is calculating")
            return self.public()
        if (
            self.current
            and self.current["status"] == "ready"
            and self.current["profile_key"] == profile_key
            and time.monotonic() - self.current["created"] < 120
            and self.sources_unchanged()
        ):
            return self.public()
        if self.expiry_task and not self.expiry_task.done():
            self.expiry_task.cancel()
        self.current = dict(
            id=uuid.uuid4().hex,
            status="calculating",
            stage="Loading team",
            profile_key=profile_key,
            created=time.monotonic(),
            result=None,
        )
        self.task = asyncio.create_task(self.run())
        return self.public()

    def sources_unchanged(self):
        from fpl_oracle.api.cache import cache_manager
        from fpl_oracle.ml.model_registry import model_registry

        if self.current.get("model_version") != model_registry.get_active_version():
            return False
        for key, value in self.current.get("sources", {}).items():
            live = cache_manager.get_with_meta(key)
            if live is None or live[0] != value:
                return False
        return True

    def public(self):
        if not self.current:
            return dict(status="absent")
        return {
            k: v for k, v in self.current.items() if k not in ("profile_key", "created", "sources", "model_version")
        }

    async def run(self):
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store
        from fpl_oracle.ml.holdout import compute_manifest_sha256
        from fpl_oracle.ml.model_registry import model_registry
        from fpl_oracle.server.routes.api import get_contingency_plans, get_squad

        model_before = model_registry.get_active_version()
        manifest_before = compute_manifest_sha256()
        original_key = self.current["profile_key"]
        token = read_context.set({})
        try:

            async def work():
                self.current["stage"] = "Checking transfers and chips"
                squad = await get_squad()
                self.current["stage"] = "Preparing advice"
                from fpl_oracle.briefing.decision_card import decision_card_generator

                card = await decision_card_generator.generate_decision_card(include_league=False)
                plans = await get_contingency_plans(detailed=False)
                if not squad.get("captain") or not card.get("captain") or not plans.get("plan_a"):
                    raise ValueError("Core advice incomplete")
                if squad["captain"]["element"] != card["captain"]["element"]:
                    raise ValueError("Captain mismatch in core publication")
                if profile_key(data_store.get_profile()) != original_key:
                    raise ValueError("Profile changed during calculation; refresh")
                if model_registry.get_active_version() != model_before or compute_manifest_sha256() != manifest_before:
                    raise ValueError("Model changed during core calculation; refresh")
                if {r["element"] for r in squad.get("starters", [])} != {r["element"] for r in card.get("xi", [])}:
                    raise ValueError("Lineup mismatch in core publication")
                context = read_context.get() or {}
                # Timestamp changes alone are not changed content. Guard real facts.
                from fpl_oracle.api.cache import cache_manager

                for key, (value, _, _) in context.items():
                    live = cache_manager.get_with_meta(key)
                    if live is None or live[0] != value:
                        raise ValueError("Source content changed during calculation; refresh")
                snapshot = hashlib.sha256(
                    json.dumps({k: v[0] for k, v in context.items()}, sort_keys=True, default=str).encode()
                ).hexdigest()
                return dict(
                    squadData=squad,
                    decisionCard=card,
                    contingencyPlans=plans,
                    snapshot_id=snapshot,
                    data_as_of=datetime.now(UTC).isoformat(),
                    source_revision=fpl_client.revision,
                )

            result = await asyncio.wait_for(work(), timeout=1800)
            from fpl_oracle.ml.model_registry import model_registry

            result["expiryResearch"] = dict(status="pending")
            self.current.update(
                status="ready",
                stage="Ready",
                result=result,
                sources={k: v[0] for k, v in (read_context.get() or {}).items()},
                model_version=model_registry.get_active_version(),
            )
            saved_recommendation = result["decisionCard"].pop("confirmation_recommendation", None)
            if saved_recommendation:
                from fpl_oracle.domain.team_confirmation import team_confirmation

                team_confirmation.write(
                    "recommendation:" + team_confirmation.identity(saved_recommendation.pop("manager_id", None)),
                    saved_recommendation,
                )
            publication = self.current
            cache_key = json.dumps(
                [original_key, result.get("snapshot_id"), str(model_before), str(manifest_before)], default=str
            )
            if cache_key in self.expiry_cache:
                result["expiryResearch"] = self.expiry_cache[cache_key]
            else:
                self.expiry_task = asyncio.create_task(self.run_expiry(publication, cache_key))
        except Exception as error:
            self.current.update(status="failed", stage="Calculation stopped", error=str(error), result=None)
        finally:
            read_context.reset(token)

    async def run_expiry(self, publication, cache_key):
        """Attach model-only research after core publication, never delay ready advice."""
        from fpl_oracle.data.store import data_store
        from fpl_oracle.server.analysis import analysis_service

        try:
            result = await analysis_service.expiry_sensitivity()
            if self.current is not publication:
                return
            if profile_key(data_store.get_profile()) != publication["profile_key"] or not self.sources_unchanged():
                publication["result"]["expiryResearch"] = dict(
                    status="unavailable", reason="Sources changed; refresh advice"
                )
                return
            publication["result"]["expiryResearch"] = result
            if result.get("status") == "model_expiry_sensitivity":
                self.expiry_cache[cache_key] = result
                if len(self.expiry_cache) > 4:
                    self.expiry_cache.pop(next(iter(self.expiry_cache)))
        except asyncio.CancelledError:
            return
        except Exception as error:
            if self.current is publication:
                publication["result"]["expiryResearch"] = dict(status="unavailable", reason=str(error))


def profile_key(profile):
    from fpl_oracle.domain.team_confirmation import team_confirmation

    fields = {
        k: getattr(profile, k, None)
        for k in (
            "manager_id",
            "target_league_id",
            "risk_preference",
            "bank",
            "free_transfers",
            "bank_override_enabled",
            "ft_override_enabled",
            "manual_squad",
        )
    }
    fields["team_confirmation"] = team_confirmation.read(getattr(profile, "manager_id", None))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


advice_publisher = AdvicePublisher()
