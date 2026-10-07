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
        self.current = dict(
            id=uuid.uuid4().hex,
            status="calculating",
            stage="Loading canonical snapshot",
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
                self.current["stage"] = "Calculating shared eight-week plan"
                squad = await get_squad()
                self.current["stage"] = "Formatting decision and contingency plans"
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

            result = await asyncio.wait_for(work(), timeout=600)
            from fpl_oracle.ml.model_registry import model_registry

            self.current.update(
                status="ready",
                stage="Published coherent core",
                result=result,
                sources={k: v[0] for k, v in (read_context.get() or {}).items()},
                model_version=model_registry.get_active_version(),
            )
        except Exception as error:
            self.current.update(status="failed", stage="Calculation stopped", error=str(error), result=None)
        finally:
            read_context.reset(token)


def profile_key(profile):
    return hashlib.sha256(
        json.dumps(
            {
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
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


advice_publisher = AdvicePublisher()
