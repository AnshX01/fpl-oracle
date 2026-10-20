"""Coalesced background advice publication, independent of browser request lifetime."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime

from fpl_oracle.api.read_context import read_context
from fpl_oracle.server.source_fingerprint import changes, decision_source


class SourceChanged(ValueError):
    def __init__(self, previous, categories):
        super().__init__("Decision inputs changed during calculation")
        self.previous = previous
        self.categories = categories


class AdviceStale(ValueError):
    """The inputs changed under a finished calculation. The message is written for the manager to read."""


logger = logging.getLogger("fpl_oracle.advice")

REUSE_SECONDS = 30 * 60  # a finished plan stays valid this long while no source changed
MIN_RUN_SECONDS = 2 * 3600  # a full calculation takes 30 to 90 minutes; closer to the deadline it cannot finish
# Changes in these categories do not alter the plan's squad, bank, fixtures or deadline: publish with a notice.
SOFT_CHANGES = {"prices", "availability"}


def _last_path():
    from fpl_oracle.config import DATA_DIR

    return DATA_DIR / "last_advice.json"


def load_last_advice(profile_key_value):
    """Last published advice for this manager, if its deadline has not passed (shown, labelled, while updating)."""
    try:
        saved = json.loads(_last_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("profile_key") != json.dumps(profile_key_value, sort_keys=True, default=str):
        return None
    if _deadline_passed(saved.get("result")):
        return None
    return saved


def _deadline_passed(result):
    deadline = ((result or {}).get("decisionCard") or {}).get("deadline")
    if isinstance(deadline, dict):
        deadline = deadline.get("deadline_time")
    try:
        return bool(deadline) and datetime.fromisoformat(str(deadline).replace("Z", "+00:00")) <= datetime.now(UTC)
    except ValueError:
        return True


def _next_deadline(bootstrap):
    """Deadline of the upcoming gameweek in a bootstrap-static payload, or None when it is not known."""
    times = [
        datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
        for event in (bootstrap or {}).get("events", [])
        if event.get("is_next") and event.get("deadline_time")
    ]
    return min(times) if times else None


def _disagreement(squad, card):
    """Fields on which the squad screen and the decision card describe different teams."""

    def view(page, points_key):
        points = page.get(points_key)
        return dict(
            captain=(page.get("captain") or {}).get("element"),
            vice_captain=(page.get("vice_captain") or {}).get("element"),
            bench=[row["element"] for row in page.get("bench") or []],
            points=None if points is None else round(float(points), 1),
        )

    left, right = view(squad, "total_expected_points"), view(card, "team_points")
    return [field for field in left if left[field] != right[field]]


def save_last_advice(profile_key_value, result):
    path = _last_path()
    body = {
        "profile_key": json.dumps(profile_key_value, sort_keys=True, default=str),
        "published_at": datetime.now(UTC).isoformat(),
        "result": result,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, default=str), encoding="utf-8")
    tmp.replace(path)


class AdvicePublisher:
    def __init__(self):
        self.current = None
        self.task = None
        self.freeze_task = None

    @property
    def running(self):
        return bool(self.task and not self.task.done())

    async def invalidate(self):
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.current = None
        self.task = None

    def start(self, profile_key):
        if self.running:
            if self.current["profile_key"] == profile_key:
                return self.public()
            # The running calculation is for a team the manager no longer has.
            self.task.cancel()
        elif (
            self.current
            and self.current["status"] == "ready"
            and self.current["profile_key"] == profile_key
            and time.monotonic() - self.current["finished"] < REUSE_SECONDS
            and not _deadline_passed(self.current["result"])
            and self.sources_unchanged()
        ):
            return self.public()
        last = None
        if self.current and self.current.get("status") == "ready" and self.current["profile_key"] == profile_key:
            last = {"result": self.current["result"], "published_at": self.current.get("published_at")}
        else:
            last = load_last_advice(profile_key)
        from fpl_oracle.api.cache import cache_manager

        bootstrap = cache_manager.get_with_meta("bootstrap-static")
        deadline = _next_deadline(bootstrap[0]) if bootstrap else None
        too_close = deadline is not None and (deadline - datetime.now(UTC)).total_seconds() < MIN_RUN_SECONDS
        self.current = dict(
            id=uuid.uuid4().hex,
            status="unavailable" if too_close else "calculating",
            stage="Deadline too close" if too_close else "Loading team",
            profile_key=profile_key,
            created=time.monotonic(),
            started_at=datetime.now(UTC).isoformat(),
            result=None,
            previous_result=last["result"] if last else None,
            previous_published_at=last["published_at"] if last else None,
        )
        if too_close:
            self.current["reason"] = "Deadline too close for a full calculation (it takes 30-90 minutes)"
        else:
            self.task = asyncio.create_task(self.run())
        return self.public()

    def sources_unchanged(self):
        from fpl_oracle.api.cache import cache_manager
        from fpl_oracle.ml.model_registry import model_registry

        if self.current.get("model_version") != model_registry.get_active_version():
            return False
        for key, value in self.current.get("sources", {}).items():
            live = cache_manager.get_with_meta(key)
            if live is None or changes(key, value, live[0]):
                return False
        return True

    def public(self):
        if not self.current:
            return dict(status="absent")
        hidden = {"profile_key", "created", "finished", "sources", "model_version", "previous_sent"}
        if self.current.get("previous_sent"):
            hidden.add("previous_result")
        return {k: v for k, v in self.current.items() if k not in hidden}

    def poll(self):
        """A status poll: the previous advice goes out once, not again every 1.5 seconds."""
        state = self.public()
        if self.current:
            self.current["previous_sent"] = True
        return state

    async def run(self):
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store
        from fpl_oracle.ml.holdout import compute_manifest_sha256
        from fpl_oracle.ml.model_registry import model_registry
        from fpl_oracle.server.retraining import ensure_latest_data
        from fpl_oracle.server.routes.api import get_contingency_plans, get_squad

        self.current["stage"] = "Getting the latest match data"
        try:
            data_basis = await ensure_latest_data()
        except Exception as error:
            logger.warning("Latest-data check failed (%s); advising on the data already held", error)
            data_basis = {"warning": "Could not check for newer match data"}
        model_before = model_registry.get_active_version()
        manifest_before = compute_manifest_sha256()
        from fpl_oracle.config import HISTORICAL_DIR

        history_file = HISTORICAL_DIR / "master_history.csv"

        def history_revision():
            return hashlib.sha256(history_file.read_bytes()).hexdigest() if history_file.exists() else None

        history_before = history_revision()
        original_key = self.current["profile_key"]
        from fpl_oracle.server.analysis import publication_projections

        projection_token = publication_projections.set({})
        token = read_context.set({})
        from fpl_oracle.optimise.progress import search_progress

        loop = asyncio.get_running_loop()
        publication = self.current

        def update_stage(stage):
            if self.current is publication and publication["status"] == "calculating":
                publication["stage"] = stage

        progress_token = search_progress.set(lambda stage: loop.call_soon_threadsafe(update_stage, stage))
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
                if differ := _disagreement(squad, card):
                    raise ValueError(f"Squad and decision card disagree on {', '.join(differ)}")
                if profile_key(data_store.get_profile()) != original_key:
                    raise AdviceStale("Profile changed during calculation; refresh")
                if model_registry.get_active_version() != model_before or compute_manifest_sha256() != manifest_before:
                    raise AdviceStale("Model changed during calculation; refresh")
                if history_revision() != history_before:
                    raise AdviceStale("History changed during calculation; refresh")
                if {r["element"] for r in squad.get("starters", [])} != {r["element"] for r in card.get("xi", [])}:
                    raise ValueError("Lineup mismatch in core publication")
                candidate = dict(
                    dataBasis=data_basis,
                    squadData=squad,
                    decisionCard=card,
                    contingencyPlans=plans,
                    source_revision=fpl_client.revision,
                )
                self.current["stage"] = "Checking the latest team data"
                context = read_context.get() or {}
                # Timestamp changes alone are not changed content. Guard real facts.
                from fpl_oracle.api.cache import cache_manager

                found = []
                for key, (value, _, _) in list(context.items()):
                    live = cache_manager.get_with_meta(key)
                    if live is None:
                        # Expiry is not a content change. Fetch current facts outside
                        # the pinned context before accepting an old-but-identical read.
                        request = getattr(fpl_client, "_source_requests", {}).get(key)
                        if request is not None:
                            validation_token = read_context.set(None)
                            try:
                                endpoint, ttl = request
                                fresh, stale = await fpl_client._fetch_json(endpoint, key, ttl, force_refresh=True)
                                live = None if stale else (fresh, None)
                            finally:
                                read_context.reset(validation_token)
                    categories = [] if live is None else changes(key, value, live[0])
                    if live is None or categories:
                        logger.warning(
                            "Advice source validation failed: key=%s outcome=%s categories=%s",
                            key,
                            "unverifiable" if live is None else "content_changed",
                            categories,
                        )
                        found += categories or ["unverifiable"]
                if found and not set(found) <= SOFT_CHANGES:
                    raise SourceChanged(candidate, sorted(set(found)))
                if found:
                    # Prices and team news move all the time; a finished hour-long plan is not thrown away for them.
                    candidate["notice"] = "Team news changed while calculating. Refresh to include it."
                # Revalidation itself may take time; owner and model must still agree.
                if profile_key(data_store.get_profile()) != original_key:
                    raise AdviceStale("Profile changed during validation; refresh")
                if model_registry.get_active_version() != model_before or compute_manifest_sha256() != manifest_before:
                    raise AdviceStale("Model changed during validation; refresh")
                if history_revision() != history_before:
                    raise AdviceStale("History changed during validation; refresh")
                snapshot = hashlib.sha256(
                    json.dumps(
                        {k: decision_source(k, v[0]) for k, v in context.items()}, sort_keys=True, default=str
                    ).encode()
                ).hexdigest()
                bootstrap = context.get("bootstrap-static", ({}, False, None))[0]
                deadline = _next_deadline(bootstrap)
                if deadline and datetime.now(UTC) >= deadline:
                    raise AdviceStale("Deadline passed during calculation; refresh")
                candidate["data_as_of"] = datetime.now(UTC).isoformat()
                candidate["snapshot_id"] = snapshot
                return candidate

            # CPU work is not discarded because a laptop is slow. Individual
            # network requests retain their own timeouts; explicit refresh can cancel.
            for attempt in range(2):
                try:
                    result = await work()
                    break
                except SourceChanged as changed:
                    changed.previous["decisionCard"].pop("confirmation_recommendation", None)
                    self.current.update(
                        previous_result=changed.previous,
                        changed_categories=changed.categories,
                        retry_count=attempt + 1,
                        previous_sent=False,
                        stage="Team data changed. Updating advice",
                    )
                    if attempt == 1 or "unverifiable" in changed.categories:
                        self.current.update(
                            status="refresh_required", result=None, stage="Team data changed again. Refresh when ready"
                        )
                        return
                    # A new pinned snapshot uses the current cache facts. Derived
                    # projection caches reuse unchanged model features where safe.
                    read_context.get().clear()
                    publication_projections.get().clear()
            self.current.update(
                status="ready",
                stage="Ready",
                result=result,
                previous_result=None,
                previous_published_at=None,
                published_at=datetime.now(UTC).isoformat(),
                finished=time.monotonic(),
                sources={k: v[0] for k, v in (read_context.get() or {}).items()},
                model_version=model_registry.get_active_version(),
            )
            try:
                save_last_advice(original_key, result)
            except OSError as error:
                logger.warning("Could not save the last advice: %s", error)
            # Every published advice before a deadline also refreshes the frozen forecast that the weekly track
            # record scores, so the record never depends on the app being open in the final 24 hours.
            self.freeze_task = asyncio.create_task(_freeze_for_track_record(result))
            saved_recommendation = result["decisionCard"].pop("confirmation_recommendation", None)
            if saved_recommendation:
                from fpl_oracle.domain.team_confirmation import team_confirmation

                team_confirmation.write(
                    "recommendation:" + team_confirmation.identity(saved_recommendation.pop("manager_id", None)),
                    saved_recommendation,
                )
        except Exception as error:
            logger.exception(
                "Advice job failed: type=%s stage=%s elapsed_seconds=%.1f",
                type(error).__name__,
                self.current.get("stage"),
                time.monotonic() - self.current["created"],
            )
            from fastapi import HTTPException

            # A 409 is a reason the manager can act on (e.g. a saved bank that disagrees with FPL): show it.
            reason = (
                error.detail
                if isinstance(error, HTTPException) and error.status_code == 409
                else str(error)
                if isinstance(error, AdviceStale)
                else None
            )
            self.current.update(
                status="failed",
                stage="Calculation stopped",
                error=reason or "Could not prepare advice. Check the application log.",
                error_type=type(error).__name__,
                fixable="saved_override" if reason and "override" in reason else None,
                result=None,
            )
        finally:
            publication_projections.reset(projection_token)
            search_progress.reset(progress_token)
            read_context.reset(token)


async def _freeze_for_track_record(result):
    from fpl_oracle.ml.holdout import freeze_predictions

    try:
        deadline = (result.get("decisionCard") or {}).get("deadline") or {}
        when = datetime.fromisoformat(str(deadline.get("deadline_time")).replace("Z", "+00:00"))
        if datetime.now(UTC) < when:
            await freeze_predictions(target_gw=int(result["decisionCard"]["gameweek"]))
    except Exception as error:
        logger.warning("Track-record freeze skipped: %s", error)


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
            "keep_ids",
            "avoid_ids",
        )
    }
    fields["team_confirmation"] = team_confirmation.read(getattr(profile, "manager_id", None))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


advice_publisher = AdvicePublisher()
