import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fpl_oracle.server.advice_job import AdvicePublisher, profile_key


async def _job_coalesces_survives_polling_and_deferred_league(monkeypatch):
    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import data_store
    from fpl_oracle.server.routes import api

    profile = SimpleNamespace(manager_id=1, target_league_id=2)
    monkeypatch.setattr(data_store, "get_profile", lambda: profile)
    release = asyncio.Event()

    async def slow_squad():
        await release.wait()
        return dict(captain=dict(element=12))

    monkeypatch.setattr(api, "get_squad", slow_squad)
    card = AsyncMock(return_value=dict(captain=dict(element=12)))
    monkeypatch.setattr(decision_card_generator, "generate_decision_card", card)
    monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="shared"))))
    publisher = AdvicePublisher()
    key = profile_key(profile)
    first = publisher.start(key)
    task = publisher.task
    assert publisher.start(key)["id"] == first["id"]
    assert publisher.task is task
    await asyncio.sleep(0)
    assert publisher.public()["status"] == "calculating"
    release.set()
    await publisher.task
    assert publisher.public()["status"] == "ready"
    card.assert_awaited_once_with(include_league=False)
    assert publisher.start(key)["id"] == first["id"]


async def _changed_profile_fails_without_publishing(monkeypatch):
    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import data_store
    from fpl_oracle.server.routes import api

    profile = SimpleNamespace(manager_id=1, target_league_id=2)
    monkeypatch.setattr(data_store, "get_profile", lambda: profile)

    async def squad():
        profile.manager_id = 3
        return dict(captain=dict(element=12))

    monkeypatch.setattr(api, "get_squad", squad)
    monkeypatch.setattr(
        decision_card_generator, "generate_decision_card", AsyncMock(return_value=dict(captain=dict(element=12)))
    )
    monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="shared"))))
    publisher = AdvicePublisher()
    publisher.start(profile_key(profile))
    await publisher.task
    assert publisher.public()["status"] == "failed"
    assert publisher.public()["result"] is None


def test_job_coalesces_survives_polling_and_deferred_league(monkeypatch):
    asyncio.run(_job_coalesces_survives_polling_and_deferred_league(monkeypatch))


def test_changed_profile_fails_without_publishing(monkeypatch):
    asyncio.run(_changed_profile_fails_without_publishing(monkeypatch))


def test_matching_captain_but_different_xi_is_rejected(monkeypatch):
    async def run():
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1, target_league_id=2)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        monkeypatch.setattr(
            api, "get_squad", AsyncMock(return_value=dict(captain=dict(element=12), starters=[dict(element=12)]))
        )
        monkeypatch.setattr(
            decision_card_generator,
            "generate_decision_card",
            AsyncMock(return_value=dict(captain=dict(element=12), xi=[dict(element=99)])),
        )
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="same"))))
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        await publisher.task
        assert publisher.public()["status"] == "failed"
        assert publisher.public()["error_type"] == "ValueError"

    asyncio.run(run())


def test_expiry_research_runs_in_background_after_core_ready(monkeypatch):
    async def run():
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.server.analysis import analysis_service
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1, target_league_id=2)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        monkeypatch.setattr(
            api,
            "get_squad",
            AsyncMock(return_value=dict(captain=dict(element=12), starters=[dict(element=12)])),
        )
        monkeypatch.setattr(
            decision_card_generator,
            "generate_decision_card",
            AsyncMock(return_value=dict(captain=dict(element=12), xi=[dict(element=12)])),
        )
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="p"))))
        release = asyncio.Event()
        calls = []

        async def slow_expiry():
            calls.append(1)
            await release.wait()
            return dict(status="model_expiry_sensitivity", first_chip_changes=False)

        monkeypatch.setattr(analysis_service, "expiry_sensitivity", slow_expiry)
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        await publisher.task
        # Core advice is ready while the long-horizon check is still pending.
        state = publisher.public()
        assert state["status"] == "ready"
        assert state["result"]["expiryResearch"]["status"] == "pending"
        release.set()
        await publisher.expiry_task
        assert publisher.public()["result"]["expiryResearch"]["status"] == "model_expiry_sensitivity"
        assert len(calls) == 1

    asyncio.run(run())


def test_confirmation_change_changes_profile_identity(monkeypatch):
    from fpl_oracle.domain.team_confirmation import team_confirmation

    profile = SimpleNamespace(manager_id=4)
    monkeypatch.setattr(team_confirmation, "read", lambda mid: None)
    first = profile_key(profile)
    monkeypatch.setattr(team_confirmation, "read", lambda mid: dict(confirmed_at="new"))
    assert profile_key(profile) != first


def test_invalidate_cancels_old_publication():
    async def run():
        publisher = AdvicePublisher()
        publisher.current = dict(status="calculating")
        publisher.task = asyncio.create_task(asyncio.sleep(100))
        old = publisher.task
        await publisher.invalidate()
        assert old.cancelled()
        assert publisher.public()["status"] == "absent"

    asyncio.run(run())


def test_only_successful_core_saves_published_recommendation(monkeypatch):
    async def run():
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.domain.team_confirmation import team_confirmation
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1, target_league_id=None)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        monkeypatch.setattr(team_confirmation, "read", lambda mid: None)
        captured = []
        monkeypatch.setattr(team_confirmation, "write", lambda mid, r: captured.append((mid, r)))
        monkeypatch.setattr(api, "get_squad", AsyncMock(return_value=dict(captain=dict(element=12))))
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="same"))))
        monkeypatch.setattr(
            decision_card_generator,
            "generate_decision_card",
            AsyncMock(
                return_value=dict(
                    captain=dict(element=12),
                    confirmation_recommendation=dict(manager_id=1, gameweek=6, ins=[], outs=[]),
                )
            ),
        )
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        await publisher.task
        assert publisher.public()["status"] == "ready"
        assert captured == [("recommendation:1", dict(gameweek=6, ins=[], outs=[]))]
        assert "confirmation_recommendation" not in publisher.public()["result"]["decisionCard"]

    asyncio.run(run())


def test_real_search_progress_reaches_publication_from_worker(monkeypatch):
    async def run():
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.optimise.progress import report_search_progress, search_progress
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1, target_league_id=None)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        release = asyncio.Event()

        async def squad():
            await asyncio.to_thread(report_search_progress, "Comparing chip dates (2 of 28)")
            await release.wait()
            return dict(captain=dict(element=12))

        monkeypatch.setattr(api, "get_squad", squad)
        monkeypatch.setattr(decision_card_generator, "generate_decision_card", AsyncMock(return_value=dict(captain=dict(element=12))))
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="same"))))
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        for _ in range(100):
            if publisher.public().get("stage") == "Comparing chip dates (2 of 28)":
                break
            await asyncio.sleep(.005)
        assert publisher.public()["stage"] == "Comparing chip dates (2 of 28)"
        assert publisher.public()["status"] == "calculating"
        release.set()
        await publisher.task
        assert publisher.public()["stage"] == "Ready"
        assert search_progress.get() is None

    asyncio.run(run())



def test_expired_source_is_revalidated_not_treated_as_changed(monkeypatch):
    async def run(changed):
        from fpl_oracle.api.cache import cache_manager
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.api.read_context import read_context, remember
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        monkeypatch.setattr(cache_manager, "get_with_meta", lambda key: None)
        monkeypatch.setattr(fpl_client, "_source_requests", {"entry:1:history": ("entry/1/history/", 300)}, raising=False)
        calls = []

        async def fresh(endpoint, key, ttl, force_refresh=False):
            assert read_context.get() is None
            assert force_refresh
            calls.append(key)
            return {"facts": 2 if changed else 1}, False

        monkeypatch.setattr(fpl_client, "_fetch_json", fresh)

        async def squad():
            remember("entry:1:history", {"facts": 1}, False, None)
            return dict(captain=dict(element=12))

        monkeypatch.setattr(api, "get_squad", squad)
        monkeypatch.setattr(decision_card_generator, "generate_decision_card", AsyncMock(return_value=dict(captain=dict(element=12))))
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="same"))))
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        await publisher.task
        assert calls == ["entry:1:history"]
        assert publisher.public()["status"] == ("failed" if changed else "ready")
        if changed:
            assert publisher.public()["result"] is None
        if publisher.expiry_task:
            publisher.expiry_task.cancel()

    asyncio.run(run(False))
    asyncio.run(run(True))
