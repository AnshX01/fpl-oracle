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
