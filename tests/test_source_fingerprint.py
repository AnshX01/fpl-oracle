import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.server.source_fingerprint import changes


def boot():
    return dict(
        elements=[dict(id=1, now_cost=50, status="a", news="", team=1, element_type=3)],
        events=[dict(id=6, is_next=True, deadline_time="2026-12-31T12:00:00Z")],
        teams=[],
        chips=[],
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("selected_by_percent", "20.1"),
        ("transfers_in_event", 999),
        ("event_points", 10),
        ("form", "9.0"),
        ("total_points", 99),
        ("minutes", 700),
    ],
)
def test_display_churn_not_a_decision_change(field, value):
    old = boot()
    new = copy.deepcopy(old)
    new["elements"][0][field] = value
    assert changes("bootstrap-static", old, new) == []


@pytest.mark.parametrize(
    "field,value,category",
    [
        ("now_cost", 51, "prices"),
        ("status", "i", "availability"),
        ("news", "Out for two weeks", "availability"),
        ("chance_of_playing_next_round", 0, "availability"),
        ("team", 2, "players"),
    ],
)
def test_relevant_changes_categorised(field, value, category):
    old = boot()
    new = copy.deepcopy(old)
    new["elements"][0][field] = value
    assert changes("bootstrap-static", old, new) == [category]


def test_deadline_and_fixture_change_not_ignored():
    old = boot()
    new = copy.deepcopy(old)
    new["events"][0]["deadline_time"] = "2026-12-30T12:00:00Z"
    assert changes("bootstrap-static", old, new) == ["events"]
    assert changes("fixtures:all", [dict(id=1, event=6)], [dict(id=1, event=7)]) == ["fixtures"]
    assert not changes("fixtures:all", [dict(id=1, event=6, minutes=0)], [dict(id=1, event=6, minutes=40)])


def test_real_change_retries_once_retains_previous_no_saved_done(monkeypatch):
    from fpl_oracle.api.cache import cache_manager
    from fpl_oracle.api.read_context import remember
    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import data_store
    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.advice_job import AdvicePublisher, profile_key
    from fpl_oracle.server.routes import api

    async def run(repeat):
        old = boot()
        current = copy.deepcopy(old)
        current["elements"][0]["now_cost"] = 51
        calls = []
        profile = SimpleNamespace(manager_id=1)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        monkeypatch.setattr(team_confirmation, "read", lambda *a: None)
        monkeypatch.setattr(cache_manager, "get_with_meta", lambda key: (current, None))
        writes = []
        monkeypatch.setattr(team_confirmation, "write", lambda *args: writes.append(args))

        async def squad():
            calls.append(1)
            if len(calls) == 1 or repeat:
                remember("bootstrap-static", old, False, None)
            else:
                remember("bootstrap-static", current, False, None)
            return dict(captain=dict(element=12))

        monkeypatch.setattr(api, "get_squad", squad)
        monkeypatch.setattr(
            decision_card_generator, "generate_decision_card", AsyncMock(return_value=dict(captain=dict(element=12)))
        )
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="same"))))
        publisher = AdvicePublisher()
        first = publisher.start(profile_key(profile))
        assert publisher.start(profile_key(profile))["id"] == first["id"]
        await publisher.task
        assert len(calls) == 2
        state = publisher.public()
        assert state["status"] == ("refresh_required" if repeat else "ready")
        assert bool(state["previous_result"]) == repeat
        assert writes == []
        if publisher.expiry_task:
            publisher.expiry_task.cancel()

    asyncio.run(run(False))
    asyncio.run(run(True))


def test_server_blocks_done_for_outdated_result(monkeypatch):
    from fastapi import HTTPException

    from fpl_oracle.server.advice_job import advice_publisher
    from fpl_oracle.server.routes.api import confirm_followed_team

    monkeypatch.setattr(advice_publisher, "current", dict(status="refresh_required", previous_result={"old": True}))
    with pytest.raises(HTTPException, match="Wait for updated advice"):
        asyncio.run(confirm_followed_team())


def test_rule_cosmetic_metadata_ignored_but_squad_rules_not():
    old = boot()
    new = copy.deepcopy(old)
    new["game_settings"] = {"ui_timestamp": "changed"}
    assert not changes("bootstrap-static", old, new)
    new["game_settings"]["squad_team_limit"] = 4
    assert changes("bootstrap-static", old, new) == ["rules"]
