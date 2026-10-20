import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.server import advice_job
from fpl_oracle.server.advice_job import AdvicePublisher, _disagreement, profile_key


def _ready(publisher, key, finished_ago=0, deadline=None):
    card = dict(deadline=dict(deadline_time=deadline.isoformat())) if deadline else {}
    publisher.current = dict(
        id="old",
        status="ready",
        profile_key=key,
        created=0,
        finished=advice_job.time.monotonic() - finished_ago,
        result=dict(decisionCard=card),
        sources={},
        model_version=None,
    )


@pytest.fixture
def quiet(monkeypatch):
    from fpl_oracle.ml.model_registry import model_registry

    monkeypatch.setattr(model_registry, "get_active_version", lambda: None)
    monkeypatch.setattr(advice_job, "load_last_advice", lambda key: None)


def test_ready_advice_reused_for_thirty_minutes_only(quiet):
    async def run():
        publisher = AdvicePublisher()
        _ready(publisher, "k", finished_ago=20 * 60)
        assert publisher.start("k")["id"] == "old"
        _ready(publisher, "k", finished_ago=31 * 60)
        assert publisher.start("k")["id"] != "old"
        publisher.task.cancel()
        await asyncio.sleep(0)
        _ready(publisher, "k", deadline=datetime.now(UTC) - timedelta(minutes=1))
        assert publisher.start("k")["id"] != "old"
        publisher.task.cancel()

    asyncio.run(run())


def test_no_full_run_when_deadline_is_under_two_hours(quiet, monkeypatch):
    from fpl_oracle.api.cache import cache_manager

    soon = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    monkeypatch.setattr(
        cache_manager, "get_with_meta", lambda key: (dict(events=[dict(is_next=True, deadline_time=soon)]), 0)
    )

    async def run():
        publisher = AdvicePublisher()
        state = publisher.start("k")
        assert state["status"] == "unavailable"
        assert state["reason"].startswith("Deadline too close for a full calculation")
        assert publisher.task is None
        assert state["started_at"]

    asyncio.run(run())


def test_new_profile_replaces_running_calculation(quiet):
    async def run():
        publisher = AdvicePublisher()
        publisher.current = dict(status="calculating", profile_key="a")
        publisher.task = asyncio.create_task(asyncio.sleep(100))
        old = publisher.task
        state = publisher.start("b")
        await asyncio.sleep(0)
        assert old.cancelled()
        assert state["status"] == "calculating" and publisher.task is not old
        publisher.task.cancel()

    asyncio.run(run())


def test_previous_advice_is_sent_on_the_first_poll_only():
    publisher = AdvicePublisher()
    publisher.current = dict(id="x", status="calculating", profile_key="k", previous_result=dict(a=1))
    assert publisher.public()["previous_result"] == dict(a=1)
    assert publisher.poll()["previous_result"] == dict(a=1)
    assert "previous_result" not in publisher.poll()


def test_squad_and_card_must_agree_on_captains_bench_and_points():
    squad = dict(
        captain=dict(element=1),
        vice_captain=dict(element=2),
        bench=[dict(element=3)],
        total_expected_points=61.04,
    )
    card = dict(captain=dict(element=1), vice_captain=dict(element=9), bench=[dict(element=3)], team_points=61.0)
    assert _disagreement(squad, card) == ["vice_captain"]
    assert _disagreement(squad, dict(card, vice_captain=dict(element=2), team_points=60.0)) == ["points"]


def test_team_news_change_publishes_finished_plan_with_notice(monkeypatch, quiet):
    async def run():
        from fpl_oracle.api.cache import cache_manager
        from fpl_oracle.api.read_context import remember
        from fpl_oracle.briefing.decision_card import decision_card_generator
        from fpl_oracle.data.store import data_store
        from fpl_oracle.server.routes import api

        profile = SimpleNamespace(manager_id=1)
        monkeypatch.setattr(data_store, "get_profile", lambda: profile)
        before = {"elements": [dict(id=1, news="")], "events": [], "teams": []}
        after = {"elements": [dict(id=1, news="Knock")], "events": [], "teams": []}
        monkeypatch.setattr(cache_manager, "get_with_meta", lambda key: (after, None))

        async def squad():
            remember("bootstrap-static", before, False, None)
            return dict(captain=dict(element=12))

        monkeypatch.setattr(api, "get_squad", squad)
        monkeypatch.setattr(
            decision_card_generator, "generate_decision_card", AsyncMock(return_value=dict(captain=dict(element=12)))
        )
        monkeypatch.setattr(api, "get_contingency_plans", AsyncMock(return_value=dict(plan_a=dict(title="p"))))
        publisher = AdvicePublisher()
        publisher.start(profile_key(profile))
        await publisher.task
        state = publisher.public()
        assert state["status"] == "ready"
        assert state["result"]["notice"] == "Team news changed while calculating. Refresh to include it."
        assert state["published_at"] and state["result"]["data_as_of"]

    asyncio.run(run())


def test_second_solve_refused_while_advice_runs():
    from fastapi import HTTPException

    from fpl_oracle.server.advice_job import advice_publisher
    from fpl_oracle.server.analysis import AnalysisService, publication_projections

    async def run():
        advice_publisher.task = asyncio.create_task(asyncio.sleep(100))
        try:
            with pytest.raises(HTTPException) as refused:
                AnalysisService._refuse_second_solve()
            assert refused.value.status_code == 409
            assert "already running" in refused.value.detail
            token = publication_projections.set({})  # the advice job's own calls
            AnalysisService._refuse_second_solve()
            publication_projections.reset(token)
        finally:
            advice_publisher.task.cancel()
            advice_publisher.task = None
        AnalysisService._refuse_second_solve()

    asyncio.run(run())


def test_beam_fallback_never_runs_chip_date_searches(monkeypatch):
    from fpl_oracle.optimise import multiperiod, transfers_beam
    from fpl_oracle.server.analysis import execute_authoritative_milp_plan

    seen = []

    def beam(self, **kwargs):
        seen.append(kwargs.get("measure_chip_values"))
        if len(seen) == 1:
            raise RuntimeError("first beam failed")
        return {}

    def milp(**kwargs):
        raise RuntimeError("no plan")

    monkeypatch.setattr(transfers_beam.TransferOptimizer, "evaluate_joint_transfer_and_chip_plan", beam)
    monkeypatch.setattr(multiperiod.multiperiod_optimizer, "evaluate_transfer_options", milp)
    result = execute_authoritative_milp_plan(target_gw=6, player_pool_df=object())
    assert seen == [False, False]
    assert result["fallback_used"] and result["decision_scope"]["global_optimum_proven"] is False


def test_background_jobs_wait_while_advice_runs(monkeypatch):
    from fpl_oracle.server import jobs
    from fpl_oracle.server.advice_job import advice_publisher

    async def run():
        advice_publisher.task = asyncio.create_task(asyncio.sleep(100))
        retrain = AsyncMock()
        monkeypatch.setattr("fpl_oracle.server.retraining.retrain_finalized_gameweeks", retrain)
        try:
            await jobs.retrain_trigger_job()
            assert await jobs.history_refresh_job() is None
            await jobs.odds_snapshot_job()
            retrain.assert_not_awaited()
            assert jobs.tracker.jobs_state["retrain_trigger"]["run_count"] == 0
        finally:
            advice_publisher.task.cancel()
            advice_publisher.task = None

    asyncio.run(run())
