"""Forced refresh and stale provenance regressions, using deterministic upstream records."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.domain.manager_state import ManagerStateService


def test_manager_force_refresh_propagates_and_keeps_old_timestamp(monkeypatch):
    profile = SimpleNamespace(
        manager_id=7,
        target_league_id=None,
        manual_squad=None,
        bank=0,
        free_transfers=1,
        bank_override_enabled=False,
        ft_override_enabled=False,
    )
    from fpl_oracle.data.store import data_store

    monkeypatch.setattr(data_store, "get_profile", lambda: profile)
    boot = BootstrapStatic.model_validate({"elements": [], "events": [], "teams": [], "element_types": []})
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=([], True)))
    monkeypatch.setattr(fpl_client, "get_current_and_next_gw", AsyncMock(return_value=(5, 6)))
    for name in ("entry", "picks", "history", "transfers"):
        monkeypatch.setattr(fpl_client, f"get_manager_{name}", AsyncMock(return_value=(None, True)))
    stamp = datetime(2026, 10, 1, tzinfo=UTC)
    monkeypatch.setattr(fpl_client, "_cache_timestamps", {"bootstrap-static": stamp})
    state = asyncio.run(ManagerStateService().get_current_state(force_refresh=True))
    assert state.is_stale and not state.squad
    assert state.source_timestamp == stamp.isoformat()
    assert "not substituted" in state.error_message
    fpl_client.get_manager_picks.assert_awaited_once_with(7, 5, force_refresh=True)
    for name in ("entry", "history", "transfers"):
        getattr(fpl_client, f"get_manager_{name}").assert_awaited_once_with(7, force_refresh=True)


def test_official_failure_does_not_substitute_saved_manual_team():
    profile = SimpleNamespace(
        manager_id=7, target_league_id=None, manual_squad=list(range(1, 16)), bank=None, free_transfers=None
    )
    state = ManagerStateService.build_effective_state(profile, SimpleNamespace(elements=[]), [], is_stale=True)
    assert not state.squad


def test_analysis_manual_invalidation_clears_dependent_snapshots():
    from fpl_oracle.server.analysis import AnalysisService

    service = AnalysisService()
    service._snapshots["old"] = {}
    service._plans["old"] = {}
    service._news_key = "old"
    service._news_at = 500
    asyncio.run(service.invalidate())
    assert not service._snapshots and not service._plans and service._news_key is None


def test_repeated_sync_triggers_coalesce_before_task_starts(monkeypatch):
    from fpl_oracle.server.pipeline import SyncPipeline

    service = SyncPipeline()
    monkeypatch.setattr(service, "run_pipeline", AsyncMock())

    async def run():
        first = service.trigger_sync()
        second = service.trigger_sync()
        await service._task
        assert first["status"] == "started" and second["status"] == "already_running"
        assert service.run_pipeline.await_count == 1

    asyncio.run(run())


def test_weekly_briefing_uses_canonical_zero_ft_bank_and_joint_action(configured_advisor, monkeypatch):
    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.briefing.weekly import weekly_briefing_generator
    from fpl_oracle.data.store import data_store

    state, _, _ = configured_advisor
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )

    async def run():
        card = await decision_card_generator.generate_decision_card()
        briefing = await weekly_briefing_generator.generate_briefing()
        assert briefing["free_transfers"] == state.free_transfers == 0
        assert briefing["bank_tenths"] == state.bank_tenths
        assert briefing["captain"]["element"] == card["captain"]["element"]
        assert briefing["chips"]["recommended_chip"] == card["chip"].get("recommended_chip")

    asyncio.run(run())


def test_chat_team_uses_canonical_bank_and_zero_ft(configured_advisor):
    from fpl_oracle.llm.tools import tool_executor

    state, _, _ = configured_advisor
    result = asyncio.run(tool_executor._tool_get_my_team({}))
    assert result["bank_millions"] == state.bank_millions
    assert result["free_transfers"] == 0 and len(result["squad"]) == 15


def test_joint_plan_canonicalizes_surface_columns_and_coalesces(configured_advisor, monkeypatch):
    from unittest.mock import Mock

    from fpl_oracle.optimise.transfers import transfer_optimizer
    from fpl_oracle.server.analysis import AnalysisService

    state, pool, horizon = configured_advisor
    service = AnalysisService()
    monkeypatch.setattr(service, "league_context", AsyncMock(return_value=None))
    solve = Mock(return_value={"recommended_plan": {}})
    monkeypatch.setattr(transfer_optimizer, "evaluate_joint_transfer_and_chip_plan", solve)
    first = state.to_squad_dataframe()
    second = first.copy()
    second["expected_points"] = 0.0
    second["p10"] = 0.0
    second = second.reindex(columns=list(reversed(second.columns)))

    async def run():
        kwargs = dict(
            player_pool_df=pool,
            horizon_projections=horizon,
            current_gw=5,
            target_gw=6,
            free_transfers=1,
            available_chips=[],
        )
        await asyncio.gather(
            service.joint_plan(current_squad_df=first, bank=18.0, **kwargs),
            service.joint_plan(current_squad_df=second, bank=18, **kwargs),
        )

    asyncio.run(run())
    assert solve.call_count == 1


def test_chip_strategy_and_news_reuse(configured_advisor, monkeypatch):
    import hashlib
    import json
    from unittest.mock import Mock

    from fpl_oracle.api.models import BootstrapStatic
    from fpl_oracle.chips.planner import chip_planner
    from fpl_oracle.server.analysis import AnalysisService

    state, pool, horizon = configured_advisor
    service = AnalysisService()
    solve = Mock(return_value={"chip_plan_table": []})
    monkeypatch.setattr(chip_planner, "generate_chip_strategy", solve)
    boot = BootstrapStatic.model_validate({"elements": [], "teams": [], "events": [], "element_types": []})
    service._news_key = (
        hashlib.sha256(json.dumps(boot.model_dump(mode="json"), sort_keys=True).encode()).hexdigest(),
        6,
    )
    service._news_signals = [{"element_id": 1}]

    async def run():
        kwargs = dict(
            current_gw=5,
            current_squad_df=state.to_squad_dataframe(),
            horizon_projections=horizon,
            fixtures=[],
            bootstrap=boot,
            manager_history=None,
        )
        second = dict(kwargs)
        second["current_squad_df"] = kwargs["current_squad_df"].iloc[::-1].reset_index(drop=True)
        second["current_squad_df"]["value"] = second["current_squad_df"]["value"].astype(float)
        await asyncio.gather(service.chip_strategy(**kwargs), service.chip_strategy(**second))
        news = await service.news_signals(boot, 6)
        news[0]["element_id"] = 99
        assert service._news_signals[0]["element_id"] == 1

    asyncio.run(run())
    assert solve.call_count == 1
