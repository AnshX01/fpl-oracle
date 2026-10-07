"""Bounded real-engine cross-surface matrix. This is not exhaustive certification."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.briefing.decision_card import decision_card_generator
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.data.store import data_store
from fpl_oracle.server.analysis import analysis_service
from fpl_oracle.server.routes.api import (
    OptimizeRequest,
    get_chip_strategy,
    get_contingency_plans,
    get_squad,
    run_optimizer,
)


@pytest.mark.parametrize(
    "bank,ft,chip,target,stale,manual",
    [
        (0, 0, None, 6, False, False),
        (18, 1, None, 6, False, False),
        (0, 5, None, 6, True, False),
        (10, 1, None, 6, False, True),
        (10, 1, "3xc", 6, False, False),
        (10, 1, "bboost", 6, False, False),
        (10, 1, "wildcard", 19, False, False),
        (10, 1, "freehit", 20, False, False),
    ],
)
def test_concurrent_surface_case(configured_advisor, monkeypatch, bank, ft, chip, target, stale, manual):
    state, pool, horizon = configured_advisor
    state.bank_tenths = bank
    state.free_transfers = ft
    state.is_stale = stale
    state.bank_source = state.ft_source = "manual_override" if manual else "fpl_api"
    state.has_bank_override = state.has_ft_override = manual
    state.current_gw = target - 1
    state.target_gw = target
    state.chips_remaining_set_1 = [chip] if chip and target <= 19 else []
    state.chips_remaining_set_2 = [chip] if chip and target > 19 else []
    forecast = {g: pool.copy() for g in range(target, target + 2)}
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value=forecast))
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    monkeypatch.setattr(analysis_service, "news_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(fpl_client, "get_current_and_next_gw", AsyncMock(return_value=(target - 1, target)))
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )
    analysis_service._chip_plans.clear()
    analysis_service._chip_tasks.clear()

    async def run():
        squad, card, opt, chips, plans, brief = await asyncio.gather(
            get_squad(),
            decision_card_generator.generate_decision_card(),
            run_optimizer(OptimizeRequest()),
            get_chip_strategy(),
            get_contingency_plans(),
            weekly_briefing_generator.generate_briefing(),
        )
        selected = opt["recommended_plan"]

        def ids(rows):
            return sorted(p["element"] for p in rows)

        assert squad["bank_millions"] == bank / 10
        assert squad["free_transfers"] == brief["free_transfers"] == ft
        assert brief["bank_tenths"] == bank
        assert squad["has_bank_override"] == manual
        assert squad["captain"]["element"] == card["captain"]["element"] == brief["captain"]["element"]
        assert squad["vice_captain"]["element"] == card["vice_captain"]["element"] == brief["vice_captain"]["element"]
        assert ids(squad["starters"]) == ids(card["xi"]) == ids(brief["lineup"]["starters"])
        assert ids(squad["bench"]) == ids(card["bench"]) == ids(brief["lineup"]["bench"])
        assert squad["formation"] == card["formation"] == brief["lineup"]["formation"]
        assert squad["total_expected_points"] == brief["lineup"]["total_gameweek_expected_points"]
        assert ids(selected["transfers_in"]) == ids(card["transfers"]["in"]) == ids(plans["plan_a"]["transfers_in"])
        assert ids(selected["transfers_out"]) == ids(card["transfers"]["out"]) == ids(plans["plan_a"]["transfers_out"])
        assert selected["transfers_count"] == len(selected["transfers_in"]) == plans["plan_a"]["transfers_count"]
        assert selected["hits"] == card["transfers"]["hits_count"] == plans["plan_a"]["hits"]
        assert (
            opt["recommended_chip"]
            == chips["recommended_chip"]
            == card["chip"]["chip_name"]
            == brief["chips"]["recommended_chip"]
        )
        assert card["transfers"]["ft_used"] == (
            0 if opt["recommended_chip"] in ("wildcard", "freehit") else min(ft, len(selected["transfers_in"]))
        )
        assert selected["next_banked_ft"] == card["transfers"]["ft_next_gw"]
        assert (
            card["is_stale"]
            == squad["is_stale"]
            == brief["is_stale"]
            == opt["is_stale"]
            == plans["is_stale"]
            == chips["is_stale"]
            == stale
        )

    asyncio.run(run())


@pytest.mark.parametrize("missing", ["squad", "projection", "partial_projection", "invalid_projection"])
def test_no_personal_advice_on_missing_source(configured_advisor, monkeypatch, missing):
    from fastapi import HTTPException

    state, pool, horizon = configured_advisor
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )
    if missing == "squad":
        state.squad = []
    elif missing == "partial_projection":
        monkeypatch.setattr(
            analysis_service, "projections", AsyncMock(return_value={g: pool.iloc[1:].copy() for g in horizon})
        )
    elif missing == "invalid_projection":
        bad = pool.copy()
        bad.loc[0, "expected_points"] = float("inf")
        monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value={g: bad.copy() for g in horizon}))
    else:
        monkeypatch.setattr(
            analysis_service, "projections", AsyncMock(return_value={g: pool.iloc[:0].copy() for g in horizon})
        )

    async def run():
        for name, func in [
            ("squad", get_squad),
            ("card", decision_card_generator.generate_decision_card),
            ("optimizer", lambda: run_optimizer(OptimizeRequest())),
            ("chips", get_chip_strategy),
            ("plans", get_contingency_plans),
            ("briefing", weekly_briefing_generator.generate_briefing),
        ]:
            try:
                result = await func()
            except HTTPException as error:
                assert error.status_code == 409
                print(name, type(error).__name__)
                continue
            assert not result.get("captain"), (name, missing)
            assert not result.get("recommended_plan"), (name, missing)
            assert not result.get("plan_a"), (name, missing)
            assert result.get("status") == "unavailable", (name, missing, result)

    asyncio.run(run())


def test_chat_captain_is_configured_lineup_not_market_leader(configured_advisor, monkeypatch):
    from fpl_oracle.llm.tools import tool_executor

    state, pool, horizon = configured_advisor
    outsider = pool.iloc[0].copy()
    outsider["element"] = 999999
    outsider["web_name"] = "Unowned Market Leader"
    outsider["expected_points"] = 100.0
    outsider["value"] = 9999
    import pandas as pd

    market = pd.concat([pool, pd.DataFrame([outsider])], ignore_index=True)
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value={6: market, 7: market.copy()}))
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )

    async def run():
        squad = await get_squad()
        captain = await tool_executor.execute("captain_options", {})
        assert captain["safe_captain"] == squad["captain"]["web_name"]
        assert captain["captain"]["element"] == squad["captain"]["element"]
        assert all(p["element"] in {r["element"] for r in squad["starters"]} for p in captain["candidates"])

    asyncio.run(run())


def test_pipeline_summary_uses_joint_selected_action(configured_advisor, monkeypatch):
    from fpl_oracle.api.game_state import GameState, game_state_manager
    from fpl_oracle.api.rules_checker import rules_checker
    from fpl_oracle.server.pipeline import SyncPipeline

    state, pool, horizon = configured_advisor
    state.chips_remaining_set_1 = ["3xc"]
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )
    monkeypatch.setattr(
        game_state_manager, "get_game_state", AsyncMock(return_value=GameState(current_gw=5, next_gw=6))
    )
    monkeypatch.setattr(rules_checker, "verify", lambda boot: SimpleNamespace(verified=True))
    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    pipeline = SyncPipeline()

    async def run():
        opt = await run_optimizer(OptimizeRequest())
        await pipeline.run_pipeline()
        summary = pipeline.get_status()["summary"]
        assert not pipeline.get_status()["last_error"]
        assert summary["transfer_plan"]["plan_type"] == opt["recommended_plan"]["plan_type"]
        assert summary["captain"] == opt["recommended_plan"]["lineup"]["captain"]["web_name"]
        assert summary["recommended_chip"] == opt["recommended_chip"]
        assert summary["bank_tenths"] == state.bank_tenths
        assert summary["free_transfers"] == state.free_transfers
        assert summary["chips_plan"].get(6) == opt["recommended_chip"]

    asyncio.run(run())


def test_offline_chat_matches_squad_captain(configured_advisor, monkeypatch):
    from fpl_oracle.llm.provider import OfflineExpertProvider

    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )

    async def run():
        squad = await get_squad()
        text = await OfflineExpertProvider().chat(
            messages=[{"role": "user", "content": "who should I captain?"}], system_prompt=""
        )
        assert f"Captain **{squad['captain']['web_name']}**" in text
        assert f"Vice-captain **{squad['vice_captain']['web_name']}**" in text

    asyncio.run(run())


@pytest.mark.parametrize("ft", [0, 1])
def test_profitable_replacement_hits_and_sale_prices_across_surfaces(configured_advisor, monkeypatch, ft):
    import pandas as pd

    from fpl_oracle.llm.tools import tool_executor

    state, pool, horizon = configured_advisor
    state.free_transfers = ft
    state.bank_tenths = 0
    old = state.squad[2]
    old.purchase_price = old.now_cost - 2
    old.selling_price = old.now_cost - 1
    replacement = pool.iloc[2].copy()
    replacement["element"] = 999998
    replacement["web_name"] = "Fixture Replacement"
    replacement["value"] = old.selling_price
    replacement["expected_points"] = 25.0
    market = pd.concat([pool, pd.DataFrame([replacement])], ignore_index=True)
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value={6: market, 7: market.copy()}))
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )

    async def run():
        squad, card, opt, plans, brief, chat = await asyncio.gather(
            get_squad(),
            decision_card_generator.generate_decision_card(),
            run_optimizer(OptimizeRequest()),
            get_contingency_plans(),
            weekly_briefing_generator.generate_briefing(),
            tool_executor.execute("optimise_transfers", {}),
        )
        plan = opt["recommended_plan"]
        assert [p["element"] for p in plan["transfers_in"]] == [999998]
        assert plan["hits"] == 1 - ft
        assert card["transfers"]["hit_cost"] == 4 * (1 - ft)
        assert card["transfers"]["ft_used"] == ft
        assert plan["remaining_bank"] == card["transfers"]["bank_after"]
        for other in [plans["plan_a"], brief["transfers"]["recommended_plan"], chat["recommended_plan"]]:
            assert other["transfers_in"] == plan["transfers_in"]
            assert other["hits"] == plan["hits"]
        assert squad["total_expected_points"] == brief["lineup"]["total_gameweek_expected_points"]

    asyncio.run(run())


def test_chat_captain_other_gw_and_missing_squad_are_explicit(configured_advisor, monkeypatch):
    from fpl_oracle.llm.provider import OfflineExpertProvider
    from fpl_oracle.llm.tools import tool_executor

    state, _, _ = configured_advisor
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )

    async def run():
        other = await tool_executor.execute("captain_options", {"gameweek": 8})
        assert other["status"] == "unavailable"
        state.squad = []
        text = await OfflineExpertProvider().chat(
            messages=[{"role": "user", "content": "who should I captain?"}], system_prompt=""
        )
        assert text.startswith("Captain advice unavailable:")

    asyncio.run(run())


@pytest.mark.parametrize("constraint", ["keep_owned", "exclude_buy", "exclude_team"])
def test_constrained_chat_optimizer_agree_without_changing_default(configured_advisor, monkeypatch, constraint):
    import pandas as pd

    from fpl_oracle.llm.tools import tool_executor

    state, pool, horizon = configured_advisor
    state.bank_tenths = 0
    state.free_transfers = 1
    replacement = pool.iloc[2].copy()
    owned = state.squad[2]
    replacement["element"] = 999998
    replacement["web_name"] = "Fixture Replacement"
    replacement["value"] = owned.selling_price
    replacement["expected_points"] = 25.0
    market = pd.concat([pool, pd.DataFrame([replacement])], ignore_index=True)
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value={6: market, 7: market.copy()}))
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )
    kwargs = (
        {"locked_in": [owned.element]}
        if constraint == "keep_owned"
        else {"locked_out": [999998]}
        if constraint == "exclude_buy"
        else {"excluded_teams": [int(replacement["team"])]}
    )

    async def run():
        plain = await run_optimizer(OptimizeRequest())
        constrained = await run_optimizer(OptimizeRequest(**kwargs))
        assert [p["element"] for p in plain["recommended_plan"]["transfers_in"]] == [999998]
        assert not constrained["recommended_plan"]["transfers_in"]
        if constraint != "exclude_team":
            chat = await tool_executor.execute("optimise_transfers", kwargs)
            assert chat["recommended_plan"] == constrained["recommended_plan"]
        again = await run_optimizer(OptimizeRequest())
        assert again["recommended_plan"] == plain["recommended_plan"]

    asyncio.run(run())


def test_stale_fixtures_propagate_every_advice_surface(configured_advisor, monkeypatch):
    monkeypatch.setattr(
        data_store,
        "get_profile",
        lambda: SimpleNamespace(manager_id=None, target_league_id=None, risk_preference="balanced"),
    )
    original = fpl_client.get_fixtures

    async def stale_fixtures(**kwargs):
        rows, _ = await original()
        return rows, True

    monkeypatch.setattr(fpl_client, "get_fixtures", stale_fixtures)

    async def run():
        results = await asyncio.gather(
            get_squad(),
            decision_card_generator.generate_decision_card(),
            run_optimizer(OptimizeRequest()),
            get_chip_strategy(),
            get_contingency_plans(),
            weekly_briefing_generator.generate_briefing(),
        )
        for name, result in zip(["squad", "card", "optimizer", "chips", "plans", "briefing"], results, strict=True):
            assert result["is_stale"], name

    asyncio.run(run())
