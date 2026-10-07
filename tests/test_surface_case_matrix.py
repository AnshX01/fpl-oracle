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
