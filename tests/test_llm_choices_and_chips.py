"""Chat tools respect keep / don't-buy choices, solve once per turn, and chip answers never print None."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd

from fpl_oracle.advisor.critique import PlanCritiqueService
from fpl_oracle.llm import tools
from fpl_oracle.llm.provider import OfflineExpertProvider
from fpl_oracle.llm.tools import SOLVE_NOTE, chat_tool_cache, tool_executor


def _choices(monkeypatch, keep, avoid):
    monkeypatch.setattr(tools.data_store, "get_profile", lambda: SimpleNamespace(keep_ids=keep, avoid_ids=avoid))
    boot = SimpleNamespace(
        elements=[SimpleNamespace(id=7, web_name="Saka"), SimpleNamespace(id=9, web_name="Isak")], teams=[]
    )
    monkeypatch.setattr(tools.fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))


def test_evaluate_plan_refuses_avoided_buy_and_kept_sale(monkeypatch):
    _choices(monkeypatch, keep=[7], avoid=[9])
    buy = asyncio.run(tool_executor.execute("evaluate_plan", {"proposed_transfers": [{"out": 1, "in": 9}]}))
    sell = asyncio.run(tool_executor.execute("evaluate_plan", {"proposed_transfers": [{"out": 7, "in": 2}]}))
    assert buy == {"error": "Isak is on your don't-buy list"}
    assert sell == {"error": "Saka is on your keep list"}


def test_affordability_path_refuses_avoided_target(monkeypatch):
    _choices(monkeypatch, keep=[], avoid=[9])
    res = asyncio.run(tool_executor.execute("affordability_path", {"target_element": 9}))
    assert res == {"error": "Isak is on your don't-buy list"}


def test_critique_baselines_carry_keep_and_avoid():
    calls = []

    class Plan:
        total_objective = 40.0
        recommended_action = SimpleNamespace(captain=1, transfers_out=[], transfers_in=[])

    class Optimizer:
        def optimize(self, **kw):
            calls.append(kw)
            return Plan()

    pool = pd.DataFrame(
        {"element": [1, 2], "value": [50, 50], "selling_price": [50, 50], "expected_points": [5.0, 5.0]}
    )
    PlanCritiqueService(optimizer=Optimizer()).evaluate_plan(
        proposed_transfers=[],
        proposed_chip="wildcard",
        proposed_captain=None,
        current_squad_df=pool,
        player_pool_df=pool,
        bank=5,
        free_transfers=1,
        horizon_projections={6: pool},
        current_gw=5,
        target_gw=6,
        chips_available=["wildcard"],
        locked_in_ids=[1],
        locked_out_ids=[2],
    )
    assert len(calls) == 3  # roll, best alternative, chip baseline
    assert all(c["locked_in_ids"] == [1] and c["locked_out_ids"] == [2] for c in calls)


def test_one_planner_rerun_per_chat_turn_and_it_says_so(monkeypatch):
    execute = AsyncMock(side_effect=lambda *a: {"verdict": "GOOD"})
    monkeypatch.setattr(tool_executor, "_execute", execute)

    async def turn():
        token = chat_tool_cache.set({})
        try:
            first = await tool_executor.execute("evaluate_plan", {})
            second = await tool_executor.execute("compare_options", {"options": []})
            other = await tool_executor.execute("get_news", {"player_name": "x"})
            return first, second, other
        finally:
            chat_tool_cache.reset(token)

    first, second, other = asyncio.run(turn())
    assert first["note"] == SOLVE_NOTE
    assert "already re-run" in second["error"]
    assert "note" not in other
    assert execute.await_count == 2  # evaluate_plan and get_news


def test_offline_chip_answer_never_prints_none(monkeypatch):
    rec = {
        "status": "ready",
        "gameweek": 8,
        "chip": {
            "play_now": False,
            "chip": None,
            "later_provisional": [{"chip_display_name": "Bench Boost", "gameweek": 12}],
        },
        "team": {},
    }

    async def execute(name, args):
        assert name == "get_current_decision"  # no planner re-run for a chip question
        return rec

    monkeypatch.setattr(tools.tool_executor, "execute", execute)
    monkeypatch.setattr("fpl_oracle.llm.provider.tool_executor.execute", execute)
    text = asyncio.run(OfflineExpertProvider().chat([{"role": "user", "content": "When do I play bench boost?"}], ""))
    assert "None" not in text
    assert "| Bench Boost | gameweek 12 |" in text
