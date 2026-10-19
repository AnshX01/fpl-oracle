"""Chat knows the published decision: prompt injection, tools, provider tool loops, offline answers."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.llm import agent as agent_module
from fpl_oracle.llm import decision_record
from fpl_oracle.llm.agent import ExpertAgent
from fpl_oracle.llm.anthropic import AnthropicProvider
from fpl_oracle.llm.openai import OpenAIProvider
from fpl_oracle.llm.provider import OFFLINE_MODE_PREFIX, OfflineExpertProvider, build_system_prompt
from fpl_oracle.llm.tools import TOOL_DEFINITIONS, tool_executor
from fpl_oracle.server.advice_job import advice_publisher


def _player(eid, name, pos, xp, **extra):
    return {
        "element": eid,
        "web_name": name,
        "team_short": "ARS",
        "position": pos,
        "now_cost": 8.5,
        "selling_price": 8.0,
        "expected_points": xp,
        "p10": 2.0,
        "p90": 11.0,
        "chance_of_playing": 100,
        "status": "a",
        "news": "",
        "next_fixtures": [{"event": 8, "opp_short": "CHE", "is_home": True, "difficulty": 3}],
        **extra,
    }


def published_result():
    cap = _player(1, "Saka", "MID", 7.4, is_captain=True)
    vice = _player(2, "Palmer", "MID", 6.9, is_vice_captain=True)
    return {
        "snapshot_id": "abc123def456789",
        "data_as_of": "2026-10-18T10:00:00+00:00",
        "squadData": {
            "target_gameweek": 8,
            "manager_name": "Test Manager",
            "bank_millions": 0.1,
            "free_transfers": 2,
            "chips_used": ["bboost"],
            "formation": "3-4-3",
            "captain": cap,
            "vice_captain": vice,
            "starters": [cap, vice],
            "bench": [_player(3, "Benchy", "DEF", 1.2, bench_order=1)],
            "starters_expected_points": 14.3,
            "total_expected_points": 21.7,
        },
        "decisionCard": {
            "gameweek": 8,
            "deadline": {"label": "GW8 Deadline: 2026-10-23T17:30:00Z"},
            "model_version": "v9.9",
            "transfers": {
                "action": "SINGLE_TRANSFER",
                "is_roll": False,
                "in": [{"element": 10, "web_name": "Haaland", "cost": 14.0, "expected_points": 8.8}],
                "out": [{"element": 11, "web_name": "Wood", "cost": 7.0, "expected_points": 4.1}],
                "bank_after": 0.1,
                "ft_used": 1,
                "ft_remaining": 1,
                "ft_next_gw": 2,
                "hit_cost": 0,
                "hits_count": 0,
                "net_gain_vs_roll": 3.3,
                "expected_gain_gw": 4.7,
                "horizon_pts": 61.0,
            },
            "chip": {"recommend": False, "chip_name": None, "gain_vs_hold": 0.0},
            "ranked_options": [
                {"rank": 1, "chip": "HOLD", "net_gain_vs_hold": 0.0, "recommended": True},
                {"rank": 2, "chip": "wildcard", "net_gain_vs_hold": 1.9, "recommended": False},
            ],
            "two_line_reasoning": "Wood -> Haaland: +4.7 predicted pts this week.",
            "caveats": ["Confirm your team in FPL before the deadline."],
        },
        "contingencyPlans": {
            "plan_a": {"title": "Plan A (Primary)", "net_expected_points": 61.0, "delta_vs_plan_a": 0.0},
            "plan_b": {
                "title": "Plan B (Injury Pivot): Pivot to Isak",
                "net_expected_points": 59.5,
                "delta_vs_plan_a": -1.5,
                "trigger_condition": "Press conference confirms Haaland is doubtful",
            },
        },
    }


@pytest.fixture
def publication(monkeypatch):
    monkeypatch.setattr(
        advice_publisher, "current", {"status": "ready", "result": published_result(), "profile_key": "k"}
    )
    monkeypatch.setattr(decision_record, "_extras", AsyncMock(return_value={}))


@pytest.fixture
def no_publication(monkeypatch):
    monkeypatch.setattr(advice_publisher, "current", None)


def test_record_is_faithful_and_compact():
    rec = decision_record.build_decision_record(
        published_result(), {"chips": {"set_1_gw1_19": ["wildcard"], "set_2_gw20_38": []}}
    )
    assert rec["gameweek"] == 8 and rec["snapshot_id"] == "abc123def456789"
    assert rec["transfers"]["in"][0]["name"] == "Haaland"
    assert rec["transfers"]["gain_vs_roll_over_horizon_xp"] == 3.3
    assert rec["team"]["bank_millions"] == 0.1 and rec["team"]["chips_remaining"]["set_1_gw1_19"] == ["wildcard"]
    assert rec["lineup"]["captain"]["web_name"] == "Saka"
    assert rec["lineup"]["xi"][0]["fixtures"] == ["GW8 CHE(H,fdr3)"]
    assert rec["alternatives"]["contingency"]["plan_b"]["delta_vs_plan_a"] == -1.5
    assert rec["chip"]["now_vs_later"][1]["chip"] == "wildcard"
    assert len(json.dumps(rec)) < 20000


def test_decision_record_reaches_every_turn_prompt(publication, monkeypatch):
    seen = {}

    class FakeProvider:
        async def chat(self, messages, system_prompt):
            seen["messages"], seen["system"] = messages, system_prompt
            return "Haaland in, captain Saka."

    monkeypatch.setattr(agent_module, "get_llm_provider", lambda: FakeProvider())
    monkeypatch.setattr(agent_module.data_store, "add_chat_message", lambda **kw: None)
    # History window starts mid-conversation with an offline-labelled assistant turn.
    monkeypatch.setattr(
        agent_module.data_store,
        "get_chat_history",
        lambda **kw: [
            {"role": "assistant", "content": OFFLINE_MODE_PREFIX + "old"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": OFFLINE_MODE_PREFIX + "Captain Saka."},
            {"role": "user", "content": "why Haaland?"},
        ],
    )
    assert asyncio.run(ExpertAgent().answer("why Haaland?")) == "Haaland in, captain Saka."
    system = seen["system"]
    assert "PUBLISHED DECISION RECORD (GW8" in system
    assert "Haaland" in system and "Plan B (Injury Pivot)" in system and "Test Manager" in system
    assert "Minimal Sufficient Intervention" in system  # voice contract is loaded
    assert seen["messages"][0] == {"role": "user", "content": "hi"}
    assert seen["messages"][1]["content"] == "Captain Saka."  # offline banner stripped
    assert seen["messages"][-1] == {"role": "user", "content": "why Haaland?"}


def test_prompt_says_so_when_nothing_is_published(no_publication):
    system = asyncio.run(build_system_prompt())
    assert "PUBLISHED DECISION: NOT READY" in system
    assert "use the tools" in system


def test_get_current_decision_tool_sections(publication):
    async def run():
        full = await tool_executor.execute("get_current_decision", {})
        part = await tool_executor.execute("get_current_decision", {"section": "transfers"})
        raw = await tool_executor.execute("get_current_decision", {"section": "raw"})
        bad = await tool_executor.execute("get_current_decision", {"section": "nope"})
        return full, part, raw, bad

    full, part, raw, bad = asyncio.run(run())
    assert full["status"] == "ready" and full["lineup"]["vice_captain"]["web_name"] == "Palmer"
    assert set(part) == {"status", "gameweek", "snapshot_id", "transfers"}
    assert raw["decision_card"]["two_line_reasoning"].startswith("Wood")
    assert bad["status"] == "error"


def test_get_current_decision_unavailable_without_publication(no_publication):
    res = asyncio.run(tool_executor.execute("get_current_decision", {}))
    assert res["status"] == "unavailable" and "No advice has been published" in res["reason"]


# ---- provider tool loops ------------------------------------------------------------------


class _Resp:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


def _client(responses, payloads):
    class Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json, headers):
            payloads.append(copy.deepcopy(json))
            return responses.pop(0)

    return Client


def _tool_spy(monkeypatch, module):
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return {"ok": name}

    monkeypatch.setattr(f"fpl_oracle.llm.{module}.tool_executor.execute", execute)
    return calls


def test_anthropic_text_plus_multi_tool_use_and_second_round(monkeypatch):
    payloads, responses = (
        [],
        [
            _Resp(
                {
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "t1", "name": "get_current_decision", "input": {}},
                        {"type": "tool_use", "id": "t2", "name": "captain_options", "input": {}},
                    ]
                }
            ),
            _Resp(
                {"content": [{"type": "tool_use", "id": "t3", "name": "get_news", "input": {"player_name": "Saka"}}]}
            ),
            _Resp({"content": [{"type": "text", "text": "Captain Saka at 7.4 xP."}]}),
        ],
    )
    monkeypatch.setattr("fpl_oracle.llm.anthropic.httpx.AsyncClient", _client(responses, payloads))
    calls = _tool_spy(monkeypatch, "anthropic")
    text = asyncio.run(AnthropicProvider("k").chat([{"role": "user", "content": "captain?"}], "sys"))
    assert text == "Captain Saka at 7.4 xP."
    assert [c[0] for c in calls] == ["get_current_decision", "captain_options", "get_news"]
    assert all(p["tools"] and p["model"] == "claude-sonnet-5-5" for p in payloads)
    answered = payloads[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in answered] == ["t1", "t2"]  # one tool_result per tool_use
    assert payloads[2]["messages"][-1]["content"][0]["tool_use_id"] == "t3"


def test_anthropic_errors_never_become_fake_answers(monkeypatch):
    for resp in (_Resp({}, status=400), _Resp({"content": [{"type": "text", "text": "  "}]})):
        monkeypatch.setattr("fpl_oracle.llm.anthropic.httpx.AsyncClient", _client([resp], []))
        with pytest.raises(RuntimeError):
            asyncio.run(AnthropicProvider("k").chat([{"role": "user", "content": "x"}], "sys"))


def test_anthropic_model_is_configurable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")
    assert AnthropicProvider("k").model == "claude-test-model"
    assert AnthropicProvider("k", model="explicit").model == "explicit"


def test_openai_answers_every_tool_call_over_several_rounds(monkeypatch):
    def call(cid, name):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}

    payloads, responses = (
        [],
        [
            _Resp(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [call("a", "get_my_team"), call("b", "plan_chips")],
                            }
                        }
                    ]
                }
            ),
            _Resp(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [call("c", "captain_options")],
                            }
                        }
                    ]
                }
            ),
            _Resp({"choices": [{"message": {"role": "assistant", "content": "Hold the chip."}}]}),
        ],
    )
    monkeypatch.setattr("fpl_oracle.llm.openai.httpx.AsyncClient", _client(responses, payloads))
    calls = _tool_spy(monkeypatch, "openai")
    assert asyncio.run(OpenAIProvider("k").chat([{"role": "user", "content": "chips?"}], "sys")) == "Hold the chip."
    assert len(calls) == 3
    tool_ids = [m["tool_call_id"] for m in payloads[-1]["messages"] if m["role"] == "tool"]
    assert tool_ids == ["a", "b", "c"]
    assert all("tools" in p for p in payloads)


def test_gemini_schemas_are_valid_and_tool_use_is_optional():
    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object" and path:  # nested objects need properties
                assert node.get("properties"), path
            for k, v in node.items():
                walk(v, path + [k])
        elif isinstance(node, list):
            for v in node:
                walk(v, path)

    for t in TOOL_DEFINITIONS:
        walk(t["parameters"], [])
    assert "get_current_decision" in {t["name"] for t in TOOL_DEFINITIONS}


# ---- critique and tools use real, consistent inputs ---------------------------------------


def test_evaluate_plan_tool_uses_tenths_remaining_chips_and_plan_horizon(configured_advisor, monkeypatch):
    from fpl_oracle.advisor.critique import PlanEvaluationResult, plan_critique_service

    state, _, _ = configured_advisor
    state.bank_tenths, state.free_transfers = 15, 2
    state.chips_used = [{"name": "bboost", "event": 3}]
    state.chips_remaining_set_1 = ["wildcard", "freehit"]
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return PlanEvaluationResult()

    monkeypatch.setattr(plan_critique_service, "evaluate_plan", fake)
    asyncio.run(tool_executor.execute("evaluate_plan", {"proposed_chip": "wildcard"}))
    assert seen["bank"] == 15.0  # tenths, not 1.5
    assert seen["chips_available"] == ["wildcard", "freehit"]  # remaining, not the used bboost
    assert seen["proposed_chip"] == "wildcard"
    assert seen["horizon_len"] == 14  # GW6..19, the same window the live plan searches
    assert seen["free_transfers"] == 2


def test_what_if_runs_off_the_event_loop_and_respects_keep_and_avoid(configured_advisor, monkeypatch):
    import threading
    from types import SimpleNamespace

    from fpl_oracle.advisor.critique import PlanEvaluationResult, plan_critique_service
    from fpl_oracle.llm import tools

    seen = {}

    def fake(**kw):
        seen.update(kw, thread=threading.current_thread())
        return PlanEvaluationResult()

    monkeypatch.setattr(plan_critique_service, "evaluate_plan", fake)
    monkeypatch.setattr(tools.data_store, "get_profile", lambda: SimpleNamespace(keep_ids=[7], avoid_ids=[9, 7]))

    async def run():
        loop_thread = threading.current_thread()
        await tool_executor.execute("evaluate_plan", {})
        return loop_thread

    loop_thread = asyncio.run(run())
    assert seen["thread"] is not loop_thread  # a minute-long solve must not freeze the app
    assert seen["locked_in_ids"] == [7]
    assert seen["locked_out_ids"] == [9]  # a player both kept and avoided is kept, as in the main plan


def test_critique_honours_proposed_chip_and_chip_availability():
    import pandas as pd

    from fpl_oracle.advisor.critique import PlanCritiqueService

    class Plan:
        def __init__(self, xp):
            self.total_objective = xp
            self.recommended_action = type("A", (), {"captain": 1, "transfers_out": [], "transfers_in": []})()

    calls = []

    class Optimizer:
        def optimize(self, **kw):
            calls.append(kw)
            return Plan(50.0 if kw.get("forced_w1_chip") == "wildcard" else 40.0)

    svc = PlanCritiqueService(optimizer=Optimizer())
    pool = pd.DataFrame(
        {"element": [1, 2], "value": [50, 50], "selling_price": [50, 50], "expected_points": [5.0, 5.0]}
    )
    kw = dict(
        proposed_transfers=[],
        proposed_captain=None,
        current_squad_df=pool,
        player_pool_df=pool,
        bank=5,
        free_transfers=1,
        horizon_projections={6: pool},
        current_gw=5,
        target_gw=6,
    )
    ok = svc.evaluate_plan(proposed_chip="Wildcard", chips_available=["wildcard"], **kw)
    assert ok.proposed_chip == "wildcard" and ok.proposed_net_xp == 50.0  # chip plan, not the roll baseline
    assert any(c.get("forced_w1_chip") == "wildcard" for c in calls)
    bad = svc.evaluate_plan(proposed_chip="wildcard", chips_available=["bboost"], **kw)
    assert bad.verdict == "WORSE THAN ALTERNATIVE" and "not available" in bad.numeric_reason


def test_explain_why_not_uses_planner_numbers_and_never_invents_them():
    from fpl_oracle.advisor.critique import PlanCritiqueService

    svc = PlanCritiqueService()
    plans = [
        {"label": "ROLL_TRANSFER", "chip": None, "hits": 0, "gain_vs_roll": 0.0, "recommended": True},
        {"label": "2_TRANSFERS", "chip": None, "hits": 1, "gain_vs_roll": -1.7, "recommended": False},
        {"label": "wildcard plan", "chip": "wildcard", "hits": 0, "gain_vs_roll": 2.4, "recommended": False},
    ]
    hit = svc.explain_why_not("hit", plans=plans)
    assert hit.numbers["option_gain_vs_roll_xp"] == -1.7 and hit.numbers["option_hit_cost"] == 4
    assert hit.numbers["opportunity_cost_xp"] == 1.7
    wc = svc.explain_why_not("wildcard", plans=plans)
    assert wc.numbers["option_gain_vs_roll_xp"] == 2.4 and wc.numbers["recommended_gain_vs_roll_xp"] == 0.0
    assert wc.verdict == "NO BETTER THAN THE RECOMMENDED PLAN"
    assert svc.explain_why_not("wildcard", plans=None).numbers == {}


def test_get_projections_ranks_best_first_and_sums_horizon(configured_advisor):
    res = asyncio.run(tool_executor.execute("get_projections", {"horizon": 2, "limit": 3}))
    xp = [p["expected_points_horizon"] for p in res["players"]]
    assert xp == sorted(xp, reverse=True) and len(xp) == 3
    best = res["players"][0]
    assert best["expected_points_horizon"] == pytest.approx(sum(best["xp_by_gw"].values()), abs=0.02)
    assert res["gameweeks"] == [6, 7]


def test_get_my_team_reports_chips_remaining_and_player_context(configured_advisor):
    state, _, _ = configured_advisor
    state.chips_remaining_set_1 = ["wildcard"]
    res = asyncio.run(tool_executor.execute("get_my_team", {}))
    assert res["chips_remaining_set_1"] == ["wildcard"] and "chips_remaining_set_2" in res
    assert {"form", "next_fixtures", "minutes"} <= set(res["squad"][0])


def test_captain_tool_reads_published_decision(publication, configured_advisor):
    res = asyncio.run(tool_executor.execute("captain_options", {}))
    assert res["source"] == "published_decision"
    assert res["captain_name"] == "Saka" and res["vice_captain_name"] == "Palmer"
    assert "differential_captain" not in res


# ---- offline provider ----------------------------------------------------------------------


def _fake_tools(monkeypatch, record):
    pool = [
        {
            "element": 10,
            "web_name": "Haaland",
            "position": "FWD",
            "cost": 14.0,
            "ownership_pct": 55.0,
            "expected_points_gw": 8.8,
            "expected_points_horizon": 40.0,
            "p10_floor": 2.0,
            "p90_ceiling": 14.0,
        },
        {
            "element": 20,
            "web_name": "Isak",
            "position": "FWD",
            "cost": 8.5,
            "ownership_pct": 9.0,
            "expected_points_gw": 6.1,
            "expected_points_horizon": 31.5,
            "p10_floor": 1.0,
            "p90_ceiling": 11.0,
        },
    ]
    seen = []

    async def execute(name, args):
        seen.append(name)
        if name == "get_current_decision":
            return record
        if name == "get_projections":
            return {"gameweek": 8, "gameweeks": [8, 9, 10, 11, 12], "players": pool}
        if name == "evaluate_plan":
            return {"verdict": "WORSE THAN ALTERNATIVE", "numeric_reason": "Underperforms by 8.5 xP.", "regret_xp": 8.5}
        raise AssertionError(name)

    monkeypatch.setattr("fpl_oracle.llm.provider.tool_executor.execute", execute)
    return seen


def test_offline_explains_decision_from_record(publication, monkeypatch):
    record = decision_record.build_decision_record(published_result())
    _fake_tools(monkeypatch, record)
    text = asyncio.run(OfflineExpertProvider().chat([{"role": "user", "content": "Why did you recommend this?"}], ""))
    assert text.startswith(OFFLINE_MODE_PREFIX)
    assert "sell Wood" in text and "buy Haaland" in text and "captain Saka" in text
    assert "+3.3 xP over the planning horizon" in text
    assert "Plan B (Injury Pivot)" in text and "wildcard: 1.9 xP vs holding" in text
    assert "Press conference confirms Haaland is doubtful" in text


def test_offline_why_not_compares_with_recommended_pick(publication, monkeypatch):
    record = decision_record.build_decision_record(published_result())
    seen = _fake_tools(monkeypatch, record)
    messages = [{"role": "user", "content": "Why not Isak instead?"}]
    text = asyncio.run(OfflineExpertProvider().chat(messages, ""))
    assert "Why not Isak?" in text
    assert "31.5 xP over GW8-12" in text and "40.0 xP over GW8-12" in text
    assert "Haaland leads by +8.50 xP" in text
    assert "Underperforms by 8.5 xP" in text and "evaluate_plan" in seen
    assert asyncio.run(OfflineExpertProvider().chat(messages, "")) == text  # deterministic


def test_offline_without_publication_still_uses_tools(no_publication, monkeypatch):
    async def execute(name, args):
        if name == "get_current_decision":
            return {"status": "unavailable", "reason": "none"}
        assert name == "get_projections"
        return {
            "gameweek": 8,
            "players": [{"web_name": "Haaland", "expected_points_gw": 8.8, "exp_defcon_points": 0.0}],
        }

    monkeypatch.setattr("fpl_oracle.llm.provider.tool_executor.execute", execute)
    text = asyncio.run(OfflineExpertProvider().chat([{"role": "user", "content": "hello there"}], ""))
    assert "No published decision is ready" in text and "Haaland" in text


def test_offline_hold_over_the_weekend_is_not_a_comparison(no_publication, monkeypatch):
    names = []

    async def execute(name, args):
        names.append(name)
        if name == "get_current_decision":
            return {"status": "unavailable", "reason": "none"}
        return {"gameweek": 8, "players": [], "gameweeks": [8]}

    monkeypatch.setattr("fpl_oracle.llm.provider.tool_executor.execute", execute)
    asyncio.run(OfflineExpertProvider().chat([{"role": "user", "content": "should I hold over the weekend"}], ""))
    assert "compare_players" not in names
