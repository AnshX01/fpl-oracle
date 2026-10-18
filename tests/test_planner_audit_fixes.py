"""Planner audit regressions: chips are real candidates, plans are executable, rules and solver status are honest."""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
from tests.optimise.test_multiperiod import _generate_candidate_pool, _generate_valid_random_squad

GWS = (8, 9, 10)


def scenario(squad_xp=None, pool_xp=None, extra=50, gws=GWS, seed=3):
    """A legal 15-man squad plus a candidate pool; the same projection in every gameweek."""
    rng = np.random.default_rng(seed)
    squad = _generate_valid_random_squad(rng)
    pool = _generate_candidate_pool(squad, rng, extra_players=extra)
    if squad_xp is not None:
        pool.loc[pool.element.isin(squad.element), "expected_points"] = squad_xp
    if pool_xp is not None:
        pool.loc[~pool.element.isin(squad.element), "expected_points"] = pool_xp
        pool.loc[~pool.element.isin(squad.element), ["value", "purchase_price", "selling_price"]] = 50
    pool["p10"] = pool.expected_points * 0.3
    pool["p90"] = pool.expected_points * 1.8
    squad = pool[pool.element.isin(squad.element)].reset_index(drop=True)
    return squad, pool, {g: pool.copy() for g in gws}


def evaluate(squad, pool, proj, horizon, cfg=None, **kwargs):
    opt = MultiPeriodOptimizer(MultiPeriodConfig(time_limit_sec=30, **(cfg or {})))
    args = dict(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=10,
        free_transfers=2,
        horizon_projections=proj,
        current_gw=7,
        target_gw=GWS[0],
        horizon_len=horizon,
    )
    args.update(kwargs)
    return opt.evaluate_transfer_options(**args)


def test_chip_without_transfers_is_a_candidate_and_can_be_recommended():
    # Free Hit and Triple Captain have no transfers: they used to collide with Roll and vanish
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    res = evaluate(squad, pool, proj, 1, chips_available=["freehit"])
    assert res["recommended_chip"] == "freehit"
    assert {c["chip"] for c in res["candidate_plans"]} >= {"freehit", None}

    squad, pool, proj = scenario(squad_xp=3.0, pool_xp=1.0)
    star = int(squad.element.iloc[0])
    for frame in [pool, *proj.values()]:
        frame.loc[frame.element == star, "expected_points"] = 14.0
    squad = pool[pool.element.isin(squad.element)].reset_index(drop=True)
    res = evaluate(squad, pool, proj, 1, chips_available=["3xc"])
    assert res["recommended_chip"] == "3xc"
    assert res["recommended_plan"]["captain_id"] == star


def test_free_hit_exposes_a_temporary_squad_and_reverts():
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    # FH now and FH next week tie here; the chip-now margin would keep it, this test is about FH mechanics
    res = evaluate(squad, pool, proj, 2, stability_threshold=0.0, chips_available=["freehit"])
    plan = res["recommended_plan"]
    owned = set(squad.element)
    fh = plan["free_hit"]
    assert plan["chip"] == "freehit" and len(fh["squad"]) == 15
    assert set(fh["reverts_to"]) == owned
    # Players to buy / sell are listed, and the XI and captain come from the Free Hit squad
    assert plan["transfers_in"] and all(t["temporary"] for t in plan["transfers_in"])
    assert set(plan["starting_xi"]) | set(plan["bench_ids"]) == set(fh["squad"])
    assert plan["captain_id"] in fh["squad"]
    assert plan["trajectory"][0]["elements"] == fh["squad"]
    assert plan["trajectory"][1]["elements"] == sorted(owned) or plan["trajectory"][1]["transfers_count"] > 0
    assert plan["hits"] == 0 and plan["remaining_bank"] == 1.0


def test_free_hit_budget_values_owned_players_at_selling_price():
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    # Owned players have risen 1.0m since purchase: only half of that is spendable
    owned = pool.element.isin(squad.element)
    for frame in [pool, *proj.values()]:
        frame.loc[frame.element.isin(squad.element), "value"] += 10
    squad = pool[owned].reset_index(drop=True)
    plan = evaluate(squad, pool, proj, 1, chips_available=["freehit"])["recommended_plan"]
    sell = {int(r.element): r.purchase_price + (r.value - r.purchase_price) // 2 for r in squad.itertuples()}
    budget = sum(sell.values()) + 10
    cost = sum(sell.get(e, float(pool.set_index("element").value[e])) for e in plan["free_hit"]["squad"])
    assert cost <= budget


@pytest.mark.parametrize("forced", [1, 3])
def test_forced_transfers_are_real_swaps_of_owned_players(forced):
    # tin and tout could both be 1 for the same (even unowned) player: a fake "forced k transfers"
    squad, pool, proj = scenario(squad_xp=7.0, pool_xp=2.0)
    opt = MultiPeriodOptimizer(MultiPeriodConfig(time_limit_sec=30))
    plan = opt.optimize(
        squad,
        pool,
        bank=10,
        free_transfers=2,
        horizon_projections=proj,
        current_gw=7,
        target_gw=8,
        horizon_len=2,
        chips_available=[],
        forced_w1_transfers=forced,
    )
    act = plan.recommended_action
    ins = {t["element"] for t in act.transfers_in}
    outs = {t["element"] for t in act.transfers_out}
    assert len(ins) == len(outs) == forced
    assert not ins & outs and outs <= set(squad.element) and not ins & set(squad.element)
    assert act.hits == max(0, forced - 2)
    assert len(act.squad) == 15


def test_alternative_moves_are_generated():
    squad, pool, proj = scenario()
    res = evaluate(squad, pool, proj, 2, chips_available=[])
    assert any(c["source"].startswith("alt:") for c in res["candidate_plans"])
    moves = {
        (
            c["chip"],
            tuple(sorted(t["element"] for t in c["transfers_in"])),
            tuple(sorted(t["element"] for t in c["transfers_out"])),
        )
        for c in res["candidate_plans"]
    }
    assert len(moves) == len(res["candidate_plans"])


def test_hit_cooldown_blocks_hits():
    squad, pool, proj = scenario(squad_xp=6.0, pool_xp=6.0)
    # six dead outfielders: three fill the bench, three dead starters need transfers (one free, two hits)
    dead = [int(e) for pos in ("DEF", "MID") for e in squad[squad.position == pos].element.iloc[:3]]
    for frame in [pool, *proj.values()]:
        frame.loc[frame.element.isin(dead), ["expected_points", "p10", "p90"]] = 0.0
    squad = pool[pool.element.isin(squad.element)].reset_index(drop=True)
    free = evaluate(squad, pool, proj, 3, chips_available=[], free_transfers=1)
    assert free["recommended_plan"]["hits"] >= 1  # dead starters are worth a hit
    blocked = evaluate(squad, pool, proj, 3, chips_available=[], free_transfers=1, hit_cooldown_until=GWS[0] + 1)
    assert blocked["recommended_plan"]["hits"] == 0
    assert all(c["hits"] == 0 for c in blocked["candidate_plans"])


def test_chips_already_used_are_never_planned():
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    res = evaluate(
        squad,
        pool,
        proj,
        3,
        chips_available=["freehit", "wildcard", "3xc", "bboost"],
        chips_already_used=["freehit", "wildcard"],
    )
    planned = {step["chip"] for c in res["candidate_plans"] for step in c["trajectory"]}
    assert not planned & {"freehit", "wildcard"}
    assert res["recommended_chip"] not in {"freehit", "wildcard"}
    # per-set availability: set 2 chips are not spent in a set 1 window
    res = evaluate(squad, pool, proj, 3, chips_by_set={1: ["3xc"], 2: ["freehit", "wildcard"]})
    planned = {step["chip"] for c in res["candidate_plans"] for step in c["trajectory"]}
    assert planned <= {"3xc", None}


def test_active_chip_is_forced_and_hold_is_unavailable():
    squad, pool, proj = scenario()
    res = evaluate(squad, pool, proj, 2, chips_available=["bboost", "3xc"], forced_current_chip="bboost")
    assert res["recommended_chip"] == "bboost"
    assert res["normal_plan_feasible"] is False
    assert [r["chip"] for r in res["chip_comparison_table"]] == ["bboost"]
    assert all(c["chip"] == "bboost" for c in res["candidate_plans"])


def test_chip_table_compares_playing_each_chip_now_against_holding():
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0)
    res = evaluate(squad, pool, proj, 2, chips_available=["freehit", "3xc"])
    table = {r["chip"]: r for r in res["chip_comparison_table"]}
    assert set(table) == {"HOLD", "freehit", "3xc"}
    assert table["HOLD"]["net_gain_vs_hold"] == 0.0
    assert sum(r["is_recommended"] for r in table.values()) == 1
    assert res["best_candidate"]["is_recommended"]
    roadmap = {r["code"]: r for r in res["measured_chip_roadmap"]}
    assert set(roadmap) == {"freehit", "3xc"}
    assert roadmap["freehit"]["recommended_gw"] is not None and roadmap["freehit"]["expected_gain"] > 0
    assert res["resource_frontier"] and res["chip_measurement_status"] in {
        "incomplete_window",
        "measured_through_expiry",
    }


def test_wildcard_keeps_banked_transfers():
    squad, pool, proj = scenario(squad_xp=3.0, pool_xp=5.0)
    opt = MultiPeriodOptimizer(MultiPeriodConfig(time_limit_sec=30))
    plan = opt.optimize(
        squad,
        pool,
        bank=10,
        free_transfers=4,
        horizon_projections=proj,
        current_gw=7,
        target_gw=8,
        horizon_len=2,
        chips_available=["wildcard"],
        forced_w1_chip="wildcard",
    )
    assert plan.trajectory[0].active_chip == "wildcard"
    assert plan.trajectory[1].free_transfers_available == 4


def test_solver_status_reaches_the_decision_scope(monkeypatch):
    from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
    from fpl_oracle.server.analysis import execute_authoritative_milp_plan

    fake = {
        "recommended_plan": {"chip": "wildcard", "trajectory": [{"chip": "wildcard"}]},
        "candidate_plans": [],
        "horizon_gameweeks": [8, 9],
        "solver": {"global_optimum_proven": False, "max_gap": 0.03, "time_limit_sec": 60, "plans": {"opt": {}}},
    }

    def solve(current_squad_df=None, player_pool_df=None, target_gw=None, hit_cooldown_until=0, xp_estimation_sd=None):
        return fake

    monkeypatch.setattr(multiperiod_optimizer, "evaluate_transfer_options", solve)
    scope = execute_authoritative_milp_plan(
        current_squad_df=pd.DataFrame(),
        player_pool_df=pd.DataFrame(),
        target_gw=8,
        rival_context={"rivals": [1]},
        hit_cooldown_until=9,
    )["decision_scope"]
    assert scope["global_optimum_proven"] is False and scope["solver_max_gap"] == 0.03
    assert scope["horizon_gameweeks"] == [8, 9] and scope["fallback_used"] is False
    assert scope["ignored_inputs"] == ["rival_context"]  # named, not silently dropped
    assert "UNCONFIRMED" in scope["assumptions"][0]


def test_milp_failure_is_labelled_as_beam_fallback(monkeypatch):
    from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
    from fpl_oracle.optimise.transfers_beam import TransferOptimizer
    from fpl_oracle.server.analysis import execute_authoritative_milp_plan

    def boom(**kw):
        raise RuntimeError("no plan")

    monkeypatch.setattr(multiperiod_optimizer, "evaluate_transfer_options", boom)
    monkeypatch.setattr(TransferOptimizer, "evaluate_joint_transfer_and_chip_plan", lambda self, **kw: {})
    res = execute_authoritative_milp_plan(current_squad_df=pd.DataFrame(), target_gw=8)
    assert res["fallback_used"] and res["decision_scope"]["global_optimum_proven"] is False
    assert res["decision_scope"]["fallback_reason"] == "no plan"


def test_live_planner_receives_every_chip_and_hit_input(monkeypatch, configured_advisor):
    """plan() builds cooldown / active chip / chip sets; none may be filtered out on the way to the MILP."""
    from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
    from fpl_oracle.optimise.progress import search_progress  # noqa: F401
    from fpl_oracle.server.analysis import analysis_service

    state, pool, horizon = configured_advisor
    state.active_chip = "bboost"
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(multiperiod_optimizer, "evaluate_transfer_options", spy)
    from fpl_oracle.optimise.transfers_beam import TransferOptimizer

    monkeypatch.setattr(TransferOptimizer, "evaluate_joint_transfer_and_chip_plan", lambda self, **kw: {})
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    asyncio.run(
        analysis_service.joint_plan(
            current_squad_df=pool,
            player_pool_df=pool,
            bank=state.bank_tenths,
            free_transfers=state.free_transfers,
            horizon_projections=horizon,
            target_gw=6,
            current_gw=5,
            available_chips=["wildcard"],
            chips_by_set={1: ["wildcard"], 2: []},
            chips_already_used=["freehit"],
        )
    )
    assert seen["forced_current_chip"] == "bboost"
    assert {"wildcard", "bboost"} <= set(seen["chips_available"])
    assert seen["chips_by_set"] == {1: ["wildcard"], 2: []} and seen["chips_already_used"] == ["freehit"]
    assert "hit_cooldown_until" in seen


def test_none_and_empty_locks_share_one_cached_plan(monkeypatch, configured_advisor):
    from fpl_oracle.server import analysis as analysis_module
    from fpl_oracle.server.analysis import analysis_service

    state, pool, horizon = configured_advisor
    calls = []
    monkeypatch.setattr(
        analysis_module,
        "execute_authoritative_milp_plan",
        lambda **kw: calls.append(kw) or {"recommended_plan": {}},
    )
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    base = dict(
        current_squad_df=pool,
        player_pool_df=pool,
        bank=state.bank_tenths,
        free_transfers=state.free_transfers,
        horizon_projections=horizon,
        target_gw=6,
        available_chips=[],
    )

    async def run():
        await analysis_service.joint_plan(**base, current_gw=5, locked_in_ids=None)
        await analysis_service.joint_plan(**base, current_gw=1)  # a different current_gw fallback

    asyncio.run(run())
    assert len(calls) == 1


def test_chip_planner_never_starts_in_the_locked_gameweek(monkeypatch):
    from fpl_oracle.chips.planner import chip_planner
    from fpl_oracle.server.analysis import analysis_service

    squad, pool, proj = scenario(gws=(8, 9))
    seen = {}
    monkeypatch.setattr(chip_planner, "generate_chip_strategy", lambda **kw: seen.update(kw) or {})
    asyncio.run(
        analysis_service.chip_strategy(
            current_gw=7, current_squad_df=squad, horizon_projections=proj, fixtures=[], bootstrap=None, bank=0.0
        )
    )
    assert seen["current_gw"] == 8


def test_router_passes_chip_limits_to_the_milp(monkeypatch):
    from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
    from fpl_oracle.optimise.transfers import transfer_optimizer

    squad, pool, proj = scenario()
    seen = {}
    monkeypatch.setattr(multiperiod_optimizer, "evaluate_transfer_options", lambda **kw: seen.update(kw) or {})
    transfer_optimizer.evaluate_transfer_options(
        squad, pool, 10, free_transfers=3, horizon_projections=proj, current_gw=7, target_gw=8, previous_chip="freehit"
    )
    assert seen["chips_available"] == [] and seen["previous_chip"] == "freehit"


def test_wildcard_and_free_hit_cannot_be_played_in_gw1():
    from fpl_oracle.rules.transfers import validate_chip_legality

    assert not validate_chip_legality("wildcard", 1)[0]
    assert not validate_chip_legality("freehit", 1)[0]
    assert validate_chip_legality("3xc", 1)[0] and validate_chip_legality("wildcard", 2)[0]


def test_target_gw_is_never_guessed():
    from fastapi import HTTPException

    from fpl_oracle.server.routes.api import _target_gw

    assert _target_gw(7, 8) == 8 and _target_gw(7, None) == 8
    with pytest.raises(HTTPException):
        _target_gw(None, None)
    with pytest.raises(HTTPException):
        _target_gw(38, None)


def test_cli_chips_renders_measured_roadmap_rows(monkeypatch):
    from fpl_oracle import cli
    from fpl_oracle.server.routes import api

    rows = [
        {"chip": "Wildcard", "code": "wildcard", "recommended_gw": None, "expected_gain": None, "status": "x"},
        {"chip": "Free Hit", "code": "freehit", "recommended_gw": 9, "expected_gain": 4.2, "status": "measured"},
    ]
    monkeypatch.setattr(api, "get_chip_strategy", AsyncMock(return_value={"chip_plan_table": rows}))
    cli.chips()


def test_postponed_fixture_team_is_not_counted_as_blank():
    from fpl_oracle.api.models import BootstrapStatic, Fixture, Team
    from fpl_oracle.chips.calendar import fixture_calendar

    boot = BootstrapStatic(teams=[Team(id=i, name=f"Team{i}", short_name=f"T{i}") for i in range(1, 6)])
    fixtures = [
        Fixture(id=1, code=1, event=5, team_h=1, team_a=2),
        Fixture(id=2, code=2, event=None, team_h=3, team_a=4),  # postponed, no date
    ]
    cal = fixture_calendar.analyze_calendar(fixtures, boot)
    gw5 = next(b for b in cal["blank_gameweeks"] if b["gameweek"] == 5)
    # Only the team with no fixture at all is a confirmed blank; the undated ones are uncertain
    assert {t["team_id"] for t in gw5["blanking_teams"]} == {5}
    assert set(gw5["uncertain_teams"]) == {3, 4}

    fixtures.append(Fixture(id=3, code=3, event=5, team_h=5, team_a=1))
    cal = fixture_calendar.analyze_calendar(fixtures, boot)
    assert not any(b["gameweek"] == 5 for b in cal["blank_gameweeks"])


def test_chip_simulator_does_not_invent_gains(monkeypatch):
    from fpl_oracle.chips import simulate
    from fpl_oracle.chips.simulate import chip_simulator

    squad, pool, _ = scenario()
    assert chip_simulator.evaluate_bench_boost(8, squad.head(10))["expected_gain"] is None
    monkeypatch.setattr(
        simulate.squad_optimizer, "solve_best_squad", lambda **kw: (_ for _ in ()).throw(ValueError("x"))
    )
    assert chip_simulator.evaluate_free_hit(8, squad, pool)["expected_gain"] is None
    assert chip_simulator.evaluate_wildcard(8, squad, pool)["expected_gain"] is None


def test_planner_bench_boost_and_triple_captain_follow_each_gameweek():
    from fpl_oracle.chips.planner import chip_planner

    squad, pool, proj = scenario(gws=(8, 9))
    proj[9] = proj[9].copy()
    proj[9]["expected_points"] = proj[9]["expected_points"] * 2
    for chip in ("3xc", "bboost"):
        g8 = chip_planner.compute_chip_utility(chip, 8, squad, proj, [], [], current_gw=7)[0]
        g9 = chip_planner.compute_chip_utility(chip, 9, squad, proj, [], [], current_gw=7)[0]
        assert g9 > g8


def test_squad_tab_values_the_squad_that_is_owned(configured_advisor, monkeypatch):
    from unittest.mock import patch

    from fpl_oracle.server.analysis import analysis_service
    from fpl_oracle.server.routes.api import get_squad

    state, _, _ = configured_advisor
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    with patch("fpl_oracle.news.ingest.news_ingestion.fetch_rss_articles", new=AsyncMock(return_value=[])):
        squad = asyncio.run(get_squad())
    owned_sell = round(sum(p.selling_price for p in state.squad) / 10.0, 1)
    assert squad["total_selling_value"] == owned_sell
    assert squad["total_team_value"] == round(owned_sell + state.bank_millions, 1)
