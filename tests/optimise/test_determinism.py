"""Identical inputs give identical advice: canonical ties, a fixed solver set-up and one gain baseline."""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.optimise import multiperiod
from fpl_oracle.optimise.lineup import LineupOptimizer, best_lineup
from fpl_oracle.optimise.multiperiod import MilpNoSolution, MultiPeriodConfig, MultiPeriodOptimizer
from tests.test_planner_audit_fixes import evaluate, scenario


def flat_squad(**overrides):
    """15 players that all score the same, so every choice is a tie."""
    rows = [("GKP", 1), ("GKP", 2)] + [("DEF", i) for i in range(3, 8)] + [("MID", i) for i in range(8, 13)]
    rows += [("FWD", i) for i in range(13, 16)]
    df = pd.DataFrame(rows, columns=["position", "element"])
    df["expected_points"] = 4.0
    df["web_name"] = df["element"].map(lambda e: f"P{e}")
    return df.assign(**overrides)


def test_lineup_ties_go_to_the_lowest_ids_in_any_row_order():
    squad = flat_squad()
    first = LineupOptimizer().select_lineup_and_captain(squad, simulate_captaincy=False)
    shuffled = squad.sample(frac=1.0, random_state=5).reset_index(drop=True)
    second = LineupOptimizer().select_lineup_and_captain(shuffled, simulate_captaincy=False)
    assert sorted(first["starters"]["element"]) == sorted(second["starters"]["element"])
    assert first["captain"]["element"] == second["captain"]["element"] == 1
    assert first["vice_captain"]["element"] == second["vice_captain"]["element"] == 3
    assert first["formation"] == second["formation"]
    assert list(first["bench"]["element"]) == list(second["bench"]["element"])


def test_lineup_honours_unavailable_players_and_chip_multipliers():
    squad = flat_squad()
    squad.loc[squad.element == 3, "expected_points"] = 9.0  # the captain, unless he cannot start
    res = LineupOptimizer().select_lineup_and_captain(squad, excluded_ids={3}, simulate_captaincy=False)
    assert 3 not in set(res["starters"]["element"]) and res["captain"]["element"] != 3
    res = LineupOptimizer().select_lineup_and_captain(squad, is_triple_captain=True, simulate_captaincy=False)
    assert res["captain"]["multiplier"] == 3 and res["captain"]["element"] == 3
    assert res["total_gameweek_expected_points"] == pytest.approx(9.0 + 10 * 4.0 + 2 * 9.0)
    res = LineupOptimizer().select_lineup_and_captain(squad, is_bench_boost=True, simulate_captaincy=False)
    assert res["total_gameweek_expected_points"] == pytest.approx(9.0 + 14 * 4.0 + 9.0)


def test_best_lineup_breaks_a_captain_tilt_by_score_then_xp():
    squad = flat_squad()
    position = dict(zip(squad.element, squad.position, strict=True))
    xp = dict(zip(squad.element, squad.expected_points, strict=True))
    _, _, captain, _ = best_lineup(list(squad.element), position, xp, 0.0, cap_score={8: 6.0})
    assert captain == 8


def test_highs_runs_single_threaded_with_a_fixed_seed():
    solver = MultiPeriodOptimizer()._get_solver()
    assert solver.threads == 1 and solver.optionsDict["random_seed"] == 0
    assert solver.gapRel == 1e-4 and solver.timeLimit == 3600


def test_missing_highs_is_an_error_not_a_silent_switch(monkeypatch):
    monkeypatch.setattr(multiperiod._WarmStartHiGHS, "available", lambda self: False)
    with pytest.raises(RuntimeError, match="HiGHS"):
        MultiPeriodOptimizer()._get_solver()
    assert MultiPeriodOptimizer(MultiPeriodConfig(solver_name="cbc"))._get_solver().name == "PULP_CBC_CMD"


def test_free_hit_squad_that_cannot_be_solved_raises():
    pool = scenario()[1]
    xp = dict(zip(pool.element, pool.expected_points, strict=True))
    costs = dict(zip(pool.element, pool.value, strict=True))
    with pytest.raises(MilpNoSolution):
        MultiPeriodOptimizer()._free_hit_squad(pool, xp, costs, 0.0, set(), set(), set())


def test_candidate_table_is_on_one_roll_baseline_and_progress_is_reported(monkeypatch):
    seen = []
    monkeypatch.setattr(multiperiod, "report_search_progress", seen.append)
    squad, pool, proj = scenario(squad_xp=2.0, pool_xp=5.0)
    res = evaluate(squad, pool, proj, 2, chips_available=["wildcard"])
    roll = res["roll_plan"]
    assert len(res["candidate_plans"]) > 2
    for cand in res["candidate_plans"] + [res["recommended_plan"]]:
        assert cand["horizon_gain_vs_roll"] == round(cand["horizon_net_xp"] - roll["horizon_net_xp"], 2)
        assert cand["expected_gain"] == round(cand["trajectory"][0]["net_xp"] - roll["trajectory"][0]["net_xp"], 2)
        assert "objective_gain_vs_roll" not in cand and "solver_status" not in cand
    wave = [m for m in seen if m.startswith("Comparing transfers and chips from Gameweek 8: ")]
    assert wave and wave[-1].endswith(f"{len(wave)} of {len(wave)} searches finished")


def test_same_inputs_give_the_same_recommendation():
    squad, pool, proj = scenario(squad_xp=2.0, pool_xp=3.0)

    def run():
        res = evaluate(squad.sample(frac=1.0, random_state=1), pool, proj, 2, chips_available=["wildcard"])
        plan = res["recommended_plan"]
        return plan["total_objective"], plan["captain_id"], sorted(t["element"] for t in plan["transfers_in"])

    assert run() == run()
    assert np.isfinite(run()[0])
