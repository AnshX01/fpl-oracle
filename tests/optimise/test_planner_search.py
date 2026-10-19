"""Planner search quality: chip spacing, chip-now margin, ruled-out players, local search and dominance."""

from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
from tests.test_planner_audit_fixes import evaluate, scenario, with_double


def chip_weeks(plan):
    return [a["gameweek"] for a in plan["trajectory"] if a.get("chip")]


def test_chips_are_never_back_to_back():
    # Weak squad, strong pool: every chip is worth playing, so an unspaced plan would stack them
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0, gws=(8, 9, 10, 11))
    res = evaluate(squad, pool, proj, 4, chips_available=["wildcard", "freehit", "bboost", "3xc"])
    for cand in res["candidate_plans"]:
        weeks = chip_weeks(cand)
        assert all(b - a >= 2 for a, b in zip(weeks, weeks[1:], strict=False)), weeks


def test_chip_now_must_beat_keeping_it_by_more_than_the_forecast_uncertainty():
    # Free Hit now vs next week: now is better only through discounting, a gain smaller than the uncertainty
    # of the 30 player-weeks it changes once each projection carries a 2-point estimation error (playing it now also
    # keeps the banked transfers that keeping it spends)
    squad, pool, proj = scenario(squad_xp=0.5, pool_xp=5.0, gws=(8, 9))
    proj = with_double(with_double(proj, 8), 9)  # a Free Hit is only planned in blank / double weeks
    held = evaluate(squad, pool, proj, 2, chips_available=["freehit"], xp_estimation_sd=2.0)
    assert held["recommended_chip"] is None
    assert held["recommended_plan"]["rejected_alternative"]["chip"] == "freehit"
    played = evaluate(squad, pool, proj, 2, chips_available=["freehit"], xp_estimation_sd=0.0)
    assert played["recommended_chip"] == "freehit"


def test_ruled_out_player_keeps_the_milp_and_sells_him():
    squad, pool, proj = scenario()
    out = int(squad.element.iloc[3])
    # Roll (no transfer) is infeasible here; the baseline is the fewest moves, not an abort to the beam search
    res = evaluate(squad, pool, proj, 2, chips_available=[], locked_out_ids={out})
    assert out in {t["element"] for t in res["recommended_plan"]["transfers_out"]}


def test_obvious_replacement_beats_rolling():
    # A zero-xP owned player and a strong affordable replacement: the plan must not roll
    squad, pool, proj = scenario(squad_xp=3.0, pool_xp=1.0)
    dead = int(squad[squad.position == "MID"].element.iloc[0])
    star = int(pool[(pool.position == "MID") & ~pool.element.isin(squad.element)].element.iloc[0])
    for frame in [pool, *proj.values()]:
        frame.loc[frame.element == dead, "expected_points"] = 0.0
        frame.loc[frame.element == star, ["expected_points", "value"]] = (8.0, 40)
    squad = pool[pool.element.isin(squad.element)].reset_index(drop=True)
    res = evaluate(squad, pool, proj, 3, chips_available=[])
    rec = res["recommended_plan"]
    assert star in {t["element"] for t in rec["transfers_in"]}
    assert dead in {t["element"] for t in rec["transfers_out"]}
    # Dominance: nothing in the candidate set beats the recommendation without failing a gate
    best_free = max(c["total_objective"] for c in res["candidate_plans"] if c["hits"] == 0 and not c["chip"])
    assert rec["total_objective"] >= best_free - 1e-6


def test_polish_never_makes_a_plan_worse():
    squad, pool, proj = scenario(gws=(8, 9, 10, 11, 12))
    opt = MultiPeriodOptimizer(MultiPeriodConfig(time_limit_sec=5, polish_window=2))
    common = dict(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=10,
        free_transfers=1,
        horizon_projections=proj,
        current_gw=7,
        target_gw=8,
        horizon_len=5,
        chips_by_set={1: [], 2: []},
    )
    start = opt.optimize(**common, time_limit_sec=2)

    def run(name, extra, lim):
        return opt.optimize(**common, **extra, time_limit_sec=lim)

    polished = opt._polish(run, {}, start, "opt", None)
    assert polished.total_objective >= start.total_objective - 1e-6
