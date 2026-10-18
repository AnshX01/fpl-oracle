import pandas as pd


def test_free_hit_only_in_blank_double_or_last_week_of_set():
    from fpl_oracle.optimise.multiperiod import free_hit_week

    normal = pd.DataFrame({"team": range(20), "is_bgw": False, "is_dgw": False})
    assert not free_hit_week(normal, 10)
    assert free_hit_week(normal.assign(is_dgw=[True] + [False] * 19), 10)
    assert free_hit_week(normal.iloc[2:], 10)  # two clubs without a fixture
    assert free_hit_week(normal, 19) and free_hit_week(normal, 38)


def test_wildcard_now_still_allows_a_later_free_hit():
    from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
    from tests.test_planner_audit_fixes import scenario, with_double

    squad, pool, proj = scenario(squad_xp=1.0, pool_xp=4.0, gws=(8, 9, 10))
    proj = with_double(proj, 10)
    plan = MultiPeriodOptimizer(MultiPeriodConfig(time_limit_sec=30)).optimize(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=10,
        free_transfers=1,
        horizon_projections=proj,
        current_gw=7,
        target_gw=8,
        horizon_len=3,
        forced_w1_chip="wildcard",
        forced_chip_schedule={8: "wildcard", 10: "freehit"},
    )
    assert [(a.gameweek, a.active_chip) for a in plan.trajectory if a.active_chip] == [(8, "wildcard"), (10, "freehit")]
