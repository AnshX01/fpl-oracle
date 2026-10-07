from fpl_oracle.league.scenarios import compare_plan_scenarios, evolve_rival


def test_evolution_legal_changes_are_hypothetical():
    import pandas as pd

    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    rows = [
        dict(
            element=i + 1,
            position=p,
            team=i % 5 + 1,
            value=50,
            expected_points=3.0,
            p10=1.0,
            p90=6.0,
            web_name=str(i + 1),
        )
        for i, p in enumerate(positions)
    ]
    squad = pd.DataFrame(rows)
    pool = pd.DataFrame(
        rows
        + [dict(element=16, position="MID", team=6, value=45, expected_points=7.0, p10=2.0, p90=12.0, web_name="New")]
    )
    projections = {6: pool, 7: pool, 8: pool}
    maps = {g: {int(r["element"]): r for r in df.to_dict("records")} for g, df in projections.items()}
    rival = dict(elements=list(squad.element), points=100)
    path = evolve_rival(rival, maps, "one_transfer_per_week")
    previous = set(rival["elements"])
    for row in path:
        current = set(row["elements"])
        assert len(current) == 15
        assert len(current - previous) <= 1
        positions = [maps[row["gameweek"]][e]["position"] for e in current]
        assert [positions.count(p) for p in ("GKP", "DEF", "MID", "FWD")] == [2, 5, 5, 3]
        previous = current
    state = dict(first_chip=None, history=path)
    result = compare_plan_scenarios([state], dict(user_points=100, rivals=[rival]), maps, samples=64)
    assert result["calibrated"] is False
    assert result["championship_probability"] is False
    assert [r["mode"] for r in result["scenarios"]] == ["hold_roster", "one_transfer_per_week", "restructure_at_start"]
    assert result == compare_plan_scenarios([state], dict(user_points=100, rivals=[rival]), maps, samples=64)
