from fpl_oracle.league.objective import league_threats, rival_weeks

POS = {i: "GKP" for i in (1, 2)} | {i: "DEF" for i in range(3, 8)} | {i: "MID" for i in range(8, 13)}
POS |= {i: "FWD" for i in (13, 14, 15, 99)}
SQUAD = list(range(1, 15)) + [99]


def xp(boost: int) -> dict[int, float]:
    return {p: (9.0 if p == boost else 2.0 + p / 100) for p in POS}


def test_rival_keeps_last_captain_in_first_week_only():
    weeks = rival_weeks(SQUAD, [8, 9], {8: xp(13), 9: xp(13)}, POS, last_captain=99)
    assert weeks[0][99] == 2.0 and weeks[0][13] == 1.0
    assert weeks[1][13] == 2.0


def test_threats_rank_rival_players_we_do_not_own():
    rivals = [dict(elements=SQUAD, captain=99)]
    cache: dict = {}
    top = league_threats(rivals, [8], {8: xp(13)}, POS, mine=set(range(1, 14)), cache=cache, top=2)
    assert top == [99, 14]
    assert 0 in cache


def test_win_probability_counts_shared_risk_once():
    from fpl_oracle.league.objective import win_probability

    xp_map = {8: xp(13)}
    sd = {8: {p: 3.0 for p in POS}}
    mine = rival_weeks(list(range(1, 15)) + [99], [8], xp_map, POS)
    other = [dict(points=105, elements=[*range(1, 13), 14, 15, 99])]

    def p(rivals):
        context = dict(user_points=100, rivals=rivals)
        return win_probability(mine, mine, 0.0, [8], context, xp_map, sd, POS)

    one, two = p(other), p(other * 2)
    assert abs(one["p_win"] - one["rivals"][0]["p_above"]) < 0.01
    assert abs(two["p_win"] - one["p_win"]) < 0.01  # not one squared


def test_captain_on_a_near_tie_follows_the_league():
    from tests.test_planner_audit_fixes import evaluate, scenario

    squad, pool, proj = scenario(squad_xp=2.0, pool_xp=0.5, gws=(8,))
    a, b = (int(e) for e in squad[squad.position != "GKP"].element.iloc[:2])
    for frame in [pool, *proj.values()]:
        frame.loc[frame.element == a, ["expected_points", "p10", "p90"]] = (5.0, 1.0, 11.0)
        frame.loc[frame.element == b, ["expected_points", "p10", "p90"]] = (4.9, 1.0, 11.0)
    squad = pool[pool.element.isin(squad.element)].reset_index(drop=True)
    twin = [dict(points=130, elements=[int(e) for e in squad.element], captain=a)]

    def captain(user_points):
        res = evaluate(
            squad,
            pool,
            proj,
            1,
            chips_available=[],
            xp_estimation_sd=1.0,
            rival_context=dict(user_points=user_points, rivals=twin),
        )
        return int(res["recommended_plan"]["trajectory"][0]["captain"]), res["recommended_plan"]

    leading, led = captain(160)  # 30 ahead: matching his captain locks the lead
    assert leading == a and led["captain_check"]["switched"] is None
    chasing, plan = captain(100)  # 30 behind a rival with the same team: copying his captain cannot catch him
    assert (
        chasing == b and plan["captain_check"]["switched"]["instead_of"] == plan["lineup"]["vice_captain"]["web_name"]
    )
    # the whole plan agrees with the switch: captain, vice, win chance of the pair and the points given up
    assert plan["captain"] is plan["lineup"]["captain"] and plan["captain_id"] == plan["captain"]["element"] == b
    assert plan["vice_captain"] is plan["lineup"]["vice_captain"] and plan["vice_captain_id"] == a
    assert plan["vice_captain"]["reasoning"].startswith(f"Activates if {plan['captain']['web_name']} does not")
    assert plan["captain"]["prob_outscore_next"] < 0.5  # the new captain is the lower-xP one
    assert plan["lineup"]["captain_rankings"][0]["element"] == b
    given_up = plan["captain_check"]["switched"]["points_given_up"]
    assert given_up > 0 and plan["net_expected_points"] == plan["trajectory"][0]["net_xp"]
    tie = plan["league_tiebreak"]  # a different plan than the roll, then the armband switch on top of it
    assert plan["total_objective"] == round(led["total_objective"] - tie["points_given_up"] - given_up, 2)
    assert tie["p_win_after"] == plan["league"]["p_win"] == plan["captain_check"]["switched"]["p_win_after"]
    assert f"to {tie['p_win_after']:.0%}." in plan["recommendation_summary"]
