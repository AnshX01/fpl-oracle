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
