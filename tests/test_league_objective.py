"""Win probability: symmetric rivals, shared players cancel, variance helps a chaser and hurts a leader."""

from fpl_oracle.league.objective import win_probability

GWS = [8]
POS = {i: "MID" for i in range(1, 40)}


def weeks(players):
    return [{p: 1.0 for p in players}]


def ctx(user, rival, rival_players):
    return {"user_points": user, "rivals": [{"points": rival, "elements": rival_players}]}


def p_win(mine, user, rival, rival_players, xp, sd):
    # held squad = plan, best plan = plan: only today's gap, squad quality and overlap matter
    out = win_probability(
        weeks(mine),
        weeks(mine),
        sum(xp[GWS[0]][p] for p in mine),
        GWS,
        ctx(user, rival, rival_players),
        xp,
        sd,
        POS,
        {0: weeks(rival_players)},
    )
    return out["p_win"]


def test_identical_squads_leave_only_todays_gap():
    xp = {8: {1: 5.0, 2: 5.0}}
    sd = {8: {1: 3.0, 2: 3.0}}
    assert p_win([1, 2], 50, 40, [1, 2], xp, sd) == 1.0  # same players, ahead: nothing can change it
    assert p_win([1, 2], 40, 50, [1, 2], xp, sd) == 0.0


def test_differential_helps_when_behind_and_hurts_when_ahead():
    xp = {8: {1: 5.0, 2: 5.0}}
    sd = {8: {1: 3.0, 2: 3.0}}
    chasing = p_win([2], 45, 50, [1], xp, sd)
    leading = p_win([2], 55, 50, [1], xp, sd)
    assert 0.0 < chasing < 0.5 < leading < 1.0
