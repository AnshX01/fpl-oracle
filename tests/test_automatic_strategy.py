from fpl_oracle.league.strategy import automatic_strategy


def test_gaps_include_leader_and_both_neighbours():
    result = automatic_strategy(
        dict(
            user_rank=3,
            user_points=80,
            standings=[
                dict(rank=1, points=120),
                dict(rank=2, points=85),
                dict(rank=3, points=80),
                dict(rank=4, points=79),
            ],
        )
    )
    assert result["gap_to_leader"] == 40
    assert result["gap_above"] == 5
    assert result["gap_below"] == 1
    assert result["status"] == "automatic"


def test_leader_and_unknown():
    assert automatic_strategy(None)["status"] == "unavailable"
    assert (
        automatic_strategy(
            dict(user_rank=1, user_points=90, standings=[dict(rank=1, points=90), dict(rank=2, points=80)])
        )["mode_title"]
        == "Protect your lead"
    )
