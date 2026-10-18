from fpl_oracle.config import planning_horizon


def test_horizon_runs_to_chip_expiry_within_bounds():
    assert planning_horizon(8) == 12  # GW8-19: through first-set expiry
    assert planning_horizon(17) == 6  # never myopic just before GW19
    assert planning_horizon(20) == 12  # capped for solve time
    assert planning_horizon(30) == 9  # through GW38
    assert planning_horizon(38) == 1
