import numpy as np
import pandas as pd

from fpl_oracle.optimise.information import availability_path, scenario_projections


def test_availability_matches_the_planners_chance_of_playing():
    rng = np.random.default_rng(0)
    for p in (0.3, 0.5, 0.8, 0.95):
        share = np.mean([availability_path(p, 3, rng).mean() for _ in range(6000)])
        assert abs(share - p) < 0.01, (p, share)


def test_scenarios_keep_the_forecast_on_average_and_leave_week_one():
    frame = pd.DataFrame({"element": [1, 2, 3], "expected_points": [6.0, 2.0, 0.4], "p_play": [0.9, 0.6, 0.1]})
    proj = {8: frame, 9: frame, 10: frame}
    rng = np.random.default_rng(1)
    draws = [scenario_projections(proj, 8, rng) for _ in range(3000)]
    assert all(d[8] is frame for d in draws)
    mean9 = np.mean([d[9]["expected_points"].to_numpy() for d in draws], axis=0)
    assert np.allclose(mean9, [6.0, 2.0, 0.4], rtol=0.05)
    assert set(np.round(draws[0][9]["expected_points"], 3)) <= {0.0, 0.4, round(6 / 0.9, 3), round(2 / 0.6, 3)}


def test_free_hit_only_in_blank_double_or_last_week_of_set():
    from fpl_oracle.optimise.multiperiod import free_hit_week

    normal = pd.DataFrame({"team": range(20), "is_bgw": False, "is_dgw": False})
    assert not free_hit_week(normal, 10)
    assert free_hit_week(normal.assign(is_dgw=[True] + [False] * 19), 10)
    assert free_hit_week(normal.iloc[2:], 10)  # two clubs without a fixture
    assert free_hit_week(normal, 19) and free_hit_week(normal, 38)


def test_card_says_the_chip_gain_allows_for_what_waiting_teaches():
    from fpl_oracle.briefing.decision_card import _comparison_sentences

    plan = {"decision_check": {"gain": 3.4, "versus": "keeping the chip", "information": {"mean": 3.4}}}
    chip_text, _ = _comparison_sentences(plan, "Wildcard")
    assert chip_text.endswith("allowing for the injuries you would see before a later Wildcard.")
