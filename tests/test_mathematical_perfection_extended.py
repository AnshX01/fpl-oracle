import numpy as np
import pandas as pd

from fpl_oracle.ml.components.congestion import turnaround_congestion_engine
from fpl_oracle.ml.ensemble import ScoringEnsemble
from fpl_oracle.news.reconcile import _return_gameweek, official_chance_of_playing
from fpl_oracle.optimise.turnaround import turnaround_drag_service


def test_rested_player_zero_fatigue_discount():
    # Haaland (element 411) rested in GW9 Carabao Cup
    df = pd.DataFrame(
        [
            {
                "element": 411,
                "web_name": "Haaland",
                "position": "FWD",
                "team": 15,
                "team_short": "MCI",
                "expected_points": 6.49,
                "expected_minutes": 88.0,
                "p_starts": 0.98,
                "kickoff_time": "2026-10-31T15:00:00Z",
            }
        ]
    )

    # 1. Congestion engine projection pass
    projs = turnaround_congestion_engine.apply_congestion_to_projections(df, gameweek=9)
    assert projs.loc[0, "congestion_fatigue_factor"] == 1.0
    assert projs.loc[0, "expected_points"] == 6.49

    # 2. Lineup optimizer turnaround drag pass
    lineup_df = turnaround_drag_service.apply_turnaround_drag(projs)
    assert lineup_df.loc[0, "expected_points"] == 6.49
    assert not lineup_df.loc[0, "midweek_fatigue_risk"]


def test_no_double_counting_for_active_midweek_player():
    df = pd.DataFrame(
        [
            {
                "element": 999,
                "web_name": "ActivePlayer",
                "position": "MID",
                "team": 15,
                "team_short": "MCI",
                "expected_points": 6.00,
                "expected_minutes": 90.0,
                "p_starts": 0.90,
                "midweek_minutes": 90.0,
                "kickoff_time": "2026-10-31T15:00:00Z",
            }
        ]
    )

    # 1. Congestion engine pass
    projs = turnaround_congestion_engine.apply_congestion_to_projections(df, gameweek=9)
    m1 = projs.loc[0, "congestion_fatigue_factor"]
    xp1 = projs.loc[0, "expected_points"]
    assert m1 < 1.0

    # 2. Lineup optimizer pass - must NOT double discount!
    lineup_df = turnaround_drag_service.apply_turnaround_drag(projs)
    assert lineup_df.loc[0, "expected_points"] == xp1


def test_return_gameweek_year_rollover_2027():
    class MockElement:
        news = "Expected back 15 Feb"
        status = "i"

    class MockEvent:
        def __init__(self, eid, deadline):
            self.id = eid
            self.deadline_time = deadline

    # Gameweek deadlines in 2027 (second half of 2026/27 season)
    events = [
        MockEvent(24, "2027-02-06T11:00:00Z"),
        MockEvent(25, "2027-02-13T11:00:00Z"),
        MockEvent(26, "2027-02-20T11:00:00Z"),
    ]

    due_gw = _return_gameweek(MockElement(), events)
    assert due_gw == 26  # Feb 15 deadline is before Feb 20 (GW26)


def test_suspended_player_no_injury_ramp():
    class MockElement:
        news = "Suspended until 24 Oct"
        status = "s"

    class MockEvent:
        def __init__(self, eid, deadline):
            self.id = eid
            self.deadline_time = deadline

    events = [MockEvent(8, "2026-10-24T11:00:00Z"), MockEvent(9, "2026-10-31T11:00:00Z")]
    elem = MockElement()

    # Target GW9 is after suspension deadline -> must be 100%, not ramped at 50%
    cop = official_chance_of_playing(elem, target_gw=9, first_gw=8, events=events)
    assert cop == 100.0


def test_ensemble_minutes_limit_algebra():
    model = ScoringEnsemble()
    X = pd.DataFrame(
        [
            {
                "chance_of_playing": 50.0,
                "news_minutes_limit": 30.0,
                "news_start_probability": 0.05,
            }
        ]
    )

    # With 90 base minutes, limit 30, and 50% doubt:
    # limit ratio = 30 / 90 = 0.3333
    # exp_mins = 90 * 0.3333 * 0.50 = 15.0 (NOT 30.0!)
    components = {
        "expected_minutes": np.array([90.0]),
        "p_starts": np.array([0.90]),
        "p_min60": np.array([0.80]),
        "expected_goals": np.array([0.40]),
        "expected_assists": np.array([0.20]),
        "p_clean_sheet": np.array([0.30]),
        "expected_goals_conceded": np.array([1.20]),
        "expected_saves": np.array([0.0]),
        "expected_bonus": np.array([0.50]),
        "expected_card_deduction": np.array([0.10]),
        "expected_penalties_saved": np.array([0.0]),
        "expected_own_goals": np.array([0.0]),
        "expected_penalties_missed": np.array([0.0]),
        "p_defcon": np.array([0.0]),
        "position": ["FWD"],
    }

    res = model.aggregate_components(components, X)
    assert round(float(res["expected_minutes"][0]), 1) == 15.0
    assert float(res["p_starts"][0]) <= 0.05
