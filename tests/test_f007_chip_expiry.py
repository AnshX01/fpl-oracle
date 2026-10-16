import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, GameweekEvent
from fpl_oracle.chips.planner import ChipPlanner


@pytest.fixture
def mock_squad():
    squad_rows = (
        [
            {"element": i, "web_name": f"GKP_{i}", "position": "GKP", "expected_points": 3.0, "is_starter": i == 1}
            for i in range(1, 3)
        ]
        + [
            {"element": i, "web_name": f"DEF_{i}", "position": "DEF", "expected_points": 3.0, "is_starter": i <= 5}
            for i in range(3, 8)
        ]
        + [
            {"element": i, "web_name": f"MID_{i}", "position": "MID", "expected_points": 3.5, "is_starter": i <= 12}
            for i in range(8, 13)
        ]
        + [
            {"element": i, "web_name": f"FWD_{i}", "position": "FWD", "expected_points": 3.0, "is_starter": i <= 14}
            for i in range(13, 16)
        ]
    )
    return pd.DataFrame(squad_rows)


@pytest.fixture
def bootstrap_with_events():
    events = [
        GameweekEvent(
            id=19,
            name="Gameweek 19",
            deadline_time="2027-01-02T11:00:00Z",
            is_next=True,
        )
    ]
    return BootstrapStatic(
        events=events,
        game_settings={},
        phases=[],
        teams=[],
        total_players=0,
        elements=[],
        element_types=[],
    )


def test_gw19_forces_chip_usage_on_sub_threshold_positive_gain(mock_squad, bootstrap_with_events, monkeypatch):
    """
    F-007: At GW19 (final GW of Set 1), an unused chip with any positive expected gain
    (even below static threshold, e.g. 2.5 xP vs 4.0 threshold) must be recommended.
    """
    planner = ChipPlanner()

    # Monkeypatch compute_chip_utility to return sub-threshold positive gain for 3xc
    monkeypatch.setattr(
        planner,
        "compute_chip_utility",
        lambda chip, gw, *args, **kwargs: (2.5, "MEDIUM", 40.0, 42.5),
    )
    monkeypatch.setattr(
        planner,
        "get_remaining_chips",
        lambda *args, **kwargs: {"set_1_remaining": ["3xc"], "set_2_remaining": []},
    )

    res = planner.generate_chip_strategy(
        current_gw=19,
        current_squad_df=mock_squad,
        horizon_projections={19: mock_squad},
        fixtures=[],
        bootstrap=bootstrap_with_events,
        manager_history=None,
    )

    assert "set_1_warning" in res
    assert res["set_1_warning"] is not None
    assert res["recommend_chip_this_gw"] is True, (
        "Sub-threshold positive gain chip was NOT recommended at GW19 deadline!"
    )
    tc_plan = next(c for c in res["chip_plan_table"] if c["code"] == "3xc")
    assert tc_plan["is_recommended_this_gw"] is True
    assert tc_plan["recommendation_threshold"] == 0.0


def test_gw18_lowers_threshold_by_half(mock_squad, bootstrap_with_events, monkeypatch):
    """
    F-007: At GW18 (one gameweek before deadline), threshold must be explicitly lowered by 50%.
    """
    planner = ChipPlanner()

    monkeypatch.setattr(
        planner,
        "compute_chip_utility",
        lambda chip, gw, *args, **kwargs: (3.0, "MEDIUM", 40.0, 43.0),
    )
    monkeypatch.setattr(
        planner,
        "get_remaining_chips",
        lambda *args, **kwargs: {"set_1_remaining": ["3xc"], "set_2_remaining": []},
    )

    res = planner.generate_chip_strategy(
        current_gw=18,
        current_squad_df=mock_squad,
        horizon_projections={18: mock_squad, 19: mock_squad},
        fixtures=[],
        bootstrap=bootstrap_with_events,
        manager_history=None,
    )

    tc_plan = next(c for c in res["chip_plan_table"] if c["code"] == "3xc")
    # Base threshold for 3xc is 4.0; at GW18 it must be 2.0
    assert tc_plan["recommendation_threshold"] == 2.0


def test_set_1_warning_non_null_from_gw17_and_uses_api_deadline(mock_squad, bootstrap_with_events, monkeypatch):
    """
    F-007: set_1_warning must be non-null from GW17 onwards if Set 1 chips remain,
    and must format the real API deadline rather than a hardcoded constant.
    """
    planner = ChipPlanner()

    monkeypatch.setattr(
        planner,
        "get_remaining_chips",
        lambda *args, **kwargs: {"set_1_remaining": ["3xc"], "set_2_remaining": []},
    )

    # GW17, GW18, GW19: must be non-null and format the API deadline
    for gw in [17, 18, 19]:
        res = planner.generate_chip_strategy(
            current_gw=gw,
            current_squad_df=mock_squad,
            horizon_projections={gw: mock_squad},
            fixtures=[],
            bootstrap=bootstrap_with_events,
        )
        assert res.get("set_1_warning") is not None
        assert "2027-01-02T11:00:00Z" in res["set_1_warning"]
        assert "set_1_deadline_warning" in res
        assert res["set_1_deadline_warning"] == res["set_1_warning"]


def test_four_unused_set1_chips_at_gw16_to_gw19_schedule_count(mock_squad, bootstrap_with_events, monkeypatch):
    """
    F-007: With 4 unused Set 1 chips:
    - At GW16 (4 GWs left: 16, 17, 18, 19): schedules all 4 chips before expiry (1 per GW).
    - At GW17 (3 GWs left: 17, 18, 19): schedules 3 chips (1 must expire).
    - At GW18 (2 GWs left: 18, 19): schedules 2 chips (2 must expire).
    - At GW19 (1 GW left: 19): schedules 1 chip (3 must expire).
    """
    planner = ChipPlanner()
    all_four_chips = ["wildcard", "freehit", "3xc", "bboost"]

    monkeypatch.setattr(
        planner,
        "get_remaining_chips",
        lambda *args, **kwargs: {"set_1_remaining": list(all_four_chips), "set_2_remaining": []},
    )

    # GW16: 4 available GWs -> schedules all 4
    projs_gw16 = {gw: mock_squad for gw in range(16, 20)}
    res16 = planner.generate_chip_strategy(
        current_gw=16,
        current_squad_df=mock_squad,
        horizon_projections=projs_gw16,
        fixtures=[],
        bootstrap=bootstrap_with_events,
    )
    scheduled_16 = [c for c in res16["chip_plan_table"] if c["recommended_gw"] in range(16, 20)]
    assert len(scheduled_16) == 4, f"Expected 4 chips scheduled at GW16, got {len(scheduled_16)}"
    assert res16["set_1_warning"] is not None
    assert "recommended:" in res16["set_1_warning"]
    assert "latest feasible:" in res16["set_1_warning"]

    # GW17: 3 available GWs -> schedules 3 chips
    projs_gw17 = {gw: mock_squad for gw in range(17, 20)}
    res17 = planner.generate_chip_strategy(
        current_gw=17,
        current_squad_df=mock_squad,
        horizon_projections=projs_gw17,
        fixtures=[],
        bootstrap=bootstrap_with_events,
    )
    scheduled_17 = [c for c in res17["chip_plan_table"] if c["recommended_gw"] in range(17, 20)]
    assert len(scheduled_17) == 3, f"Expected 3 chips scheduled at GW17, got {len(scheduled_17)}"
    assert "CRITICAL CHIP CONGESTION" in res17["set_1_warning"]

    # GW18: 2 available GWs -> schedules 2 chips
    projs_gw18 = {gw: mock_squad for gw in range(18, 20)}
    res18 = planner.generate_chip_strategy(
        current_gw=18,
        current_squad_df=mock_squad,
        horizon_projections=projs_gw18,
        fixtures=[],
        bootstrap=bootstrap_with_events,
    )
    scheduled_18 = [c for c in res18["chip_plan_table"] if c["recommended_gw"] in range(18, 20)]
    assert len(scheduled_18) == 2, f"Expected 2 chips scheduled at GW18, got {len(scheduled_18)}"

    # GW19: 1 available GW -> schedules 1 chip
    projs_gw19 = {19: mock_squad}
    res19 = planner.generate_chip_strategy(
        current_gw=19,
        current_squad_df=mock_squad,
        horizon_projections=projs_gw19,
        fixtures=[],
        bootstrap=bootstrap_with_events,
    )
    scheduled_19 = [c for c in res19["chip_plan_table"] if c["recommended_gw"] == 19]
    assert len(scheduled_19) == 1, f"Expected 1 chip scheduled at GW19, got {len(scheduled_19)}"


def test_set_1_warning_fires_from_gw14(mock_squad, bootstrap_with_events, monkeypatch):
    """
    F-007: set_1_warning must fire from GW14 onwards when Set 1 chips remain.
    """
    planner = ChipPlanner()
    monkeypatch.setattr(
        planner,
        "get_remaining_chips",
        lambda *args, **kwargs: {"set_1_remaining": ["3xc"], "set_2_remaining": []},
    )

    projs = {gw: mock_squad for gw in range(14, 20)}
    res14 = planner.generate_chip_strategy(
        current_gw=14,
        current_squad_df=mock_squad,
        horizon_projections=projs,
        fixtures=[],
        bootstrap=bootstrap_with_events,
    )
    assert res14.get("set_1_warning") is not None, "set_1_warning should fire from GW14 onwards"
    assert "Triple Captain" in res14["set_1_warning"]
