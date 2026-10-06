"""
Comprehensive test suite for Phase 4: Expert Behaviour & Contingency Engine.
Verifies:
1. Plan A, Plan B, Plan C precomputation with explicit triggers and xP deltas.
2. Injury & Rotation Contingency Matrix (Auto-sub vs Emergency Transfer vs Trust Bench).
3. Panic Button re-optimization under sudden crisis news.
4. Pre-deadline operational checklist generator.
"""

import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Element, GameweekEvent, Team
from fpl_oracle.optimise.contingency import contingency_engine


@pytest.fixture
def dummy_bootstrap() -> BootstrapStatic:
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS"),
        Team(id=2, name="Aston Villa", short_name="AVL"),
        Team(id=3, name="Liverpool", short_name="LIV"),
        Team(id=4, name="Man City", short_name="MCI"),
    ]
    elements = []
    # 2 GKPs
    elements.append(Element(id=1, web_name="Raya", element_type=1, team=1, now_cost=55, status="a"))
    elements.append(Element(id=2, web_name="Alisson", element_type=1, team=3, now_cost=55, status="a"))
    # 5 DEFs
    elements.append(Element(id=3, web_name="Gabriel", element_type=2, team=1, now_cost=60, status="a"))
    elements.append(Element(id=4, web_name="Saliba", element_type=2, team=1, now_cost=60, status="a"))
    elements.append(Element(id=5, web_name="Van Dijk", element_type=2, team=3, now_cost=60, status="a"))
    elements.append(Element(id=6, web_name="Konsa", element_type=2, team=2, now_cost=45, status="a"))
    elements.append(
        Element(
            id=7,
            web_name="Lewis",
            element_type=2,
            team=4,
            now_cost=48,
            status="d",
            chance_of_playing_next_round=50,
            news="Knock",
        )
    )
    # 5 MIDs
    elements.append(Element(id=8, web_name="Saka", element_type=3, team=1, now_cost=100, status="a"))
    elements.append(Element(id=9, web_name="Salah", element_type=3, team=3, now_cost=125, status="a"))
    elements.append(Element(id=10, web_name="Foden", element_type=3, team=4, now_cost=95, status="a"))
    elements.append(Element(id=11, web_name="Rogers", element_type=3, team=2, now_cost=52, status="a"))
    elements.append(Element(id=12, web_name="Diaz", element_type=3, team=3, now_cost=75, status="a"))
    # 3 FWDs
    elements.append(Element(id=13, web_name="Haaland", element_type=4, team=4, now_cost=150, status="a"))
    elements.append(Element(id=14, web_name="Watkins", element_type=4, team=2, now_cost=90, status="a"))
    elements.append(Element(id=15, web_name="Havertz", element_type=4, team=1, now_cost=80, status="a"))
    # Additional market replacement players
    elements.append(Element(id=16, web_name="Mbeumo", element_type=3, team=2, now_cost=72, status="a"))
    elements.append(Element(id=17, web_name="Wood", element_type=4, team=2, now_cost=62, status="a"))

    events = [
        GameweekEvent(id=6, name="Gameweek 6", deadline_time="2026-10-10T10:00:00Z", is_current=False, is_next=True)
    ]
    return BootstrapStatic(events=events, teams=teams, elements=elements)


@pytest.fixture
def mock_squad_and_pool(dummy_bootstrap):
    elem_dict = {e.id: e for e in dummy_bootstrap.elements}
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    xp_map = {
        1: 4.5,
        2: 4.0,
        3: 4.8,
        4: 4.6,
        5: 4.4,
        6: 3.5,
        7: 2.5,
        8: 7.5,
        9: 8.5,
        10: 6.5,
        11: 4.5,
        12: 5.5,
        13: 9.5,
        14: 6.8,
        15: 5.8,
        16: 6.2,
        17: 5.0,
    }

    squad_rows = []
    for elem_id in range(1, 16):
        e = elem_dict[elem_id]
        squad_rows.append(
            {
                "element": e.id,
                "web_name": e.web_name,
                "team": e.team,
                "position": pos_map[e.element_type],
                "value": e.now_cost,
                "purchase_price": e.now_cost,
                "selling_price": e.now_cost,
                "expected_points": xp_map[e.id],
                "p10": xp_map[e.id] * 0.4,
                "p90": xp_map[e.id] * 1.6,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    pool_rows = list(squad_rows)
    for elem_id in [16, 17]:
        e = elem_dict[elem_id]
        pool_rows.append(
            {
                "element": e.id,
                "web_name": e.web_name,
                "team": e.team,
                "position": pos_map[e.element_type],
                "value": e.now_cost,
                "purchase_price": e.now_cost,
                "selling_price": e.now_cost,
                "expected_points": xp_map[e.id],
                "p10": xp_map[e.id] * 0.4,
                "p90": xp_map[e.id] * 1.6,
            }
        )
    pool_df = pd.DataFrame(pool_rows)

    return squad_df, pool_df


def test_contingency_plans_precomputation(mock_squad_and_pool):
    squad_df, pool_df = mock_squad_and_pool
    horizon_proj = {6: pool_df, 7: pool_df, 8: pool_df}

    plans = contingency_engine.generate_contingency_plans(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_proj,
        current_gw=5,
        target_gw=6,
        risk_preference="balanced",
    )

    assert "plan_a" in plans
    assert "plan_b" in plans
    assert "plan_c" in plans
    assert "trigger_condition" in plans["plan_a"]
    assert "trigger_condition" in plans["plan_b"]
    assert "trigger_condition" in plans["plan_c"]
    assert "delta_vs_plan_a" in plans["plan_b"]
    assert "delta_vs_plan_a" in plans["plan_c"]


def test_injury_contingency_matrix(mock_squad_and_pool, dummy_bootstrap):
    squad_df, pool_df = mock_squad_and_pool
    matrix = contingency_engine.compute_injury_matrix(
        squad_df=squad_df, player_pool_df=pool_df, bank=5.0, free_transfers=1, bootstrap=dummy_bootstrap
    )

    assert len(matrix) == 11  # All 11 starters analyzed
    for row in matrix:
        assert "web_name" in row
        assert "autosub_player" in row
        assert "autosub_points_delta" in row
        assert "action_verdict" in row
        assert row["action_verdict"] in ["TRUST_BENCH", "EXECUTE_TRANSFER", "MONITOR_PRESS_CONFERENCE"]
        assert "wait_vs_commit" in row


def test_panic_button_reoptimize(mock_squad_and_pool):
    squad_df, pool_df = mock_squad_and_pool
    # Test query-based panic: "Haaland broken foot out 8 weeks"
    panic_res = contingency_engine.panic_button_reoptimize(
        query="Haaland broken foot out 8 weeks", squad_df=squad_df, player_pool_df=pool_df, bank=10.0, free_transfers=1
    )

    assert panic_res["status"] == "crisis_resolved"
    assert panic_res["ruled_out_player"]["web_name"] == "Haaland"
    assert panic_res["lineup_action"]["captain"]["web_name"] != "Haaland"
    assert panic_res["lineup_action"]["promoted_player"] is not None
    assert "1-Click Recommendation" in panic_res["recommendation"]


def test_pre_deadline_checklist(mock_squad_and_pool, dummy_bootstrap):
    squad_df, _ = mock_squad_and_pool
    game_state_data = {"current_gameweek": 5, "seconds_to_deadline": 72000.0, "phase": "BETWEEN_GWS"}
    chips_status = {
        "set_1_remaining": ["wildcard", "freehit", "3xc", "bboost"],
        "set_1_used": [],
        "set_2_remaining": ["wildcard", "freehit", "3xc", "bboost"],
    }

    checklist = contingency_engine.generate_pre_deadline_checklist(
        squad_df=squad_df, bootstrap=dummy_bootstrap, game_state_data=game_state_data, chips_status=chips_status
    )

    assert len(checklist) == 5
    items = [c["item"] for c in checklist]
    assert "Starting XI Fitness & Availability" in items
    assert "Vice-Captain Failsafe" in items
    assert "Autosub Hierarchy Order" in items
    assert "Chip Set 1 Expiry Deadline" in items
    assert "Pre-Deadline Lock Time" in items
