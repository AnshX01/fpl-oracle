"""
Unit tests for TrajectoryBoardEngine.
Verifies multi-period visual roadmap generation, formation extraction,
cumulative xP tracking, and transfer card structuring.
"""

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Team
from fpl_oracle.optimise.trajectory_board import trajectory_board_engine


def make_mock_bootstrap():
    elements = [
        Element(id=1, web_name="PlayerA", element_type=1, team=1, now_cost=50),  # GKP
        Element(id=2, web_name="PlayerB", element_type=2, team=1, now_cost=55),  # DEF
        Element(id=3, web_name="PlayerC", element_type=2, team=1, now_cost=50),  # DEF
        Element(id=4, web_name="PlayerD", element_type=2, team=1, now_cost=45),  # DEF
        Element(id=5, web_name="PlayerE", element_type=2, team=1, now_cost=40),  # DEF
        Element(id=6, web_name="PlayerF", element_type=3, team=2, now_cost=80),  # MID
        Element(id=7, web_name="PlayerG", element_type=3, team=2, now_cost=75),  # MID
        Element(id=8, web_name="PlayerH", element_type=3, team=2, now_cost=70),  # MID
        Element(id=9, web_name="PlayerI", element_type=3, team=2, now_cost=65),  # MID
        Element(id=10, web_name="PlayerJ", element_type=4, team=2, now_cost=140),  # FWD
        Element(id=11, web_name="PlayerK", element_type=4, team=2, now_cost=60),  # FWD
    ]
    element_types = [
        ElementType(
            id=1,
            singular_name_short="GKP",
            singular_name="Goalkeeper",
            plural_name="Goalkeepers",
            plural_name_short="GKP",
        ),
        ElementType(
            id=2, singular_name_short="DEF", singular_name="Defender", plural_name="Defenders", plural_name_short="DEF"
        ),
        ElementType(
            id=3,
            singular_name_short="MID",
            singular_name="Midfielder",
            plural_name="Midfielders",
            plural_name_short="MID",
        ),
        ElementType(
            id=4, singular_name_short="FWD", singular_name="Forward", plural_name="Forwards", plural_name_short="FWD"
        ),
    ]
    teams = [
        Team(id=1, name="Team1", short_name="T1"),
        Team(id=2, name="Team2", short_name="T2"),
    ]
    return BootstrapStatic(elements=elements, element_types=element_types, teams=teams, events=[])


def test_build_trajectory_board_structure():
    boot = make_mock_bootstrap()

    # Mock plan with 2 gameweeks
    plan = {
        "trajectory": [
            {
                "gameweek": 9,
                "bank_end": 5,  # 0.5m
                "free_transfers_available": 1,
                "free_transfers_next": 2,
                "hit_cost": 0,
                "active_chip": None,
                "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "bench": [],
                "captain": 10,
                "vice_captain": 6,
                "transfers_in": [],
                "transfers_out": [],
            },
            {
                "gameweek": 10,
                "bank_end": 2,  # 0.2m
                "free_transfers_available": 2,
                "free_transfers_next": 1,
                "hit_cost": 0,
                "active_chip": "bboost",
                "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "bench": [],
                "captain": 10,
                "vice_captain": 6,
                "transfers_in": [{"element": 8}],
                "transfers_out": [{"element": 9}],
            },
        ]
    }

    horizon_proj = {
        9: pd.DataFrame([{"element": i, "expected_points": 5.0} for i in range(1, 12)]),
        10: pd.DataFrame([{"element": i, "expected_points": 6.0} for i in range(1, 12)]),
    }

    board = trajectory_board_engine.build_trajectory_board(plan, boot, horizon_proj)

    assert board.start_gw == 9
    assert board.end_gw == 10
    assert board.total_horizon_weeks == 2
    assert len(board.steps) == 2

    step1 = board.steps[0]
    assert step1.gameweek == 9
    assert step1.formation == "4-4-2"
    assert step1.free_transfers_available == 1
    assert step1.free_transfers_next == 2
    assert step1.captain.element == 10
    assert step1.predicted_points > 50.0

    step2 = board.steps[1]
    assert step2.gameweek == 10
    assert step2.active_chip == "bboost"
    assert len(step2.transfers_in) == 1
    assert len(step2.transfers_out) == 1
    assert step2.transfers_in[0].element == 8
    assert board.total_cumulative_xp == step1.predicted_points + step2.predicted_points
