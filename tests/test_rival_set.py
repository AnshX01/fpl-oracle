"""
Unit and AST enforcement tests for proximity rival selection and uncapped league retrieval (Requirement F5).
"""

import ast
from pathlib import Path

import pytest

from fpl_oracle.league.rivals import get_rival_set


def test_rival_set_120_managers_exact_window():
    """
    Test a 120-manager league:
    - User is at rank 35 (500 pts).
    - Managers 1..34 (above user) must all be included.
    - Managers below user within 20 points (480..499) must all be included.
    - Managers with <= 479 points must not be included.
    - User entry itself must be excluded from rivals.
    """
    standings = []
    # Rank 1..34: 510 to 600 pts
    for r in range(1, 35):
        standings.append({
            "entry": 1000 + r,
            "player_name": f"Manager {r}",
            "rank": r,
            "total": 600 - (r * 2),
        })

    # User at Rank 35: 500 pts
    user_id = 9999
    standings.append({
        "entry": user_id,
        "player_name": "User Manager",
        "rank": 35,
        "total": 500,
    })

    # Rank 36..60: 480 to 499 pts (within 20 pts) -> 25 managers
    for r in range(36, 61):
        standings.append({
            "entry": 2000 + r,
            "player_name": f"Manager {r}",
            "rank": r,
            "total": 499 - ((r - 36) * (19 / 24)),  # all between 480 and 499
        })

    # Rank 61..120: 300 to 475 pts (outside 20 pts) -> 60 managers
    for r in range(61, 121):
        standings.append({
            "entry": 3000 + r,
            "player_name": f"Manager {r}",
            "rank": r,
            "total": 475 - (r - 61),
        })

    assert len(standings) == 120

    rivals, mode, user_rank = get_rival_set(standings, user_manager_id=user_id, points_window=20)

    assert mode == "PROXIMITY_WINDOW"
    assert user_rank == 35

    # Check user is excluded
    assert not any(r["entry"] == user_id for r in rivals)

    # Check ALL managers above (1..34) are included
    above_entries = {1000 + r for r in range(1, 35)}
    rival_entries = {r["entry"] for r in rivals}
    assert above_entries.issubset(rival_entries), "All managers above user must be included without caps"

    # Check managers within 20 points below are included
    below_within_entries = {2000 + r for r in range(36, 61)}
    assert below_within_entries.issubset(rival_entries), "All managers within 20 pts below must be included"

    # Check NO managers > 20 points below are included
    outside_entries = {3000 + r for r in range(61, 121)}
    assert outside_entries.isdisjoint(rival_entries), "Managers >20 pts below must not be included"

    # Total rivals = 34 + 25 = 59
    assert len(rivals) == 59


def test_rival_set_user_rank_1():
    """When user is in 1st place, at least top 10 chasers are included."""
    user_id = 7777
    standings = [{"entry": user_id, "player_name": "Leader", "rank": 1, "total": 600}]
    for r in range(2, 25):
        standings.append({
            "entry": 1000 + r,
            "player_name": f"Manager {r}",
            "rank": r,
            "total": 550 - (r * 5),  # All > 20 pts below (540, 535, ...)
        })

    rivals, mode, user_rank = get_rival_set(standings, user_manager_id=user_id, points_window=20)
    assert user_rank == 1
    # Fallback guarantees at least 10 chasers
    assert len(rivals) >= 10
    assert rivals[0]["rank"] == 2


def test_ast_check_no_reintroduced_caps():
    """
    AST inspection: verify that no calls to get_league_standings or analyze_rivals
    reintroduce hardcoded max_pages=1, max_pages=2, max_rivals=6, or max_rivals=8.
    """
    src_dir = Path(__file__).resolve().parent.parent / "src" / "fpl_oracle"
    forbidden_page_caps = {1, 2}
    forbidden_rival_caps = {6, 8}

    violations = []

    for py_file in src_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("get_league_standings", "analyze_rivals", "select_proximity_rivals"):
                    for kw in node.keywords:
                        if kw.arg == "max_pages" and isinstance(kw.value, ast.Constant):
                            if kw.value.value in forbidden_page_caps:
                                violations.append(f"{py_file.name}:{node.lineno} {kw.arg}={kw.value.value}")
                        if kw.arg in ("max_rivals", "max_rivals_to_inspect") and isinstance(kw.value, ast.Constant):
                            if kw.value.value in forbidden_rival_caps:
                                violations.append(f"{py_file.name}:{node.lineno} {kw.arg}={kw.value.value}")

    assert not violations, f"Forbidden artificial caps found in codebase: {violations}"
