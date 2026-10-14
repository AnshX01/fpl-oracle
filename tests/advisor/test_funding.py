"""
Tests for Affordability Path Finder (WI-8.3).
Validates direct swaps, two-step funding plans, selling-price arithmetic,
and ensures zero hard-coded player names in production code.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.advisor.funding import AffordabilityPathFinder


@pytest.fixture
def s1_fixture():
    fixture_path = Path(__file__).parents[1] / "fixtures" / "s1_gw6_state.json"
    with open(fixture_path) as f:
        return json.load(f)


def test_s1_single_free_transfer_swap_path_found(s1_fixture):
    """
    WI-8.3 Acceptance:
    On S1 fixture, the direct single free transfer swap path to a premium target (e.g. Saka eid=350, cost=96)
    MUST be discovered using the selling price oracle and £1.8m bank balance.
    """
    squad_df = pd.DataFrame(s1_fixture["squad"])

    # Synthetic player pool containing target premium mid
    target_eid = 999
    pool_rows = list(s1_fixture["squad"])
    pool_rows.append({
        "element": target_eid,
        "position": "MID",
        "team": 1,
        "value": 96.0,  # £9.6m
        "expected_points": 6.5,
    })
    pool_df = pd.DataFrame(pool_rows)

    finder = AffordabilityPathFinder()
    paths = finder.find_cheapest_paths(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank_tenths=s1_fixture["bank_tenths"],  # 18 = £1.8m
        target_element=target_eid,
        free_transfers=1,
        top_k=3,
    )

    assert len(paths) >= 1
    best_path = paths[0]

    # Must find a 1-transfer DIRECT_SWAP with 0 hits
    assert best_path.path_type == "DIRECT_SWAP"
    assert best_path.transfers_count == 1
    assert best_path.hits == 0
    assert best_path.final_bank_tenths >= 0
    assert len(best_path.moves) == 1

    move = best_path.moves[0]
    assert move.in_element == target_eid
    # Outgoing player must be a midfielder from the squad whose sell price + 1.8m >= 9.6m
    assert (s1_fixture["bank_tenths"] + move.out_sell_price_tenths) >= move.in_cost_tenths
    assert "Direct swap" in best_path.arithmetic_explanation


def test_two_step_plan_found_when_direct_swap_insufficient():
    """When bank is short, funding finder must identify an enabler downgrade + target upgrade pair."""
    # 15 player squad with minimal value and 0 bank
    squad_rows = [
        {"element": 1, "position": "GKP", "team": 1, "value": 45, "purchase_price": 45},
        {"element": 2, "position": "GKP", "team": 2, "value": 40, "purchase_price": 40},
        {"element": 3, "position": "DEF", "team": 3, "value": 60, "purchase_price": 60},  # Expensive DEF
        {"element": 4, "position": "DEF", "team": 4, "value": 45, "purchase_price": 45},
        {"element": 5, "position": "DEF", "team": 5, "value": 45, "purchase_price": 45},
        {"element": 6, "position": "DEF", "team": 6, "value": 40, "purchase_price": 40},
        {"element": 7, "position": "DEF", "team": 7, "value": 40, "purchase_price": 40},
        {"element": 8, "position": "MID", "team": 8, "value": 50, "purchase_price": 50},  # Target swap out
        {"element": 9, "position": "MID", "team": 9, "value": 50, "purchase_price": 50},
        {"element": 10, "position": "MID", "team": 10, "value": 50, "purchase_price": 50},
        {"element": 11, "position": "MID", "team": 11, "value": 45, "purchase_price": 45},
        {"element": 12, "position": "MID", "team": 12, "value": 45, "purchase_price": 45},
        {"element": 13, "position": "FWD", "team": 13, "value": 75, "purchase_price": 75},
        {"element": 14, "position": "FWD", "team": 14, "value": 55, "purchase_price": 55},
        {"element": 15, "position": "FWD", "team": 15, "value": 45, "purchase_price": 45},
    ]
    squad_df = pd.DataFrame(squad_rows)

    # Target is £6.5m MID (65 tenths), but element 8 is only £5.0m (50 tenths) and bank is 0.
    # Shortfall is £1.5m.
    # Element 3 (DEF, £6.0m) can downgrade to a £4.0m enabler (saving £2.0m), funding the MID upgrade!
    pool_rows = list(squad_rows)
    target_eid = 100
    pool_rows.append({"element": target_eid, "position": "MID", "team": 16, "value": 65})
    enabler_eid = 101
    pool_rows.append({"element": enabler_eid, "position": "DEF", "team": 17, "value": 40})
    pool_df = pd.DataFrame(pool_rows)

    finder = AffordabilityPathFinder()
    paths = finder.find_cheapest_paths(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank_tenths=0,
        target_element=target_eid,
        free_transfers=1,
        top_k=2,
    )

    assert len(paths) >= 1
    two_step = paths[0]
    assert two_step.path_type == "TWO_STEP_PLAN"
    assert two_step.transfers_count == 2
    assert two_step.final_bank_tenths >= 0
    assert len(two_step.moves) == 2


def test_no_hardcoded_player_names_in_funding():
    """Principle: zero hardcoded player names allowed in production code."""
    path = Path(__file__).parents[2] / "src" / "fpl_oracle" / "advisor" / "funding.py"
    content = path.read_text(encoding="utf-8").lower()
    for forbidden in ["haaland", "saka", "palmer", "mbeumo", "tarkowski", "saliba", "foden"]:
        assert forbidden not in content, f"Found hardcoded player name '{forbidden}' in funding.py"
