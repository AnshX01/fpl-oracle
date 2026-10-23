"""
Unit tests for Mini-League Game Theory: Rival Shadowing (Lead Defense) vs Divergence (Catch-up Mode).
"""

import pandas as pd
import pytest

from fpl_oracle.league.strategy import LeagueStrategyAdvisor
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def test_rival_shadow_adjustment_defending_lead():
    """Verify that selling a chaser's player when defending a lead incurs a shadow penalty."""
    advisor = LeagueStrategyAdvisor()
    rival_elements = {10, 11, 12, 13}  # Chaser owns these players

    # Case A: We sell player 10 (owned by rival)
    transfers_out = [{"element": 10}, {"element": 5}]
    transfers_in = [{"element": 20}, {"element": 21}]

    adj, reasons = advisor.compute_rival_shadow_adjustment(
        transfers_out=transfers_out,
        transfers_in=transfers_in,
        rival_elements=rival_elements,
        posture="DEFENDING_LEAD",
    )
    assert adj == -0.60
    assert any("selling player 10" in r.lower() for r in reasons)


def test_rival_differential_bonus_chasing_pack():
    """Verify that buying an unowned player when chasing awards a differential bonus."""
    advisor = LeagueStrategyAdvisor()
    rival_elements = {10, 11, 12, 13}  # Leader owns these players

    # We buy players 20 and 21 (neither owned by leader)
    transfers_out = [{"element": 5}]
    transfers_in = [{"element": 20}, {"element": 21}]

    adj, reasons = advisor.compute_rival_shadow_adjustment(
        transfers_out=transfers_out,
        transfers_in=transfers_in,
        rival_elements=rival_elements,
        posture="CHASING_PACK",
    )
    assert adj == 0.80  # 2 x 0.40
    assert any("not owned by the leader" in r.lower() for r in reasons)


def test_evaluate_captain_shadowing_lead_and_chase():
    """Verify captain shadowing recommends aligning armband when leading and diverging when chasing."""
    advisor = LeagueStrategyAdvisor()
    starters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Defending lead: rival captains player 9 (who is in our starters)
    # Our raw top pick is player 8
    res_lead = advisor.evaluate_captain_shadowing(
        top_captain_id=8,
        starters_ids=starters,
        rival_captain_id=9,
        posture="DEFENDING_LEAD",
    )
    assert res_lead["action"] == "SHADOW_RIVAL_CAPTAIN"
    assert res_lead["captain_id"] == 9
    assert res_lead["should_switch"] is True

    # Chasing: leader captains player 8 (which is also our top pick)
    # Advisor should recommend diverging to alternative starter
    res_chase = advisor.evaluate_captain_shadowing(
        top_captain_id=8,
        starters_ids=starters,
        rival_captain_id=8,
        posture="CHASING_PACK",
    )
    assert res_chase["action"] == "DIVERGE_FROM_RIVAL_CAPTAIN"
    assert res_chase["captain_id"] != 8
    assert res_chase["should_switch"] is True
