"""
Unit tests for Feature 3: Cognitive Bias & FOMO / Regression-to-Mean Blocker.
Tests detection of chasing past points (overperformance trap) and premature rage-selling
(dumping elite underlying assets at a variance trough).
Zero hardcoded player names in production code.
"""

import pandas as pd

from fpl_oracle.advisor.critique import PlanCritiqueService


def test_detects_chasing_past_points_fomo():
    """Incoming target with 5 goals from 1.5 xG (+3.5 overperformance) triggers CHASING_PAST_POINTS warning."""
    critique = PlanCritiqueService()
    pool_df = pd.DataFrame([
        {"element": 101, "goals_scored": 5, "expected_goals": 1.5, "expected_points": 5.0, "value": 65},
    ])
    squad_df = pd.DataFrame([
        {"element": 201, "goals_scored": 1, "expected_goals": 1.0, "expected_points": 4.0, "value": 65},
    ])

    warnings = critique.detect_regression_traps(
        proposed_transfers=[{"in": 101, "out": 201}],
        current_squad_df=squad_df,
        player_pool_df=pool_df,
    )

    assert len(warnings) == 1
    assert "CHASING_PAST_POINTS" in warnings[0]
    assert "element 101" in warnings[0]
    assert "regression to the mean" in warnings[0].lower()


def test_detects_premature_rage_selling():
    """Outgoing player with 0 returns despite 2.4 xGI over 270 mins (0.80 xGI/90) triggers PREMATURE_SALE_WARNING."""
    critique = PlanCritiqueService()
    pool_df = pd.DataFrame([
        {"element": 102, "goals_scored": 1, "expected_goals": 1.0, "expected_points": 5.0, "value": 75},
    ])
    squad_df = pd.DataFrame([
        {
            "element": 202,
            "goals_scored": 0,
            "assists": 0,
            "expected_goal_involvements": 2.4,
            "minutes": 270,
            "expected_points": 5.5,
            "value": 75,
        },
    ])

    warnings = critique.detect_regression_traps(
        proposed_transfers=[{"in": 102, "out": 202}],
        current_squad_df=squad_df,
        player_pool_df=pool_df,
    )

    assert len(warnings) == 1
    assert "PREMATURE_SALE_WARNING" in warnings[0]
    assert "element 202" in warnings[0]
    assert "negative variance" in warnings[0].lower()


def test_no_warnings_for_sound_transfers():
    """Normal sustainable transfers produce zero cognitive bias warnings."""
    critique = PlanCritiqueService()
    pool_df = pd.DataFrame([
        {"element": 103, "goals_scored": 2, "expected_goals": 1.9, "expected_points": 5.0, "value": 60},
    ])
    squad_df = pd.DataFrame([
        {
            "element": 203,
            "goals_scored": 0,
            "assists": 0,
            "expected_goal_involvements": 0.3,
            "minutes": 270,
            "expected_points": 2.5,
            "value": 60,
        },
    ])

    warnings = critique.detect_regression_traps(
        proposed_transfers=[{"in": 103, "out": 203}],
        current_squad_df=squad_df,
        player_pool_df=pool_df,
    )

    assert len(warnings) == 0


def test_both_warnings_triggered_simultaneously():
    """When a manager rage-sells an unlucky asset to FOMO-buy an unsustainable hauler, both warnings trigger."""
    critique = PlanCritiqueService()
    pool_df = pd.DataFrame([
        {"element": 104, "goals_scored": 4, "expected_goals": 1.1, "expected_points": 6.0, "value": 80},
    ])
    squad_df = pd.DataFrame([
        {
            "element": 204,
            "goals_scored": 0,
            "assists": 0,
            "expected_goal_involvements": 2.1,
            "minutes": 270,
            "expected_points": 5.8,
            "value": 80,
        },
    ])

    warnings = critique.detect_regression_traps(
        proposed_transfers=[{"in": 104, "out": 204}],
        current_squad_df=squad_df,
        player_pool_df=pool_df,
    )

    assert len(warnings) == 2
    types = [w.split(":")[0] for w in warnings]
    assert "CHASING_PAST_POINTS" in types
    assert "PREMATURE_SALE_WARNING" in types
