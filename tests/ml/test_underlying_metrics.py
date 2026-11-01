"""
Unit tests for Advanced Underlying Event Metrics & Spatial Threat Engine (Pillar 3).
Validates xT calculation, box touch intensity, and Bayesian finishing regression.
Zero hardcoded player names.
"""

import pandas as pd

from fpl_oracle.ml.components.underlying_metrics import (
    PlayerUnderlyingProfile,
    UnderlyingMetricsEngine,
    underlying_metrics_engine,
)


def test_underlying_profile_calculation():
    """Validates calculation of xT, box touches, set-piece share, and regression."""
    engine = UnderlyingMetricsEngine(shrinkage_weight=0.35)

    profile = engine.calculate_underlying_profile(
        element=101,
        web_name="Test Playmaker",
        position="MID",
        raw_xg_per_90=0.35,
        raw_xa_per_90=0.45,
        ict_threat=40.0,
        ict_creativity=65.0,
        actual_goals=4,
        actual_assists=3,
        minutes_played=540,
    )

    assert isinstance(profile, PlayerUnderlyingProfile)
    assert 0.05 <= profile.expected_threat_xt <= 0.95
    assert profile.box_touch_intensity > 0.0
    assert 0.5 <= profile.finishing_sustainability_factor <= 2.0
    assert profile.adjusted_xg_per_90 > 0.0
    assert profile.adjusted_xa_per_90 > 0.0


def test_defender_set_piece_threat_share():
    """Defenders with high aerial xG have elevated set piece threat share."""
    engine = UnderlyingMetricsEngine()

    def_profile = engine.calculate_underlying_profile(
        element=102,
        web_name="Test CentreBack",
        position="DEF",
        raw_xg_per_90=0.20,
        raw_xa_per_90=0.05,
        ict_threat=25.0,
        ict_creativity=5.0,
        actual_goals=1,
        actual_assists=0,
        minutes_played=720,
    )

    assert def_profile.set_piece_threat_share >= 0.20


def test_enhance_projections_dataframe():
    """Enhance projections adds underlying metric columns without mutating row count."""
    df = pd.DataFrame(
        {
            "element": [1, 2, 3],
            "web_name": ["Player A", "Player B", "Player C"],
            "position": ["DEF", "MID", "FWD"],
            "expected_points": [4.0, 5.5, 6.2],
            "exp_goals_pts": [0.8, 2.0, 3.2],
            "exp_assists_pts": [0.3, 1.8, 0.9],
            "threat": [15.0, 35.0, 55.0],
            "creativity": [10.0, 45.0, 20.0],
            "minutes": [90, 90, 90],
        }
    )

    enhanced = underlying_metrics_engine.enhance_projections(df)

    assert len(enhanced) == 3
    assert "expected_threat_xt" in enhanced.columns
    assert "box_touch_intensity" in enhanced.columns
    assert "finishing_sustainability" in enhanced.columns
    assert (enhanced["expected_threat_xt"] > 0.0).all()

    assert (enhanced["finishing_sustainability"] == 1.0).all()


def test_finishing_sustainability_with_actual_goals():

    df = pd.DataFrame(
        {
            "element": [1, 2],
            "web_name": ["Overperformer", "Underperformer"],
            "position": ["FWD", "FWD"],
            "expected_points": [6.0, 6.0],
            "exp_goals_pts": [4.0, 4.0],
            "goals_scored": [3, 0],
            "threat": [50.0, 50.0],
            "creativity": [20.0, 20.0],
            "minutes": [90, 90],
        }
    )

    enhanced = underlying_metrics_engine.enhance_projections(df)

    assert enhanced.loc[0, "finishing_sustainability"] > 1.0

    assert enhanced.loc[1, "finishing_sustainability"] < 1.0
