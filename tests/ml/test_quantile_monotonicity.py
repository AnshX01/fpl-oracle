"""
Unit tests for quantile monotonicity (p10 <= p50 <= p90) and domain scoring ensemble export.
Verifies mathematical order invariants across all projections and aggregation outputs.
"""

import numpy as np
import pandas as pd

from fpl_oracle.domain.scoring_ensemble import ScoringEnsemble, scoring_ensemble


def test_domain_scoring_ensemble_reexport():
    """Verify that fpl_oracle.domain.scoring_ensemble re-exports ScoringEnsemble and singleton instance."""
    from fpl_oracle.ml.ensemble import ScoringEnsemble as MLEnsemble
    from fpl_oracle.ml.ensemble import scoring_ensemble as ml_singleton

    assert ScoringEnsemble is MLEnsemble
    assert scoring_ensemble is ml_singleton


def test_scoring_ensemble_quantile_monotonicity():
    """Verify that aggregate_components guarantees p10 <= p50 <= p90 across diverse players."""
    rng = np.random.default_rng(42)
    n = 200

    positions = rng.choice(["GKP", "DEF", "MID", "FWD"], size=n)
    X = pd.DataFrame(
        {
            "pos_GKP": (positions == "GKP").astype(float),
            "pos_DEF": (positions == "DEF").astype(float),
            "pos_MID": (positions == "MID").astype(float),
            "pos_FWD": (positions == "FWD").astype(float),
            "chance_of_playing": rng.choice([0.0, 50.0, 75.0, 100.0], size=n),
            "opponent_difficulty": rng.integers(1, 6, size=n),
            "roll_points_5": rng.uniform(0.0, 10.0, size=n),
            "roll_starts_ratio_5": rng.uniform(0.0, 1.0, size=n),
        }
    )

    components = {
        "p_starts": rng.uniform(0.0, 1.0, size=n),
        "p_min60": rng.uniform(0.0, 0.9, size=n),
        "p_play": rng.uniform(0.1, 1.0, size=n),
        "expected_minutes": rng.uniform(0.0, 90.0, size=n),
        "expected_goals": rng.uniform(0.0, 1.5, size=n),
        "expected_assists": rng.uniform(0.0, 1.0, size=n),
        "p_clean_sheet": rng.uniform(0.05, 0.60, size=n),
        "expected_goals_conceded": rng.uniform(0.2, 3.5, size=n),
        "expected_saves": rng.uniform(0.0, 8.0, size=n),
        "p_defcon": rng.uniform(0.0, 0.5, size=n),
        "expected_card_deduction": rng.uniform(0.0, 0.5, size=n),
        "expected_bonus": rng.uniform(0.0, 2.5, size=n),
    }

    res_df = scoring_ensemble.aggregate_components(components, X)

    p10 = res_df["p10"].to_numpy()
    p50 = res_df["p50"].to_numpy()
    p90 = res_df["p90"].to_numpy()

    # Monotonicity checks: p10 <= p50 <= p90 unconditionally
    assert np.all(p10 <= p50 + 1e-6), f"Violation of p10 <= p50: min gap {np.min(p50 - p10)}"
    assert np.all(p50 <= p90 + 1e-6), f"Violation of p50 <= p90: min gap {np.min(p90 - p50)}"
    assert np.all(p10 >= 0.0), "p10 must be non-negative"


def test_extreme_quantiles_forcing_monotonicity():
    """Verify that even when raw calibration z10 > 0 or skew adjustment is large, monotonicity holds."""
    custom_ensemble = ScoringEnsemble(z10=0.5, z90=0.2)  # Inverted raw z-scores
    X = pd.DataFrame(
        {
            "pos_FWD": [1.0, 1.0],
            "pos_MID": [0.0, 0.0],
            "pos_DEF": [0.0, 0.0],
            "pos_GKP": [0.0, 0.0],
            "chance_of_playing": [100.0, 100.0],
        }
    )
    components = {
        "p_starts": np.array([0.9, 0.9]),
        "p_min60": np.array([0.9, 0.9]),
        "p_play": np.array([0.95, 0.95]),
        "expected_minutes": np.array([85.0, 85.0]),
        "expected_goals": np.array([0.8, 0.1]),
        "expected_assists": np.array([0.4, 0.0]),
        "p_clean_sheet": np.array([0.2, 0.2]),
        "expected_goals_conceded": np.array([1.2, 1.2]),
        "expected_saves": np.array([0.0, 0.0]),
        "p_defcon": np.array([0.1, 0.1]),
        "expected_card_deduction": np.array([0.1, 0.1]),
        "expected_bonus": np.array([0.5, 0.0]),
    }

    res_df = custom_ensemble.aggregate_components(components, X)
    p10 = res_df["p10"].to_numpy()
    p50 = res_df["p50"].to_numpy()
    p90 = res_df["p90"].to_numpy()

    assert np.all(p10 <= p50)
    assert np.all(p50 <= p90)
