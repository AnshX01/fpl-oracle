"""
Test Suite for Requirement F2: Tested Promotion Gate.
Verifies that:
1. Candidate models must beat naive baselines with strictly positive gain across rolling origins.
2. Latest rolling origin (e.g. 2026-27 current season holdout) must achieve positive gain.
3. Candidate models that degrade overall MAE or fail baseline gate are rejected with automatic rollback.
"""

from fpl_oracle.ml.model_registry import check_promotion_gate


def test_promotion_gate_passes_when_all_criteria_met():
    """Candidate that beats baseline overall and across rolling origins passes gate."""
    candidate_metrics = {
        "ml_mae": 1.020,
        "base_mae": 1.080,
        "rolling_origins": [
            {"ml_mae": 1.010, "best_base_mae": 1.040},
            {"ml_mae": 0.950, "best_base_mae": 0.990},
            {"ml_mae": 1.820, "best_base_mae": 1.952},
        ],
    }
    active_metrics = {"ml_mae": 1.045}

    passed, reason = check_promotion_gate(candidate_metrics, active_metrics=active_metrics)
    assert passed is True
    assert "passed" in reason.lower()


def test_promotion_gate_rejects_when_mean_rolling_gain_negative():
    """Candidate with negative average gain across origins is rejected."""
    candidate_metrics = {
        "ml_mae": 1.050,
        "base_mae": 1.080,
        "rolling_origins": [
            {"ml_mae": 1.100, "best_base_mae": 1.040},  # -0.060
            {"ml_mae": 1.050, "best_base_mae": 0.990},  # -0.060
            {"ml_mae": 1.900, "best_base_mae": 1.952},  # +0.052
        ],
    }
    # Mean gain = (-0.060 - 0.060 + 0.052) / 3 = -0.0227 <= 0
    passed, reason = check_promotion_gate(candidate_metrics)
    assert passed is False
    assert "mean gain" in reason.lower()


def test_promotion_gate_rejects_when_latest_origin_gain_negative():
    """Candidate where latest origin (current season) loses to baseline is rejected."""
    candidate_metrics = {
        "ml_mae": 1.020,
        "base_mae": 1.080,
        "rolling_origins": [
            {"ml_mae": 1.010, "best_base_mae": 1.040},  # +0.030
            {"ml_mae": 0.950, "best_base_mae": 0.990},  # +0.040
            {"ml_mae": 2.000, "best_base_mae": 1.952},  # -0.048 (LOSES to baseline)
        ],
    }
    # Mean gain is (+0.030 + 0.040 - 0.048) / 3 = +0.0073 > 0, BUT latest origin loses!
    passed, reason = check_promotion_gate(candidate_metrics)
    assert passed is False
    assert "latest origin gain" in reason.lower()


def test_promotion_gate_rejects_when_degraded_vs_active_production():
    """Candidate that degrades vs active production by more than tolerance is rejected."""
    candidate_metrics = {
        "ml_mae": 1.250,
        "base_mae": 1.300,
        "rolling_origins": [
            {"ml_mae": 1.010, "best_base_mae": 1.040},
            {"ml_mae": 0.950, "best_base_mae": 0.990},
            {"ml_mae": 1.800, "best_base_mae": 1.952},
        ],
    }
    active_metrics = {"ml_mae": 1.040}  # Candidate is +0.210 worse than active (tolerance 0.05)

    passed, reason = check_promotion_gate(candidate_metrics, active_metrics=active_metrics, tolerance=0.05)
    assert passed is False
    assert "degraded" in reason.lower()


def test_promotion_gate_rejects_when_worse_than_overall_baseline():
    """Candidate that fails overall baseline superiority is rejected."""
    candidate_metrics = {
        "ml_mae": 1.450,
        "base_mae": 1.200,  # Candidate is +0.250 worse than baseline
    }
    passed, reason = check_promotion_gate(candidate_metrics)
    assert passed is False
    assert "failed baseline gate" in reason.lower()
