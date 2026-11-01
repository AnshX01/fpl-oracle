"""
Unit tests for Calibration math and Euclidean Simplex Bonus Projection (Subagent 4):
1. Platt scaling unregularized (C=1e5, solver='lbfgs') eliminating shrinkage to prior
2. Brier score calculation (mean squared probability error)
3. Log-loss calculation with stable epsilon clipping
4. Exact L2 Euclidean simplex projection:
   min_b 0.5 * ||b - b_hat||_2^2 subject to sum(b_i) = 6.0 and 0 <= b_i <= 3.0
   - Boundary checks, feasibility, sum invariant, KKT condition checks.
"""

import numpy as np

from fpl_oracle.ml.calibration import Calibrator
from fpl_oracle.ml.components.bonus import BonusModel, project_to_box_simplex


def test_platt_scaling_unregularized_no_l2_shrinkage():
    """
    Verifies that Platt scaling with C=1e5 has no L2 shrinkage toward base rate:
    Extreme logits retain their sharp calibrated probabilities rather than shrinking to 0.5.
    """
    cal = Calibrator(method="platt")

    # Generate synthetic binary dataset where high probabilities correspond strictly to 1s
    np.random.seed(42)
    y_prob = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95] * 20)
    y_true = np.array([0, 0, 0, 1, 1, 1] * 20)

    cal.fit(y_prob, y_true)

    # Check fitted model is LogisticRegression with C=1e5
    assert hasattr(cal.model, "C")
    assert cal.model.C >= 1e4

    # Calibrate extreme values
    calibrated = cal.calibrate(np.array([0.01, 0.99]))
    # For sharp data, calibrated[1] should remain very close to 1.0 (no shrinkage towards prior 0.5)
    assert calibrated[1] > 0.95
    assert calibrated[0] < 0.05


def test_calibrator_brier_score_and_log_loss():
    """
    Verifies exact mathematical formulas for Brier score and Log-loss:
    - Brier = mean((p - y)^2)
    - Log-loss = -mean(y * log(p) + (1 - y) * log(1 - p))
    """
    cal = Calibrator()

    p = np.array([0.9, 0.1, 0.8, 0.4])
    y = np.array([1, 0, 1, 0])

    expected_brier = float(np.mean((p - y) ** 2))
    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    expected_log_loss = float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))

    assert np.isclose(cal.brier_score(p, y), expected_brier)
    assert np.isclose(cal.log_loss(p, y), expected_log_loss)

    metrics = cal.evaluate(p, y)
    assert np.isclose(metrics["brier_score"], expected_brier)
    assert np.isclose(metrics["log_loss"], expected_log_loss)


def test_calibrator_numerical_stability_edge_cases():
    """Tests log-loss stability on extreme inputs (0.0 and 1.0)."""
    cal = Calibrator()
    # Predicted 1.0 for true 0: without clipping, log(0) would produce -inf/NaN
    loss = cal.log_loss(np.array([1.0, 0.0]), np.array([0, 1]))
    assert np.isfinite(loss)
    assert loss > 0.0

    # Empty inputs
    assert cal.brier_score(np.array([]), np.array([])) == 0.0
    assert cal.log_loss(np.array([]), np.array([])) == 0.0


def test_euclidean_simplex_projection_exact_kkt():
    """
    Verifies that project_to_box_simplex satisfies the exact KKT conditions:
    min_b 0.5 * ||b - b_hat||_2^2 s.t. sum(b_i) = 6.0 and 0 <= b_i <= 3.0.
    """
    # 6 players with raw predictions summing to 9.0 > 6.0
    b_hat = np.array([2.8, 2.2, 1.8, 1.2, 0.7, 0.3])
    b_opt = project_to_box_simplex(b_hat, target_sum=6.0, lower=0.0, upper=3.0)

    # 1. Sum constraint holds exactly
    assert np.isclose(np.sum(b_opt), 6.0, atol=1e-10)

    # 2. Box constraints hold strictly
    assert np.all(b_opt >= 0.0)
    assert np.all(b_opt <= 3.0)

    # 3. KKT Stationarity:
    # b_i = clip(b_hat_i - lambda, 0, 3)
    # Active coordinates (0 < b_i < 3) must all share the EXACT same shift (b_hat_i - b_i = lambda)
    active = (b_opt > 1e-7) & (b_opt < 3.0 - 1e-7)
    shifts = b_hat[active] - b_opt[active]
    assert np.allclose(shifts, shifts[0], atol=1e-10)


def test_euclidean_simplex_projection_boundary_and_infeasible():
    """Verifies projection behavior on boundary conditions and extreme caps."""
    # Under-sum projection: raw sum 3.0 < target_sum 6.0
    b_hat_low = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    b_opt_low = project_to_box_simplex(b_hat_low, target_sum=6.0, lower=0.0, upper=3.0)
    assert np.isclose(np.sum(b_opt_low), 6.0, atol=1e-10)
    assert np.all(b_opt_low >= 0.0)
    assert np.all(b_opt_low <= 3.0)

    # Infeasible: target_sum > K * upper (e.g. target_sum = 10.0 for 2 players with upper=3.0)
    b_opt_inf = project_to_box_simplex(np.array([2.0, 2.0]), target_sum=10.0, lower=0.0, upper=3.0)
    assert np.allclose(b_opt_inf, np.array([3.0, 3.0]))

    # Infeasible: target_sum < K * lower (e.g. target_sum = -2.0 with lower=0.0)
    b_opt_zero = project_to_box_simplex(np.array([2.0, 2.0]), target_sum=-2.0, lower=0.0, upper=3.0)
    assert np.allclose(b_opt_zero, np.array([0.0, 0.0]))

    # Empty array
    assert len(project_to_box_simplex(np.array([]))) == 0

    # Attached staticmethod on BonusModel
    bm = BonusModel()
    res = bm.project_to_box_simplex(np.array([1.5, 1.5, 1.5, 1.5]), target_sum=3.0)
    assert np.isclose(np.sum(res), 3.0)
    assert np.allclose(res, np.array([0.75, 0.75, 0.75, 0.75]))
