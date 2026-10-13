"""
Regression test for F-012: Architectural Claims of 'Dixon-Coles Model' Contradicted by Fixed Poisson Implementation.

In a true bivariate Dixon-Coles (1997) model:
The joint probability P(X=x, Y=y) adjusts low scores (0-0, 1-0, 0-1, 1-1) via correlation parameter rho:
  tau_00 = 1 - lambda * mu * rho
  tau_10 = 1 + mu * rho
  tau_01 = 1 + lambda * rho
  tau_11 = 1 - rho
The current codebase relies solely on independent univariate Poisson expectations with static multipliers.

Deferred to spec WI-3.1 (True bivariate Dixon-Coles model).
"""

import pytest


@pytest.mark.xfail(strict=True, reason="Deferred to spec WI-3.1 (True bivariate Dixon-Coles model)")
def test_dixon_coles_fits_low_score_rho():
    # Attempt to import genuine Dixon-Coles model or verify rho adjustment parameter
    try:
        from fpl_oracle.ml.components.dixon_coles import DixonColesModel
    except ImportError:
        # If the class does not exist, fail as expected
        raise AssertionError(
            "DixonColesModel does not exist: codebase uses independent univariate Poisson instead of bivariate Dixon-Coles"
        ) from None

    model = DixonColesModel()
    assert hasattr(model, "rho"), "Dixon-Coles model must include low-score dependency parameter rho"
    assert hasattr(model, "fit"), "Dixon-Coles model must fit attack, defence, and rho via maximum likelihood"

    # Verify non-zero adjustment on 0-0 and 1-0 scorelines
    prob_00, prob_indep_00 = model.evaluate_scoreline_adjustment(home_team="Arsenal", away_team="Chelsea", score=(0, 0))
    assert prob_00 != prob_indep_00, "Dixon-Coles must adjust 0-0 joint probability relative to independent Poisson"
