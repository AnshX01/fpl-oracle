import numpy as np
from scipy.stats import poisson

from fpl_oracle.domain.scoring import expected_floor_div


def test_expected_floor_div_matches_brute_force_rates_0_to_60():
    """
    F-004: Validate expected_floor_div against exact brute-force summation
    over Poisson rates 0.01 to 60 for divisors 2 and 3.
    """
    rates = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]

    for divisor in [2, 3]:
        for rate in rates:
            computed = expected_floor_div(rate, divisor)

            # Brute-force summation up to k=300
            k_arr = np.arange(0, 300)
            p_arr = poisson.pmf(k_arr, rate)
            exact = float(np.sum((k_arr // divisor) * p_arr))

            assert abs(computed - exact) < 1e-6, (
                f"Rate {rate}, divisor {divisor}: computed={computed}, exact={exact}, diff={abs(computed - exact)}"
            )


def test_expected_floor_div_monotonic_high_lambda():
    """
    F-004: Ensure expected_floor_div increases monotonically with rate.
    """
    rates = np.linspace(0.1, 40.0, 40)
    vals_div2 = [expected_floor_div(r, 2) for r in rates]
    vals_div3 = [expected_floor_div(r, 3) for r in rates]

    for i in range(len(rates) - 1):
        assert vals_div2[i + 1] > vals_div2[i], f"Non-monotonic at rate {rates[i]} -> {rates[i + 1]} (div 2)"
        assert vals_div3[i + 1] > vals_div3[i], f"Non-monotonic at rate {rates[i]} -> {rates[i + 1]} (div 3)"
