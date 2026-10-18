"""
Deterministic Match Scoring Engine & Exact Rule Implementation for FPL.
Computes deterministic points from actual match event counts according to verified 2026/27 rules.
Also provides expected points calculation under discrete count approximations for thresholded stats.
"""

import math


def expected_floor_div(rate: float, divisor: int, max_k: int | None = None, tol: float = 1e-12) -> float:
    """
    Compute E[floor(K / divisor)] where K ~ Poisson(rate).
    Accounts for the Jensen inequality gap between floor(E[K]/divisor) and E[floor(K/divisor)].
    Uses dynamic upper bound when max_k is None, summing until residual mass < tol.
    """
    if rate <= 0.0 or divisor <= 0:
        return 0.0

    cap = max(100, int(rate + 10 * math.sqrt(rate) + 20)) if max_k is None else max_k
    prob_k = math.exp(-rate)
    cdf = prob_k
    exp_val = 0.0
    k = 0
    while k <= cap:
        if k > 0:
            prob_k = prob_k * (rate / k)
            cdf += prob_k
        exp_val += (k // divisor) * prob_k
        if max_k is None and k > rate and (1.0 - cdf) < tol:
            break
        k += 1
    return exp_val
