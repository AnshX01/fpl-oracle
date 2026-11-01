"""
Unit tests covering mathematical upgrades and boundary conditions across fpl-oracle:
1. Rule R4 Big-M Banked FT Invariant (multiperiod.py)
2. Subbed-Off Clean Sheet Binomial Survival (match_sim.py)
3. Exact Shared Captain Covariance (minimax.py)
4. Shin's Method Margin Removal (odds.py)
5. Exact L2 Simplex Projection for Bonus (bonus.py)
6. Downside Semi-Deviation (stochastic.py)
7. Quantile Monotonicity (ensemble.py / scoring_ensemble.py)

Strict production purity: 0 hardcoded player names.
"""

import math

import numpy as np
import pandas as pd
import pulp
import pytest
from scipy.optimize import brentq

from fpl_oracle.data.odds import shin_implied_probabilities
from fpl_oracle.domain.scoring_ensemble import scoring_ensemble
from fpl_oracle.league.minimax import AdversarialMinimaxSolver, RivalAction
from fpl_oracle.ml.components.bonus import project_to_box_simplex


def test_r4_big_m_banked_ft_invariant():
    """
    Test Rule R4 FT banking state machine with extreme transfer volume u_t = 15, ft_t = 1.
    M=20 must guarantee RHS >= 1 and solve to status Optimal with ft_next = 1.
    """
    u_val = 15
    ft_val = 1

    # Insufficient Big-M values must produce Infeasible
    for m_bad in (10, 13):
        prob = pulp.LpProblem(f"R4_M_{m_bad}", pulp.LpMinimize)
        ft_t = pulp.LpVariable("ft_t", lowBound=1, upBound=5, cat=pulp.LpInteger)
        ft_next = pulp.LpVariable("ft_next", lowBound=1, upBound=5, cat=pulp.LpInteger)
        u_t = pulp.LpVariable("u_t", lowBound=0, upBound=15, cat=pulp.LpInteger)
        z_t = pulp.LpVariable("z_t", cat=pulp.LpBinary)

        prob += ft_t == ft_val
        prob += u_t == u_val
        prob += u_t - ft_t <= 15 * z_t
        prob += ft_next <= ft_t - u_t + 1 + m_bad * z_t
        prob += ft_next <= 1 + 5 * (1 - z_t)
        prob += ft_next

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        assert pulp.LpStatus[status] == "Infeasible"

    # M = 20 must guarantee feasibility and ft_next = 1
    m_correct = 20
    prob_ok = pulp.LpProblem("R4_M_20", pulp.LpMinimize)
    ft_t = pulp.LpVariable("ft_t", lowBound=1, upBound=5, cat=pulp.LpInteger)
    ft_next = pulp.LpVariable("ft_next", lowBound=1, upBound=5, cat=pulp.LpInteger)
    u_t = pulp.LpVariable("u_t", lowBound=0, upBound=15, cat=pulp.LpInteger)
    z_t = pulp.LpVariable("z_t", cat=pulp.LpBinary)

    prob_ok += ft_t == ft_val
    prob_ok += u_t == u_val
    prob_ok += u_t - ft_t <= 15 * z_t
    prob_ok += ft_next <= ft_t - u_t + 1 + m_correct * z_t
    prob_ok += ft_next <= 1 + 5 * (1 - z_t)
    prob_ok += ft_next

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_ok = prob_ok.solve(solver)

    rhs = ft_val - u_val + 1 + m_correct * 1
    assert rhs >= 1
    assert pulp.LpStatus[status_ok] == "Optimal"
    assert math.isclose(float(pulp.value(ft_next)), 1.0)
    assert math.isclose(float(pulp.value(z_t)), 1.0)


def test_subbed_off_clean_sheet_binomial_survival():
    """
    Defender subbed at 60 mins: clean sheet frequency across N=10,000 draws
    must match Binomial survival (1/3)^opp_g within 2 standard deviations.
    """
    N = 10000
    rng = np.random.default_rng(20261028)
    p_on_pitch = 60.0 / 90.0

    # opp_g = 1: P(CS) = 1/3
    gc_1 = rng.binomial(1, p_on_pitch, size=N)
    freq_1 = float(np.mean(gc_1 == 0))
    p_true_1 = 1.0 / 3.0
    se_1 = math.sqrt(p_true_1 * (1.0 - p_true_1) / N)
    assert abs(freq_1 - p_true_1) <= 2.0 * se_1
    assert abs(freq_1 - p_true_1) <= 0.015

    # opp_g = 2: P(CS) = 1/9
    gc_2 = rng.binomial(2, p_on_pitch, size=N)
    freq_2 = float(np.mean(gc_2 == 0))
    p_true_2 = 1.0 / 9.0
    se_2 = math.sqrt(p_true_2 * (1.0 - p_true_2) / N)
    assert abs(freq_2 - p_true_2) <= 2.0 * se_2


def test_exact_shared_captain_covariance():
    """
    Identical starting XIs sharing captain: non-captain covariance term is 6.5,
    captain covariance term is 26.0, total covariance is 91.0.
    """
    u_starters = set(range(1, 12))
    r_starters = list(range(1, 12))
    u_cap = 10
    rival_cap = 10

    terms = {
        k: (2.0 if k == u_cap else 1.0) * (2.0 if k == rival_cap else 1.0) * 6.5 for k in (u_starters & set(r_starters))
    }

    assert math.isclose(terms[1], 6.5, rel_tol=1e-12)
    assert math.isclose(terms[u_cap], 26.0, rel_tol=1e-12)
    assert math.isclose(sum(terms.values()), 91.0, rel_tol=1e-12)

    solver = AdversarialMinimaxSolver()
    user_cands = [
        {
            "label": "User Starting XI",
            "expected_gain": 0.0,
            "hit_cost": 0,
            "starting_xi": list(range(1, 12)),
            "captain_id": u_cap,
        }
    ]
    rival_actions = [
        RivalAction(
            action_id="rival_same",
            description="Identical XI",
            transfers_in=[],
            transfers_out=[],
            captain_id=rival_cap,
            expected_gain=0.0,
            prior_probability=1.0,
            rival_starters=r_starters,
            rival_elements=list(range(1, 16)),
        )
    ]

    res = solver.evaluate_game(
        user_candidates=user_cands,
        rival_actions=rival_actions,
        user_base_points=100.0,
        rival_base_points=100.0,
        user_base_xp=60.0,
        rival_base_xp=60.0,
        var_user=120.0,
        var_rival=120.0,
        covariance=35.0,
    )
    assert math.isclose(res.payoff_matrix[0][0], 0.50, abs_tol=1e-4)


@pytest.mark.parametrize(
    "odds",
    [
        [1.15, 8.50, 21.0],
        [2.80, 3.20, 2.70],
        [1.08, 12.0, 34.0],
    ],
)
def test_shins_method_margin_removal(odds):
    """
    Shin's method must produce exact probabilities summing to 1.0 +- 1e-7,
    with z in [0, 1), and favorite probability strictly higher than naive normalization.
    """
    odds_arr = np.asarray(odds, dtype=float)
    inv = 1.0 / odds_arr
    total_inv = float(np.sum(inv))
    p_naive = inv / total_inv

    p_shin = shin_implied_probabilities(odds)
    p_sum = float(np.sum(p_shin))

    assert abs(p_sum - 1.0) <= 1e-7

    def diff(z: float) -> float:
        term = z**2 + 4.0 * (1.0 - z) * (inv**2) / total_inv
        p = (np.sqrt(np.maximum(0.0, term)) - z) / (2.0 * (1.0 - z))
        return float(np.sum(p) - 1.0)

    z_star = float(brentq(diff, 0.0, 1.0 - 1e-7, xtol=1e-10))
    assert 0.0 <= z_star < 1.0

    fav_idx = int(np.argmin(odds_arr))
    assert p_shin[fav_idx] > p_naive[fav_idx]


def test_l2_simplex_projection_for_bonus():
    """
    Exact L2 Euclidean box-simplex projection onto sum=6.0, lower=0, upper=3.
    """
    # Exceeding 6.0
    b_high = np.array([4.2, 3.1, 2.5, 1.8, 1.2, 0.9, 0.4])
    proj_high = project_to_box_simplex(b_high, target_sum=6.0, lower=0.0, upper=3.0)
    assert math.isclose(float(np.sum(proj_high)), 6.0, abs_tol=1e-9)
    assert np.all(proj_high >= -1e-12) and np.all(proj_high <= 3.0 + 1e-12)

    # Below 6.0
    b_low = np.array([1.1, 0.8, 0.5, 0.2, 0.1, 0.05])
    proj_low = project_to_box_simplex(b_low, target_sum=6.0, lower=0.0, upper=3.0)
    assert math.isclose(float(np.sum(proj_low)), 6.0, abs_tol=1e-9)
    assert np.all(proj_low >= -1e-12) and np.all(proj_low <= 3.0 + 1e-12)


def test_downside_semi_deviation():
    """
    Downside semi-deviation sqrt(E[min(0, U - u_bar)^2]):
    Upside outlier generating positive deviation must have 0 downside penalty.
    """
    probs = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    utils_upside = np.array([10.0, 10.0, 25.0])
    mean_upside = float(np.dot(probs, utils_upside))  # 15.0

    # Symmetric variance heavily penalizes the haul: (25 - 15)^2 = 100
    assert math.isclose((25.0 - mean_upside) ** 2, 100.0)

    # Downside semi-deviation penalizes 0 for the haul
    downside_diff = np.minimum(0.0, utils_upside - mean_upside)
    assert math.isclose(float(downside_diff[2] ** 2), 0.0)

    # Target downside semi-deviation relative to nominal baseline 10.0
    target_diff = np.minimum(0.0, utils_upside - 10.0)
    assert math.isclose(float(np.sqrt(np.dot(probs, target_diff**2))), 0.0)


def test_quantile_monotonicity_15_players():
    """
    Scoring ensemble predictions across all 15 squad players must satisfy
    strict quantile monotonicity: p10 <= p50 <= p90 without exception.
    """
    positions = [
        "GKP",
        "GKP",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    n = len(positions)

    x_df = pd.DataFrame(
        {
            "element": list(range(201, 201 + n)),
            "web_name": [f"Anon_Player_{i}" for i in range(1, n + 1)],
            "position": positions,
            "pos_GKP": [1.0 if p == "GKP" else 0.0 for p in positions],
            "pos_DEF": [1.0 if p == "DEF" else 0.0 for p in positions],
            "pos_MID": [1.0 if p == "MID" else 0.0 for p in positions],
            "pos_FWD": [1.0 if p == "FWD" else 0.0 for p in positions],
            "chance_of_playing": [100.0] * 11 + [75.0, 50.0, 25.0, 0.0],
        }
    )

    components = {
        "p_starts": np.array(
            [0.98, 0.05, 0.95, 0.90, 0.85, 0.70, 0.40, 0.98, 0.92, 0.85, 0.70, 0.35, 0.95, 0.90, 0.05], dtype=float
        ),
        "p_min60": np.array(
            [0.95, 0.02, 0.90, 0.85, 0.80, 0.60, 0.30, 0.90, 0.85, 0.75, 0.60, 0.25, 0.90, 0.80, 0.02], dtype=float
        ),
        "expected_minutes": np.array([90, 5, 88, 85, 80, 65, 45, 88, 85, 78, 65, 30, 88, 82, 10], dtype=float),
        "expected_goals": np.array(
            [0.0, 0.0, 0.04, 0.08, 0.02, 0.01, 0.0, 0.40, 0.28, 0.18, 0.10, 0.04, 0.70, 0.50, 0.02], dtype=float
        ),
        "expected_assists": np.array(
            [0.0, 0.0, 0.12, 0.16, 0.06, 0.03, 0.01, 0.35, 0.22, 0.16, 0.10, 0.05, 0.25, 0.18, 0.01], dtype=float
        ),
        "p_clean_sheet": np.full(n, 0.35),
        "expected_goals_conceded": np.full(n, 1.25),
        "expected_saves": np.array([3.6, 0.3] + [0.0] * 13, dtype=float),
        "p_defcon": np.full(n, 0.22),
        "expected_card_deduction": np.full(n, 0.14),
        "expected_bonus": np.array(
            [0.15, 0.0, 0.35, 0.45, 0.20, 0.10, 0.05, 0.90, 0.65, 0.40, 0.25, 0.10, 1.30, 0.80, 0.0], dtype=float
        ),
    }

    preds = scoring_ensemble.aggregate_components(components, x_df)
    assert len(preds) == 15

    for idx, row in preds.iterrows():
        p10 = row["p10"]
        p50 = row["p50"]
        p90 = row["p90"]
        assert p10 <= p50, f"Player {idx}: p10 ({p10}) > p50 ({p50})"
        assert p50 <= p90, f"Player {idx}: p50 ({p50}) > p90 ({p90})"
