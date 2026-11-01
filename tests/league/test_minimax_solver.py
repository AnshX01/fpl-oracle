"""
Unit tests for Adversarial Minimax Solver (Pillar 1).
Validates game-theoretic payoff matrix, worst-case guarantees, Bayesian Nash response, and regret minimization.
"""

from fpl_oracle.league.minimax import AdversarialMinimaxSolver


def test_rival_action_space_generation():
    """Rival action space respects budget ITB and positions."""
    rival = {
        "elements": list(range(1, 16)),
        "bank_itb": 3.5,  # £3.5m in the bank
        "captain": 10,
    }
    pool = {
        1: {"position": "GKP", "value": 50, "expected_points": 4.0, "team": 1, "web_name": "Keeper A"},
        10: {"position": "FWD", "value": 110, "expected_points": 7.0, "team": 2, "web_name": "Forward A"},
        11: {"position": "FWD", "value": 60, "expected_points": 2.5, "team": 3, "web_name": "Forward B"},
        99: {"position": "FWD", "value": 90, "expected_points": 6.0, "team": 4, "web_name": "Target FWD"},
    }
    # Player 11 (val 60) + bank (3.5 = 35 tenths) = 95 afford target 99 (val 90)
    solver = AdversarialMinimaxSolver()
    actions = solver.generate_rival_action_space(rival, pool, current_gw=9)

    assert len(actions) >= 1
    assert actions[0].action_id == "rival_roll"
    # Action probabilities sum to 1.0
    total_prob = sum(a.prior_probability for a in actions)
    assert abs(total_prob - 1.0) < 1e-4


def test_minimax_game_evaluation():
    """Payoff matrix and minimax worst-case selection."""
    solver = AdversarialMinimaxSolver()

    # User has 2 candidates: Roll vs Aggressive Transfer
    user_candidates = [
        {"label": "Roll", "expected_gain": 0.0, "hit_cost": 0.0},
        {"label": "Move A", "expected_gain": 3.0, "hit_cost": 0.0},
    ]

    actions = solver.generate_rival_action_space(
        rival={"elements": list(range(1, 16)), "bank_itb": 0.5},
        pool={},
        current_gw=9,
    )
    # User is 3 points behind (-3.0 margin)
    res = solver.evaluate_game(
        user_candidates=user_candidates,
        rival_actions=actions,
        user_base_points=526.0,
        rival_base_points=529.0,
        user_base_xp=50.0,
        rival_base_xp=50.0,
    )

    assert len(res.payoff_matrix) == 2
    assert len(res.payoff_matrix[0]) == 1
    # Move A (+3.0 net gain) should have higher win probability than Roll
    assert res.payoff_matrix[1][0] > res.payoff_matrix[0][0]
    assert res.best_minimax_plan["label"] == "Move A"
    assert res.best_bayesian_plan["label"] == "Move A"
    assert "guarantees" in res.summary


def test_multi_rival_minimax_game():
    """Multi-rival pack game theory: worst-case guarantees, expected rank, and joint Pareto plan."""
    solver = AdversarialMinimaxSolver()

    user_candidates = [
        {"label": "Roll", "expected_gain": 0.0, "hit_cost": 0.0, "total_objective": 50.0},
        {"label": "Balanced Move", "expected_gain": 2.5, "hit_cost": 0.0, "total_objective": 52.5},
        {"label": "Aggressive Differential", "expected_gain": 3.0, "hit_cost": 4.0, "total_objective": 49.0},
    ]

    rivals = [
        {
            "entry": 2227954,
            "player_name": "Rival Leader",
            "entry_name": "Change Name",
            "total_points": 529.0,
            "base_xp": 51.0,
            "elements": list(range(1, 16)),
            "bank_itb": 3.5,
            "captain": 10,
        },
        {
            "entry": 3333333,
            "player_name": "Chasing Rival 2",
            "entry_name": "FC Chaser",
            "total_points": 515.0,
            "base_xp": 48.0,
            "elements": list(range(5, 20)),
            "bank_itb": 0.0,
            "captain": 10,
        },
    ]

    res = solver.evaluate_multi_rival_game(
        user_candidates=user_candidates,
        rival_profiles=rivals,
        player_pool={},
        target_gw=9,
        user_base_points=526.0,
        user_base_xp=50.0,
    )

    assert len(res.global_worst_case_guarantees) == 3
    assert len(res.expected_ranks) == 3
    assert len(res.pack_win_rates) == 3
    assert len(res.bottleneck_rivals) == 3
    # Expected rank is between 1.0 (leading everyone) and 3.0 (trailing both rivals)
    for rank in res.expected_ranks:
        assert 1.0 <= rank <= 3.0
    # Balanced Move should have better expected rank than Roll
    assert res.expected_ranks[1] < res.expected_ranks[0]
    assert res.best_bayesian_rank_plan["label"] == "Balanced Move"
    assert res.joint_pareto_plan["label"] == "Balanced Move"
    assert "Multi-Rival Adversarial Duel" in res.summary
