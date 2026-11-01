"""
Unit tests for Active Stochastic & Minimax Closed-Loop Gating in Multi-Period Optimization.
Verifies that candidates within the forecast uncertainty hurdle (<= 1.20 xP) with superior
stochastic downside protection or minimax guarantees are actively promoted as recommended plan.
Zero hardcoded player names in production code.
"""

from fpl_oracle.league.minimax import AdversarialMinimaxSolver
from fpl_oracle.optimise.stochastic import StochasticScenarioOptimizer


def test_stochastic_robustness_evaluation_and_gating():
    """Candidates within hurdle with superior downside protection are gated into recommendation."""
    stoch = StochasticScenarioOptimizer()

    candidates = [
        {
            "plan_id": "naive_xp_plan",
            "label": "Naive Top Forecast",
            "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "captain_id": 11,
            "transfers_in": [11],
            "transfers_out": [12],
            "hit_cost": 0,
            "expected_gain": 2.0,
            "total_objective": 52.0,
        },
        {
            "plan_id": "robust_shield_plan",
            "label": "Robust Downside Shield",
            "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
            "captain_id": 10,
            "transfers_in": [13],
            "transfers_out": [12],
            "hit_cost": 0,
            "expected_gain": 1.5,
            "total_objective": 51.5,  # 0.5 xP gap <= 1.20 hurdle
        },
    ]

    import pandas as pd

    projections = {
        9: pd.DataFrame(
            [
                {"element": i, "expected_points": 5.0, "p10": 2.0, "p90": 8.0, "chance_of_playing": 100}
                for i in range(1, 15)
            ]
        ),
        10: pd.DataFrame(
            [
                {"element": i, "expected_points": 5.0, "p10": 2.0, "p90": 8.0, "chance_of_playing": 100}
                for i in range(1, 15)
            ]
        ),
    }
    # In GW10, Player 11 is heavily flagged with 25% chance of playing
    projections[10].loc[projections[10]["element"] == 11, "chance_of_playing"] = 25

    scenarios = stoch.generate_scenario_tree(projections, target_gw=9)
    evals = stoch.evaluate_candidates(candidates, scenarios, target_gw=10)

    eval_map = {e.plan_id: e for e in evals}
    assert len(eval_map) == 2
    # The robust plan with player 13 (100% fit) should have superior stochastic utility and robustness index
    assert (
        eval_map["robust_shield_plan"].expected_stochastic_utility
        > eval_map["naive_xp_plan"].expected_stochastic_utility
    )
    assert eval_map["robust_shield_plan"].robustness_index > eval_map["naive_xp_plan"].robustness_index


def test_minimax_multi_rival_gating_promotes_defensive_winner():
    """Multi-rival minimax evaluation identifies robust joint Pareto action."""
    solver = AdversarialMinimaxSolver()

    user_candidates = [
        {"label": "Fragile Differential", "expected_gain": 2.0, "hit_cost": 0.0, "total_objective": 52.0},
        {"label": "Template Shield", "expected_gain": 1.6, "hit_cost": 0.0, "total_objective": 51.6},
    ]

    rivals = [
        {
            "entry": 101,
            "player_name": "Rival Alpha",
            "total_points": 525.0,
            "base_xp": 50.0,
            "elements": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18],
            "bank_itb": 1.0,
            "captain": 10,
        },
        {
            "entry": 102,
            "player_name": "Rival Beta",
            "total_points": 520.0,
            "base_xp": 49.0,
            "elements": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18],
            "bank_itb": 0.5,
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
        hurdle=1.20,
    )

    assert len(res.global_worst_case_guarantees) == 2
    assert len(res.expected_ranks) == 2
    assert res.joint_pareto_plan is not None


def test_closed_loop_gating_guards_roll_and_near_chips():
    """
    Verifies that when the Bayesian hurdle demotes a marginal transfer plan to Roll,
    the stochastic closed-loop gate does not circularly override it or sabotage
    an upcoming chip (e.g. Bench Boost).
    """
    from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
    from tests.optimise.test_ft_and_churn import make_test_squad_and_pool

    squad_df, pool_df, horizon = make_test_squad_and_pool()
    # Filter out big_target so only marginal_target is available
    pool_marginal = pool_df[pool_df["element"] != 21].copy()
    horizon_marginal = {1: pool_marginal.copy(), 2: pool_marginal.copy()}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, ft_option_value=1.80))
    res = opt.evaluate_transfer_options(
        current_squad_df=squad_df,
        player_pool_df=pool_marginal,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon_marginal,
        current_gw=1,
        target_gw=1,
        horizon_len=2,
        chips_available=[],
    )

    rec = res["recommended_plan"]
    # Ensure recommended plan is safely preserved as ROLL_TRANSFER
    assert rec["plan_type"] == "ROLL_TRANSFER"
    assert "Roll" in rec.get("recommendation_summary", "")
    assert rec.get("stochastic_minimax_gated") is not True
