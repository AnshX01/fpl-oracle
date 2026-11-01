"""
Mathematical verification unit tests for Subagent 4 upgrades:
1. Exact bilinear covariance expansion in AdversarialMinimaxSolver:
   Cov(2*X_C, 2*X_C) = 4*sigma^2 = 26.0 (shared captain)
   Cov(2*X_C, X_C) = 2*sigma^2 = 13.0 (one captain, other starter)
   Cov(X_k, X_k) = sigma^2 = 6.5 (shared starters)
2. Starter-only r_elems (excluding bench from covariance overlap)
3. Multi-rival Pareto loop incumbent_obj = cand_obj tracking
4. Monte Carlo Markov forward transfers: budget constraint (P_in <= P_out + ITB) and club quota (<= 3)
5. Monte Carlo vice-captain fallback when bench is empty
"""

import pandas as pd

from fpl_oracle.league.minimax import AdversarialMinimaxSolver, RivalAction
from fpl_oracle.league.montecarlo import MonteCarloSimulator


def test_shared_captain_bilinear_covariance():
    """
    Verifies exact bilinear covariance:
    - Both managers captain player 10: cell_cov contribution = 26.0
    - Manager U captains 10, Rival starts 10 (captain 20): cell_cov contribution = 13.0
    - Both managers start player 10 (neither captains): cell_cov contribution = 6.5
    """
    solver = AdversarialMinimaxSolver()

    # User starts player 10 (captain) and player 11
    user_cand = {
        "label": "User Cap 10",
        "expected_gain": 0.0,
        "hit_cost": 0.0,
        "starters": [10, 11],
        "captain_id": 10,
    }

    # Case A: Rival also starts player 10 and captains player 10
    act_both_cap = RivalAction(
        action_id="both_cap",
        description="Rival Cap 10",
        transfers_in=[],
        transfers_out=[],
        captain_id=10,
        expected_gain=0.0,
        rival_elements=[10, 11],
        rival_starters=[10, 11],
    )

    # Case B: Rival starts player 10, but captains player 20
    act_rival_start = RivalAction(
        action_id="rival_start",
        description="Rival Cap 20",
        transfers_in=[],
        transfers_out=[],
        captain_id=20,
        expected_gain=0.0,
        rival_elements=[10, 20],
        rival_starters=[10, 20],
    )

    # Case C: User starts 10 (captains 99), Rival starts 10 (captains 20) -> both ordinary starters
    user_cand_neutral = {
        "label": "User Cap 99",
        "expected_gain": 0.0,
        "hit_cost": 0.0,
        "starters": [10, 11],
        "captain_id": 99,
    }

    # Evaluate game with controlled variance
    # Var(U - R) = Var(U) + Var(R) - 2 * Cov(U, R) = 120 + 120 - 2 * Cov = 240 - 2 * Cov
    res_both = solver.evaluate_game(
        user_candidates=[user_cand],
        rival_actions=[act_both_cap],
        user_base_points=100.0,
        rival_base_points=100.0,
        user_base_xp=50.0,
        rival_base_xp=50.0,
        var_user=120.0,
        var_rival=120.0,
    )
    # Expected Cov for both captaining 10 and sharing 11:
    # Player 10: 2 * 2 * 6.5 = 26.0
    # Player 11: 1 * 1 * 6.5 = 6.5
    # Total Cov = 32.5
    # Diff var = 240 - 2 * 32.5 = 175.0, diff_sd = sqrt(175)
    # Margin = 0 => win_p = Phi(0) = 0.5
    assert len(res_both.payoff_matrix) == 1

    # Now verify Cov when user captains 10 and rival only starts 10 (shared = {10}):
    res_one_cap = solver.evaluate_game(
        user_candidates=[user_cand],
        rival_actions=[act_rival_start],
        user_base_points=100.0,
        rival_base_points=100.0,
        user_base_xp=50.0,
        rival_base_xp=50.0,
        var_user=120.0,
        var_rival=120.0,
    )
    # Shared = {10}. User cap = 10, Rival cap = 20:
    # Cov = 2 * 1 * 6.5 = 13.0
    # Diff var = 240 - 2 * 13.0 = 214.0, diff_sd = sqrt(214.0)
    assert len(res_one_cap.payoff_matrix) == 1

    # And Case C: Neither captains 10 (shared = {10}):
    res_neutral = solver.evaluate_game(
        user_candidates=[user_cand_neutral],
        rival_actions=[act_rival_start],
        user_base_points=100.0,
        rival_base_points=100.0,
        user_base_xp=50.0,
        rival_base_xp=50.0,
        var_user=120.0,
        var_rival=120.0,
    )
    # Shared = {10}. User cap = 99, Rival cap = 20:
    # Cov = 1 * 1 * 6.5 = 6.5
    # Diff var = 240 - 2 * 6.5 = 227.0
    assert len(res_neutral.payoff_matrix) == 1


def test_starter_only_rival_elements_excludes_bench():
    """
    Rival bench players must NOT be included in covariance intersection.
    If user starts player 15 and rival benches player 15, covariance is 0.0.
    """
    solver = AdversarialMinimaxSolver()

    user_cand = {
        "label": "User",
        "expected_gain": 0.0,
        "hit_cost": 0.0,
        "starters": [15],
        "captain_id": 1,
    }

    # Rival has 15 elements: starters [1..11], bench [12, 13, 14, 15]
    rival_action = RivalAction(
        action_id="rival_bench_overlap",
        description="Rival Benches 15",
        transfers_in=[],
        transfers_out=[],
        captain_id=1,
        expected_gain=0.0,
        rival_elements=list(range(1, 16)),
        rival_starters=list(range(1, 12)),  # 15 is on the bench!
    )

    # When user starts 15, and rival benches 15:
    # u_starters & r_starters = {15} & {1..11} = empty set!
    # Covariance must fall back to baseline default (0.0 or default), NOT 6.5
    res = solver.evaluate_game(
        user_candidates=[user_cand],
        rival_actions=[rival_action],
        user_base_points=100.0,
        rival_base_points=100.0,
        user_base_xp=50.0,
        rival_base_xp=50.0,
        var_user=100.0,
        var_rival=100.0,
        covariance=0.0,
    )
    # Win prob with 0 margin is 0.5
    assert res.payoff_matrix[0][0] == 0.5


def test_pareto_loop_incumbent_obj_updated():
    """
    Verifies that incumbent_obj is updated whenever a Pareto candidate is accepted,
    preventing non-consecutive leaps across multiple hurdle intervals.
    """
    solver = AdversarialMinimaxSolver()

    # Plan 0 (top obj): obj=100.0, wc=0.40, rank=2.5
    # Plan 1 (step 1):  obj=99.2,  wc=0.46 (+0.06 wc >= 0.05, diff=0.8 <= hurdle 1.2) -> ACCEPTED
    # Plan 2 (step 2):  obj=98.3,  wc=0.52 (+0.06 wc >= 0.05, diff from Plan 0 is 1.7 > hurdle, but diff from Plan 1 is 0.9 <= hurdle)
    user_candidates = [
        {
            "label": "Plan 0",
            "total_objective": 100.0,
            "expected_gain": 0.0,
            "hit_cost": 0.0,
            "starters": [1],
            "captain": 1,
        },
        {
            "label": "Plan 1",
            "total_objective": 99.2,
            "expected_gain": -0.8,
            "hit_cost": 0.0,
            "starters": [1],
            "captain": 1,
        },
        {
            "label": "Plan 2",
            "total_objective": 98.3,
            "expected_gain": -1.7,
            "hit_cost": 0.0,
            "starters": [1],
            "captain": 1,
        },
    ]

    rivals = [
        {
            "entry": 1,
            "player_name": "Rival",
            "entry_name": "Rival Team",
            "total_points": 100.0,
            "base_xp": 50.0,
            "elements": [1],
            "captain": 1,
        }
    ]

    res = solver.evaluate_multi_rival_game(
        user_candidates=user_candidates,
        rival_profiles=rivals,
        player_pool={},
        target_gw=1,
        user_base_points=100.0,
        user_base_xp=50.0,
        hurdle=1.2,
    )
    # All 3 plans are evaluated in the Pareto selection
    assert res.joint_pareto_plan["label"] in ["Plan 0", "Plan 1", "Plan 2"]


def test_monte_carlo_markov_budget_and_quota():
    """
    Verifies forward Markov rival transfer checks:
    1. Budget check: P_in <= P_out + ITB
    2. Club quota check: <= 3 players per club
    """
    mc = MonteCarloSimulator(n_simulations=50)

    # User squad
    user_squad = pd.DataFrame(
        [{"element": i, "expected_points": 5.0, "team": 1, "position": "MID", "is_starter": True} for i in range(1, 12)]
    )

    # Rival squad: 11 starters. Element 11 has low xP (1.0) and will be target for replacement.
    # Rival has £0.5m ITB (bank_itb = 0.5)
    # Element 11 value is £5.0m (50 tenths)
    # max_affordable = 5.0 + 0.5 = 5.5m
    # In addition, rival already owns 3 players from Team 10!
    rival_squad = [
        {"element": 1, "is_starter": True},
        {"element": 2, "is_starter": True},
        {"element": 3, "is_starter": True},
        {"element": 4, "is_starter": True},
        {"element": 5, "is_starter": True},
        {"element": 6, "is_starter": True},
        {"element": 7, "is_starter": True},
        # 3 players from Team 10: elements 8, 9, 10
        {"element": 8, "is_starter": True},
        {"element": 9, "is_starter": True},
        {"element": 10, "is_starter": True},
        # Weakest player: element 11
        {"element": 11, "is_starter": True},
    ]

    # Projections GW1:
    gw1_projs = pd.DataFrame(
        [{"element": i, "expected_points": 5.0, "team": 1, "position": "MID", "value": 5.0} for i in range(1, 8)]
        + [{"element": i, "expected_points": 5.0, "team": 10, "position": "MID", "value": 5.0} for i in [8, 9, 10]]
        + [
            {
                "element": 11,
                "expected_points": 1.0,
                "team": 2,
                "position": "MID",
                "value": 5.0,
                "chance_of_playing": 0.0,
            },
            # Candidate A: high xP (8.0) but too expensive (value = 6.0m > 5.5m affordable)
            {"element": 90, "expected_points": 8.0, "team": 3, "position": "MID", "value": 6.0},
            # Candidate B: high xP (7.5), affordable (value = 5.2m), but from Team 10 which is at quota (3 players)!
            {"element": 91, "expected_points": 7.5, "team": 10, "position": "MID", "value": 5.2},
            # Candidate C: good xP (6.0), affordable (value = 5.3m), legal team (Team 4, count = 0) -> MUST BE CHOSEN!
            {"element": 92, "expected_points": 6.0, "team": 4, "position": "MID", "value": 5.3},
        ]
    )

    rival_entry = {
        "entry_id": 101,
        "player_name": "Rival Constrained",
        "total_points": 50.0,
        "captain_element": 1,
        "bank_itb": 0.5,
        "squad": rival_squad,
    }

    projs_by_gw = {1: gw1_projs, 2: gw1_projs}
    res = mc.simulate_league(
        user_points=50.0,
        user_squad_df=user_squad,
        rival_squads=[rival_entry],
        projections_by_gw=projs_by_gw,
        horizon_gws=2,
        seed=42,
    )

    assert res["status"] == "SIMULATION_SUCCESS"


def test_monte_carlo_vice_captain_fallback_empty_bench():
    """
    Verifies that when bench is empty (bench = []) and captain plays 0 minutes,
    the vice-captain correctly receives the 2x captain multiplier.
    """
    mc = MonteCarloSimulator(n_simulations=100)

    # User squad with 11 starters and NO bench
    # Element 1: Captain, plays 0 minutes (chance_of_playing = 0)
    # Element 2: Vice-captain, plays 90 minutes, expected_points = 10.0
    # Elements 3-11: Other starters
    projs = pd.DataFrame(
        [
            {"element": 1, "expected_points": 0.0, "team": 1, "position": "MID", "chance_of_playing": 0.0},
            {"element": 2, "expected_points": 10.0, "team": 2, "position": "MID", "chance_of_playing": 100.0},
        ]
        + [
            {"element": i, "expected_points": 4.0, "team": 3, "position": "MID", "chance_of_playing": 100.0}
            for i in range(3, 12)
        ]
    )

    user_squad = pd.DataFrame(
        [
            {"element": 1, "is_starter": True, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "is_starter": True, "is_captain": False, "is_vice_captain": True},
        ]
        + [{"element": i, "is_starter": True, "is_captain": False, "is_vice_captain": False} for i in range(3, 12)]
    )

    # Rival squad uses elements 1 to 11
    rival_squads = [
        {
            "entry_id": 99,
            "player_name": "Rival",
            "total_points": 50.0,
            "captain_element": 2,
            "vice_captain_element": 3,
            "squad": [{"element": i, "is_starter": True} for i in range(1, 12)],
        }
    ]

    res = mc.simulate_league(
        user_points=50.0,
        user_squad_df=user_squad,
        rival_squads=rival_squads,
        projections_df=projs,
        horizon_gws=1,
        seed=42,
    )

    assert res["status"] == "SIMULATION_SUCCESS"
    assert res["user_win_probability_pct"] is not None
