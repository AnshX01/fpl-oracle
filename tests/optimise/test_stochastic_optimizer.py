"""
Unit tests for Stochastic Multi-Scenario Tree Optimizer (Pillar 2).
Validates multi-scenario branching, recourse utility, risk-adjusted robustness ranking.
"""

import pandas as pd

from fpl_oracle.optimise.stochastic import ParetoPlanPoint, StochasticScenario, StochasticScenarioOptimizer


def test_stochastic_scenario_tree_generation():
    """Validates creation of nominal, rotation, and stress scenarios."""
    projs = {
        9: pd.DataFrame(
            {
                "element": [1, 2],
                "expected_points": [5.0, 6.0],
                "value": [85.0, 50.0],
                "chance_of_playing": [100.0, 75.0],
            }
        ),
        10: pd.DataFrame(
            {
                "element": [1, 2],
                "expected_points": [5.0, 6.0],
                "value": [85.0, 50.0],
                "chance_of_playing": [100.0, 75.0],
            }
        ),
    }
    optimizer = StochasticScenarioOptimizer()
    scenarios = optimizer.generate_scenario_tree(projs, target_gw=9)

    assert len(scenarios) == 3
    names = [s.name for s in scenarios]
    assert "nominal" in names
    assert "rotation_shock" in names
    assert "availability_stress" in names

    total_prob = sum(s.probability for s in scenarios)
    assert abs(total_prob - 1.0) < 1e-4

    # GW9 should remain unperturbed (target decision week)
    rot_gw9 = next(s for s in scenarios if s.name == "rotation_shock").scenario_projections[9]
    assert rot_gw9.loc[0, "expected_points"] == 5.0

    # GW10 should have rotation reduction on premium asset (element 1, val 85)
    rot_gw10 = next(s for s in scenarios if s.name == "rotation_shock").scenario_projections[10]
    assert rot_gw10.loc[0, "expected_points"] < 5.0


def test_stochastic_candidate_evaluation_and_robustness():
    """Candidate plans are ranked by stochastic risk-adjusted utility."""
    optimizer = StochasticScenarioOptimizer()
    scenarios = optimizer.generate_scenario_tree(
        {9: pd.DataFrame({"element": [1], "expected_points": [4.0]})},
        target_gw=9,
    )
    candidates = [
        {"label": "Roll", "transfers_in": [], "transfers_out": [], "expected_gain": 0.0, "hit_cost": 0},
        {
            "label": "Aggressive Transfer",
            "transfers_in": [99],
            "transfers_out": [1],
            "expected_gain": 2.5,
            "hit_cost": 0,
        },
    ]

    evals = optimizer.evaluate_candidates(candidates, scenarios, target_gw=9)

    assert len(evals) == 2
    top = evals[0]
    assert top.label == "Aggressive Transfer"
    assert top.robustness_index > 0.0
    assert "nominal" in top.scenario_utilities


def test_downside_semi_deviation_does_not_penalize_upside():
    """
    Downside semi-deviation sqrt(E[min(0, U - u_bar)^2]) must not penalize
    plans that have high upside variance across scenarios.
    """
    scenarios = [
        StochasticScenario(name="nominal", probability=0.50, description="", scenario_projections={}),
        StochasticScenario(name="rotation_shock", probability=0.25, description="", scenario_projections={}),
        StochasticScenario(name="availability_stress", probability=0.25, description="", scenario_projections={}),
    ]

    opt = StochasticScenarioOptimizer(risk_aversion_lambda=0.5)

    # Plan A: flat utility across all scenarios
    # Plan B: strictly higher upside in stress/rotation
    candidates = [
        {
            "plan_id": "flat",
            "label": "Flat",
            "transfers_in": [],
            "transfers_out": [],
            "expected_gain": 5.0,
            "hit_cost": 0,
        },
        {
            "plan_id": "upside",
            "label": "Upside",
            "transfers_in": [],
            "transfers_out": [],
            "expected_gain": 10.0,
            "hit_cost": 0,
        },
    ]

    evals = opt.evaluate_candidates(candidates, scenarios, target_gw=9)
    eval_map = {e.plan_id: e for e in evals}

    # Plan with higher returns and upside variance should have strictly higher robustness
    assert eval_map["upside"].robustness_index > eval_map["flat"].robustness_index
    assert eval_map["upside"].expected_stochastic_utility > eval_map["flat"].expected_stochastic_utility


def test_negative_base_xp_sign_flip_guardrail():
    """
    When base xP is negative (e.g. following multiple transfer hits), stress scenarios
    must subtract a non-negative penalty rather than multiplying a negative number,
    preventing the stress scenario from appearing better than the baseline.
    """
    scenarios = [
        StochasticScenario(name="nominal", probability=0.50, description="", scenario_projections={}),
        StochasticScenario(name="rotation_shock", probability=0.25, description="", scenario_projections={}),
        StochasticScenario(name="availability_stress", probability=0.25, description="", scenario_projections={}),
    ]
    opt = StochasticScenarioOptimizer()

    # Plan with net negative base xP (-3.0 pts)
    candidates = [
        {"label": "Heavy Hit Plan", "transfers_in": [1], "transfers_out": [2], "expected_gain": 1.0, "hit_cost": 4},
    ]
    evals = opt.evaluate_candidates(candidates, scenarios, target_gw=9)
    res = evals[0]

    nom = res.scenario_utilities["nominal"]
    rot = res.scenario_utilities["rotation_shock"]
    stress = res.scenario_utilities["availability_stress"]

    assert nom == -3.0
    # Rotation shock and availability stress must be strictly less than or equal to nominal
    assert rot <= nom, f"Rotation shock ({rot}) should be <= nominal ({nom})"
    assert stress <= nom, f"Availability stress ({stress}) should be <= nominal ({nom})"


def test_pareto_frontier_pruning_and_deduplication():
    """
    filter_pareto_candidates and filter_pareto_frontier_points must prune strictly
    dominated plans and remove duplicate points.
    """
    scenarios = [
        StochasticScenario(name="nominal", probability=0.50, description="", scenario_projections={}),
        StochasticScenario(name="rotation_shock", probability=0.25, description="", scenario_projections={}),
        StochasticScenario(name="availability_stress", probability=0.25, description="", scenario_projections={}),
    ]
    opt = StochasticScenarioOptimizer()

    candidates = [
        # Dominant plan: high expected utility and high floor
        {
            "plan_id": "dominant_core",
            "label": "Dominant Core",
            "transfers_in": [],
            "transfers_out": [],
            "expected_gain": 8.0,
            "hit_cost": 0,
        },
        # Dominated plan: strictly worse utility and worse floor
        {
            "plan_id": "dominated_plan",
            "label": "Dominated Plan",
            "transfers_in": [1],
            "transfers_out": [2],
            "expected_gain": 2.0,
            "hit_cost": 4,
        },
        # Duplicate plan
        {
            "plan_id": "dominant_core",
            "label": "Dominant Core Duplicate",
            "transfers_in": [],
            "transfers_out": [],
            "expected_gain": 8.0,
            "hit_cost": 0,
        },
    ]

    # 1. Candidate filtering prunes dominated and duplicate plans
    filtered_cands = opt.filter_pareto_candidates(candidates, scenarios, target_gw=9)
    assert len(filtered_cands) == 1
    assert filtered_cands[0]["plan_id"] == "dominant_core"

    # 2. generate_pareto_frontier with deduplicate=True prunes duplicate points
    frontier_dedup = opt.generate_pareto_frontier(candidates, scenarios, target_gw=9, deduplicate=True)
    assert len(frontier_dedup) == 1
    assert frontier_dedup[0].best_plan_id == "dominant_core"

    # 3. Frontier points filtering removes strictly dominated points
    pts = [
        ParetoPlanPoint(
            lambda_risk=0.0,
            strategy_profile="A",
            best_plan_id="p1",
            label="P1",
            expected_stochastic_utility=10.0,
            downside_worst_case=6.0,
            downside_cvar=6.0,
            robustness_score=10.0,
            transfers_in=[],
            transfers_out=[],
            hit_cost=0,
            rationale="",
        ),
        ParetoPlanPoint(
            lambda_risk=0.35,
            strategy_profile="B",
            best_plan_id="p2",
            label="P2",
            expected_stochastic_utility=8.0,
            downside_worst_case=4.0,
            downside_cvar=4.0,
            robustness_score=8.0,
            transfers_in=[],
            transfers_out=[],
            hit_cost=0,
            rationale="",
        ),
    ]
    optimal_pts = opt.filter_pareto_frontier_points(pts)
    assert len(optimal_pts) == 1
    assert optimal_pts[0].best_plan_id == "p1"


def test_trajectory_relative_scenario_evaluation():
    """
    Trajectory-based candidate evaluation calculates scenario utility relative
    to the baseline roll plan, preventing dimensional mismatch from absolute
    multi-week squad degradation.
    """
    import pandas as pd

    from fpl_oracle.optimise.stochastic import StochasticScenarioOptimizer

    opt = StochasticScenarioOptimizer()
    projs = {
        9: pd.DataFrame(
            {
                "element": [1, 2],
                "expected_points": [5.0, 6.0],
                "value": [85.0, 50.0],
                "chance_of_playing": [100.0, 100.0],
            }
        ),
        10: pd.DataFrame(
            {
                "element": [1, 2],
                "expected_points": [5.0, 6.0],
                "value": [85.0, 50.0],
                "chance_of_playing": [100.0, 100.0],
            }
        ),
    }
    scenarios = opt.generate_scenario_tree(projs, target_gw=9)

    candidates = [
        {
            "plan_id": "roll",
            "label": "Roll",
            "transfers_in": [],
            "transfers_out": [],
            "expected_gain": 0.0,
            "hit_cost": 0,
            "trajectory": [
                {"gameweek": 9, "starters": [1, 2], "captain": 1},
                {"gameweek": 10, "starters": [1, 2], "captain": 1},
            ],
        },
        {
            "plan_id": "transfer_in_2",
            "label": "Bring in 2",
            "transfers_in": [2],
            "transfers_out": [1],
            "expected_gain": 1.5,
            "hit_cost": 0,
            "trajectory": [
                {"gameweek": 9, "starters": [2, 2], "captain": 2},
                {"gameweek": 10, "starters": [2, 2], "captain": 2},
            ],
        },
    ]

    evals = opt.evaluate_candidates(candidates, scenarios, target_gw=9)
    eval_map = {e.plan_id: e for e in evals}

    # Roll plan should have 0.0 utility across all scenarios and remain VIABLE
    assert eval_map["roll"].expected_stochastic_utility == 0.0
    assert eval_map["roll"].recommendation_tier == "VIABLE"

    # Robust transfer plan should maintain positive expected utility and not be FRAGILE_SUBOPTIMAL
    assert eval_map["transfer_in_2"].expected_stochastic_utility > 0.0
    assert eval_map["transfer_in_2"].recommendation_tier in ("ROBUST_DOMINANT", "VIABLE")
