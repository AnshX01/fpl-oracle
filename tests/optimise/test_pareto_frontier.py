"""
Tests for Pareto Frontier Risk Selector (Peak Frontier Pillar 4).
"""

import pandas as pd

from fpl_oracle.optimise.stochastic import StochasticScenario, stochastic_optimizer


def test_generate_pareto_frontier():
    # 2 Scenarios: Nominal and Rotation Shock
    df_nom = pd.DataFrame(
        [
            {"element": 1, "expected_points": 7.0},
            {"element": 2, "expected_points": 5.0},
        ]
    )
    df_rot = pd.DataFrame(
        [
            {"element": 1, "expected_points": 2.0},  # Volatile premium
            {"element": 2, "expected_points": 4.8},  # Ultra-steady talisman
        ]
    )

    scenarios = [
        StochasticScenario(name="nominal", probability=0.5, description="Nominal", scenario_projections={10: df_nom}),
        StochasticScenario(
            name="rotation_shock", probability=0.5, description="Rotation", scenario_projections={10: df_rot}
        ),
    ]

    # Plan 1: High variance (plays element 1)
    # Plan 2: Low variance / high floor (plays element 2)
    candidate_plans = [
        {
            "plan_id": "Plan_Ceiling",
            "label": "High Ceiling Differential",
            "transfers_in": [1],
            "transfers_out": [],
            "hit_cost": 0,
            "starting_xi": [1],
            "captain_id": 1,
            "expected_gain": 7.0,
        },
        {
            "plan_id": "Plan_Floor",
            "label": "High Floor Shield",
            "transfers_in": [2],
            "transfers_out": [],
            "hit_cost": 0,
            "starting_xi": [2],
            "captain_id": 2,
            "expected_gain": 5.0,
        },
    ]

    frontier = stochastic_optimizer.generate_pareto_frontier(
        candidate_plans,
        scenarios,
        target_gw=10,
        lambda_grid=[-0.50, 0.00, 0.35, 1.00],
    )

    assert len(frontier) == 4
    profiles = {p.strategy_profile: p for p in frontier}

    # At MAX_UPSIDE (lambda = -0.50), high-variance ceiling plan should be favoured
    assert profiles["MAX_UPSIDE"].lambda_risk == -0.50
    assert profiles["MAX_UPSIDE"].strategy_profile == "MAX_UPSIDE"

    # At DEFENSIVE_PRESERVATION (lambda = 1.00), high floor plan should dominate
    assert profiles["DEFENSIVE_PRESERVATION"].lambda_risk == 1.00
    assert profiles["DEFENSIVE_PRESERVATION"].strategy_profile == "DEFENSIVE_PRESERVATION"
    assert profiles["DEFENSIVE_PRESERVATION"].best_plan_id == "Plan_Floor"
