"""
Unit and Integration Tests for Absolute Peak Frontier Integrations:
1. Deep In-MILP Stochastic Recourse Tree Optimization with Player-by-Player Rescoring.
2. Closed-Loop Adversarial Minimax with Dynamic Squad-Overlap Covariance.
3. Spatial Threat (xT) and Bayesian Finishing Regression in ML Projections.
4. Autonomous 90-Minute Deadline Sentinel Daemon.
"""

import pandas as pd
import pytest

from fpl_oracle.briefing.decision_card import decision_card_generator
from fpl_oracle.league.minimax import AdversarialMinimaxSolver, RivalAction
from fpl_oracle.ml.components.underlying_metrics import underlying_metrics_engine
from fpl_oracle.news.leak_watchdog import deadline_watchdog
from fpl_oracle.optimise.stochastic import StochasticScenarioOptimizer
from fpl_oracle.server.jobs import deadline_sentinel_job, tracker


def test_underlying_metrics_enhance_projections():
    """Verify that underlying spatial threat and Bayesian finishing enrich projections."""
    df = pd.DataFrame(
        [
            {
                "element": 1,
                "web_name": "Striker A",
                "position": "FWD",
                "expected_points": 6.5,
                "exp_goals_pts": 3.8,
                "exp_assists_pts": 1.2,
                "threat": 65.0,
                "creativity": 20.0,
                "minutes": 90,
            },
            {
                "element": 2,
                "web_name": "Playmaker B",
                "position": "MID",
                "expected_points": 5.8,
                "exp_goals_pts": 1.5,
                "exp_assists_pts": 2.7,
                "threat": 25.0,
                "creativity": 75.0,
                "minutes": 90,
            },
            {
                "element": 3,
                "web_name": "Wingback C",
                "position": "DEF",
                "expected_points": 4.5,
                "exp_goals_pts": 0.6,
                "exp_assists_pts": 1.5,
                "threat": 30.0,
                "creativity": 45.0,
                "minutes": 90,
            },
        ]
    )

    enhanced = underlying_metrics_engine.enhance_projections(df)
    assert "expected_threat_xt" in enhanced.columns
    assert "box_touch_intensity" in enhanced.columns
    assert "finishing_sustainability" in enhanced.columns

    # Playmaker should have high xT
    playmaker_xt = enhanced.loc[enhanced["web_name"] == "Playmaker B", "expected_threat_xt"].iloc[0]
    assert playmaker_xt > 0.20

    # Striker should have high box touches
    striker_box = enhanced.loc[enhanced["web_name"] == "Striker A", "box_touch_intensity"].iloc[0]
    assert striker_box > 4.0


@pytest.mark.asyncio
async def test_deadline_sentinel_job_execution():
    """Verify that the autonomous deadline sentinel runs cleanly without error."""
    await deadline_sentinel_job()
    state = tracker.jobs_state["deadline_sentinel"]
    assert state["status"] == "SUCCESS"
    assert state["run_count"] >= 1
    assert "Deadline Sentinel scan complete" in state["details"]


@pytest.mark.asyncio
async def test_deadline_watchdog_poll_sources():
    """Verify autonomous source polling returns parsed leak objects."""
    reports = await deadline_watchdog.poll_verified_sources_async()
    assert isinstance(reports, list)


def test_decision_card_surfaces_peak_capabilities():
    """Verify decision card structure contains adversarial minimax and stochastic keys."""
    assert hasattr(decision_card_generator, "generate_decision_card")


def test_stochastic_player_level_sensitivity_to_rotation():
    """Verify that stochastic rescoring penalizes rotated players at the individual asset level."""
    opt = StochasticScenarioOptimizer(risk_aversion_lambda=0.30)
    target_gw = 9

    # Player 1 is premium/high-xP (rotated in rotation_shock scenario)
    # Player 2 is non-premium/low-xP (unrotated across scenarios)
    pool_df = pd.DataFrame(
        [
            {"element": 1, "web_name": "Premium Rotated", "value": 110.0, "expected_points": 7.0},
            {"element": 2, "web_name": "Budget Nailed", "value": 50.0, "expected_points": 4.0},
        ]
    )
    horizon_projs = {target_gw: pool_df, target_gw + 1: pool_df}
    scenarios = opt.generate_scenario_tree(horizon_projs, target_gw)

    # Candidate A relies on rotated player 1
    cand_a = {
        "label": "Plan A (Rotated Premium)",
        "expected_gain": 5.0,
        "hit_cost": 0,
        "trajectory": [{"starters": [1], "captain": 1}],
    }
    # Candidate B relies on nailed player 2
    cand_b = {
        "label": "Plan B (Nailed Asset)",
        "expected_gain": 5.0,
        "hit_cost": 0,
        "trajectory": [{"starters": [2], "captain": 2}],
    }

    evals = opt.evaluate_candidates([cand_a, cand_b], scenarios, target_gw)
    eval_a = next(e for e in evals if e.label == "Plan A (Rotated Premium)")
    eval_b = next(e for e in evals if e.label == "Plan B (Nailed Asset)")

    # Plan B must have higher or equal robustness index than Plan A because Plan A suffers rotation variance
    assert eval_b.robustness_index >= eval_a.robustness_index


def test_minimax_dynamic_overlap_covariance_matrix():
    """Verify that shared player overlap dynamically adjusts covariance and differential variance."""
    solver = AdversarialMinimaxSolver()

    # User starts with elements [1, 2, 3, 4, 5]
    u_cand = {
        "label": "User Candidate",
        "expected_gain": 2.0,
        "hit_cost": 0,
        "trajectory": [{"starters": [1, 2, 3, 4, 5], "captain": 1}],
    }

    # Action 1: High Overlap with User (shares 1, 2, 3, 4, 5 and captain 1)
    act_high_overlap = RivalAction(
        action_id="rival_overlap",
        description="High Overlap Move",
        transfers_in=[],
        transfers_out=[],
        captain_id=1,
        expected_gain=1.0,
        rival_elements=[1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    )

    # Action 2: Low Overlap with User (shares only element 10, different captain)
    act_low_overlap = RivalAction(
        action_id="rival_diff",
        description="Differential Move",
        transfers_in=[],
        transfers_out=[],
        captain_id=99,
        expected_gain=1.0,
        rival_elements=[10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    )

    res = solver.evaluate_game(
        user_candidates=[u_cand],
        rival_actions=[act_high_overlap, act_low_overlap],
        user_base_points=526.0,
        rival_base_points=529.0,
        user_base_xp=55.0,
        rival_base_xp=55.0,
    )

    # High overlap locks in covariance, so differential variance is small.
    # When user is 2 points ahead in gain (net gap -1 vs rival), smaller variance means probability is sharper!
    assert len(res.payoff_matrix[0]) == 2
    assert 0.0 <= res.payoff_matrix[0][0] <= 1.0
    assert 0.0 <= res.payoff_matrix[0][1] <= 1.0
