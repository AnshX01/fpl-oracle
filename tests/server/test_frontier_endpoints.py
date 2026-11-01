"""
Integration tests for the 2 Peak Frontier API Endpoints:
- /api/market/price_radar
- /api/optimise/pareto
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
from fastapi.testclient import TestClient

from fpl_oracle.server import main

client = TestClient(main.app)


def test_market_price_radar_endpoint():
    """Verify market price radar returns microstructure metrics."""
    resp = client.get("/api/market/price_radar?elements=1,2,3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "total_tracked" in data
    assert "radar" in data
    assert "urgent_rises" in data
    assert "urgent_falls" in data


def test_pareto_frontier_endpoint():
    """Verify pareto frontier endpoint returns calibrated risk profile points."""
    mock_proj = {
        9: pd.DataFrame([{"element": 1, "expected_points": 5.0, "value": 50.0}]),
        10: pd.DataFrame([{"element": 1, "expected_points": 5.0, "value": 50.0}]),
        11: pd.DataFrame([{"element": 1, "expected_points": 5.0, "value": 50.0}]),
    }
    with patch("fpl_oracle.server.routes.api.analysis_service.projections", new=AsyncMock(return_value=mock_proj)):
        resp = client.get("/api/optimise/pareto?gameweek=9")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["frontier"]) == 4
        profiles = [p["strategy_profile"] for p in data["frontier"]]
        assert "MAX_UPSIDE" in profiles
        assert "DEFENSIVE_PRESERVATION" in profiles


def test_elite_ownership_endpoint():
    """Verify elite ownership endpoint returns calibrated Top-10k EO metrics."""
    mock_proj = {
        9: pd.DataFrame(
            [
                {"element": 1, "expected_points": 6.0, "value": 100.0},
                {"element": 2, "expected_points": 2.0, "value": 50.0},
            ]
        )
    }
    with patch("fpl_oracle.server.routes.api.analysis_service.projections", new=AsyncMock(return_value=mock_proj)):
        resp = client.get("/api/league/elite_ownership?gameweek=9")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["gameweek"] == 9
        assert "total_tracked" in data
        assert "elite_core" in data
        assert "elite_differentials" in data
        assert "zombie_traps" in data
        assert "assets" in data


def test_what_if_endpoint():
    """Verify live what-if simulation endpoint returns recalculated ranks."""
    payload = {"events": [{"element": 1, "goals": 1, "bonus": 3}]}
    resp = client.post("/api/matchday/what_if", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "result" in data
    assert "user_rank_before" in data["result"]
    assert "user_rank_after" in data["result"]
    assert "summary_narrative" in data["result"]


def test_trajectory_endpoint():
    """Verify visual trajectory roadmap endpoint returns structured multi-GW steps."""
    mock_proj = {
        9: pd.DataFrame([{"element": i, "expected_points": 5.0, "value": 50.0} for i in range(1, 16)]),
        10: pd.DataFrame([{"element": i, "expected_points": 5.0, "value": 50.0} for i in range(1, 16)]),
    }
    mock_joint = {
        "recommended_plan": {
            "trajectory": [
                {
                    "gameweek": 9,
                    "bank_end": 5,
                    "free_transfers_available": 1,
                    "free_transfers_next": 2,
                    "hit_cost": 0,
                    "active_chip": None,
                    "starters": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    "bench": [12, 13, 14, 15],
                    "captain": 10,
                    "vice_captain": 6,
                    "transfers_in": [],
                    "transfers_out": [],
                }
            ]
        }
    }
    with (
        patch("fpl_oracle.server.routes.api.analysis_service.projections", new=AsyncMock(return_value=mock_proj)),
        patch("fpl_oracle.server.routes.api.analysis_service.joint_plan", new=AsyncMock(return_value=mock_joint)),
    ):
        resp = client.get("/api/optimise/trajectory?gameweek=9&horizon=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "board" in data
        assert data["board"]["start_gw"] == 9
        assert len(data["board"]["steps"]) == 1
        assert isinstance(data["board"]["steps"][0]["formation"], str)
        assert len(data["board"]["steps"][0]["formation"]) >= 5


def test_covariance_endpoint():
    """Verify Markowitz portfolio covariance endpoint returns matrix and risk metrics."""
    mock_proj = {
        9: pd.DataFrame([{"element": i, "expected_points": 5.0, "value": 50.0} for i in range(1, 16)]),
    }
    with patch("fpl_oracle.server.routes.api.analysis_service.projections", new=AsyncMock(return_value=mock_proj)):
        resp = client.get("/api/optimise/covariance?gameweek=9")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "covariance" in data
        cov = data["covariance"]
        assert "player_metrics" in cov
        assert "covariance_matrix" in cov
        assert "correlation_matrix" in cov
        assert "portfolio_expected_points" in cov
        assert "portfolio_volatility" in cov
        assert "sharpe_ratio" in cov
        assert "portfolio_type" in cov
        assert cov["portfolio_type"] in ("CONCENTRATED", "BALANCED", "HEDGED")
