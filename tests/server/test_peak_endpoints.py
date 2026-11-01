"""
Integration tests for the 4 Frontier Peak API Endpoints:
- /api/league/minimax
- /api/optimise/stochastic
- /api/ml/underlying
- /api/news/leak_watchdog and /api/news/leak_watchdog/ingest
"""

import pytest
from fastapi.testclient import TestClient

from fpl_oracle.server import main

client = TestClient(main.app)

pytestmark = pytest.mark.slow


def test_leak_watchdog_ingest_and_status():
    """Ingest a leak and verify it appears in status."""
    # 1. Ingest
    resp = client.post(
        "/api/news/leak_watchdog/ingest",
        json={
            "raw_text": "Exclusive: Striker Test benched for today's match.",
            "source": "verified_itk",
            "confidence": 0.95,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingested"
    assert data["leaks_parsed"] == 1
    assert data["reports"][0]["status"] == "BENCHED"

    # 2. Status
    status_resp = client.get("/api/news/leak_watchdog")
    assert status_resp.status_code == 200
    s_data = status_resp.json()
    assert s_data["status"] == "active"
    assert s_data["total_leaks_recorded"] >= 1


def test_stochastic_endpoint():
    """Verify stochastic scenario tree endpoint returns 3 calibrated scenarios."""
    resp = client.get("/api/optimise/stochastic")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["scenarios"]) == 3
    assert len(data["evaluations"]) == 3
    assert any(e["label"].startswith("Roll") for e in data["evaluations"])


def test_underlying_metrics_endpoint():
    """Verify underlying metrics endpoint returns xT and spatial profiles."""
    resp = client.get("/api/ml/underlying")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "top_underlying_creators" in data


def test_league_minimax_endpoint():
    """Verify adversarial minimax endpoint returns valid status."""
    resp = client.get("/api/league/minimax")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "unavailable")
