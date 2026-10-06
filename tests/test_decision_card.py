"""
Tests for Per-Gameweek Unified Decision Card (D1, D2, D3).
Verifies:
- /api/decision-card endpoint schema completeness and contract
- /api/decision-card/export plain text markdown output
- Consistency with underlying transfer and lineup engines
- Graceful handling of unconfigured league and edge states
"""

import asyncio
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fpl_oracle.briefing.decision_card import decision_card_generator, format_decision_card_markdown
from fpl_oracle.server.routes.api import router as api_router


def test_decision_card_generator_contract():
    async def _run():
        card = await decision_card_generator.generate_decision_card()

        # Gameweek & Metadata
        assert "gameweek" in card
        assert isinstance(card["gameweek"], int)
        assert card["gameweek"] >= 1

        # Deadline
        assert "deadline" in card
        dl = card["deadline"]
        assert "deadline_time" in dl
        assert "seconds_to_deadline" in dl
        assert "label" in dl

        # Data freshness
        assert "data_freshness" in card
        df = card["data_freshness"]
        assert "projections_as_of" in df
        assert "news_as_of" in df
        assert "rivals_as_of" in df
        assert "data_as_of" in df

        # Model Version
        assert "model_version" in card
        assert isinstance(card["model_version"], str)

        # Transfers
        assert "transfers" in card
        t = card["transfers"]
        assert t["action"] in ["ROLL", "SINGLE_TRANSFER", "MULTIPLE_TRANSFERS"]
        assert "is_roll" in t
        assert "in" in t
        assert "out" in t
        assert "bank_after" in t
        assert "ft_used" in t
        assert "ft_remaining" in t
        assert "hit_cost" in t
        assert "net_gain_vs_roll" in t
        assert "no_regret_flag" in t
        assert "action_summary" in t

        # Starting XI
        assert "xi" in card
        assert len(card["xi"]) == 11
        captain_count = sum(1 for p in card["xi"] if p["is_captain"])
        vice_count = sum(1 for p in card["xi"] if p["is_vice_captain"])
        assert captain_count == 1
        assert vice_count == 1

        # Captain and Vice
        assert "captain" in card
        assert "vice_captain" in card
        assert card["captain"]["element"] != card["vice_captain"]["element"]

        # Bench
        assert "bench" in card
        assert len(card["bench"]) == 4

        # Formation
        assert "formation" in card
        assert len(card["formation"].split("-")) == 3

        # Chip
        assert "chip" in card
        chip = card["chip"]
        assert "recommend" in chip
        assert isinstance(chip["recommend"], bool)
        assert "reason" in chip
        assert "gain_vs_hold" in chip

        # Rivals & Win Prob
        assert "rivals" in card
        assert card["rivals"]["status"] in ["configured", "unconfigured"]
        assert "win_prob" in card
        assert "p_first" in card["win_prob"]

        # Two-line reasoning & caveats
        assert "two_line_reasoning" in card
        assert len(card["two_line_reasoning"]) > 10
        assert "caveats" in card
        assert isinstance(card["caveats"], list)
        assert "what_changed" in card

        # Disclaimer
        assert card["advice_disclaimer"] == "Advice only — nothing is submitted to FPL."

    asyncio.run(_run())


def test_decision_card_api_endpoints():
    mock_card = {
        "gameweek": 6,
        "deadline": {"label": "GW6 Deadline: Friday 18:30 UTC", "seconds_to_deadline": 7200.0, "is_live": False},
        "data_freshness": {
            "projections_as_of": "2026-10-06T12:00:00Z",
            "news_as_of": "2026-10-06T12:00:00Z",
            "rivals_as_of": "2026-10-06T12:00:00Z",
            "data_as_of": "2026-10-06T12:00:00Z",
        },
        "data_as_of": "2026-10-06T12:00:00Z",
        "model_version": "v2.1.0",
        "advice_disclaimer": "Advice only — nothing is submitted to FPL.",
        "chip": {
            "recommend": False,
            "chip_name": None,
            "chip_display_name": None,
            "gain_vs_hold": 0.0,
            "reason": "Save chips for Double Gameweek runs.",
            "next_best_window": 34,
            "set_1_deadline_warning": "Set 1 chips expire GW19.",
        },
        "transfers": {
            "is_roll": True,
            "action": "ROLL",
            "in": [],
            "out": [],
            "action_summary": "Roll transfer to bank 2 FTs.",
            "ft_used": 0,
            "ft_remaining": 2,
            "expected_gain_gw": 0.0,
            "net_gain_vs_roll": 0.0,
            "horizon_pts": 240.5,
            "hit_cost": 0,
            "hits_count": 0,
            "bank_after": 1.5,
            "no_regret_flag": True,
        },
        "formation": "3-5-2",
        "captain": {"element": 10, "web_name": "Haaland", "team_short": "MCI", "expected_points": 7.4, "p10": 4.0, "p90": 11.2},
        "vice_captain": {"element": 20, "web_name": "Salah", "team_short": "LIV", "expected_points": 6.8, "p10": 3.5, "p90": 10.5},
        "xi": [
            {"element": 10, "position": "FWD", "web_name": "Haaland", "team_short": "MCI", "expected_points": 7.4, "p10": 4.0, "p90": 11.2, "is_captain": True, "is_vice_captain": False},
            {"element": 20, "position": "MID", "web_name": "Salah", "team_short": "LIV", "expected_points": 6.8, "p10": 3.5, "p90": 10.5, "is_captain": False, "is_vice_captain": True},
        ] + [
            {"element": i, "position": "DEF", "web_name": f"Def_{i}", "team_short": "ARS", "expected_points": 4.5, "p10": 2.0, "p90": 7.0, "is_captain": False, "is_vice_captain": False}
            for i in range(30, 39)
        ],
        "bench": [
            {"element": i, "bench_order": idx, "web_name": f"Bench_{i}", "position": "MID", "team_short": "AVL", "expected_points": 3.2}
            for idx, i in enumerate(range(40, 44), start=1)
        ],
        "rivals": {
            "league_name": "Legends League",
            "status": "configured",
            "posture": "BALANCED_ATTACK",
            "posture_reason": "Maintain lead.",
            "exposure_players": ["Saka"],
            "differential_players": ["Diaz"],
            "nearest_above_gap": 0.0,
            "nearest_below_gap": 15.0,
            "rival_count": 6,
        },
        "win_prob": {
            "p_first": 24.5,
            "p_above_key_rivals": 78.0,
            "expected_rank": 2.1,
            "mc_se": 0.45,
            "simulation_note": "Joint Bernoulli Monte Carlo simulation.",
        },
        "two_line_reasoning": "Roll transfer banks flexibility. Haaland captaincy leads projections.",
        "caveats": ["Check Friday pressers."],
        "what_changed": "Refreshed team news.",
        "is_stale": False,
        "stale": False,
    }

    async def _run():
        test_app = FastAPI()
        test_app.include_router(api_router)

        with patch("fpl_oracle.briefing.decision_card.decision_card_generator.generate_decision_card", new=AsyncMock(return_value=mock_card)):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
                # 1. JSON endpoint (D1)
                resp = await ac.get("/api/decision-card")
                assert resp.status_code == 200
                data = resp.json()
                assert data["gameweek"] == 6
                assert data["transfers"]["action"] == "ROLL"
                assert len(data["xi"]) == 11
                assert data["captain"]["web_name"] == "Haaland"
                assert data["advice_disclaimer"] == "Advice only — nothing is submitted to FPL."

                # 2. Export Markdown endpoint (D2)
                export_resp = await ac.get("/api/decision-card/export")
                assert export_resp.status_code == 200
                assert "text/plain" in export_resp.headers["content-type"]
                text = export_resp.text
                assert "# FPL Oracle — Gameweek 6 Decision Card" in text
                assert "## 1. Chip Strategy" in text
                assert "## 2. Transfer Plan" in text
                assert "## 3. Starting XI & Captaincy" in text
                assert "## 4. Mini-League & Rivals" in text
                assert "Haaland" in text
                assert "Advice only — nothing is submitted to FPL." in text

    asyncio.run(_run())


def test_format_decision_card_markdown():
    mock_card = {
        "gameweek": 6,
        "deadline": {"label": "GW6 Deadline: Friday 18:30 UTC", "seconds_to_deadline": 7200.0},
        "data_as_of": "2026-10-06T12:00:00Z",
        "model_version": "v2.1.0",
        "advice_disclaimer": "Advice only — nothing is submitted to FPL.",
        "chip": {
            "recommend": False,
            "chip_name": None,
            "chip_display_name": None,
            "gain_vs_hold": 0.0,
            "reason": "Save chips for Double Gameweek runs.",
            "next_best_window": 34,
            "set_1_deadline_warning": "Set 1 chips expire GW19.",
        },
        "transfers": {
            "is_roll": True,
            "action_summary": "Roll transfer to bank 2 FTs.",
            "ft_remaining": 2,
            "expected_gain_gw": 0.0,
            "net_gain_vs_roll": 0.0,
            "hit_cost": 0,
            "hits_count": 0,
            "bank_after": 1.5,
            "no_regret_flag": True,
        },
        "formation": "3-5-2",
        "captain": {"web_name": "Haaland", "team_short": "MCI", "expected_points": 7.4, "p10": 4.0, "p90": 11.2},
        "vice_captain": {"web_name": "Salah", "team_short": "LIV", "expected_points": 6.8, "p10": 3.5, "p90": 10.5},
        "xi": [
            {"position": "FWD", "web_name": "Haaland", "team_short": "MCI", "expected_points": 7.4, "p10": 4.0, "p90": 11.2, "is_captain": True, "is_vice_captain": False},
            {"position": "MID", "web_name": "Salah", "team_short": "LIV", "expected_points": 6.8, "p10": 3.5, "p90": 10.5, "is_captain": False, "is_vice_captain": True},
        ],
        "bench": [
            {"bench_order": 1, "web_name": "Rogers", "position": "MID", "team_short": "AVL", "expected_points": 3.2}
        ],
        "rivals": {
            "league_name": "Legends League",
            "status": "configured",
            "posture": "BALANCED_ATTACK",
            "posture_reason": "Maintain lead.",
            "exposure_players": ["Saka"],
            "differential_players": ["Diaz"],
        },
        "win_prob": {
            "p_first": 24.5,
            "expected_rank": 2.1,
        },
        "two_line_reasoning": "Roll transfer banks flexibility. Haaland captaincy leads projections.",
        "caveats": ["Check Friday pressers."],
        "what_changed": "Refreshed team news.",
    }

    md = format_decision_card_markdown(mock_card)
    assert "# FPL Oracle — Gameweek 6 Decision Card" in md
    assert "Haaland" in md
    assert "Salah" in md
    assert "Set 1 chips expire GW19" in md
    assert "Legends League" in md


def test_decision_card_unmocked_happy_path():
    """Verify unmocked end-to-end decision card execution and honest metrics."""
    async def _run():
        card = await decision_card_generator.generate_decision_card()
        assert isinstance(card["gameweek"], int)
        assert card["gameweek"] >= 1
        assert "transfers" in card
        assert "xi" in card
        assert len(card["xi"]) == 11
        # Honest win prob reporting: not a silent fallback
        wp = card["win_prob"]
        assert wp["status"] in ["simulated", "unconfigured", "error"]
        if wp["status"] == "simulated":
            assert wp["p_first"] is not None
            assert wp["expected_rank"] is not None
        elif wp["status"] == "error":
            assert wp["p_first"] is None
            assert wp["expected_rank"] is None
            assert "error_message" in wp
        elif wp["status"] == "unconfigured":
            assert wp["p_first"] is None
            assert wp["expected_rank"] is None

    asyncio.run(_run())


def test_decision_card_error_path_honest_reporting():
    """Verify that Monte Carlo exceptions emit status='error' with null metrics rather than swallowing."""
    async def _run():
        with patch(
            "fpl_oracle.league.standings.league_standings_manager.get_league_standings",
            new=AsyncMock(return_value={"league_name": "Test League", "standings": [{"entry": 1, "rank": 1, "total": 500}]}),
        ), patch(
            "fpl_oracle.league.rivals.rival_analyzer.analyze_rivals",
            new=AsyncMock(return_value={"rival_squads": [{"entry_id": 2, "player_name": "Rival Leader", "rank": 1, "total_points": 490, "squad": []}], "template_players": [], "differential_players": []}),
        ), patch(
            "fpl_oracle.league.montecarlo.monte_carlo_simulator.simulate_league",
            side_effect=RuntimeError("Simulated Monte Carlo numerical breakdown"),
        ):
            card = await decision_card_generator.generate_decision_card()
            wp = card["win_prob"]
            assert wp["status"] == "error"
            assert wp["p_first"] is None, "Failed simulation must emit p_first=None, not 0.0"
            assert wp["expected_rank"] is None, "Failed simulation must emit expected_rank=None, not 1.0"
            assert "Simulated Monte Carlo numerical breakdown" in wp["error_message"]

    asyncio.run(_run())

