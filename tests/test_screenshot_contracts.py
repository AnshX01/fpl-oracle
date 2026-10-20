import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from fpl_oracle.briefing.decision_card import decision_card_generator
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.llm.agent import ExpertAgent
from fpl_oracle.server.analysis import analysis_service
from fpl_oracle.server.routes.api import get_contingency_matrix, get_pre_deadline_checklist, get_squad


def test_chip_schedule_comes_from_selected_trajectory():
    joint = {
        "recommended_plan": {
            "trajectory": [
                {"gameweek": 6, "chip": "wildcard"},
                {"gameweek": 7, "chip": None},
                {"gameweek": 8, "chip": "3xc"},
            ]
        },
        "measured_chip_roadmap": [
            {"chip": "Wildcard", "code": "wildcard", "recommended_gw": 6, "expected_gain": None, "status": "planned"},
            {"chip": "Triple Captain", "code": "3xc", "recommended_gw": 8, "expected_gain": None, "status": "planned"},
        ],
        "chip_measurement_status": "planned",
        "recommended_chip": "wildcard",
        "chip_comparison_table": [],
    }
    result = analysis_service.bind_chip_schedule({"chip_plan_table": [{"recommended_gw": 5}]}, joint)
    assert [r["recommended_gw"] for r in result["chip_plan_table"]] == [6, 8]
    assert all(r["status"] == "planned" and r["expected_gain"] is None for r in result["chip_plan_table"])
    assert result["schedule_status"] == "planned"


def test_chip_roadmap_renders_without_none(monkeypatch):
    from io import StringIO

    from rich.console import Console

    from fpl_oracle import cli
    from fpl_oracle.server.routes import api

    rows = [{"chip": "Wildcard", "code": "wildcard", "recommended_gw": 6, "expected_gain": None, "status": "planned"}]
    monkeypatch.setattr(api, "get_chip_strategy", AsyncMock(return_value={"chip_plan_table": rows}))
    out = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=100))
    cli.chips()
    assert "None" not in out.getvalue() and "Gameweek 6" in out.getvalue() and "n/a" not in out.getvalue()


@pytest.mark.slow
def test_card_payload_for_the_screens(configured_advisor):
    card = asyncio.run(decision_card_generator.generate_decision_card())
    cap = card["captain"]
    assert cap["reason"] and cap["as_captain_points"] == round(cap["multiplier"] * cap["expected_points"], 1)
    assert "bench_note" in card and "minutes_note" in card
    assert all("gain" not in later for later in card["chip"]["later"])
    assert "None" not in str(card["chip"]["later"]) + str(card["chip"]["reason"])
    assert card["chip"]["chip_name"] != "3xc" or cap["multiplier"] == 3


def test_empty_chat_provider_falls_back_and_persists_nonempty():
    async def run():
        provider = type("Provider", (), {"chat": AsyncMock(return_value="")})()
        with (
            patch("fpl_oracle.llm.agent.get_llm_provider", return_value=provider),
            patch("fpl_oracle.llm.agent.data_store.add_chat_message") as save,
            patch("fpl_oracle.llm.agent.data_store.get_chat_history", return_value=[]),
            patch("fpl_oracle.llm.provider.OfflineExpertProvider.chat", new=AsyncMock(return_value="Captain Saka.")),
        ):
            assert (await ExpertAgent().answer("captain?")).endswith("Captain Saka.")
            assert save.call_args.kwargs["content"].endswith("Captain Saka.")

    asyncio.run(run())


@pytest.mark.slow
def test_auxiliary_team_matches_selected_pitch(configured_advisor):
    async def run():
        with patch("fpl_oracle.news.ingest.news_ingestion.fetch_rss_articles", new=AsyncMock(return_value=[])):
            squad = await get_squad()
            matrix = await get_contingency_matrix()
            checklist = await get_pre_deadline_checklist()
            briefing = await weekly_briefing_generator.generate_briefing()
        assert {r["element"] for r in matrix["contingency_matrix"]} == {r["element"] for r in squad["starters"]}
        assert all("emergency_net_expected_points" in r for r in matrix["contingency_matrix"])
        vc = next(r for r in checklist["checklist"] if r["item"] == "Vice-Captain Failsafe")
        assert squad["vice_captain"]["web_name"] in vc["detail"]
        assert vc["status"] in ("PASS", "WARNING")
        assert "Imminent Rises" not in briefing["markdown"]
        assert matrix["scope"] == "selected_plan_not_submitted"

    asyncio.run(run())


def test_review_does_not_invent_calibration_or_autosub_evidence(configured_advisor):
    from fpl_oracle.briefing.review import post_gameweek_reviewer

    review = asyncio.run(post_gameweek_reviewer.generate_gameweek_review())
    assert "Variance remains within calibrated" not in review["review_markdown"]
    assert "Autosub and bench hierarchy operated" not in review["review_markdown"]
    assert "Prediction comparison unavailable" in review["review_markdown"]


def test_recent_chat_history_returns_newest_window(configured_advisor):
    import uuid

    from fpl_oracle.data.store import data_store

    session = str(uuid.uuid4())
    for i in range(25):
        data_store.add_chat_message(role="user", content=str(i), session_id=session)
    messages = data_store.get_chat_history(session_id=session, limit=20)
    assert [m["content"] for m in messages] == [str(i) for i in range(5, 25)]
