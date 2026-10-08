import asyncio
from unittest.mock import AsyncMock, patch

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
            {"chip": "Wildcard", "code": "wildcard", "recommended_gw": 6, "expected_gain": 12.5},
            {"chip": "Triple Captain", "code": "3xc", "recommended_gw": 8, "expected_gain": 5.0},
        ],
        "chip_measurement_status": "measured_through_expiry",
        "recommended_chip": "wildcard",
        "chip_comparison_table": [],
    }
    result = analysis_service.bind_chip_schedule({"chip_plan_table": [{"recommended_gw": 5}]}, joint)
    assert [r["recommended_gw"] for r in result["chip_plan_table"]] == [6, 8]
    assert result["chip_plan_table"][0]["expected_gain"] == 12.5


def test_empty_chat_provider_falls_back_and_persists_nonempty():
    async def run():
        provider = type("Provider", (), {"chat": AsyncMock(return_value="")})()
        with (
            patch("fpl_oracle.llm.agent.get_llm_provider", return_value=provider),
            patch("fpl_oracle.llm.agent.data_store.add_chat_message") as save,
            patch("fpl_oracle.llm.agent.data_store.get_chat_history", return_value=[]),
            patch("fpl_oracle.llm.provider.OfflineExpertProvider.chat", new=AsyncMock(return_value="Captain Saka.")),
        ):
            assert await ExpertAgent().answer("captain?") == "Captain Saka."
            assert save.call_args.kwargs["content"] == "Captain Saka."

    asyncio.run(run())


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
