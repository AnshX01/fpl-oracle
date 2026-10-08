import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx

from fpl_oracle.utils.logging import SecretRedactingFilter


def test_basic_values_are_millions_and_keep_published_captain(configured_advisor):
    state, _, _ = configured_advisor
    from fpl_oracle.server.routes.api import get_basic_squad

    result = asyncio.run(get_basic_squad())
    assert len(result["starters"]) == 11 and len(result["bench"]) == 4
    assert result["published_gameweek"] == state.current_gw
    assert result["team_confirmed"] is False
    assert result["total_squad_value"] == round(sum(p.now_cost for p in state.squad) / 10, 1)
    assert result["total_selling_value"] == round(sum(p.selling_price for p in state.squad) / 10, 1)
    assert result["starters"][0]["now_cost"] == state.squad[0].now_cost / 10
    assert result["captain"]["element"] == state.squad[0].element


def test_secret_redaction_formats_url_and_exception(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-secret-credential")
    record = logging.LogRecord(
        "httpx", logging.INFO, "", 0, "POST %s", ("https://example.test/?key=mock-secret-credential&other=ok",), None
    )
    assert SecretRedactingFilter().filter(record)
    assert "mock-secret" not in record.getMessage()
    assert "key=[REDACTED]&other=ok" in record.getMessage()
    try:
        raise ValueError("https://example.test/?api_key=other-secret")
    except ValueError:
        import sys

        record = logging.LogRecord("app", logging.ERROR, "", 0, "failed", (), sys.exc_info())
    SecretRedactingFilter().filter(record)
    assert "other-secret" not in record.getMessage()
    assert "ValueError" in record.getMessage()


def test_news_gemini_key_is_header_not_url(monkeypatch):
    from fpl_oracle.news.gemini_extractor import GeminiEvidenceExtractor

    monkeypatch.setenv("GEMINI_API_KEY", "mock-secret-credential")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")
    extractor = GeminiEvidenceExtractor()
    response = httpx.Response(200, json={"candidates": []})
    post = AsyncMock(return_value=response)
    with (
        patch.object(extractor.budget, "check_and_increment", return_value=(True, "ok")),
        patch("httpx.AsyncClient.post", post),
    ):
        asyncio.run(extractor.extract_evidence_from_text("Player trained today.", "https://example.test/news"))
    assert "key=" not in post.call_args.args[0]
    assert "mock-secret" not in post.call_args.args[0]
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "mock-secret-credential"}


def test_update_after_done_does_not_call_news_or_predictions(configured_advisor, monkeypatch):
    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.news.analyse import news_analyzer
    from fpl_oracle.server.analysis import analysis_service
    from fpl_oracle.server.pipeline import SyncPipeline

    monkeypatch.setattr(team_confirmation, "locked", lambda state: True)
    news = AsyncMock()
    monkeypatch.setattr(news_analyzer, "get_player_news_signals", news)
    projections = AsyncMock()
    monkeypatch.setattr(analysis_service, "projections", projections)
    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    pipeline = SyncPipeline()
    asyncio.run(pipeline.run_pipeline())
    news.assert_not_awaited()
    projections.assert_not_awaited()
    assert pipeline.get_status()["last_error"] is None
    assert pipeline.get_status()["summary"]["status"] == "followed"


def test_update_team_endpoint_has_player_roster(configured_advisor, monkeypatch):
    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.routes.api import get_team_confirmation

    monkeypatch.setattr(team_confirmation, "read", lambda mid: None)
    monkeypatch.setattr(team_confirmation, "recommendation", lambda mid: None)
    result = asyncio.run(get_team_confirmation())
    roster = {p["element"]: p for p in result["players"]}
    assert all(p.element in roster for p in configured_advisor[0].squad)
    assert all(p["name"] and p["price"] > 0 for p in roster.values())
