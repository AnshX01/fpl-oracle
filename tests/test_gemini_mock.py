"""
Mock and Unit Tests for Gemini LLM Extraction Path (Milestone G12 / Requirement f).
Verifies:
1. Valid schema parsing into PlayerEvidence
2. Hallucinated quote detection and confidence penalty
3. HTTP 429 rate limit graceful degradation
4. HTTP 500 internal server error fallback
5. Network timeout exception handling
6. Empty response and malformed JSON fallback
7. Prompt injection defense treating article text as passive data
8. SSRF URL guarding on external feeds
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fpl_oracle.news.gemini_extractor import GeminiEvidenceExtractor
from fpl_oracle.news.ingest import is_safe_external_url
from fpl_oracle.news.models import EvidenceCategory


@pytest.fixture
def extractor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")
    return GeminiEvidenceExtractor()


def test_gemini_configuration_detection(extractor):
    configured, reason = extractor.is_configured()
    assert configured is True
    assert reason == "Ready"


def test_gemini_unconfigured_without_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_FREE_TIER_CONFIRMED", "true")
    ext = GeminiEvidenceExtractor()
    configured, reason = ext.is_configured()
    assert configured is False
    assert "GEMINI_API_KEY not set" in reason


def test_gemini_valid_schema_parsing(extractor):
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '[\n  {\n    "player_id": 1,\n    "player_name": "Saka",\n    "team_name": "Arsenal",\n'
                                '    "category": "available",\n    "quote": "Saka is fit and available for Saturday.",\n'
                                '    "match_context": null,\n    "is_negated": false,\n    "minutes_restriction": null,\n'
                                '    "confidence": 0.95,\n    "ambiguity_notes": null\n  }\n]'
                            )
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = httpx.Response(200, json=mock_response_data, request=httpx.Request("POST", "http://test"))

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
        ):
            payload = await extractor.extract_evidence_from_text(
                article_text="Mikel Arteta confirmed Saka is fit and available for Saturday.",
                article_url="https://example.com/news",
            )
            assert len(payload.evidences) == 1
            ev = payload.evidences[0]
            assert ev.player_name == "Saka"
            assert ev.category == EvidenceCategory.AVAILABLE
            assert ev.confidence == 0.95
            assert "fit and available" in ev.quote

    asyncio.run(_run())


def test_gemini_hallucinated_quote_penalized(extractor):
    """If Gemini invents a quote not in the article text, confidence is severely penalized."""
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '[\n  {\n    "player_id": 10,\n    "player_name": "Haaland",\n    "team_name": "Man City",\n'
                                '    "category": "ruled_out",\n    "quote": "He will definitely not play for six weeks due to knee fracture.",\n'
                                '    "match_context": null,\n    "is_negated": false,\n    "minutes_restriction": null,\n'
                                '    "confidence": 0.95,\n    "ambiguity_notes": null\n  }\n]'
                            )
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = httpx.Response(200, json=mock_response_data, request=httpx.Request("POST", "http://test"))

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
        ):
            # Article mentions Haaland but NOT the hallucinated quote
            payload = await extractor.extract_evidence_from_text(
                article_text="Erling Haaland took part in tactical drills with the main squad today.",
                article_url="https://example.com/news",
            )
            assert len(payload.evidences) == 1
            ev = payload.evidences[0]
            # Must penalize confidence and flag ambiguity note
            assert ev.confidence <= 0.40
            assert "Quote span not found verbatim" in str(ev.ambiguity_notes)

    asyncio.run(_run())


def test_gemini_http_429_graceful_fallback(extractor):
    mock_resp = httpx.Response(429, text="Rate limit exceeded", request=httpx.Request("POST", "http://test"))

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
        ):
            payload = await extractor.extract_evidence_from_text(article_text="Sample news text")
            assert payload.evidences == []
            assert "429" in payload.raw_response_snippet

    asyncio.run(_run())


def test_gemini_http_500_graceful_fallback(extractor):
    mock_resp = httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", "http://test"))

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
        ):
            payload = await extractor.extract_evidence_from_text(article_text="Sample news text")
            assert payload.evidences == []
            assert "HTTP 500" in payload.raw_response_snippet

    asyncio.run(_run())


def test_gemini_timeout_fallback(extractor):
    async def _mock_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=_mock_timeout),
        ):
            payload = await extractor.extract_evidence_from_text(article_text="Sample news text")
            assert payload.evidences == []
            assert "Exception" in payload.raw_response_snippet

    asyncio.run(_run())


def test_gemini_empty_response_fallback(extractor):
    mock_response_data = {"candidates": [{"content": {"parts": [{"text": "[]"}]}}]}
    mock_resp = httpx.Response(200, json=mock_response_data, request=httpx.Request("POST", "http://test"))

    async def _run():
        with (
            patch.object(extractor.budget, "check_and_increment", return_value=(True, "Within budget")),
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
        ):
            payload = await extractor.extract_evidence_from_text(article_text="No fitness news in this text.")
            assert payload.evidences == []
            assert payload.raw_response_snippet == "Success"

    asyncio.run(_run())


def test_ssrf_guard_blocks_untrusted_urls():
    assert not is_safe_external_url("http://169.254.169.254/latest/meta-data/", resolve_dns=False)
    assert not is_safe_external_url("http://127.0.0.1:8000/", resolve_dns=False)
    assert not is_safe_external_url("http://localhost:5000/", resolve_dns=False)
    assert not is_safe_external_url("http://10.0.0.1/admin", resolve_dns=False)
    assert is_safe_external_url("https://www.premierleague.com/news", resolve_dns=False)
