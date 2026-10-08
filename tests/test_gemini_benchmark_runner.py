import asyncio
import json
from types import SimpleNamespace

import pytest

from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from scripts import benchmark_gemini as runner
from scripts.run_news_benchmark import run_benchmark


def test_unconfigured_and_failed_calls_never_publish(monkeypatch):
    class Extractor:
        def is_configured(self):
            return False, "Missing key"

    with pytest.raises(RuntimeError, match="Missing key"):
        asyncio.run(runner.measure(Extractor()))

    class Failed:
        def is_configured(self):
            return True, "Ready"

        async def extract_evidence_from_text(self, **kwargs):
            return SimpleNamespace(raw_response_snippet="HTTP 429", evidences=[])

    monkeypatch.setattr(
        runner, "HELD_OUT_BENCHMARK_CASES", [{"id": "x", "text": "a", "player_id": 1, "player_name": "A"}]
    )
    with pytest.raises(RuntimeError, match="No active artifact written"):
        asyncio.run(runner.measure(Failed()))


def test_publish_measured_model_preserves_fallback_and_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    (tmp_path / "news_benchmark.json").write_text(
        json.dumps({"extractors": {"deterministic_fallback": {"active": True}}})
    )
    report = {
        "production_gate": {"gate_passed": False},
        "benchmark_timestamp": "2026-10-08T00:00:00Z",
        "cases": [],
        "extractors": {},
    }
    result = runner.publish(report)
    assert result["active"] is False
    assert result["model"] == "test-model"
    written = json.loads((tmp_path / "news_benchmark.json").read_text())
    assert written["extractors"]["deterministic_fallback"]["active"]
    assert not written["extractors"]["gemini"]["active"]


def test_benchmark_category_errors_are_false_negatives_too(monkeypatch):
    from scripts import run_news_benchmark as script

    cases = [
        dict(
            id="x",
            description="category mismatch",
            group="general",
            text="A trained",
            player_name="A",
            player_id=1,
            expected_category=EvidenceCategory.RULED_OUT,
        )
    ]
    monkeypatch.setattr(script, "HELD_OUT_BENCHMARK_CASES", cases)
    class Reconciler:
        def __init__(self,**kwargs):pass
        def reconcile_player_fixture(self,*args,**kwargs):return SimpleNamespace(effective_chance_of_playing=100)
    monkeypatch.setattr(script,"AvailabilityReconciler",Reconciler)
    report = run_benchmark(
        {"x": [PlayerEvidence(player_id=1, player_name="A", category=EvidenceCategory.AVAILABLE, quote="A trained")]},
        write_reports=False,
    )
    assert report["confusion_matrix"]["false_positives"] == 1
    assert report["confusion_matrix"]["false_negatives"] == 1
    assert not report["production_gate"]["gate_passed"]
