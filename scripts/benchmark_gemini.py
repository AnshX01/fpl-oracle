"""Measure actual Gemini responses; never substitute fallback responses."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from fpl_oracle.config import REPORTS_DIR
from fpl_oracle.news.gemini_extractor import DEFAULT_GEMINI_MODEL, GeminiEvidenceExtractor
from fpl_oracle.news.held_out_benchmark_cases import HELD_OUT_BENCHMARK_CASES

if __package__ in (None, ""):
    from run_news_benchmark import run_benchmark
else:
    from scripts.run_news_benchmark import run_benchmark


async def measure(extractor):
    configured, reason = extractor.is_configured()
    if not configured:
        raise RuntimeError(reason)
    evidence = {}
    for case in HELD_OUT_BENCHMARK_CASES:
        if case.get("group") == "ssrf":
            continue
        payload = await extractor.extract_evidence_from_text(
            article_text=case["text"],
            candidate_players=[{"id": case["player_id"], "web_name": case["player_name"], "team": 1}],
            target_gw=10,
        )
        if payload.raw_response_snippet != "Success":
            raise RuntimeError(
                f"{case['id']}: Gemini did not return a measured response ({payload.raw_response_snippet}). No active artifact written."
            )
        evidence[case["id"]] = [ev for ev in payload.evidences if ev.player_id == case["player_id"]]
        await asyncio.sleep(4)
    report = run_benchmark(evidence_by_case=evidence, write_reports=False)
    return report


def publish(report):
    path = REPORTS_DIR / "news_benchmark.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    summary = {k: v for k, v in report.items() if k not in ("extractors", "cases")}
    summary.update(
        active=report["production_gate"]["gate_passed"],
        status="measured",
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        measured_at=report["benchmark_timestamp"],
        cases_sha256=hashlib.sha256(json.dumps(HELD_OUT_BENCHMARK_CASES, sort_keys=True).encode()).hexdigest(),
    )
    prior.setdefault("extractors", {})["gemini"] = summary
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(prior, indent=2))
    temp.replace(path)
    (REPORTS_DIR / "gemini_benchmark_results.json").write_text(json.dumps(report, indent=2))
    return summary


if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        report = asyncio.run(measure(GeminiEvidenceExtractor()))
        result = publish(report)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["active"] else 1)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1) from None
