"""
Structured news and injury signal analysis.
Combines official FPL status signals with verified LLM evidence extractions.
CRITICAL INTEGRITY INVARIANT: Crude 50/85% guess heuristics are permanently removed.
RSS articles are strictly discovery/display-only. All availability adjustments are processed
via the single auditable AvailabilityReconciler.
"""

import logging
from typing import Any

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.news.extract import text_extractor
from fpl_oracle.news.gemini_extractor import gemini_extractor
from fpl_oracle.news.ingest import news_ingestion
from fpl_oracle.news.reconcile import availability_reconciler

logger = logging.getLogger("fpl_oracle.news.analyse")


class NewsAnalyzer:
    def __init__(self):
        self.reconciler = availability_reconciler

    async def get_player_news_signals(
        self, bootstrap: BootstrapStatic, player_filter_id: int | None = None, target_gw: int = 6
    ) -> list[dict[str, Any]]:
        """
        Combine authoritative official FPL status with validated LLM evidence extractions.
        Zero crude percentage guesses.
        """
        signals: list[dict[str, Any]] = []

        # 1. Authoritative Official Signals
        fpl_signals = news_ingestion.fetch_official_fpl_signals(bootstrap)
        for s in fpl_signals:
            if player_filter_id and s["element_id"] != player_filter_id:
                continue

            cop = s.get("chance_of_playing_next_round")
            if cop is None and s.get("chance_of_playing_this_round") is not None:
                cop = s.get("chance_of_playing_this_round")

            status_code = s["status"]
            signal_type = (
                "INJURY" if status_code in ["i", "u"] else ("SUSPENSION" if status_code == "s" else "DOUBTFUL")
            )

            # Reconciled via single availability reconciler (no guessing!)
            reconciled = self.reconciler.reconcile_player_fixture(
                element=type("Elem", (), s)(),
                target_gw=target_gw,
            )

            signals.append(
                {
                    "element_id": s["element_id"],
                    "web_name": s["web_name"],
                    "team_name": s["team_name"],
                    "signal_type": signal_type,
                    "status_code": status_code,
                    "headline": s["news"] or f"Official availability status: {status_code}",
                    "chance_of_playing": reconciled.effective_chance_of_playing,
                    "source": "Official Premier League / FPL Update",
                    "source_url": "https://fantasy.premierleague.com/",
                    "confidence": 1.0,
                    "timestamp": s.get("published_at") or s.get("fetched_at"),
                    "reconciliation_reason": reconciled.reconciliation_reason,
                    "applied_to_production": reconciled.applied_to_production,
                }
            )

        # 2. RSS Articles & Gemini Free-Tier Extractor (Discovery & Evidence Extraction)
        try:
            articles = await news_ingestion.fetch_rss_articles()
            roster_sample = [
                {"id": e.id, "web_name": e.web_name, "team_name": str(e.team)}
                for e in bootstrap.elements[:50]
            ]

            is_gemini_ready, _ = gemini_extractor.is_configured()

            for art in articles[:10]:
                cleaned_text = text_extractor.clean_html(art["title"] + " " + art.get("summary", ""))

                # If Gemini is configured and confirmed non-billable Free Tier, run evidence extraction
                if is_gemini_ready:
                    try:
                        extracted_payload = await gemini_extractor.extract_evidence_from_text(
                            article_text=cleaned_text,
                            article_url=art.get("link", ""),
                            published_at=art.get("published_at"),
                            candidate_players=roster_sample,
                            target_gw=target_gw,
                        )
                        for ev in extracted_payload.evidences:
                            signals.append(
                                {
                                    "element_id": ev.player_id,
                                    "web_name": ev.player_name,
                                    "team_name": ev.team_name,
                                    "signal_type": f"GEMINI_{ev.category.value.upper()}",
                                    "status_code": "evidence",
                                    "headline": f"Quote: \"{ev.quote[:80]}...\"",
                                    "chance_of_playing": None,  # Candidate only, not a hardcoded guess!
                                    "source": art["source_name"],
                                    "source_url": ev.source_url or art["link"],
                                    "confidence": ev.confidence,
                                    "timestamp": art.get("published_at") or art.get("fetched_at"),
                                    "quote": ev.quote,
                                    "is_shadow": True,
                                    "reconciliation_reason": f"Candidate extraction ({ev.category.value})",
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Error extracting evidence via Gemini: {e}")
                else:
                    # In no-key mode: articles remain discovery/display-only headlines.
                    # CRITICAL: We do NOT assign fake 50% or 85% numbers!
                    pass

        except Exception as e:
            logger.warning(f"News analysis warning: {e}")

        return signals


news_analyzer = NewsAnalyzer()
