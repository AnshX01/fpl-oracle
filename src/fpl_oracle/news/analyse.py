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
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence
from fpl_oracle.news.reconcile import availability_reconciler

logger = logging.getLogger("fpl_oracle.news.analyse")


class NewsAnalyzer:
    def __init__(self):
        self.reconciler = availability_reconciler
        self._last_candidate_evidences: list[PlayerEvidence] = []
        self._last_reconciled_map: dict[int, float] = {}

    async def get_player_news_signals(
        self, bootstrap: BootstrapStatic, player_filter_id: int | None = None, target_gw: int = 6
    ) -> list[dict[str, Any]]:
        """
        Combine authoritative official FPL status with validated LLM evidence extractions.
        Executes candidate evidence extraction BEFORE reconciliation (N1, N4).
        """
        candidate_evidences: list[PlayerEvidence] = []

        # 1. Candidate Evidence Extraction via RSS & Gemini Free-Tier (if configured)
        try:
            articles = await news_ingestion.fetch_rss_articles()
            # Batch all roster players with canonical integer IDs (N4: no [:50] truncation)
            roster_batch = [
                {"id": int(e.id), "web_name": e.web_name, "team_name": str(e.team)} for e in bootstrap.elements
            ]

            is_gemini_ready, _ = gemini_extractor.is_configured()
            self._active_extractor = "gemini" if is_gemini_ready else "deterministic_fallback"

            if articles:
                for art in articles[:10]:
                    cleaned_text = text_extractor.clean_html(art["title"] + " " + art.get("summary", ""))
                    if not cleaned_text:
                        continue

                    if is_gemini_ready:
                        try:
                            extracted_payload = await gemini_extractor.extract_evidence_from_text(
                                article_text=cleaned_text,
                                article_url=art.get("link", ""),
                                published_at=art.get("published_at"),
                                candidate_players=roster_batch,
                                target_gw=target_gw,
                            )
                            candidate_evidences.extend(extracted_payload.evidences)
                        except Exception as e:
                            logger.warning(f"Error extracting evidence via Gemini: {e}")
                    else:
                        try:
                            # Production path: deterministic fallback extractor
                            for p in roster_batch:
                                p_evs = text_extractor.extract_evidence_from_text(
                                    text=cleaned_text,
                                    player_name=str(p["web_name"]),
                                    player_id=int(str(p["id"])),
                                    target_gw=target_gw,
                                )
                                for ev in p_evs:
                                    if ev.category != EvidenceCategory.UNKNOWN:
                                        ev.source_url = art.get("link", "")
                                        candidate_evidences.append(ev)
                        except Exception as e:
                            logger.warning(f"Error in deterministic fallback extraction: {e}")
        except Exception as e:
            logger.warning(f"Error ingesting news feeds: {e}")

        from datetime import UTC, datetime, timedelta

        prior = [
            ev
            for ev in self._last_candidate_evidences
            if ev.published_at and ev.published_at.tzinfo and datetime.now(UTC) - ev.published_at < timedelta(hours=48)
        ]
        combined = {ev.model_dump_json(): ev for ev in prior + candidate_evidences}
        candidate_evidences = list(combined.values())
        self._last_candidate_evidences = candidate_evidences

        # 2. Reconcile players through single AvailabilityReconciler
        elem_map = {int(e.id): e for e in bootstrap.elements}
        signals: list[dict[str, Any]] = []
        reconciled_map: dict[int, float] = {}

        # Collect official FPL signals
        fpl_signals = news_ingestion.fetch_official_fpl_signals(bootstrap)
        fpl_elem_ids = {s["element_id"] for s in fpl_signals}

        # Also collect candidate evidence element IDs
        cand_elem_ids = {ev.player_id for ev in candidate_evidences if ev.player_id > 0}
        all_notable_ids = fpl_elem_ids | cand_elem_ids

        for elem_id in all_notable_ids:
            if player_filter_id and elem_id != player_filter_id:
                continue
            elem = elem_map.get(elem_id)
            if not elem:
                continue

            reconciled = self.reconciler.reconcile_player_fixture(
                element=elem,
                target_gw=target_gw,
                candidate_evidence=candidate_evidences,
                extractor_name=getattr(self, "_active_extractor", "deterministic_fallback"),
            )
            reconciled_map[elem_id] = reconciled.effective_chance_of_playing

            status_code = getattr(elem, "status", "a")
            signal_type = (
                "INJURY" if status_code in ["i", "u"] else ("SUSPENSION" if status_code == "s" else "DOUBTFUL")
            )
            if reconciled.is_shadow_override and reconciled.evidence_category:
                signal_type = f"PRESS_{reconciled.evidence_category.value.upper()}"

            signals.append(
                {
                    "element_id": elem_id,
                    "web_name": elem.web_name,
                    "team_name": getattr(elem, "team_name", str(elem.team)),
                    "signal_type": signal_type,
                    "status_code": status_code,
                    "headline": getattr(elem, "news", "") or f"Reconciled status: {reconciled.reconciliation_reason}",
                    "chance_of_playing": reconciled.effective_chance_of_playing,
                    "baseline_chance": reconciled.baseline_chance_of_playing,
                    "source": "Official FPL & Press Intelligence",
                    "source_url": reconciled.source_url or "https://fantasy.premierleague.com/",
                    "quote": reconciled.source_quote,
                    "confidence": 1.0,
                    "reconciliation_reason": reconciled.reconciliation_reason,
                    "applied_to_production": reconciled.applied_to_production,
                    "is_shadow": not reconciled.applied_to_production,
                    "expected_minutes_limit": reconciled.expected_minutes_limit,
                    "active_extractor": getattr(self, "_active_extractor", "deterministic_fallback"),
                }
            )

        # Build comprehensive map for all players
        for elem in bootstrap.elements:
            eid = int(elem.id)
            if eid not in reconciled_map:
                rec = self.reconciler.reconcile_player_fixture(
                    element=elem,
                    target_gw=target_gw,
                    candidate_evidence=candidate_evidences,
                    extractor_name=getattr(self, "_active_extractor", "deterministic_fallback"),
                )
                reconciled_map[eid] = rec.effective_chance_of_playing

        self._last_reconciled_map = reconciled_map
        return signals

    def get_reconciled_inputs(self, bootstrap, target_gw):
        return {
            int(e.id): self.reconciler.reconcile_player_fixture(
                e,
                target_gw,
                candidate_evidence=self._last_candidate_evidences,
                extractor_name=getattr(self, "_active_extractor", "deterministic_fallback"),
            )
            for e in bootstrap.elements
        }

    def get_reconciled_availabilities_map(
        self,
        bootstrap: BootstrapStatic,
        target_gw: int = 6,
        candidate_evidence: list[PlayerEvidence] | None = None,
    ) -> dict[int, float]:
        """Return map of element_id -> effective_chance_of_playing across all players (N1)."""
        evidences = candidate_evidence if candidate_evidence is not None else self._last_candidate_evidences
        avail_map: dict[int, float] = {}
        for elem in bootstrap.elements:
            eid = int(elem.id)
            rec = self.reconciler.reconcile_player_fixture(
                element=elem,
                target_gw=target_gw,
                candidate_evidence=evidences,
                extractor_name=getattr(self, "_active_extractor", "deterministic_fallback"),
            )
            avail_map[eid] = rec.effective_chance_of_playing
        return avail_map


news_analyzer = NewsAnalyzer()
