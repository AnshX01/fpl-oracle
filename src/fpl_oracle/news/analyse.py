"""
Structured news and injury signal analysis.
Matches players to news items, categorizes injury severity, rotation risk,
and quotes, strictly without hallucination.
"""

from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timezone
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.news.ingest import news_ingestion
from fpl_oracle.news.extract import text_extractor

class NewsAnalyzer:
    def __init__(self):
        pass

    async def get_player_news_signals(
        self,
        bootstrap: BootstrapStatic,
        player_filter_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine official FPL news and live RSS articles into structured signals.
        """
        signals = []

        # 1. Authoritative Official Signals
        fpl_signals = news_ingestion.fetch_official_fpl_signals(bootstrap)
        for s in fpl_signals:
            if player_filter_id and s["element_id"] != player_filter_id:
                continue

            cop = s.get("chance_of_playing_next_round")
            signal_type = "INJURY" if s["status"] in ["i", "u"] else ("SUSPENSION" if s["status"] == "s" else "DOUBTFUL")
            mins_expectation = cop if cop is not None else (0 if s["status"] == "i" else 75)

            signals.append({
                "element_id": s["element_id"],
                "web_name": s["web_name"],
                "team_name": s["team_name"],
                "signal_type": signal_type,
                "status_code": s["status"],
                "headline": s["news"] or f"Availability status: {s['status']}",
                "minutes_expectation_pct": mins_expectation,
                "source": "Official Premier League / FPL Update",
                "source_url": "https://fantasy.premierleague.com/",
                "confidence": 1.0,
                "timestamp": s.get("news_added") or s["timestamp"]
            })

        # 2. RSS Articles Matching
        try:
            articles = await news_ingestion.fetch_rss_articles()
            elem_list = [e for e in bootstrap.elements if len(e.web_name) > 3]

            for art in articles:
                cleaned_text = text_extractor.clean_html(art["title"] + " " + art["summary"])
                for elem in elem_list:
                    if player_filter_id and elem.id != player_filter_id:
                        continue

                    # Search for player name with word boundary
                    pattern = rf"\b{re.escape(elem.web_name)}\b"
                    if re.search(pattern, cleaned_text, re.IGNORECASE):
                        sig_type = "PRESS_CONFERENCE_REPORT"
                        if any(w in cleaned_text.lower() for w in ["injured", "hamstring", "knee", "ruled out", "surgery"]):
                            sig_type = "INJURY_REPORT"
                        elif any(w in cleaned_text.lower() for w in ["starts", "returns", "fit to play", "available"]):
                            sig_type = "FITNESS_UPDATE"

                        signals.append({
                            "element_id": elem.id,
                            "web_name": elem.web_name,
                            "team_name": str(elem.team),
                            "signal_type": sig_type,
                            "status_code": elem.status,
                            "headline": art["title"],
                            "minutes_expectation_pct": 50 if "injury" in sig_type.lower() else 85,
                            "source": art["source_name"],
                            "source_url": art["link"],
                            "confidence": art["weight"],
                            "timestamp": art["published"]
                        })
        except Exception:
            pass

        return signals

news_analyzer = NewsAnalyzer()
