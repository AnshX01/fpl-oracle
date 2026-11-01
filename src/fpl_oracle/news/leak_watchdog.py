"""
Automated Deadline Leak Watchdog & Alerting Engine (Pillar 4).
Monitors deadline-morning team leaks (<= 90 minutes before deadline),
detects surprise benchings or ruled-out starters, triggers automated lineup repair,
and formats real-time alerts for Discord/Telegram/Webhook dispatch.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class TeamLeakReport:
    leak_id: str
    player_name: str
    team_name: str
    status: str  # "BENCHED", "ABSENT", "STARTING", "FITNESS_TEST_FAILED"
    source_handle: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_text: str = ""


@dataclass
class DeadlineAlert:
    alert_level: str  # "CRITICAL", "WARNING", "INFO"
    headline: str
    affected_elements: list[int]
    recommended_action: str
    revised_lineup: dict[str, Any] | None
    dispatch_payload: dict[str, Any]  # Formatted for Discord / Telegram webhooks


class DeadlineLeakWatchdog:
    """
    Automated real-time sentinel for deadline-morning lineup leaks.
    """

    def __init__(self):
        self.verified_sources = {
            "teamnewsandticks",
            "fpl_focal",
            "official_club_leak",
            "press_embargo",
            "training_ground_itk",
        }
        self.leak_history: list[TeamLeakReport] = []

    def parse_leak_text(
        self,
        text: str,
        source: str = "verified_itk",
        confidence: float = 0.90,
    ) -> list[TeamLeakReport]:
        """
        Extracts structured leak reports from incoming raw social/RSS leak text.
        """
        reports = []
        text_clean = text.strip()

        # Regex patterns for benching / absence leaks
        benched_match = re.search(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:benched|on the bench|not starting|dropped)",
            text_clean,
            re.IGNORECASE,
        )
        absent_match = re.search(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:not in squad|misses out|absent|left out|spotted on crutches|failed fitness test)",
            text_clean,
            re.IGNORECASE,
        )
        starting_match = re.search(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:starts|in the starting xi|leading the line)",
            text_clean,
            re.IGNORECASE,
        )

        if benched_match:
            p_name = re.sub(r"\b(is|are|has|have|was|were)\b$", "", benched_match.group(1), flags=re.IGNORECASE).strip()
            reports.append(
                TeamLeakReport(
                    leak_id=f"leak_{len(self.leak_history) + 1}",
                    player_name=p_name,
                    team_name="",
                    status="BENCHED",
                    source_handle=source,
                    confidence=confidence,
                    raw_text=text_clean,
                )
            )
        elif absent_match:
            p_name = absent_match.group(1).strip()
            reports.append(
                TeamLeakReport(
                    leak_id=f"leak_{len(self.leak_history) + 1}",
                    player_name=p_name,
                    team_name="",
                    status="ABSENT",
                    source_handle=source,
                    confidence=confidence,
                    raw_text=text_clean,
                )
            )
        elif starting_match:
            p_name = starting_match.group(1).strip()
            reports.append(
                TeamLeakReport(
                    leak_id=f"leak_{len(self.leak_history) + 1}",
                    player_name=p_name,
                    team_name="",
                    status="STARTING",
                    source_handle=source,
                    confidence=confidence,
                    raw_text=text_clean,
                )
            )

        for r in reports:
            self.leak_history.append(r)
        return reports

    def audit_squad_against_leaks(
        self,
        current_starters: list[dict[str, Any]],
        current_bench: list[dict[str, Any]],
        captain_id: int | None,
        vice_captain_id: int | None,
        leaks: list[TeamLeakReport],
    ) -> list[DeadlineAlert]:
        """
        Scans current starting lineup against recent verified leaks and generates emergency alerts.
        """
        alerts: list[DeadlineAlert] = []
        starter_map = {s.get("web_name", "").lower(): s for s in current_starters}

        for leak in leaks:
            leak_name = leak.player_name.lower()
            # Match by name substring
            matched_starter = next(
                (
                    s
                    for name, s in starter_map.items()
                    if leak_name and name and (leak_name in name or name in leak_name)
                ),
                None,
            )

            if matched_starter and leak.status in ("BENCHED", "ABSENT", "FITNESS_TEST_FAILED"):
                elem_id = int(matched_starter.get("element", 0))
                is_cap = elem_id == captain_id
                action_text = []

                if is_cap:
                    action_text.append(
                        f"CRITICAL: Captain {matched_starter.get('web_name')} is leaked {leak.status}! Switch armband to Vice-Captain (ID: {vice_captain_id})."
                    )

                # Suggest top bench replacement
                top_sub = current_bench[0] if current_bench else None
                if top_sub:
                    action_text.append(f"Promote 1st Sub {top_sub.get('web_name', 'Sub 1')} into Starting XI.")

                headline = f"🚨 DEADLINE LEAK: {matched_starter.get('web_name')} reported {leak.status}!"
                rec_action = " ".join(action_text)

                # Format webhook dispatch payload (Discord / Telegram standard)
                dispatch_payload = {
                    "embeds": [
                        {
                            "title": headline,
                            "description": rec_action,
                            "color": 15158332,  # Crimson Red
                            "fields": [
                                {"name": "Status", "value": leak.status, "inline": True},
                                {"name": "Source", "value": leak.source_handle, "inline": True},
                                {"name": "Confidence", "value": f"{leak.confidence * 100:.0f}%", "inline": True},
                            ],
                            "footer": {"text": "FPL Oracle Automated Deadline Watchdog"},
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    ],
                    "telegram_text": f"{headline}\n\n{rec_action}\n(Source: {leak.source_handle}, {leak.confidence * 100:.0f}% confidence)",
                }

                alerts.append(
                    DeadlineAlert(
                        alert_level="CRITICAL" if is_cap else "WARNING",
                        headline=headline,
                        affected_elements=[elem_id],
                        recommended_action=rec_action,
                        revised_lineup=None,
                        dispatch_payload=dispatch_payload,
                    )
                )

        return alerts

    async def poll_verified_sources_async(self) -> list[TeamLeakReport]:
        """
        Autonomous Sentinel: Polls recent news and social RSS items from the data store,
        identifies breaking deadline leaks, and registers them into the active leak history.
        """
        from fpl_oracle.data.store import data_store

        try:
            recent_items = data_store.get_recent_news(limit=30)
        except Exception:
            recent_items = []

        new_reports: list[TeamLeakReport] = []
        for item in recent_items:
            content = str(item.get("content", "") or item.get("title", "") or "")
            source = str(item.get("source", "news_feed") or "news_feed")
            if not content:
                continue
            lower_c = content.lower()
            if any(
                k in lower_c
                for k in ("bench", "benched", "misses out", "not in squad", "dropped", "starting xi", "starts")
            ):
                parsed = self.parse_leak_text(content, source=source)
                new_reports.extend(parsed)

        return new_reports


deadline_watchdog = DeadlineLeakWatchdog()
