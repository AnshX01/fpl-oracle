"""
Unit tests for Automated Deadline Leak Watchdog & Alerting Engine (Pillar 4).
Validates leak parsing, squad vulnerability auditing, captaincy emergency promotion,
and webhook dispatch formatting.
"""

from fpl_oracle.news.leak_watchdog import (
    DeadlineLeakWatchdog,
    TeamLeakReport,
)


def test_parse_leak_text_benching_and_absence():
    """Validates regex parsing of typical deadline morning leak tweets."""
    watchdog = DeadlineLeakWatchdog()

    t1 = "Exclusive: Premium Midfielder benched for today's match."
    reports = watchdog.parse_leak_text(t1, source="teamnewsandticks")
    assert len(reports) == 1
    assert reports[0].status == "BENCHED"
    assert reports[0].player_name == "Premium Midfielder"

    t2 = "Defender One not in squad after failing late fitness test."
    reports2 = watchdog.parse_leak_text(t2, source="training_ground_itk")
    assert len(reports2) == 1
    assert reports2[0].status == "ABSENT"
    assert reports2[0].player_name == "Defender One"


def test_audit_squad_against_leaks_critical_captain_alert():
    """When a starter/captain is leaked benched, critical alert is generated."""
    watchdog = DeadlineLeakWatchdog()

    current_starters = [
        {"element": 10, "web_name": "Talisman Striker", "position": "FWD"},
        {"element": 11, "web_name": "Wing Defender", "position": "DEF"},
    ]
    current_bench = [
        {"element": 20, "web_name": "Solid Sub", "position": "DEF"},
    ]

    leaks = [
        TeamLeakReport(
            leak_id="leak_test_1",
            player_name="Talisman Striker",
            team_name="",
            status="BENCHED",
            source_handle="fpl_focal",
            confidence=0.95,
        )
    ]

    alerts = watchdog.audit_squad_against_leaks(
        current_starters=current_starters,
        current_bench=current_bench,
        captain_id=10,  # Talisman Striker is captain!
        vice_captain_id=11,
        leaks=leaks,
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.alert_level == "CRITICAL"
    assert 10 in alert.affected_elements
    assert "Switch armband" in alert.recommended_action
    assert "Solid Sub" in alert.recommended_action
    assert "telegram_text" in alert.dispatch_payload
    assert "embeds" in alert.dispatch_payload
