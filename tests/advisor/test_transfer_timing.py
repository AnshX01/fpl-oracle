"""
Unit tests for Feature 2: Transfer Timing & Expected Value of Information Engine.
Tests decision gating between early execution (priced out) and holding for Friday press conferences.
Zero hardcoded player names in production code.
"""

import re

from fpl_oracle.advisor.timing import TransferTimingAdvisor


def test_hold_for_pressers_when_bank_buffer_exists():
    """Even with 95% price rise likelihood, bank buffer £0.5m absorbs the rise; holding for pressers has positive EV."""
    advisor = TransferTimingAdvisor()
    rec = advisor.evaluate_timing(
        bank=0.5,
        target_rise_prob=0.95,
        outgoing_fall_prob=0.0,
        pressers_concluded=False,
        midweek_matches_pending=0,
    )

    assert rec.urgency == "HOLD_FOR_PRESSERS"
    assert rec.ev_waiting_pts > 0.0
    assert "absorbs" in rec.detailed_reason.lower()


def test_execute_tonight_when_zero_bank_and_imminent_price_rise():
    """Zero bank buffer (< £0.1m) + imminent price rise (85%) triggers early execution to prevent being priced out."""
    advisor = TransferTimingAdvisor()
    rec = advisor.evaluate_timing(
        bank=0.0,
        target_rise_prob=0.85,
        outgoing_fall_prob=0.0,
        pressers_concluded=False,
        midweek_matches_pending=0,
    )

    assert rec.urgency == "EXECUTE_TONIGHT_PRICED_OUT"
    assert "priced out" in rec.detailed_reason.lower()


def test_flexible_safe_when_pressers_concluded_and_no_midweek():
    """When press conferences are finished and 0 midweek games remain, transfer timing is safe and flexible."""
    advisor = TransferTimingAdvisor()
    rec = advisor.evaluate_timing(
        bank=0.2,
        target_rise_prob=0.10,
        outgoing_fall_prob=0.0,
        pressers_concluded=True,
        midweek_matches_pending=0,
    )

    assert rec.urgency == "FLEXIBLE_SAFE"
    assert "concluded" in rec.headline.lower()


def test_midweek_congestion_increases_waiting_ev():
    """Pending European/midweek fixtures increase injury knock probability, elevating EV of waiting."""
    advisor = TransferTimingAdvisor()
    rec_no_midweek = advisor.evaluate_timing(bank=0.5, target_rise_prob=0.50, midweek_matches_pending=0)
    rec_with_midweek = advisor.evaluate_timing(bank=0.5, target_rise_prob=0.50, midweek_matches_pending=2)

    assert rec_with_midweek.ev_waiting_pts > rec_no_midweek.ev_waiting_pts


def test_no_hardcoded_player_names_in_timing_module():
    """Enforces zero hardcoded player names in src/fpl_oracle/advisor/timing.py."""
    with open("src/fpl_oracle/advisor/timing.py", encoding="utf-8") as f:
        content = f.read().lower()

    for forbidden in ["haaland", "salah", "palmer", "saka", "watkins", "gabriel", "son"]:
        assert not re.search(
            rf"\b{forbidden}\b", content
        ), f"Forbidden hardcoded player name '{forbidden}' found in timing.py"

