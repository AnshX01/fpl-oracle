"""
Unit tests for Complete Decision Card Synthesis (Pillar 6).
Verifies that all 5 pillars (Transfers, Captaincy, Timing, Sentinel, Spatial Threat, Minimax)
are synthesized cleanly in the Decision Card dictionary and printable Markdown export.
"""

from fpl_oracle.briefing.decision_card import format_decision_card_markdown


def test_decision_card_synthesis_markdown_formatting():
    """Markdown formatter renders all 5 analytical pillars seamlessly."""
    card = {
        "gameweek": 9,
        "deadline": {"deadline_time": "2026-10-24 11:00:00", "label": "GW9 Deadline"},
        "formation": "3-5-2",
        "transfers": {
            "is_roll": False,
            "in": [{"web_name": "Striker In"}],
            "out": [{"web_name": "Striker Out"}],
            "hit_cost": 0,
            "bank_after": 0.5,
            "ft_remaining": 1,
        },
        "timing": {
            "urgency": "HOLD_FOR_PRESSERS",
            "headline": "HOLD FOR PRESSERS: Available bank buffer absorbs price fluctuations",
            "detailed_reason": "Bank buffer of £0.50m absorbs £0.1m rise. Wait for Friday press conferences.",
            "ev_waiting_pts": 0.42,
            "cash_buffer": 0.50,
            "price_change_prob": 0.35,
        },
        "captain": {"web_name": "Captain Pick", "expected_points": 7.5},
        "vice_captain": {"web_name": "Vice Pick", "expected_points": 6.0},
        "xi": [
            {"web_name": "Keeper A", "position": "GKP", "expected_points": 4.5},
            {"web_name": "Defender A", "position": "DEF", "expected_points": 5.0},
            {"web_name": "Midfielder A", "position": "MID", "expected_points": 6.5},
        ],
        "bench": [{"bench_order": 1, "web_name": "Sub 1"}],
        "chip": {"recommend": False, "chip_display_name": "Save chips"},
        "rivals": {"league_name": "IceMakers League", "status": "configured", "summary": "Chasing leader by 3 pts"},
        "win_prob": {"p_first": 68.0},
        "deadline_sentinel": {
            "status": "ALL_CLEAR",
            "active_alerts_count": 0,
        },
        "underlying_metrics": {
            "xt_leaders": [
                {"web_name": "Midfielder A", "expected_threat_xt": 0.42},
                {"web_name": "Striker In", "expected_threat_xt": 0.38},
            ]
        },
    }

    md = format_decision_card_markdown(card)

    # Core sections check
    assert "FPL Oracle - Gameweek 9" in md
    assert "Striker Out -> Striker In" in md
    assert "Transfer Timing" in md
    assert "HOLD FOR PRESSERS" in md
    assert "Spatial Threat (xT) Leaders" in md
    assert "Midfielder A (0.42 xT)" in md
    assert "**Deadline Sentinel:** ALL_CLEAR" in md
    assert "IceMakers League" in md
    assert "Chance of finishing first: 68%" in md
