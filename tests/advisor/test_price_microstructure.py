"""
Tests for Calibrated Nightly Price Change Microstructure Engine.
"""

from datetime import UTC, datetime
from types import SimpleNamespace

from fpl_oracle.advisor.timing import (
    get_squad_price_radar,
    predict_player_price_microstructure,
)


def test_price_microstructure_urgent_rise():
    elem = SimpleNamespace(
        id=101,
        web_name="HighVelocityTarget",
        now_cost=85,
        transfers_in_event=85000,
        transfers_out_event=5000,
        selected_by_percent=15.0,
        cost_change_event=0,
        status="a",
        is_price_locked=False,
    )
    profile = predict_player_price_microstructure(elem, now_dt=datetime(2026, 10, 27, 20, 0, tzinfo=UTC))
    assert profile.net_transfers == 80000
    assert profile.p_rise >= 0.70
    assert profile.action_tier == "URGENT_BUY_TONIGHT"
    assert profile.hours_to_cutoff == 5.5
    assert not profile.is_locked


def test_price_microstructure_locked_player():
    elem = SimpleNamespace(
        id=102,
        web_name="ReturningStar",
        now_cost=120,
        transfers_in_event=120000,
        transfers_out_event=1000,
        selected_by_percent=25.0,
        cost_change_event=0,
        status="a",
        is_price_locked=True,  # 8-day price protection lock
    )
    profile = predict_player_price_microstructure(elem)
    assert profile.is_locked is True
    assert profile.p_rise == 0.0
    assert profile.action_tier == "PRICE_LOCKED"


def test_get_squad_price_radar():
    boot = SimpleNamespace(
        elements=[
            SimpleNamespace(
                id=1,
                web_name="P1",
                now_cost=60,
                transfers_in_event=50000,
                transfers_out_event=2000,
                selected_by_percent=10.0,
                cost_change_event=0,
                status="a",
                is_price_locked=False,
            ),
            SimpleNamespace(
                id=2,
                web_name="P2",
                now_cost=70,
                transfers_in_event=1000,
                transfers_out_event=50000,
                selected_by_percent=15.0,
                cost_change_event=0,
                status="a",
                is_price_locked=False,
            ),
            SimpleNamespace(
                id=3,
                web_name="P3",
                now_cost=50,
                transfers_in_event=5000,
                transfers_out_event=5000,
                selected_by_percent=5.0,
                cost_change_event=0,
                status="a",
                is_price_locked=False,
            ),
        ]
    )
    radar = get_squad_price_radar([1, 2, 3], boot)
    assert len(radar) == 3
    # Ordered by movement urgency
    assert max(radar[0]["p_rise"], radar[0]["p_fall"]) >= max(radar[2]["p_rise"], radar[2]["p_fall"])
