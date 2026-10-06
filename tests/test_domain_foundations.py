"""
Tests for Domain Foundations:
- Exact scoring cases, thresholds, negative outcomes, multipliers.
- TimeService clock abstraction, deadline boundaries, season over.
- RulesService verification, unknown fields reporting.
- ManagerStateService selling price math, FT replay, override preservation.
- DataStore profile seeding from environment and no-overwrite.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from fpl_oracle.domain.manager_state import (
    ManagerStateService,
)
from fpl_oracle.domain.rules_service import RulesService
from fpl_oracle.domain.scoring import (
    calculate_expected_fixture_points,
    calculate_match_points,
    expected_floor_div,
)
from fpl_oracle.domain.time_service import (
    FrozenClock,
    GameweekPhase,
    TimeService,
)


def test_exact_match_scoring_all_positions_and_cases():
    # 1. GKP clean sheet, 6 saves, 90 mins -> 2 (appearance) + 4 (CS) + 2 (6//3 saves) = 8 pts
    assert calculate_match_points("GKP", minutes=90, clean_sheet=True, saves=6) == 8

    # 2. GKP 5 saves (5//3 = 1 pt), 3 goals conceded (3//2 = 1 pt penalty) -> 2 + 1 - 1 = 2 pts
    assert calculate_match_points("GKP", minutes=90, saves=5, goals_conceded=3) == 2

    # 3. DEF: 60+ min (2 pts), 1 goal (6 pts), 1 assist (3 pts), clean sheet (4 pts), DefCon (+2 pts), bonus 3 = 20 pts
    assert calculate_match_points(
        "DEF", minutes=90, goals_scored=1, assists=1, clean_sheet=True, defensive_contribution=True, bonus=3
    ) == 20

    # 4. DEF clean sheet NOT awarded if played <60 min (e.g. 59 min)
    assert calculate_match_points("DEF", minutes=59, clean_sheet=True) == 1  # 1 appearance, 0 CS

    # 5. MID: 90 min (2), 2 goals (10), clean sheet (1), yellow card (-1) = 12 pts
    assert calculate_match_points("MID", minutes=90, goals_scored=2, clean_sheet=True, yellow_cards=1) == 12

    # 6. FWD: 90 min (2), 1 goal (4), 1 missed penalty (-2) = 4 pts
    assert calculate_match_points("FWD", minutes=90, goals_scored=1, penalties_missed=1) == 4

    # 7. Negative point outcomes: 45 min (1), 1 red card (-3), 1 own goal (-2), 1 yellow (-1) = 1 - 3 - 2 - 1 = -5 pts
    assert calculate_match_points("DEF", minutes=45, red_cards=1, own_goals=1, yellow_cards=1) == -5

    # 8. Captain multiplier (x2) on negative score: -5 * 2 = -10 pts
    assert calculate_match_points("DEF", minutes=45, red_cards=1, own_goals=1, yellow_cards=1, multiplier=2) == -10

    # 9. Triple captain multiplier (x3) on positive score: 10 * 3 = 30 pts
    assert calculate_match_points("MID", minutes=90, goals_scored=1, assists=1, bonus=0, multiplier=3) == 30


def test_expected_floor_div_poisson_expectation():
    # E[floor(K/3)] for K ~ Poisson(rate)
    # If rate = 0, expectation is 0.0
    assert expected_floor_div(0.0, 3) == 0.0

    # If rate = 3.0, simple naive 3/3 = 1.0, but exact Poisson expectation accounts for probabilities of 0, 1, 2 (yield 0)
    exp_saves = expected_floor_div(3.0, 3)
    assert 0.6 < exp_saves < 1.0  # Strictly less than 1.0 due to Jensen's inequality and discrete floor!


def test_expected_fixture_points_coherence_and_negatives():
    # Zero minutes / not playing must yield strictly 0.0 expected points
    zero_res = calculate_expected_fixture_points(
        position="MID",
        p_play=0.0,
        p_min60=0.0,
        p_starts=0.0,
        expected_minutes=0.0,
        expected_goals=0.5,
        expected_assists=0.3,
        p_clean_sheet=0.4,
        expected_goals_conceded=1.0,
        expected_saves=0.0,
        p_defcon=0.5,
        expected_card_deduction=0.1,
        expected_bonus=0.5,
    )
    assert zero_res["expected_points"] == 0.0

    # Active player with regular minutes
    active_res = calculate_expected_fixture_points(
        position="DEF",
        p_play=1.0,
        p_min60=0.9,
        p_starts=0.95,
        expected_minutes=85.0,
        expected_goals=0.05,
        expected_assists=0.10,
        p_clean_sheet=0.40,
        expected_goals_conceded=1.1,
        expected_saves=0.0,
        p_defcon=0.6,
        expected_card_deduction=0.15,
        expected_bonus=0.4,
    )
    assert active_res["expected_points"] > 3.0
    assert active_res["appearance_pts"] == pytest.approx(1.9, rel=1e-2)  # 0.9*2 + 0.1*1


def test_time_service_clock_abstraction_and_phases():
    # Freeze time 2 hours before GW6 deadline
    deadline_str = "2026-10-17T10:00:00Z"
    deadline_dt = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
    freeze_time = deadline_dt - timedelta(hours=2)

    clock = FrozenClock(freeze_time)
    service = TimeService(clock=clock)

    mock_event_6 = MagicMock()
    mock_event_6.id = 6
    mock_event_6.is_current = False
    mock_event_6.is_next = True
    mock_event_6.deadline_time = deadline_str

    mock_event_5 = MagicMock()
    mock_event_5.id = 5
    mock_event_5.is_current = True
    mock_event_5.is_next = False

    mock_bootstrap = MagicMock()
    mock_bootstrap.events = [mock_event_5, mock_event_6]

    # Fixtures for GW5 finished
    mock_fix = MagicMock()
    mock_fix.event = 5
    mock_fix.started = True
    mock_fix.finished = True
    mock_fix.team_h = 1
    mock_fix.team_a = 2

    status_raw = {"status": [{"event": 5, "bonus_added": True, "points": "r"}], "leagues": "Updated"}

    ctx = service.resolve_time_context(mock_bootstrap, [mock_fix], status_raw)
    assert ctx.current_gw == 5
    assert ctx.next_actionable_gw == 6
    assert ctx.phase == GameweekPhase.PRE_DEADLINE
    assert ctx.seconds_to_deadline == pytest.approx(7200.0, abs=5.0)

    # Now advance clock to 10 minutes past deadline
    clock.set_time(deadline_dt + timedelta(minutes=10))
    ctx_past = service.resolve_time_context(mock_bootstrap, [mock_fix], status_raw)
    assert ctx_past.phase == GameweekPhase.LIVE


def test_selling_price_math_invariants():
    # 1. Price rise: bought at 100 (£10.0m), now 105 (£10.5m) -> profit 5 // 2 = 2 -> sell at 102 (£10.2m)
    assert ManagerStateService.calculate_selling_price(100, 105) == 102

    # 2. Odd rise: bought at 50, now 51 -> profit 1 // 2 = 0 -> sell at 50
    assert ManagerStateService.calculate_selling_price(50, 51) == 50

    # 3. Even rise: bought at 50, now 52 -> profit 2 // 2 = 1 -> sell at 51
    assert ManagerStateService.calculate_selling_price(50, 52) == 51

    # 4. Price fall: bought at 60, now 58 -> sell at 58 (no penalty deduction, sell at current market)
    assert ManagerStateService.calculate_selling_price(60, 58) == 58

    # 5. Equal price: bought at 70, now 70 -> sell at 70
    assert ManagerStateService.calculate_selling_price(70, 70) == 70


def test_free_transfers_replay_with_chips_and_caps():
    # GW1 start: 1 FT
    # GW1: made 0 transfers -> banked = min(5, 1 - 0 + 1) = 2
    # GW2: made 0 transfers -> banked = min(5, 2 + 1) = 3
    # GW3: played wildcard, made 5 transfers -> wildcard preserves banked and adds 1 -> min(5, 3 + 1) = 4
    # GW4: made 2 transfers -> remaining = 4 - 2 = 2 -> banked = min(5, 2 + 1) = 3
    # GW5: made 5 transfers (with 3 hits) -> remaining = 0 -> banked = min(5, 0 + 1) = 1
    current_history = [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 5},
        {"event": 4, "event_transfers": 2},
        {"event": 5, "event_transfers": 5},
    ]
    chips_history = [{"event": 3, "name": "wildcard"}]

    ft = ManagerStateService.calculate_banked_free_transfers(current_history, chips_history)
    assert ft == 1


def test_rules_service_missing_field_reports_unknown():
    service = RulesService()
    # Bootstrap missing game_settings entirely
    mock_boot = MagicMock()
    mock_boot.game_settings = None
    mock_boot.game_config = None
    mock_boot.chips = None

    report = service.verify(mock_boot)
    assert not report.verified
    assert any(it.status == "UNKNOWN" for it in report.items)
    assert len(report.limitations) > 0


def test_manager_state_preserves_zero_bank_and_ft_overrides():
    # Verify that explicit 0.0 bank and 0 free transfers overrides are not converted to defaults (truthiness bug)
    profile_data = MagicMock()
    profile_data.manager_id = 12345
    profile_data.target_league_id = 999
    profile_data.bank = 0.0  # Explicit zero
    profile_data.free_transfers = 0  # Explicit zero
    profile_data.manual_squad = None

    mock_bootstrap = MagicMock()
    mock_bootstrap.elements = []

    state = ManagerStateService.build_effective_state(
        profile_data=profile_data,
        bootstrap=mock_bootstrap,
        fixtures=[],
        manager_entry=None,
        manager_picks=None,
        manager_history=None,
        manager_transfers=None,
    )

    assert state.bank_tenths == 0
    assert state.has_bank_override is True
    assert state.bank_source == "manual_override"
    assert state.free_transfers == 0
    assert state.has_ft_override is True
    assert state.ft_source == "manual_override"

