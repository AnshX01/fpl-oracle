"""
Tests for Domain Foundations:
- ManagerStateService selling price math, FT replay, override preservation.
- DataStore profile seeding from environment and no-overwrite.
"""

from unittest.mock import MagicMock

from fpl_oracle.domain.manager_state import (
    ManagerStateService,
)
from fpl_oracle.domain.scoring import expected_floor_div


def test_expected_floor_div_poisson_expectation():
    # E[floor(K/3)] for K ~ Poisson(rate)
    # If rate = 0, expectation is 0.0
    assert expected_floor_div(0.0, 3) == 0.0

    # If rate = 3.0, simple naive 3/3 = 1.0, but exact Poisson expectation accounts for probabilities of 0, 1, 2 (yield 0)
    exp_saves = expected_floor_div(3.0, 3)
    assert 0.6 < exp_saves < 1.0  # Strictly less than 1.0 due to Jensen's inequality and discrete floor!


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
    # Initial deadline unlimited: next GW starts at one FT.
    # GW2 hold banks two; GW3 wildcard preserves two; GW4 uses two and resets one.
    # GW5 five transfers leaves one FT for GW6.
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


def test_failed_history_fetch_is_reported_not_silently_defaulted():
    profile = MagicMock()
    profile.manager_id = 12345
    profile.target_league_id = None
    profile.bank = None
    profile.free_transfers = None
    profile.manual_squad = None
    picks = MagicMock()
    picks.picks = []
    picks.active_chip = None
    picks.entry_history = None
    boot = MagicMock()
    boot.elements = []

    state = ManagerStateService.build_effective_state(
        profile_data=profile, bootstrap=boot, manager_picks=picks, manager_history=None
    )
    assert "history unavailable" in (state.error_message or "")
    assert state.ft_source == "default"


def test_override_that_disagrees_with_official_state_is_flagged():
    profile = MagicMock()
    profile.manager_id = 12345
    profile.target_league_id = None
    profile.bank = 1.5
    profile.free_transfers = 2
    profile.bank_override_enabled = True
    profile.ft_override_enabled = True
    profile.manual_squad = None
    picks = MagicMock()
    picks.picks = []
    picks.active_chip = None
    picks.entry_history = MagicMock(bank=1)
    history = MagicMock()
    history.current = []
    history.chips = []
    boot = MagicMock()
    boot.elements = []

    state = ManagerStateService.build_effective_state(
        profile_data=profile, bootstrap=boot, manager_picks=picks, manager_history=history
    )
    assert state.bank_tenths == 15 and state.bank_source == "manual_override"
    assert "£1.5m differs from the official £0.1m" in state.error_message
    assert "free-transfer override 2" in state.error_message
