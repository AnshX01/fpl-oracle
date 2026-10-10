"""
Tests for Centralised FPL Transfer Rules and Selling Price Oracle (WI-1.3, WI-1.4).
"""

import pytest

from fpl_oracle.rules.transfers import (
    TransferStateConfig,
    next_free_transfers,
    sell_price,
    sell_price_millions,
    validate_chip_legality,
)


@pytest.mark.parametrize(
    "purchase_tenths,current_tenths,expected_sell_tenths",
    [
        (50, 50, 50),   # No change (£5.0m -> £5.0m)
        (50, 51, 50),   # +0.1m rise -> sell at purchase £5.0m
        (50, 52, 51),   # +0.2m rise -> sell at £5.1m (+0.1m profit)
        (50, 53, 51),   # +0.3m rise -> sell at £5.1m (+0.1m profit)
        (50, 54, 52),   # +0.4m rise -> sell at £5.2m (+0.2m profit)
        (50, 49, 49),   # -0.1m fall -> full fall passes through (£4.9m)
        (50, 48, 48),   # -0.2m fall -> full fall passes through (£4.8m)
        (100, 105, 102), # £10.0m -> £10.5m (+0.5m rise -> £10.2m)
    ],
)
def test_selling_price_oracle_tenths(purchase_tenths, current_tenths, expected_sell_tenths):
    assert sell_price(purchase_tenths, current_tenths) == expected_sell_tenths


def test_selling_price_oracle_millions():
    assert sell_price_millions(5.0, 5.1) == 5.0
    assert sell_price_millions(5.0, 5.2) == 5.1
    assert sell_price_millions(5.0, 5.3) == 5.1
    assert sell_price_millions(5.0, 4.8) == 4.8


@pytest.mark.parametrize(
    "ft_in,transfers_made,chip,expected_ft_next,expected_hit_points",
    [
        # Table-driven cases from WI-1.3
        (1, 3, None, 1, 8),   # 1 FT, 3 transfers -> 2 hits (8 pts), next FT = 1
        (5, 0, None, 5, 0),   # 5 FT, 0 transfers -> roll, capped at 5 FT
        (5, 2, None, 4, 0),   # 5 FT, 2 transfers -> 3 left + 1 = 4 FT, 0 hits
        (2, 1, None, 2, 0),   # 2 FT, 1 transfer -> 1 left + 1 = 2 FT, 0 hits
        (2, 0, None, 3, 0),   # 2 FT, 0 transfers -> roll to 3 FT, 0 hits
        (1, 1, None, 1, 0),   # 1 FT, 1 transfer -> 0 left + 1 = 1 FT, 0 hits
        (1, 0, None, 2, 0),   # 1 FT, 0 transfers -> roll to 2 FT, 0 hits
        (3, 5, None, 1, 8),   # 3 FT, 5 transfers -> 2 hits (8 pts), next FT = 1
        # Wildcard / Free Hit weeks (preserve FT, 0 hits)
        (2, 10, "wildcard", 2, 0),  # Wildcard preserves 2 banked FTs, 0 hits
        (4, 12, "freehit", 4, 0),   # Free Hit preserves 4 banked FTs, 0 hits
        (5, 8, "wildcard", 5, 0),   # Wildcard with 5 FTs preserves 5 FTs
        # 3xC and Bench Boost are normal weeks regarding transfer costs
        (2, 2, "3xc", 1, 0),        # 3xC uses 2 FTs, banks to 1
        (1, 2, "bboost", 1, 4),     # Bench Boost with 2 transfers on 1 FT -> 1 hit (4 pts), banks to 1
    ],
)
def test_next_free_transfers_table_driven(ft_in, transfers_made, chip, expected_ft_next, expected_hit_points):
    ft_next, hits = next_free_transfers(ft_in, transfers_made, chip)
    assert ft_next == expected_ft_next
    assert hits == expected_hit_points


def test_next_free_transfers_preserve_plus_one_option():
    cfg = TransferStateConfig(chip_week_ft_rule="preserve_plus_one")
    ft_next, hits = next_free_transfers(2, 10, "wildcard", config=cfg)
    assert ft_next == 3
    assert hits == 0


def test_chip_legality_r2_and_boundaries():
    # R2: Free Hit cannot be played in GW1
    valid, msg = validate_chip_legality("freehit", 1)
    assert valid is False
    assert "GW1" in msg or "Gameweek 1" in msg

    # Legal in GW2
    valid, msg = validate_chip_legality("freehit", 2)
    assert valid is True

    # R2: If FH played in GW19, cannot be played in GW20
    used_gw19_fh = [{"name": "freehit", "event": 19}]
    valid, msg = validate_chip_legality("freehit", 20, chips_already_used=used_gw19_fh)
    assert valid is False
    assert "GW20" in msg

    # Set 1 expiration check (same chip cannot be played twice in same set)
    used_set1_wc = [{"name": "wildcard", "event": 5}]
    valid, msg = validate_chip_legality("wildcard", 10, chips_already_used=used_set1_wc)
    assert valid is False
    assert "Set 1" in msg

    # Wildcard in Set 2 is legal if only used in Set 1
    valid, msg = validate_chip_legality("wildcard", 25, chips_already_used=used_set1_wc)
    assert valid is True
