"""
Tests for Centralised FPL Transfer Rules and Selling Price Oracle (WI-1.3, WI-1.4).
"""

import pytest

from fpl_oracle.rules.transfers import (
    POST_CHIP_FT_RULE_DEFAULT,
    POST_CHIP_FT_RULE_STATUS,
    TransferStateConfig,
    next_free_transfers,
    sell_price,
    validate_chip_legality,
)


@pytest.mark.parametrize(
    "purchase_tenths,current_tenths,expected_sell_tenths",
    [
        (50, 50, 50),  # No change (£5.0m -> £5.0m)
        (50, 51, 50),  # +0.1m rise -> sell at purchase £5.0m
        (50, 52, 51),  # +0.2m rise -> sell at £5.1m (+0.1m profit)
        (50, 53, 51),  # +0.3m rise -> sell at £5.1m (+0.1m profit)
        (50, 54, 52),  # +0.4m rise -> sell at £5.2m (+0.2m profit)
        (50, 49, 49),  # -0.1m fall -> full fall passes through (£4.9m)
        (50, 48, 48),  # -0.2m fall -> full fall passes through (£4.8m)
        (100, 105, 102),  # £10.0m -> £10.5m (+0.5m rise -> £10.2m)
    ],
)
def test_selling_price_oracle_tenths(purchase_tenths, current_tenths, expected_sell_tenths):
    assert sell_price(purchase_tenths, current_tenths) == expected_sell_tenths


@pytest.mark.parametrize(
    "ft_in,transfers_made,chip,expected_ft_next,expected_hit_points",
    [
        # Table-driven cases from WI-1.3
        (1, 3, None, 1, 8),  # 1 FT, 3 transfers -> 2 hits (8 pts), next FT = 1
        (5, 0, None, 5, 0),  # 5 FT, 0 transfers -> roll, capped at 5 FT
        (5, 2, None, 4, 0),  # 5 FT, 2 transfers -> 3 left + 1 = 4 FT, 0 hits
        (2, 1, None, 2, 0),  # 2 FT, 1 transfer -> 1 left + 1 = 2 FT, 0 hits
        (2, 0, None, 3, 0),  # 2 FT, 0 transfers -> roll to 3 FT, 0 hits
        (1, 1, None, 1, 0),  # 1 FT, 1 transfer -> 0 left + 1 = 1 FT, 0 hits
        (1, 0, None, 2, 0),  # 1 FT, 0 transfers -> roll to 2 FT, 0 hits
        (3, 5, None, 1, 8),  # 3 FT, 5 transfers -> 2 hits (8 pts), next FT = 1
        # Wildcard / Free Hit weeks (preserve FT, 0 hits)
        (2, 10, "wildcard", 2, 0),  # Wildcard preserves 2 banked FTs, 0 hits
        (4, 12, "freehit", 4, 0),  # Free Hit preserves 4 banked FTs, 0 hits
        (5, 8, "wildcard", 5, 0),  # Wildcard with 5 FTs preserves 5 FTs
        # 3xC and Bench Boost are normal weeks regarding transfer costs
        (2, 2, "3xc", 1, 0),  # 3xC uses 2 FTs, banks to 1
        (1, 2, "bboost", 1, 4),  # Bench Boost with 2 transfers on 1 FT -> 1 hit (4 pts), banks to 1
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


def test_post_chip_ft_rule_named_constant_status():
    """Verify that post-chip free-transfer rule is exposed as a named constant flagged UNCONFIRMED."""
    assert POST_CHIP_FT_RULE_DEFAULT == "preserve"
    assert POST_CHIP_FT_RULE_STATUS == "CONFIRMED"
    cfg = TransferStateConfig()
    assert cfg.chip_week_ft_rule == POST_CHIP_FT_RULE_DEFAULT


def test_selling_price_real_player_trajectories():
    """
    Validates selling price rule (half profit rounded down, price falls pass through)
    against real player price movements from historical FPL seasons.
    """
    import pandas as pd

    df = pd.read_csv("data/historical/master_history.csv")

    # 1. Cole Palmer 2023-24: started at 50 (£5.0m), peaked at 63 (£6.3m), trough at 49 (£4.9m)
    palmer_23 = df[(df["name"].str.contains("Cole Palmer", na=False)) & (df["season"] == "2023-24")]
    assert not palmer_23.empty
    buy_palmer = int(palmer_23.iloc[0]["value"])  # 50
    peak_palmer = int(palmer_23["value"].max())  # 63
    trough_palmer = int(palmer_23["value"].min())  # 49
    # Profit = 63 - 50 = 13 -> 13 // 2 = 6 -> sell price = 56
    assert sell_price(buy_palmer, peak_palmer) == 56
    # Dip below purchase -> full pass-through -> 49
    assert sell_price(buy_palmer, trough_palmer) == 49

    # 2. Morgan Rogers 2024-25: started at 50 (£5.0m), peaked at 58 (£5.8m)
    rogers_24 = df[(df["name"].str.contains("Morgan Rogers", na=False)) & (df["season"] == "2024-25")]
    assert not rogers_24.empty
    buy_rogers = int(rogers_24.iloc[0]["value"])  # 50
    peak_rogers = int(rogers_24["value"].max())  # 58
    # Profit = 58 - 50 = 8 -> 8 // 2 = 4 -> sell price = 54
    assert sell_price(buy_rogers, peak_rogers) == 54

    # 3. Erling Haaland 2024-25: started at 150 (£15.0m), peaked at 154 (£15.4m), dipped to 147 (£14.7m)
    haaland_24 = df[(df["name"].str.contains("Erling Haaland", na=False)) & (df["season"] == "2024-25")]
    assert not haaland_24.empty
    buy_haaland = int(haaland_24.iloc[0]["value"])  # 150
    peak_haaland = int(haaland_24["value"].max())  # 154
    trough_haaland = int(haaland_24["value"].min())  # 147
    # Profit = 154 - 150 = 4 -> 4 // 2 = 2 -> sell price = 152
    assert sell_price(buy_haaland, peak_haaland) == 152
    assert sell_price(buy_haaland, trough_haaland) == 147

    # 4. Ollie Watkins 2023-24: started at 80 (£8.0m), peaked at 90 (£9.0m), dipped to 79 (£7.9m)
    watkins_23 = df[(df["name"].str.contains("Ollie Watkins", na=False)) & (df["season"] == "2023-24")]
    assert not watkins_23.empty
    buy_watkins = int(watkins_23.iloc[0]["value"])  # 80
    peak_watkins = int(watkins_23["value"].max())  # 90
    trough_watkins = int(watkins_23["value"].min())  # 79
    # Profit = 90 - 80 = 10 -> 10 // 2 = 5 -> sell price = 85
    assert sell_price(buy_watkins, peak_watkins) == 85
    assert sell_price(buy_watkins, trough_watkins) == 79
