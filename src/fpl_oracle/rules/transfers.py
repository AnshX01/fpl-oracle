"""
Centralised FPL Transfer & Chip Rules Module (WI-1.3, WI-1.4).
Deterministic functions for:
1. Selling price oracle with 50% profit retention rounded down (tenths of a million).
2. Free transfer banking state machine across normal gameweeks and chip deployments.
3. Chip legality validation (Set 1 GW19 boundary, GW1 WC/FH restriction, GW19/GW20 FH adjacency).
"""

from dataclasses import dataclass
from typing import Any, Literal

from fpl_oracle.rules.scoring import (
    HIT_COST_POINTS,
    MAX_BANKED_FREE_TRANSFERS,
    SET_1_LAST_GW,
)


def sell_price(purchase_tenths: int, current_tenths: int) -> int:
    """
    Official FPL Selling Price oracle (WI-1.4).
    Calculates selling price in tenths of a million (e.g. 55 = £5.5m).
    Rule: Returns 50% of price rise rounded down to nearest 0.1m; full price falls pass through.
    """
    if current_tenths <= purchase_tenths:
        return current_tenths
    profit = current_tenths - purchase_tenths
    return purchase_tenths + (profit // 2)


# Named config constant for post-chip free-transfer state transition rule (WI-1.3, WI-1.4).
# Banked transfers are preserved through a Wildcard or Free Hit with no extra transfer: confirmed in the 2026-27
# game (a mini-league rival who played a GW5 Wildcard had 1 free transfer in GW6 and took a hit for a second).
POST_CHIP_FT_RULE_DEFAULT: Literal["preserve", "preserve_plus_one"] = "preserve"
POST_CHIP_FT_RULE_STATUS: str = "CONFIRMED"


# First gameweek each chip can be played (bootstrap `chips[].start_event`): the GW1 squad is built
# with unlimited free transfers, so Wildcard and Free Hit open in GW2. Other chips open in GW1.
CHIP_FIRST_GW: dict[str, int] = {"wildcard": 2, "freehit": 2}


@dataclass(frozen=True)
class TransferStateConfig:
    chip_week_ft_rule: Literal["preserve", "preserve_plus_one"] = POST_CHIP_FT_RULE_DEFAULT
    max_banked: int = MAX_BANKED_FREE_TRANSFERS
    hit_cost: int = HIT_COST_POINTS


def next_free_transfers(
    ft: int,
    transfers_made: int,
    chip: str | None = None,
    *,
    active_chip: str | None = None,
    config: TransferStateConfig | None = None,
) -> tuple[int, int]:
    """
    Pure transfer state transition function (WI-1.3).
    Computes (next_banked_ft, hit_points).

    Rules:
    - Normal week: free_used = min(transfers_made, ft)
                   hit_points = 4 * max(0, transfers_made - ft)
                   ft_next = min(max_banked, ft - free_used + 1)
    - Wildcard/Free Hit: zero hits. Banked FTs preserved (R4).
      Under default 'preserve', ft_next = ft.
      Under 'preserve_plus_one', ft_next = min(max_banked, ft + 1).
    """
    cfg = config or TransferStateConfig()
    effective_chip = chip if chip is not None else active_chip
    chip_lower = effective_chip.lower().strip() if effective_chip else None

    if chip_lower in ("wildcard", "freehit"):
        hit_points = 0
        if cfg.chip_week_ft_rule == "preserve_plus_one":
            ft_next = min(cfg.max_banked, max(1, ft + 1))
        else:
            ft_next = min(cfg.max_banked, max(1, ft))
        return ft_next, hit_points

    # Normal gameweek (or 3xc / bboost)
    free_used = min(transfers_made, ft)
    hits_count = max(0, transfers_made - ft)
    hit_points = hits_count * cfg.hit_cost
    ft_next = min(cfg.max_banked, max(1, ft - free_used + 1))

    return ft_next, hit_points


def validate_chip_legality(
    chip: str,
    target_gw: int,
    chips_already_used: set[str] | list[str] | list[dict[str, Any]] | None = None,
    previous_chip: str | None = None,
) -> tuple[bool, str]:
    """
    Validates legality of chip deployment for target_gw according to 2026/27 rules:
    - Valid chips: 3xc, bboost, freehit, wildcard
    - Set 1 (GW 1-19) / Set 2 (GW 20-38) with hard Set 1 expiration after GW19
    - Rule R2: Free Hit cannot be played in GW1
    - Rule R2: If Free Hit is played in GW19, Free Hit cannot be played in GW20
    - Exactly 1 chip deployment per set (or 1 per half)
    """
    chip_lower = chip.lower().strip()
    valid_chips = {"3xc", "bboost", "freehit", "wildcard"}
    if chip_lower not in valid_chips:
        return False, f"Unknown chip '{chip}'. Valid chips are: {sorted(valid_chips)}."

    if target_gw < 1 or target_gw > 38:
        return False, f"Invalid gameweek {target_gw}. Gameweeks must be between 1 and 38."

    # R2: Free Hit cannot be played in GW1 (nor Wildcard: both open in GW2)
    first_gw = CHIP_FIRST_GW.get(chip_lower, 1)
    if target_gw < first_gw:
        if chip_lower == "freehit":
            return False, "Free Hit cannot be played in Gameweek 1 (Rule R2)."
        return False, f"{chip} cannot be played before Gameweek {first_gw}."

    current_set = 1 if target_gw <= SET_1_LAST_GW else 2

    # Parse chips already used in the current set
    used_in_set = set()
    used_in_all = set()
    if chips_already_used:
        for item in chips_already_used:
            if isinstance(item, dict):
                c_name = str(item.get("name", "")).lower()
                c_ev = item.get("event")
                if c_ev is not None:
                    c_set = 1 if int(c_ev) <= SET_1_LAST_GW else 2
                    if c_set == current_set:
                        used_in_set.add(c_name)
                    if c_name == "freehit" and int(c_ev) == 19 and target_gw == 20:
                        return False, "Free Hit cannot be played in GW20 if used in GW19 (Rule R2)."
                used_in_all.add(c_name)
            else:
                c_name = str(item).lower()
                used_in_set.add(c_name)

    # R2 check for GW20 Free Hit following GW19 Free Hit
    if chip_lower == "freehit" and target_gw == 20 and previous_chip and previous_chip.lower() == "freehit":
        return False, "Free Hit cannot be played in GW20 if used in GW19 (Rule R2)."

    if chip_lower in used_in_set:
        return False, f"Chip '{chip}' has already been deployed in Set {current_set}."

    return True, f"Chip '{chip}' is legal for deployment in GW{target_gw} (Set {current_set})."
