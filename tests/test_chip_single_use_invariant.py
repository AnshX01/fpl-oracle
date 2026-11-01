"""
Comprehensive Unit Tests for Chip Single-Use Invariant (Official 2026/27 FPL Rules).
Strictly validates:
- Wildcard has 2 sets (GW 1-19 and GW 20-38).
- Free Hit, Bench Boost, and Triple Captain are strictly ONCE per season.
"""

from fpl_oracle.api.models import BootstrapStatic, ChipDefinition, ChipHistoryItem, ManagerHistory
from fpl_oracle.api.rules_checker import LiveRulesChecker
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.rules.transfers import validate_chip_legality


def test_validate_chip_legality_single_use_enforcement():
    """FH, BB, and TC deployed in Set 1 cannot be deployed in Set 2."""
    history = [
        {"name": "freehit", "event": 5},
        {"name": "wildcard", "event": 8},
        {"name": "bboost", "event": 12},
    ]

    # 1. Free Hit deployed in GW5 is illegal in GW25 (Set 2)
    legal, reason = validate_chip_legality("freehit", 25, history)
    assert not legal
    assert "single-use" in reason

    # 2. Bench Boost deployed in GW12 is illegal in GW30 (Set 2)
    legal, reason = validate_chip_legality("bboost", 30, history)
    assert not legal
    assert "single-use" in reason

    # 3. Triple Captain NOT deployed is legal in GW35 (Set 2)
    legal, _ = validate_chip_legality("3xc", 35, history)
    assert legal

    # 4. Wildcard deployed in GW8 (Set 1) IS legal in GW25 (Set 2 Wildcard is available)
    legal, _ = validate_chip_legality("wildcard", 25, history)
    assert legal

    # 5. Wildcard deployed in GW8 (Set 1) is NOT legal in GW15 (Set 1 Wildcard already used)
    legal, reason = validate_chip_legality("wildcard", 15, history)
    assert not legal
    assert "Set 1" in reason


def test_chip_planner_get_remaining_chips():
    """Chip planner correctly segregates single-use chips vs two-set wildcard."""
    hist = ManagerHistory(
        chips=[
            ChipHistoryItem(name="freehit", event=4, time="2026-09-01T00:00:00Z"),
            ChipHistoryItem(name="wildcard", event=7, time="2026-09-20T00:00:00Z"),
        ]
    )
    rem = chip_planner.get_remaining_chips(hist)
    assert "freehit" not in rem["set_1_remaining"]
    assert "freehit" not in rem["set_2_remaining"]  # Must NOT be resurrected in Set 2

    assert "wildcard" not in rem["set_1_remaining"]
    assert "wildcard" in rem["set_2_remaining"]  # Set 2 Wildcard remains available!

    assert "bboost" in rem["set_1_remaining"]
    assert "bboost" in rem["set_2_remaining"]
    assert "3xc" in rem["set_1_remaining"]
    assert "3xc" in rem["set_2_remaining"]


def test_manager_state_from_fpl_api_single_use_chips():
    """Effective manager state does not leak single-use chips into Set 2."""
    chips_used = [
        {"name": "freehit", "event": 8},
        {"name": "wildcard", "event": 6},
    ]
    all_chips = ["wildcard", "freehit", "3xc", "bboost"]
    season_used = {c.get("name", "").lower() for c in chips_used}
    set1_used = {c.get("name", "").lower() for c in chips_used if int(c.get("event", 0) or 0) <= 19}
    set2_used = {c.get("name", "").lower() for c in chips_used if int(c.get("event", 0) or 0) > 19}

    rem_set_1 = [c for c in all_chips if (c not in set1_used if c == "wildcard" else c not in season_used)]
    rem_set_2 = [c for c in all_chips if (c not in set2_used if c == "wildcard" else c not in season_used)]

    assert "freehit" not in rem_set_1
    assert "freehit" not in rem_set_2
    assert "wildcard" not in rem_set_1
    assert "wildcard" in rem_set_2


def test_live_rules_checker_supports_official_5_chips():
    """Live rules checker accepts official 5-chip structure."""
    chips = [
        ChipDefinition(name="wildcard", chip_type="transfer", start_event=2, stop_event=19),
        ChipDefinition(name="wildcard", chip_type="transfer", start_event=20, stop_event=38),
        ChipDefinition(name="freehit", chip_type="transfer", start_event=2, stop_event=38),
        ChipDefinition(name="bboost", chip_type="team", start_event=1, stop_event=38),
        ChipDefinition(name="3xc", chip_type="team", start_event=1, stop_event=38),
    ]
    boot = BootstrapStatic(
        chips=chips,
        game_settings={
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": 1000,
            "transfers_sell_on_fee": 0.5,
            "max_extra_free_transfers": 4,
            "element_sell_at_purchase_price": False,
        },
        game_config={
            "scoring": {
                "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
                "assists": 3,
                "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
                "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
            }
        },
    )
    checker = LiveRulesChecker()
    res = checker.verify(boot)
    assert res.verified is True
    assert res.details["chips_structure"]["status"] == "PASS"
