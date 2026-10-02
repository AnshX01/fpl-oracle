"""
Unit tests for Live Rules Verification Engine.
Verifies rules matching and mismatch detection for 2026/27 official FPL rules.
"""

import pytest

from fpl_oracle.api.models import BootstrapStatic, ChipDefinition
from fpl_oracle.api.rules_checker import LiveRulesChecker


@pytest.fixture
def valid_bootstrap() -> BootstrapStatic:
    chips = [
        ChipDefinition(name="wildcard", chip_type="transfer", start_event=2, stop_event=19),
        ChipDefinition(name="wildcard", chip_type="transfer", start_event=20, stop_event=38),
        ChipDefinition(name="freehit", chip_type="transfer", start_event=2, stop_event=19),
        ChipDefinition(name="freehit", chip_type="transfer", start_event=20, stop_event=38),
        ChipDefinition(name="bboost", chip_type="team", start_event=1, stop_event=19),
        ChipDefinition(name="bboost", chip_type="team", start_event=20, stop_event=38),
        ChipDefinition(name="3xc", chip_type="team", start_event=1, stop_event=19),
        ChipDefinition(name="3xc", chip_type="team", start_event=20, stop_event=38),
    ]
    game_settings = {
        "squad_squadsize": 15,
        "squad_squadplay": 11,
        "squad_team_limit": 3,
        "squad_total_spend": 1000,
        "transfers_sell_on_fee": 0.5,
        "max_extra_free_transfers": 4,
        "element_sell_at_purchase_price": False,
    }
    game_config = {
        "scoring": {
            "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
            "assists": 3,
            "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
            "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
        }
    }
    return BootstrapStatic(
        chips=chips,
        game_settings=game_settings,
        game_config=game_config,
    )


def test_rules_checker_valid(valid_bootstrap: BootstrapStatic):
    checker = LiveRulesChecker()
    res = checker.verify(valid_bootstrap)
    assert res.verified is True
    assert len(res.mismatches) == 0
    assert res.details["squad_constraints"]["status"] == "PASS"
    assert res.details["transfers_and_pricing"]["status"] == "PASS"
    assert res.details["chips_structure"]["status"] == "PASS"
    assert res.details["scoring_and_defcon"]["status"] == "PASS"


def test_rules_checker_detects_assistant_manager(valid_bootstrap: BootstrapStatic):
    checker = LiveRulesChecker()
    # Add assistant manager chip
    valid_bootstrap.chips.append(
        ChipDefinition(name="manager", chip_type="team", start_event=1, stop_event=38)
    )
    res = checker.verify(valid_bootstrap)
    assert res.verified is False
    assert any("Assistant Manager" in m or "chips mismatch" in m for m in res.mismatches)


def test_rules_checker_detects_defcon_discrepancy(valid_bootstrap: BootstrapStatic):
    checker = LiveRulesChecker()
    # Alter defcon scoring
    assert valid_bootstrap.game_config is not None
    valid_bootstrap.game_config["scoring"]["defensive_contribution"]["DEF"] = 3
    res = checker.verify(valid_bootstrap)
    assert res.verified is False
    assert any("DefCon scoring mismatch for DEF" in m for m in res.mismatches)


def test_rules_checker_detects_transfer_bank_discrepancy(valid_bootstrap: BootstrapStatic):
    checker = LiveRulesChecker()
    # Alter max extra free transfers
    valid_bootstrap.game_settings["max_extra_free_transfers"] = 2
    res = checker.verify(valid_bootstrap)
    assert res.verified is False
    assert any("Banked FTs mismatch" in m for m in res.mismatches)
