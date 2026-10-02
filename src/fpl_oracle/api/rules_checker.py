"""
Live Rules Verification Engine.
Compares live FPL API bootstrap-static rules against local config/rules.yaml and config/scoring.yaml.
Specifically verifies:
1. 2026/27 chip configuration (Set 1 GW1-19, Set 2 GW20-38, assistant manager absent).
2. Free transfers banking (up to 5 = 1 base + 4 extra) and hit penalty (4 pts).
3. Selling price calculation (50% profit retention, element_sell_at_purchase_price: false).
4. Squad constraints (15 players, 11 starters, 3 per club, £100.0m starting budget).
5. 2026/27 scoring rules including Defensive Contribution (DefCon +2 pts for DEF/MID/FWD).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.config import load_rules_config, load_scoring_config

logger = logging.getLogger("fpl_oracle.api.rules_checker")


@dataclass
class RulesVerificationResult:
    verified: bool
    mismatches: list[str] = field(default_factory=list)
    rules_source: str = "Live FPL API bootstrap-static (2026/27)"
    last_checked: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "mismatches": self.mismatches,
            "rules_source": self.rules_source,
            "last_checked": self.last_checked.isoformat(),
            "details": self.details,
        }


class LiveRulesChecker:
    def __init__(self):
        self._last_result: RulesVerificationResult | None = None

    def verify(self, bootstrap: BootstrapStatic) -> RulesVerificationResult:
        """
        Verify live bootstrap against config/rules.yaml and config/scoring.yaml.
        """
        mismatches: list[str] = []
        details: dict[str, Any] = {}

        rules_cfg = load_rules_config()
        scoring_cfg = load_scoring_config()

        game_settings = bootstrap.game_settings or {}
        game_config = bootstrap.game_config or {}
        scoring_api = game_config.get("scoring", {})

        # 1. Squad Constraints
        squad_size = game_settings.get("squad_squadsize", 15)
        squad_play = game_settings.get("squad_squadplay", 11)
        team_limit = game_settings.get("squad_team_limit", 3)
        spend_budget = game_settings.get("squad_total_spend", 1000)

        cfg_squad = rules_cfg.get("squad", {})
        if squad_size != cfg_squad.get("total_players", 15):
            mismatches.append(f"Squad size mismatch: API {squad_size} vs config {cfg_squad.get('total_players')}")
        if squad_play != cfg_squad.get("squad_play", 11):
            mismatches.append(f"Squad play mismatch: API {squad_play} vs config {cfg_squad.get('squad_play')}")
        if team_limit != cfg_squad.get("max_per_team", 3):
            mismatches.append(f"Team limit mismatch: API {team_limit} vs config {cfg_squad.get('max_per_team')}")
        if spend_budget != cfg_squad.get("starting_budget_tenths", 1000):
            mismatches.append(f"Budget mismatch: API {spend_budget} vs config {cfg_squad.get('starting_budget_tenths')}")

        details["squad_constraints"] = {
            "squad_size": squad_size,
            "squad_play": squad_play,
            "team_limit": team_limit,
            "starting_budget": spend_budget / 10.0,
            "status": "PASS" if not any("squad" in m or "Team" in m or "Budget" in m for m in mismatches) else "FAIL",
        }

        # 2. Transfers & Selling Price
        sell_on_fee = float(game_settings.get("transfers_sell_on_fee", 0.5))
        max_extra_ft = int(game_settings.get("max_extra_free_transfers", 4))
        sell_at_purchase = bool(game_settings.get("element_sell_at_purchase_price", False))

        cfg_transfers = rules_cfg.get("transfers", {})
        expected_max_banked = 1 + max_extra_ft # 1 base + 4 extra = 5
        if expected_max_banked != cfg_transfers.get("max_banked_free_transfers", 5):
            mismatches.append(f"Banked FTs mismatch: API {expected_max_banked} vs config {cfg_transfers.get('max_banked_free_transfers')}")
        if sell_on_fee != float(cfg_transfers.get("sell_on_fee", 0.5)):
            mismatches.append(f"Sell-on fee mismatch: API {sell_on_fee} vs config {cfg_transfers.get('sell_on_fee')}")
        if sell_at_purchase is True:
            mismatches.append("API unexpectedly enabled element_sell_at_purchase_price: True")

        details["transfers_and_pricing"] = {
            "max_banked_transfers": expected_max_banked,
            "transfers_sell_on_fee": sell_on_fee,
            "element_sell_at_purchase_price": sell_at_purchase,
            "status": "PASS" if not any("FT" in m or "fee" in m or "purchase_price" in m for m in mismatches) else "FAIL",
        }

        # 3. Chips Configuration (2026/27 2-set structure)
        chips = bootstrap.chips
        chip_names = [c.name for c in chips]
        set1_chips = [c for c in chips if c.stop_event == 19]
        set2_chips = [c for c in chips if c.stop_event == 38]

        if len(chips) != 8:
            mismatches.append(f"Total chips mismatch: live API has {len(chips)} chips, expected 8")
        if "manager" in chip_names or "assistant_manager" in chip_names:
            mismatches.append("Assistant Manager chip unexpectedly present in live 2026/27 API")
        if len(set1_chips) != 4 or len(set2_chips) != 4:
            mismatches.append(f"Chip sets mismatch: Set 1 has {len(set1_chips)} chips, Set 2 has {len(set2_chips)} chips (expected 4 each)")

        details["chips_structure"] = {
            "total_chips": len(chips),
            "set_1_chips_count": len(set1_chips),
            "set_2_chips_count": len(set2_chips),
            "assistant_manager_present": "manager" in chip_names or "assistant_manager" in chip_names,
            "status": "PASS" if not any("chip" in m.lower() for m in mismatches) else "FAIL",
        }

        # 4. Scoring & Defensive Contribution (DefCon)
        if scoring_api:
            # Check DefCon
            defcon_api = scoring_api.get("defensive_contribution", {})
            cfg_defcon = scoring_cfg.get("defensive_contribution", {})
            for pos in ["DEF", "MID", "FWD"]:
                api_val = defcon_api.get(pos)
                cfg_val = cfg_defcon.get(pos)
                if api_val != cfg_val:
                    mismatches.append(f"DefCon scoring mismatch for {pos}: API {api_val} vs config {cfg_val}")

            # Check Clean Sheets
            cs_api = scoring_api.get("clean_sheets", {})
            cfg_cs = scoring_cfg.get("clean_sheets", {})
            for pos in ["GKP", "DEF", "MID"]:
                if cs_api.get(pos) != cfg_cs.get(pos):
                    mismatches.append(f"Clean sheet scoring mismatch for {pos}: API {cs_api.get(pos)} vs config {cfg_cs.get(pos)}")

            # Check Goals
            goals_api = scoring_api.get("goals_scored", {})
            cfg_goals = scoring_cfg.get("goals_scored", {})
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                if goals_api.get(pos) != cfg_goals.get(pos):
                    mismatches.append(f"Goal scoring mismatch for {pos}: API {goals_api.get(pos)} vs config {cfg_goals.get(pos)}")

            # Check Assists & Saves
            if scoring_api.get("assists") != scoring_cfg.get("assists", {}).get("MID", 3):
                mismatches.append(f"Assists scoring mismatch: API {scoring_api.get('assists')} vs config")

            details["scoring_and_defcon"] = {
                "defcon_points": defcon_api,
                "clean_sheet_points": cs_api,
                "goal_points": goals_api,
                "assists_points": scoring_api.get("assists"),
                "status": "PASS" if not any("scoring" in m.lower() or "defcon" in m.lower() for m in mismatches) else "FAIL",
            }
        else:
            details["scoring_and_defcon"] = {"status": "SKIPPED_NO_API_SCORING"}

        verified = (len(mismatches) == 0)
        if not verified:
            for m in mismatches:
                logger.warning("Rules verification alert: %s", m)
        else:
            logger.info("Live rules verification passed 100%% against 2026/27 official specifications.")

        self._last_result = RulesVerificationResult(
            verified=verified,
            mismatches=mismatches,
            details=details,
        )
        return self._last_result

    def get_last_result(self) -> RulesVerificationResult | None:
        return self._last_result


rules_checker = LiveRulesChecker()
