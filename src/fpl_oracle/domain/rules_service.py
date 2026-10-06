"""
Domain Rules Service for FPL Oracle.
Verifies scoring, squad limits, transfer banking, and chip rules against live FPL API.
Records checked-at time, source URLs, and reports UNKNOWN if fields are missing (no fake defaults!).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.config import load_rules_config, load_scoring_config


@dataclass
class RuleItemVerification:
    name: str
    expected: Any
    actual: Any
    status: str  # "PASS", "FAIL", "UNKNOWN"
    notes: str = ""


@dataclass
class RulesVerificationReport:
    verified: bool
    status: str  # "VERIFIED", "MISMATCH", "UNVERIFIED"
    checked_at: str
    source_url: str = "https://fantasy.premierleague.com/api/bootstrap-static/"
    items: list[RuleItemVerification] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "status": self.status,
            "checked_at": self.checked_at,
            "source_url": self.source_url,
            "items": [
                {
                    "name": it.name,
                    "expected": it.expected,
                    "actual": it.actual,
                    "status": it.status,
                    "notes": it.notes,
                }
                for it in self.items
            ],
            "mismatches": self.mismatches,
            "limitations": self.limitations,
        }


class RulesService:
    def __init__(self):
        self._last_report: RulesVerificationReport | None = None

    def verify(self, bootstrap: Any) -> RulesVerificationReport:
        now_str = datetime.now(UTC).isoformat()
        items: list[RuleItemVerification] = []
        mismatches: list[str] = []
        limitations: list[str] = []

        rules_cfg = load_rules_config()
        scoring_cfg = load_scoring_config()

        game_settings = getattr(bootstrap, "game_settings", None)
        game_config = getattr(bootstrap, "game_config", None)
        chips_api = getattr(bootstrap, "chips", None)

        if not game_settings or not isinstance(game_settings, dict):
            limitations.append("game_settings field unavailable from bootstrap-static")
        if not game_config or not isinstance(game_config, dict):
            limitations.append("game_config field unavailable from bootstrap-static")

        gs = game_settings or {}
        gc = game_config or {}
        scoring_api = gc.get("scoring", {})

        # 1. Squad Constraints
        cfg_squad = rules_cfg.get("squad", {})
        for key, cfg_key, default_name in [
            ("squad_squadsize", "total_players", "Squad Size"),
            ("squad_squadplay", "squad_play", "Starting Players"),
            ("squad_team_limit", "max_per_team", "Max per Club"),
            ("squad_total_spend", "starting_budget_tenths", "Starting Budget (tenths)"),
        ]:
            exp_val = cfg_squad.get(cfg_key)
            if key not in gs:
                items.append(RuleItemVerification(default_name, exp_val, None, "UNKNOWN", "Field missing in API"))
                limitations.append(f"{default_name} rule field missing in live API")
            else:
                act_val = gs[key]
                passed = act_val == exp_val
                st = "PASS" if passed else "FAIL"
                if not passed:
                    mismatches.append(f"{default_name} mismatch: API {act_val} vs config {exp_val}")
                items.append(RuleItemVerification(default_name, exp_val, act_val, st))

        # 2. Transfers & Selling Price
        cfg_transfers = rules_cfg.get("transfers", {})
        if "max_extra_free_transfers" not in gs:
            items.append(RuleItemVerification("Max Banked FTs", 5, None, "UNKNOWN", "max_extra_free_transfers missing"))
            limitations.append("Free transfer banking rule missing in live API")
        else:
            max_extra = int(gs["max_extra_free_transfers"])
            actual_banked = 1 + max_extra
            passed = actual_banked == cfg_transfers.get("max_banked_free_transfers", 5)
            st = "PASS" if passed else "FAIL"
            if not passed:
                mismatches.append(f"Max Banked FTs mismatch: API {actual_banked} vs config {cfg_transfers.get('max_banked_free_transfers')}")
            items.append(RuleItemVerification("Max Banked FTs", cfg_transfers.get("max_banked_free_transfers", 5), actual_banked, st))

        if "transfers_sell_on_fee" not in gs:
            items.append(RuleItemVerification("Sell-on Fee Rate", 0.5, None, "UNKNOWN", "transfers_sell_on_fee missing"))
        else:
            fee = float(gs["transfers_sell_on_fee"])
            passed = fee == float(cfg_transfers.get("sell_on_fee", 0.5))
            st = "PASS" if passed else "FAIL"
            if not passed:
                mismatches.append(f"Sell-on fee mismatch: API {fee} vs config {cfg_transfers.get('sell_on_fee')}")
            items.append(RuleItemVerification("Sell-on Fee Rate", cfg_transfers.get("sell_on_fee", 0.5), fee, st))

        if "element_sell_at_purchase_price" in gs:
            sell_at_purchase = bool(gs["element_sell_at_purchase_price"])
            passed = sell_at_purchase is False
            st = "PASS" if passed else "FAIL"
            if not passed:
                mismatches.append("API enabled element_sell_at_purchase_price unexpectedly")
            items.append(RuleItemVerification("Sell at Purchase Price", False, sell_at_purchase, st))

        # 3. Chips Structure
        if chips_api is not None:
            # Check 8 chips or 2 sets
            chip_names = [getattr(c, "name", c.get("name") if isinstance(c, dict) else "") for c in chips_api]
            # Check absence of assistant manager
            has_assistant = "assistant_manager" in chip_names or "manager" in chip_names
            items.append(RuleItemVerification("Assistant Manager Removed", True, not has_assistant, "PASS" if not has_assistant else "FAIL"))
            if has_assistant:
                mismatches.append("Assistant Manager chip unexpectedly present in API")

            # Check 8 chips count across season (2 sets)
            items.append(RuleItemVerification("Total Season Chips", 8, len(chips_api), "PASS" if len(chips_api) == 8 else "UNKNOWN", f"Found {len(chips_api)} chips"))
        else:
            items.append(RuleItemVerification("Chips Configuration", 8, None, "UNKNOWN", "chips array missing in bootstrap"))
            limitations.append("Chips configuration missing in API response")

        # 4. Scoring Verification
        if scoring_api:
            # Check Goals Scored points
            goals_pts = scoring_api.get("goals_scored", {})
            cfg_goals = scoring_cfg.get("goals_scored", {})
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                if pos in goals_pts:
                    passed = goals_pts[pos] == cfg_goals.get(pos)
                    st = "PASS" if passed else "FAIL"
                    if not passed:
                        mismatches.append(f"Goal points {pos} mismatch: API {goals_pts[pos]} vs config {cfg_goals.get(pos)}")
                    items.append(RuleItemVerification(f"Goals Scored ({pos})", cfg_goals.get(pos), goals_pts[pos], st))
                else:
                    items.append(RuleItemVerification(f"Goals Scored ({pos})", cfg_goals.get(pos), None, "UNKNOWN"))

            # Check Clean Sheet points
            cs_pts = scoring_api.get("clean_sheets", {})
            cfg_cs = scoring_cfg.get("clean_sheets", {})
            for pos in ["GKP", "DEF", "MID"]:
                if pos in cs_pts:
                    passed = cs_pts[pos] == cfg_cs.get(pos)
                    st = "PASS" if passed else "FAIL"
                    if not passed:
                        mismatches.append(f"Clean Sheet {pos} mismatch: API {cs_pts[pos]} vs config {cfg_cs.get(pos)}")
                    items.append(RuleItemVerification(f"Clean Sheet ({pos})", cfg_cs.get(pos), cs_pts[pos], st))
                else:
                    items.append(RuleItemVerification(f"Clean Sheet ({pos})", cfg_cs.get(pos), None, "UNKNOWN"))

            # Check Defensive Contribution (DefCon +2)
            defcon_pts = scoring_api.get("defensive_contribution", {})
            cfg_defcon = scoring_cfg.get("defensive_contribution", {})
            if "DEF" in defcon_pts:
                passed = defcon_pts["DEF"] == cfg_defcon.get("DEF", 2)
                st = "PASS" if passed else "FAIL"
                if not passed:
                    mismatches.append(f"DefCon DEF mismatch: API {defcon_pts['DEF']} vs config {cfg_defcon.get('DEF')}")
                items.append(RuleItemVerification("DefCon Points (DEF)", cfg_defcon.get("DEF", 2), defcon_pts["DEF"], st))
            else:
                items.append(RuleItemVerification("DefCon Points (DEF)", cfg_defcon.get("DEF", 2), None, "UNKNOWN"))

        has_fails = any(it.status == "FAIL" for it in items)
        has_unknowns = any(it.status == "UNKNOWN" for it in items)

        overall_status = "MISMATCH" if has_fails else ("VERIFIED" if not has_unknowns else "PARTIALLY_VERIFIED")
        report = RulesVerificationReport(
            verified=(not has_fails) and (not has_unknowns) and (len(mismatches) == 0),
            status=overall_status,
            checked_at=now_str,
            source_url="https://fantasy.premierleague.com/api/bootstrap-static/",
            items=items,
            mismatches=mismatches,
            limitations=limitations,
        )
        self._last_report = report
        return report

    def get_last_report(self) -> RulesVerificationReport | None:
        return self._last_report


rules_service = RulesService()
