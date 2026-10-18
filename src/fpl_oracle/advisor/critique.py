"""
Plan Critique, What-If Sensitivity, and Decision Defense (Phase 8: WI-8.6).
Critiques manager-proposed transfers, chips, and captaincy choices against the efficient frontier,
evaluates regret in expected points and P(beat roll), provides numeric reasons,
identifies the strongest alternative, and offers concessions when proposals have qualitative merit.
Zero hardcoded player names in production code.
"""

import logging
import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.health import SquadHealthDiagnosis
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer

logger = logging.getLogger("fpl_oracle.advisor.critique")

_FREE_TRANSFER_CHIPS = {"wildcard", "freehit"}
_CHIP_CODES = {
    "wc": "wildcard",
    "wildcard": "wildcard",
    "fh": "freehit",
    "free hit": "freehit",
    "freehit": "freehit",
    "bb": "bboost",
    "bench boost": "bboost",
    "bboost": "bboost",
    "tc": "3xc",
    "triple captain": "3xc",
    "3xc": "3xc",
}


class PlanEvaluationResult(BaseModel):
    proposed_transfers: list[dict[str, Any]] = Field(default_factory=list)
    proposed_chip: str | None = None
    proposed_captain: int | None = None
    proposed_net_xp: float = 0.0
    roll_net_xp: float = 0.0
    best_alternative_net_xp: float = 0.0
    regret_xp: float = 0.0
    prob_beat_roll: float = 0.50
    verdict: Literal["GOOD", "ACCEPTABLE", "WORSE THAN ALTERNATIVE"] = "ACCEPTABLE"
    strongest_alternative: str = ""
    numeric_reason: str = ""
    concession: str = ""


class OptionComparisonItem(BaseModel):
    name: str
    description: str
    horizon_net_xp: float
    hits: int = 0
    hit_cost: int = 0
    risk_summary: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class OptionComparisonReport(BaseModel):
    options: list[OptionComparisonItem] = Field(default_factory=list)
    recommended_option: str = ""
    summary_verdict: str = ""


class WhyNotExplanation(BaseModel):
    query_action: str
    verdict: str
    explanation: str
    numbers: dict[str, Any] = Field(default_factory=dict)
    alternative_suggested: str = ""


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class PlanCritiqueService:
    """
    Evaluates and critiques manager proposals, providing what-if sensitivity analysis (WI-8.6).
    Enforces Principle P8: Decisive expert voice with numbers.
    """

    def __init__(
        self,
        optimizer: MultiPeriodOptimizer | None = None,
        health_service: SquadHealthDiagnosis | None = None,
        funding_finder: AffordabilityPathFinder | None = None,
    ) -> None:
        self.optimizer = optimizer or MultiPeriodOptimizer(MultiPeriodConfig())
        self.health_service = health_service or SquadHealthDiagnosis()
        self.funding_finder = funding_finder or AffordabilityPathFinder()

    def evaluate_plan(
        self,
        proposed_transfers: list[dict[str, int]],  # [{"out": eid, "in": eid}, ...]
        proposed_chip: str | None,
        proposed_captain: int | None,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        horizon_len: int = 5,
        chips_available: list[str] | None = None,
    ) -> PlanEvaluationResult:
        """
        Evaluates a manager proposal against the roll plan and optimal frontier alternative.
        Returns verdict (GOOD | ACCEPTABLE | WORSE THAN ALTERNATIVE), regret, and numeric justification.
        `bank` is in tenths of a million, like the optimiser and every price column.
        """
        chip = _CHIP_CODES.get((proposed_chip or "").lower().strip(), (proposed_chip or "").lower().strip()) or None
        chip_legal = chip is None or chips_available is None or chip in {c.lower().strip() for c in chips_available}
        # 1. Roll plan baseline
        roll_plan = self.optimizer.optimize(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            horizon_len=horizon_len,
            chips_available=[],
            forced_w1_transfers=0,
        )
        roll_xp = roll_plan.total_objective

        # 2. Optimal alternative plan
        best_alt_plan = self.optimizer.optimize(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            horizon_len=horizon_len,
            chips_available=chips_available or [],
        )
        best_alt_xp = best_alt_plan.total_objective

        # A proposed chip changes the baseline the manager's own moves sit on.
        # Wildcard/Free Hit transfers are free and re-optimised; Bench Boost/Triple
        # Captain keep the proposed transfers, so start from a chip plan with none.
        base_xp = roll_xp
        if chip and chip_legal:
            chip_plan = self.optimizer.optimize(
                current_squad_df=current_squad_df,
                player_pool_df=player_pool_df,
                bank=bank,
                free_transfers=free_transfers,
                horizon_projections=horizon_projections,
                current_gw=current_gw,
                target_gw=target_gw,
                horizon_len=horizon_len,
                chips_available=[chip],
                forced_w1_chip=chip,
                forced_w1_transfers=None if chip in _FREE_TRANSFER_CHIPS else 0,
            )
            base_xp = chip_plan.total_objective

        # 3. Calculate proposed plan's projected net xP
        # Build modified squad after week 1 proposed transfers
        k_transfers = len(proposed_transfers)
        hits = max(0, k_transfers - free_transfers) if chip not in _FREE_TRANSFER_CHIPS else 0
        hit_penalty = hits * 4.0

        pool_map = {int(r["element"]): r for _, r in player_pool_df.iterrows()}
        squad_map = {int(r["element"]): r for _, r in current_squad_df.iterrows()}

        # Verify legality and budget
        net_bank = float(bank)
        for move in proposed_transfers:
            out_id = move.get("out")
            in_id = move.get("in")
            if out_id and out_id in squad_map:
                net_bank += float(squad_map[out_id].get("selling_price", squad_map[out_id].get("value", 50)))
            if in_id and in_id in pool_map:
                net_bank -= float(pool_map[in_id].get("value", 50))

        # Wildcard/Free Hit rebuild the squad under the planner's own budget check.
        is_budget_legal = net_bank >= -0.001 or chip in _FREE_TRANSFER_CHIPS

        # Calculate estimated horizon xP for proposed plan
        # Base from roll plan adjusted for diff in transfer xPs across horizon
        transfer_delta_xp = 0.0
        for move in [] if chip in _FREE_TRANSFER_CHIPS else proposed_transfers:
            out_id = move.get("out")
            in_id = move.get("in")
            for gw in range(target_gw, target_gw + horizon_len):
                gw_df = horizon_projections.get(gw, player_pool_df)
                gw_dict = {int(r["element"]): float(r.get("expected_points", 0.0)) for _, r in gw_df.iterrows()}
                in_xp = gw_dict.get(int(in_id), 0.0) if in_id else 0.0
                out_xp = gw_dict.get(int(out_id), 0.0) if out_id else 0.0
                transfer_delta_xp += in_xp - out_xp

        # Captain adjustment
        captain_delta = 0.0
        if proposed_captain:
            opt_c = roll_plan.recommended_action.captain
            c_now_xp = (
                float(pool_map.get(proposed_captain, {}).get("expected_points", 0.0))
                if proposed_captain in pool_map
                else 0.0
            )
            opt_c_xp = float(pool_map.get(opt_c, {}).get("expected_points", 0.0)) if opt_c in pool_map else 0.0
            captain_delta = c_now_xp - opt_c_xp

        proposed_net_xp = round(base_xp + transfer_delta_xp + captain_delta - hit_penalty, 2)

        regret = round(max(0.0, best_alt_xp - proposed_net_xp), 2)
        gain_vs_roll = proposed_net_xp - roll_xp

        # Probability of beating roll
        sigma = max(1.5, math.sqrt(max(1, k_transfers)) * 2.2)
        z = gain_vs_roll / sigma
        p_beat_roll = round(_normal_cdf(z), 3)

        # Best alt summary
        alt_act = best_alt_plan.recommended_action
        alt_moves = [
            f"OUT ID:{o.get('element')} -> IN ID:{i.get('element')}"
            for o, i in zip(alt_act.transfers_out, alt_act.transfers_in, strict=False)
        ]
        alt_summary = ", ".join(alt_moves) if alt_moves else "Roll Transfer"

        # Concession logic
        concession = ""
        has_high_ceiling = False
        for move in proposed_transfers:
            in_id = move.get("in")
            if in_id and in_id in pool_map:
                p90 = float(pool_map[in_id].get("p90", 0.0))
                xp = float(pool_map[in_id].get("expected_points", 0.0))
                if p90 - xp >= 3.0:
                    has_high_ceiling = True
                    break

        if has_high_ceiling:
            concession = "Targets explosive upside with high ceiling, which could pay off in high-variance game states."
        elif hits > 0 and transfer_delta_xp > hit_penalty:
            concession = "Target delivers strong raw attacking volume, but 4-point penalty narrows the net advantage."
        else:
            concession = "Preserves preferred squad structure, though mathematical projections favor higher-efficiency alternatives."

        # Verdict logic
        if not chip_legal:
            verdict: Literal["GOOD", "ACCEPTABLE", "WORSE THAN ALTERNATIVE"] = "WORSE THAN ALTERNATIVE"
            numeric_reason = f"Plan is illegal: {chip} is not available in this chip set."
        elif not is_budget_legal:
            verdict = "WORSE THAN ALTERNATIVE"
            numeric_reason = f"Plan is illegal: exceeds available budget by £{abs(net_bank) / 10.0:.2f}m."
        elif regret <= 0.6 and (gain_vs_roll >= 0.0 or k_transfers == 0):
            verdict = "GOOD"
            numeric_reason = (
                f"Captures {proposed_net_xp:.2f} horizon xP (regret only {regret:.2f} xP vs optimal {best_alt_xp:.2f} xP). "
                f"P(beat roll) = {p_beat_roll * 100:.1f}%."
            )
        elif regret <= 2.0 and (gain_vs_roll >= -1.0 or hits == 0):
            verdict = "ACCEPTABLE"
            numeric_reason = (
                f"Delivers {proposed_net_xp:.2f} horizon xP ({regret:.2f} xP below optimal {best_alt_xp:.2f} xP). "
                f"Leaves reasonable baseline performance."
            )
        else:
            verdict = "WORSE THAN ALTERNATIVE"
            numeric_reason = (
                f"Underperforms optimal alternative ({best_alt_xp:.2f} xP) by {regret:.2f} xP "
                f"(Hits: -{hit_penalty:.0f} pts, P(beat roll) = {p_beat_roll * 100:.1f}%)."
            )

        return PlanEvaluationResult(
            proposed_transfers=proposed_transfers,
            proposed_chip=chip,
            proposed_captain=proposed_captain,
            proposed_net_xp=proposed_net_xp,
            roll_net_xp=round(roll_xp, 2),
            best_alternative_net_xp=round(best_alt_xp, 2),
            regret_xp=regret,
            prob_beat_roll=p_beat_roll,
            verdict=verdict,
            strongest_alternative=alt_summary,
            numeric_reason=numeric_reason,
            concession=concession,
        )

    def compare_options(
        self,
        options: list[dict[str, Any]],
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        horizon_len: int = 5,
        chips_available: list[str] | None = None,
    ) -> OptionComparisonReport:
        """
        Compares named alternatives side-by-side with net xP, hits, risk, and pros/cons.
        """
        items: list[OptionComparisonItem] = []
        best_name = ""
        best_xp = -1e9

        for opt in options:
            name = str(opt.get("name", "Unnamed Option"))
            desc = str(opt.get("description", ""))
            moves = opt.get("transfers", [])
            chip = opt.get("chip")
            captain = opt.get("captain")

            eval_res = self.evaluate_plan(
                proposed_transfers=moves,
                proposed_chip=chip,
                proposed_captain=captain,
                current_squad_df=current_squad_df,
                player_pool_df=player_pool_df,
                bank=bank,
                free_transfers=free_transfers,
                horizon_projections=horizon_projections,
                current_gw=current_gw,
                target_gw=target_gw,
                horizon_len=horizon_len,
                chips_available=chips_available,
            )

            hits = max(0, len(moves) - free_transfers) if eval_res.proposed_chip not in _FREE_TRANSFER_CHIPS else 0
            hit_cost = hits * 4

            pros = []
            cons = []
            if eval_res.proposed_net_xp - eval_res.roll_net_xp > 0:
                pros.append(f"+{eval_res.proposed_net_xp - eval_res.roll_net_xp:.2f} xP expected gain over rolling")
            else:
                cons.append("Zero or negative gain over roll plan")

            if hits > 0:
                cons.append(f"Incurs -{hit_cost} hit penalty")
            else:
                pros.append("Free transfer - zero hit deduction")

            if eval_res.verdict == "GOOD":
                pros.append("High mathematical efficiency close to optimal")
            elif eval_res.verdict == "WORSE THAN ALTERNATIVE":
                cons.append(f"Leaves {eval_res.regret_xp:.2f} xP on the table vs best alternative")

            risk_desc = f"P(beat roll) = {eval_res.prob_beat_roll * 100:.1f}%"

            item = OptionComparisonItem(
                name=name,
                description=desc,
                horizon_net_xp=eval_res.proposed_net_xp,
                hits=hits,
                hit_cost=hit_cost,
                risk_summary=risk_desc,
                pros=pros,
                cons=cons,
            )
            items.append(item)

            if eval_res.proposed_net_xp > best_xp:
                best_xp = eval_res.proposed_net_xp
                best_name = name

        return OptionComparisonReport(
            options=items,
            recommended_option=best_name,
            summary_verdict=f"Option '{best_name}' yields the highest net return ({best_xp:.2f} xP) balancing hits and upside.",
        )

    def explain_why_not(
        self,
        action_type: str,
        squad_df: pd.DataFrame | None = None,
        plans: list[dict[str, Any]] | None = None,
    ) -> WhyNotExplanation:
        """
        Explains why a contemplated aggressive move (wildcard, hit, any chip) is or is not the
        planner's pick, using the planner's own scored plans. `plans` rows:
        {label, chip, hits, gain_vs_roll, recommended}. No plan data -> no numbers are invented.
        """
        action = action_type.lower().strip()
        chip = _CHIP_CODES.get(action)
        rows = plans or []
        rec = next((r for r in rows if r.get("recommended")), None)

        if action == "hit":
            candidates = [r for r in rows if int(r.get("hits") or 0) > 0]
            query = "Why not take a 4-point hit?"
        elif chip or action == "chip":
            candidates = [r for r in rows if r.get("chip") and (chip is None or r["chip"] == chip)]
            query = f"Why not use {action_type} now?"
        else:
            return WhyNotExplanation(
                query_action=f"Why not {action_type}?",
                verdict="NOT EVALUATED",
                explanation="Only hits and chips can be explained here; use compare_players for player questions.",
            )
        if not rec or not candidates:
            return WhyNotExplanation(
                query_action=query,
                verdict="NOT EVALUATED",
                explanation="The planner did not score this option in the current plan set, so no comparison is available.",
            )

        best = max(candidates, key=lambda r: float(r.get("gain_vs_roll") or 0.0))
        best_gain = round(float(best.get("gain_vs_roll") or 0.0), 2)
        rec_gain = round(float(rec.get("gain_vs_roll") or 0.0), 2)
        numbers: dict[str, Any] = {
            "option": best.get("label"),
            "option_gain_vs_roll_xp": best_gain,
            "option_hits": int(best.get("hits") or 0),
            "option_hit_cost": 4 * int(best.get("hits") or 0),
            "recommended": rec.get("label"),
            "recommended_gain_vs_roll_xp": rec_gain,
        }
        if best is rec:
            return WhyNotExplanation(
                query_action=query,
                verdict="IT IS THE RECOMMENDED MOVE",
                explanation=f"{best.get('label')} is the plan the optimiser picked (+{best_gain} xP vs rolling over the horizon).",
                numbers=numbers,
            )
        numbers["opportunity_cost_xp"] = round(rec_gain - best_gain, 2)
        explanation = (
            f"{best.get('label')} scores {best_gain:+.2f} xP vs rolling over the horizon "
            f"(hit cost -{numbers['option_hit_cost']} included where it applies), "
            f"{rec.get('label')} scores {rec_gain:+.2f}."
        )
        if chip == "wildcard" and squad_df is not None:
            starters = self.health_service.diagnose_squad(squad_df).functional_starters_count
            numbers["functional_starters"] = starters
            explanation += f" The squad has {starters} functional starters."
        return WhyNotExplanation(
            query_action=query,
            verdict="BEATEN BY THE RECOMMENDED PLAN" if best_gain < rec_gain else "NO BETTER THAN THE RECOMMENDED PLAN",
            explanation=explanation,
            numbers=numbers,
            alternative_suggested=str(rec.get("label") or ""),
        )


plan_critique_service = PlanCritiqueService()
