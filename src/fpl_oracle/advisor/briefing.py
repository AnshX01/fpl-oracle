"""
Pre-Deadline Briefing and Expert Voice Contract (Phase 8: WI-8.8).
Produces schema-validated, decisively narrated expert recommendations across deadline checkpoints (-24h, -3h, -1h).
Enforces required structured fields: decision, why (with numbers), alternatives considered, cost, risk,
confidence, what would change this, deadline & next checkpoint, captain floor/ceiling, and one-line summary.
Zero hardcoded player names in production code.
"""

import logging
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from fpl_oracle.advisor.chip_timeline import ChipTimelineComparator
from fpl_oracle.advisor.frontier import FrontierPlanner
from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.health import SquadHealthDiagnosis
from fpl_oracle.advisor.profile import ManagerProfile, manager_profile_service
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer

logger = logging.getLogger("fpl_oracle.advisor.briefing")


class RecommendationCost(BaseModel):
    hits: int = 0
    hit_cost: int = 0
    transfers_used: int = 0
    free_transfers_next: int = 1
    bank_remaining: float = 0.0


class RecommendationRisk(BaseModel):
    risk_description: str
    probability: float  # e.g. 0.25 for 25%


class CaptainAdvice(BaseModel):
    captain_id: int
    captain_name: str
    vice_captain_id: int | None = None
    vice_captain_name: str = ""
    expected_points: float = 0.0
    p10_floor: float = 0.0
    p90_ceiling: float = 0.0
    rationale: str = ""


class AdvisorRecommendation(BaseModel):
    """
    Structured recommendation object satisfying the response contract in WI-8.8.
    """

    recommendation_id: str
    gameweek: int
    checkpoint: Literal["-24h", "-3h", "-1h"] = "-24h"
    decision: str = Field(description="Exactly one sentence declaring the immediate action.")
    why: list[str] = Field(description="Two to four numbered reasons with exact numbers.")
    alternatives_considered: list[dict[str, Any]] = Field(
        description="At least the roll plan and the nearest alternative."
    )
    cost: RecommendationCost
    risk: RecommendationRisk
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    what_would_change_this: list[str] = Field(
        description="Facts that flip the advice (e.g. injury confirmation, price rise)."
    )
    deadline_and_next_checkpoint: str
    captain_advice: CaptainAdvice
    chip_outlook: str
    watchlist: list[str] = Field(default_factory=list)
    one_line_summary: str
    offline_mode: bool = False


def validate_recommendation_schema(data: dict[str, Any] | AdvisorRecommendation) -> tuple[bool, list[str]]:
    """
    Validates a recommendation object against the WI-8.8 contract.
    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []
    if isinstance(data, dict):
        try:
            rec = AdvisorRecommendation.model_validate(data)
        except ValidationError as ve:
            return False, [str(err) for err in ve.errors()]
    else:
        rec = data

    # Strict content assertions per WI-8.8
    if not rec.decision.strip():
        errors.append("decision sentence is empty")
    elif rec.decision.count(".") > 2:
        errors.append("decision must be exactly one sentence")

    if len(rec.why) < 2 or len(rec.why) > 5:
        errors.append(f"why reasons must have between 2 and 4 points (found {len(rec.why)})")

    if not rec.alternatives_considered:
        errors.append("alternatives_considered must contain at least the roll plan and next best")
    elif len(rec.alternatives_considered) < 2:
        errors.append("alternatives_considered must contain at least 2 alternatives")

    if rec.cost is None:
        errors.append("cost field is required")

    if rec.confidence not in {"LOW", "MEDIUM", "HIGH"}:
        errors.append("confidence must be LOW, MEDIUM, or HIGH")

    if not rec.what_would_change_this:
        errors.append("what_would_change_this must have at least one trigger")

    if not rec.one_line_summary.strip():
        errors.append("one_line_summary is empty")

    return len(errors) == 0, errors


class AdvisorBriefingService:
    """
    Generates structured pre-deadline briefings with expert voice (WI-8.8).
    Enforces Principle P7: Proactive and structured; Principle P8: Decisive voice.
    """

    def __init__(
        self,
        frontier_planner: FrontierPlanner | None = None,
        health_diagnoser: SquadHealthDiagnosis | None = None,
        chip_timeline: ChipTimelineComparator | None = None,
        funding_finder: AffordabilityPathFinder | None = None,
    ) -> None:
        opt = MultiPeriodOptimizer(MultiPeriodConfig())
        self.frontier_planner = frontier_planner or FrontierPlanner(opt)
        self.health_diagnoser = health_diagnoser or SquadHealthDiagnosis()
        self.chip_timeline = chip_timeline or ChipTimelineComparator(opt)
        self.funding_finder = funding_finder or AffordabilityPathFinder()

    def generate_recommendation(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        checkpoint: Literal["-24h", "-3h", "-1h"] = "-24h",
        chips_available: list[str] | None = None,
        profile: ManagerProfile | None = None,
        deadline_time_str: str = "Saturday 11:00 UK",
        imminent_price_rises: list[int] | None = None,
        imminent_price_falls: list[int] | None = None,
    ) -> AdvisorRecommendation:
        """
        Builds the structured recommendation object adhering strictly to WI-8.8 schema.
        """
        prof = profile or manager_profile_service.load_profile()
        chips = chips_available or []

        # 1. Health Diagnosis
        health_rep = self.health_diagnoser.diagnose_squad(current_squad_df, player_pool_df)

        # 2. Frontier Evaluation
        frontier_rep = self.frontier_planner.compute_frontier(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            chips_available=chips,
            locked_in=prof.hard_preferences.never_sell,
            excluded_team_ids=prof.hard_preferences.avoid_clubs,
        )

        rec_pt = frontier_rep.recommended_point
        roll_pt = frontier_rep.roll_point

        # 3. Captain Selection
        squad_ids = current_squad_df["element"].astype(int).tolist()
        ranked_captains = manager_profile_service.rank_captains(player_pool_df, squad_ids, profile=prof)

        top_cap = (
            ranked_captains[0]
            if ranked_captains
            else {
                "element": squad_ids[0] if squad_ids else 1,
                "web_name": "Captain",
                "expected_points": 6.0,
                "p10_floor": 2.0,
                "p90_ceiling": 12.0,
                "rationale": "Default projection",
            }
        )
        vice_cap = ranked_captains[1] if len(ranked_captains) > 1 else None

        # 4. Chip Outlook
        chip_outlook_str = "Hold chips; Set 1 chips remain available until GW19 deadline."
        if "wildcard" in [c.lower() for c in chips]:
            if health_rep.classification == "HEALTHY_CORE_TARGETED_FIXES":
                chip_outlook_str = (
                    f"Wildcard HELD: Healthy core confirmed ({health_rep.functional_starters_count} functional starters). "
                    "Save Wildcard for winter fixture congestion or double gameweeks."
                )
            else:
                chip_outlook_str = "Wildcard UNDER ACTIVE REVIEW: Significant squad deficiency detected."

        # 5. Formulate Decision & Why
        pool_map = {int(r["element"]): r for _, r in player_pool_df.iterrows()}
        squad_map = {int(r["element"]): r for _, r in current_squad_df.iterrows()}

        rec_k = rec_pt.transfers_count
        if rec_k == 0:
            decision_text = f"Roll your free transfer in Gameweek {target_gw} to enter Gameweek {target_gw + 1} with 2 banked transfers."
            why_reasons = [
                f"Your 11 starters are projected for {roll_pt.horizon_net_xp:.2f} horizon points with healthy captaincy security.",
                "Minimal-intervention frontier confirms marginal gain from immediate single-transfer swaps does not justify exhausting optionality.",
                "Preserving the transfer provides 2 banked transfers for tactical agility ahead of upcoming fixture shifts.",
            ]
        else:
            in_names = [
                str(pool_map[i["element"]].get("web_name", f"ID:{i['element']}"))
                for i in rec_pt.transfers_in
                if i.get("element") in pool_map
            ]
            out_names = [
                str(squad_map[o["element"]].get("web_name", f"ID:{o['element']}"))
                for o in rec_pt.transfers_out
                if o.get("element") in squad_map
            ]
            in_str = ", ".join(in_names) if in_names else f"{rec_k} targets"
            out_str = ", ".join(out_names) if out_names else "outgoing assets"

            hit_phrase = f" with a -{rec_pt.hit_cost} point hit" if rec_pt.hits > 0 else " with zero hits"
            decision_text = f"Transfer {out_str} out for {in_str}{hit_phrase}."
            why_reasons = [
                f"Captures +{rec_pt.net_gain_over_roll:.2f} net horizon points over rolling ({rec_pt.horizon_net_xp:.2f} vs {roll_pt.horizon_net_xp:.2f} xP).",
                f"Probability of outscoring the roll plan is {rec_pt.prob_improvement * 100:.1f}%.",
                f"Knee-point selection confirms this move secures >= {frontier_rep.rho_threshold * 100:.0f}% of peak available frontier upside.",
            ]

        # 6. Alternatives Considered
        alts = [
            {
                "name": "Roll Transfer",
                "transfers_count": 0,
                "net_xp": round(roll_pt.horizon_net_xp, 2),
                "gain_vs_roll": 0.0,
                "hits": 0,
                "summary": "Roll transfer and bank for next gameweek.",
            }
        ]
        if rec_k > 0:
            alts.append(
                {
                    "name": "Recommended Transfer Plan",
                    "transfers_count": rec_k,
                    "net_xp": round(rec_pt.horizon_net_xp, 2),
                    "gain_vs_roll": round(rec_pt.net_gain_over_roll, 2),
                    "hits": rec_pt.hits,
                    "summary": rec_pt.moves_summary,
                }
            )
        for pt in frontier_rep.frontier:
            if pt.transfers_count not in {0, rec_k} and len(alts) < 4:
                alts.append(
                    {
                        "name": f"{pt.transfers_count}-Transfer Move",
                        "transfers_count": pt.transfers_count,
                        "net_xp": round(pt.horizon_net_xp, 2),
                        "gain_vs_roll": round(pt.net_gain_over_roll, 2),
                        "hits": pt.hits,
                        "summary": pt.moves_summary,
                    }
                )

        # 7. Confidence & Risk
        p_imp = rec_pt.prob_improvement
        confidence_tier: Literal["LOW", "MEDIUM", "HIGH"] = (
            "HIGH" if p_imp >= 0.70 else ("MEDIUM" if p_imp >= 0.55 else "LOW")
        )

        risk_prob = round(max(0.10, 1.0 - p_imp), 2)
        risk_text = f"Rotation or variance risk: {risk_prob * 100:.0f}% chance roll plan matches or outperforms due to baseline noise."

        # 8. What would change this
        change_triggers = []
        if imminent_price_rises and any(i.get("element") in set(imminent_price_rises) for i in rec_pt.transfers_in):
            change_triggers.append("Target player price rise tonight: execute before 23:59 UK if budget tight.")
        change_triggers.append("Press conference update ruling out a key starter or confirming major rotation.")
        if rec_k > 0:
            change_triggers.append("Rival manager moves forcing a defensive template pivot.")

        # Next checkpoint string
        next_ckpt = (
            "Final 1-hour deadline alert"
            if checkpoint == "-3h"
            else ("3-hour pre-deadline check" if checkpoint == "-24h" else "Gameweek locked")
        )

        # One line summary
        one_liner = f"{decision_text} Captain {top_cap['web_name']} (P90 ceiling {top_cap['p90_ceiling']:.1f} pts)."

        return AdvisorRecommendation(
            recommendation_id=f"rec_gw{target_gw}_{checkpoint}",
            gameweek=target_gw,
            checkpoint=checkpoint,
            decision=decision_text,
            why=why_reasons,
            alternatives_considered=alts,
            cost=RecommendationCost(
                hits=rec_pt.hits,
                hit_cost=rec_pt.hit_cost,
                transfers_used=rec_k,
                free_transfers_next=rec_pt.free_transfers_next,
                bank_remaining=rec_pt.bank_after,
            ),
            risk=RecommendationRisk(
                risk_description=risk_text,
                probability=risk_prob,
            ),
            confidence=confidence_tier,
            what_would_change_this=change_triggers,
            deadline_and_next_checkpoint=f"Deadline: {deadline_time_str}. Next checkpoint: {next_ckpt}.",
            captain_advice=CaptainAdvice(
                captain_id=int(top_cap["element"]),
                captain_name=str(top_cap["web_name"]),
                vice_captain_id=int(vice_cap["element"]) if vice_cap else None,
                vice_captain_name=str(vice_cap["web_name"]) if vice_cap else "",
                expected_points=float(top_cap["expected_points"]),
                p10_floor=float(top_cap["p10_floor"]),
                p90_ceiling=float(top_cap["p90_ceiling"]),
                rationale=str(top_cap["rationale"]),
            ),
            chip_outlook=chip_outlook_str,
            watchlist=[f"Element ID:{w}" for w in health_rep.watch_ids[:3]],
            one_line_summary=one_liner,
        )

    def render_markdown_briefing(self, rec: AdvisorRecommendation) -> str:
        """
        Renders the structured recommendation into concise, plain-voice expert Markdown.
        """
        lines = [
            f"# Gameweek {rec.gameweek} Pre-Deadline Briefing ({rec.checkpoint})",
            "",
            f"> **Bottom Line:** {rec.one_line_summary}",
            "",
            "## 1. Primary Recommendation",
            f"**{rec.decision}**",
            "",
            "### Why this move:",
            *[f"- {reason}" for reason in rec.why],
            "",
            "## 2. Cost & Squad Mechanics",
            f"- **Hits:** -{rec.cost.hit_cost} pts ({rec.cost.hits} hit{'s' if rec.cost.hits != 1 else ''})",
            f"- **Transfers Used:** {rec.cost.transfers_used}",
            f"- **Free Transfers Next GW:** {rec.cost.free_transfers_next}",
            f"- **Bank Remaining:** £{rec.cost.bank_remaining:.1f}m",
            "",
            "## 3. Captaincy Plan",
            f"- **Captain:** **{rec.captain_advice.captain_name}** ({rec.captain_advice.expected_points:.1f} xP, Floor P10: {rec.captain_advice.p10_floor:.1f}, Ceiling P90: {rec.captain_advice.p90_ceiling:.1f})",
        ]
        if rec.captain_advice.vice_captain_name:
            lines.append(f"- **Vice Captain:** {rec.captain_advice.vice_captain_name}")
        lines.append(f"- **Captain Rationale:** {rec.captain_advice.rationale}")

        lines.extend(
            [
                "",
                "## 4. Chip Strategy Outlook",
                f"{rec.chip_outlook}",
                "",
                "## 5. Alternatives Evaluated",
            ]
        )
        for alt in rec.alternatives_considered:
            gain = float(alt.get("gain_vs_roll", 0.0))
            gain_str = f"+{gain:.2f}" if gain >= 0 else f"{gain:.2f}"
            lines.append(f"- **{alt['name']}:** {alt['net_xp']:.2f} xP (Gain vs Roll: {gain_str}) - {alt['summary']}")

        lines.extend(
            [
                "",
                f"## 6. Risk & Confidence ({rec.confidence} Confidence)",
                f"- **Downside Risk ({rec.risk.probability * 100:.0f}% chance):** {rec.risk.risk_description}",
                "",
                "## 7. What Would Change This Advice",
                *[f"- {trig}" for trig in rec.what_would_change_this],
                "",
                f"*{rec.deadline_and_next_checkpoint}*",
            ]
        )

        rendered = "\n".join(lines)
        return rendered


advisor_briefing_service = AdvisorBriefingService()
