"""
Mini-League Strategic Advisor.
Adapts tactical recommendations based on league rank, points deficit/lead,
and rival effective ownership profiles.
"""

from typing import Any

import pandas as pd


class LeagueStrategyAdvisor:
    def determine_risk_posture(self, points_gap: float, win_probability_pct: float | None = None) -> str:
        """
        Determines risk posture:
        - DEFENDING_LEAD: Points lead >= 10 pts or win probability >= 60%
        - CHASING_PACK: Points deficit > 25 pts or win probability <= 20%
        - BALANCED_ATTACK: In contention (-25 to +9 pts)
        """
        if points_gap >= 10.0 or (win_probability_pct is not None and win_probability_pct >= 60.0):
            return "DEFENDING_LEAD"
        elif points_gap < -25.0 or (win_probability_pct is not None and win_probability_pct <= 20.0):
            return "CHASING_PACK"
        else:
            return "BALANCED_ATTACK"

    def evaluate_strategy(
        self,
        user_rank: int,
        user_total_points: int,
        rivals_analysis: dict[str, Any],
        user_squad_df: pd.DataFrame,
        win_probability_pct: float | None = None,
    ) -> dict[str, Any]:
        """
        Formulate tailored strategy mode (DEFENDING_LEAD, CHASING_PACK, BALANCED_ATTACK).
        """
        rival_squads = rivals_analysis.get("rival_squads", [])
        if not rival_squads:
            return {
                "strategy_mode": "BALANCED_ATTACK",
                "mode_title": "Balanced Optimization",
                "rationale": "No specific rival squad data loaded. Optimizing purely for global expected points.",
                "tactical_recommendations": ["Maximize baseline expected points.", "Maintain core template."],
            }

        # Compare with the strongest analysed rival by points (rival_squads is ordered by rank; when the user leads,
        # these are chasers, so the first entry is not necessarily the points leader).
        top = max(rival_squads, key=lambda r: r["total_points"])
        pts_gap = user_total_points - top["total_points"]
        above = [r["total_points"] - user_total_points for r in rival_squads if r["total_points"] > user_total_points]
        below = [user_total_points - r["total_points"] for r in rival_squads if r["total_points"] < user_total_points]

        user_elements = set(user_squad_df["element"].tolist())
        mode = self.determine_risk_posture(pts_gap, win_probability_pct=win_probability_pct)

        if pts_gap > 0:
            standing = f"You lead {top['player_name']} by {pts_gap} points."
        elif pts_gap < 0:
            standing = f"You trail {top['player_name']} by {abs(pts_gap)} points."
        else:
            standing = f"You are level on points with {top['player_name']}."
        # Posture is informational: the plan maximises expected points, so none of the tactics below
        # changes transfer or hit maths.
        if mode == "DEFENDING_LEAD":
            mode_title = "Defending the Crown (Risk-Minimizing Mode)"
            rationale = (
                f"{standing} Win probability or the points lead puts you in front. "
                "Between near-equal options, prefer lower variance and cover the players your chasers own."
            )
            tactics = [
                "Template Protection: Mirror key players owned by your immediate chasers.",
                "Safe Captaincy: Back the consensus high-floor premium captain to minimize variance.",
                "Hits: Take a -4 only when the expected-points model says it pays for itself.",
                "Conserve Chips: Save your chips to deploy reactively or alongside rivals' double gameweeks.",
            ]
        elif mode == "BALANCED_ATTACK":
            mode_title = "Contender Attack (Controlled Variance)"
            rationale = (
                f"{standing} You are within striking distance. "
                "Between near-equal options, keep the template core and add a high-ceiling differential."
            )
            tactics = [
                "Target Fixture Swings: Target clubs entering favorable 4-game runs that the leader doesn't own.",
                "Selective Captain Differential: Choose an in-form alternative captain when the leader's captain faces a tough away fixture.",
                "Bank Transfers: Build up to 2-3 banked transfers to execute a sharp mini-wildcard without hits.",
            ]
        else:  # CHASING_PACK
            mode_title = "Chasing the Leader (High-Variance Differential Mode)"
            rationale = (
                f"{standing} Standard template moves will not close this gap. "
                "Between near-equal options, prefer high-ceiling (P90) differentials and diverge on captaincy and chip timing."
            )
            tactics = [
                "Differential Captaincy: Diverge from the leader's captain when a high-ceiling option presents itself.",
                "Exploit Low-Owned Differentials: Back high-xGI assets with <15% league ownership.",
                "Counter-Chip Timing: Play your Bench Boost or Triple Captain in gameweeks where the leader has already burned their chip.",
                "Calculated Hits: A -4 hit is justified if targeting an immediate fixture swing that offers high multi-week upside.",
            ]

        # Top Threat Analysis
        top_threats = []
        for r in rival_squads[:3]:
            rival_elems = [p["element"] for p in r.get("squad", []) if p.get("is_starter", True)]
            unowned = [e for e in rival_elems if e not in user_elements]
            top_threats.append(
                {
                    "rival_name": r["player_name"],
                    "rank": r["rank"],
                    "points_gap": user_total_points - r["total_points"],
                    "unowned_threat_count": len(unowned),
                    "chips_used": r.get("chips_used", []),
                    "chips_remaining": r.get("chips_remaining", []),
                }
            )

        return {
            "strategy_mode": mode,
            "mode_title": mode_title,
            "points_gap_to_leader": pts_gap,
            "nearest_above_gap": min(above, default=None),
            "nearest_below_gap": min(below, default=None),
            "rationale": rationale,
            "tactical_recommendations": tactics,
            "top_rivals_threat_summary": top_threats,
        }


league_strategy_advisor = LeagueStrategyAdvisor()


def automatic_strategy(context):
    """Describe current complete standings, never infer rank from simulation."""
    if not context or not context.get("standings"):
        return dict(mode_title="Focus on points", status="unavailable", rationale="League standings unavailable.")
    rank = context["user_rank"]
    points = context["user_points"]
    rows = context["standings"]
    above = [row for row in rows if row["rank"] < rank]
    below = [row for row in rows if row["rank"] > rank]
    leader = max(row["points"] for row in rows)
    gap_above = min((row["points"] - points for row in above), default=None)
    gap_below = min((points - row["points"] for row in below), default=None)
    title = (
        "Protect your lead"
        if rank == 1
        else "Close the gap"
        if gap_below is None or (gap_above or 0) <= gap_below
        else "Gain ground, protect your place"
    )
    return dict(
        mode_title=title,
        status="automatic",
        rank=rank,
        gap_to_leader=max(0, leader - points),
        gap_above=gap_above,
        gap_below=gap_below,
        rationale="Maximise points; use rival coverage to choose between close options.",
    )
