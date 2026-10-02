"""
Mini-League Strategic Advisor.
Adapts tactical recommendations based on league rank, points deficit/lead,
and rival effective ownership profiles.
"""

from typing import Any

import pandas as pd


class LeagueStrategyAdvisor:
    def __init__(self):
        pass

    def evaluate_strategy(
        self,
        user_rank: int,
        user_total_points: int,
        rivals_analysis: dict[str, Any],
        user_squad_df: pd.DataFrame
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
                "tactical_recommendations": ["Maximize baseline expected points.", "Maintain core template."]
            }

        leader = rival_squads[0]
        leader_pts = leader["total_points"]
        pts_gap = user_total_points - leader_pts

        user_elements = set(user_squad_df["element"].tolist())

        if user_rank == 1 or pts_gap >= 0:
            mode = "DEFENDING_LEAD"
            mode_title = "Defending the Crown (Risk-Minimizing Mode)"
            rationale = (
                f"You are currently leading the mini-league (+{pts_gap} points ahead of 2nd place {leader['player_name']}). "
                "Your objective is to minimize variance, neutralize rival threats, and protect your lead."
            )
            tactics = [
                "Template Protection: Mirror key players owned by your immediate chasers.",
                "Safe Captaincy: Back the consensus high-floor premium captain (e.g. Haaland/Saka) to minimize variance.",
                "Avoid Hits: Do not take -4 hits unless forced by severe injuries.",
                "Conserve Chips: Save your chips to deploy reactively or alongside rivals' double gameweeks."
            ]
        elif pts_gap > -25:
            mode = "BALANCED_ATTACK"
            mode_title = "Contender Attack (Controlled Variance)"
            rationale = (
                f"You are within striking distance ({abs(pts_gap)} points behind leader {leader['player_name']}). "
                "Keep 80% of your squad anchored in elite template players while exploiting 2 high-ceiling differentials."
            )
            tactics = [
                "Target Fixture Swings: Target clubs entering favorable 4-game runs that the leader doesn't own.",
                "Selective Captain Differential: Choose an in-form alternative captain when the leader's captain faces a tough away fixture.",
                "Bank Transfers: Build up to 2-3 banked transfers to execute a sharp mini-wildcard without hits."
            ]
        else:
            mode = "CHASING_PACK"
            mode_title = "Chasing the Leader (High-Variance Differential Mode)"
            rationale = (
                f"You are trailing the leader by {abs(pts_gap)} points. Standard template moves will not close this gap. "
                "You must introduce high-ceiling differentials (P90) and diverge strategically on captaincy and chip timing."
            )
            tactics = [
                "Differential Captaincy: You MUST diverge from the leader's captain when a high-ceiling option presents itself.",
                "Exploit Low-Owned Differentials: Back high-xGI assets with <15% league ownership.",
                "Counter-Chip Timing: Play your Bench Boost or Triple Captain in gameweeks where the leader has already burned their chip.",
                "Calculated Hits: A -4 hit is justified if targeting an immediate fixture swing that offers high multi-week upside."
            ]

        # Top Threat Analysis
        top_threats = []
        for r in rival_squads[:3]:
            rival_elems = [p["element"] for p in r.get("squad", []) if p.get("is_starter", True)]
            unowned = [e for e in rival_elems if e not in user_elements]
            top_threats.append({
                "rival_name": r["player_name"],
                "rank": r["rank"],
                "points_gap": user_total_points - r["total_points"],
                "unowned_threat_count": len(unowned),
                "chips_used": r.get("chips_used", [])
            })

        return {
            "strategy_mode": mode,
            "mode_title": mode_title,
            "points_gap_to_leader": pts_gap,
            "rationale": rationale,
            "tactical_recommendations": tactics,
            "top_rivals_threat_summary": top_threats
        }

league_strategy_advisor = LeagueStrategyAdvisor()
