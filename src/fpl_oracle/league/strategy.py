"""
Mini-League Strategic Advisor.
Adapts tactical recommendations based on league rank, points deficit/lead,
and rival effective ownership profiles.
Implements bounded tie-breaker (margin <= 0.5 xP) for tactical postures (R4).
"""

from typing import Any

import pandas as pd


class LeagueStrategyAdvisor:
    def __init__(self):
        self.tie_breaker_margin_xp = 0.5

    def determine_risk_posture(
        self, points_gap: float, win_probability_pct: float | None = None
    ) -> str:
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

    def apply_tie_breaker(
        self,
        candidate_plans: list[dict[str, Any]],
        posture: str,
        rival_eo_map: dict[int, float],
        margin_xp: float = 0.5,
    ) -> dict[str, Any]:
        """
        Applies tactical risk posture strictly as a bounded tie-breaker on high-xP candidates (R4).
        If top plan beats runner-up by > margin_xp (0.5 xP), model prediction holds without alteration.
        If plans are within margin_xp, posture selects the tactical tie-breaker:
        - DEFENDING_LEAD: Maximizes template coverage (neutralizes rival threats).
        - CHASING_PACK: Maximizes differential leverage (lower rival EO).
        - BALANCED_ATTACK: Retains highest pure xP.
        """
        if not candidate_plans:
            return {
                "selected_plan": None,
                "tie_breaker_applied": False,
                "rationale": "No candidate plans provided.",
            }

        if len(candidate_plans) == 1:
            return {
                "selected_plan": candidate_plans[0],
                "tie_breaker_applied": False,
                "rationale": "Single candidate plan available.",
            }

        sorted_plans = sorted(candidate_plans, key=lambda p: p.get("expected_gain", 0.0), reverse=True)
        top_plan = sorted_plans[0]
        runner_up = sorted_plans[1]

        xp_diff = round(top_plan.get("expected_gain", 0.0) - runner_up.get("expected_gain", 0.0), 2)

        # If gap exceeds margin, model decision is final
        if xp_diff > margin_xp:
            return {
                "selected_plan": top_plan,
                "tie_breaker_applied": False,
                "rationale": f"Pure ML model prediction stands. Gap (+{xp_diff} xP) exceeds the {margin_xp} xP tie-breaker bound.",
            }

        # Candidates within margin
        close_candidates = [
            p for p in sorted_plans
            if (top_plan.get("expected_gain", 0.0) - p.get("expected_gain", 0.0)) <= margin_xp
        ]

        if posture == "DEFENDING_LEAD":
            # Score by template ownership (sum of rival EO of transfers in)
            def _template_score(plan):
                elems = [t.get("element", 0) for t in plan.get("transfers_in", [])]
                return sum(rival_eo_map.get(e, 0.0) for e in elems)

            chosen = max(close_candidates, key=_template_score)
            tie_applied = chosen != top_plan
            rationale = (
                f"Defending Lead Tie-Breaker Applied: Selected candidate matching rival template coverage within {margin_xp} xP margin."
                if tie_applied
                else "Defending Lead: Top ML prediction already aligns with optimal template coverage."
            )

        elif posture == "CHASING_PACK":
            # Score by differential leverage (lower rival EO of transfers in)
            def _diff_score(plan):
                elems = [t.get("element", 0) for t in plan.get("transfers_in", [])]
                return sum(100.0 - rival_eo_map.get(e, 0.0) for e in elems)

            chosen = max(close_candidates, key=_diff_score)
            tie_applied = chosen != top_plan
            rationale = (
                f"Chasing Pack Tie-Breaker Applied: Selected high-differential candidate within {margin_xp} xP margin to exploit rival divergence."
                if tie_applied
                else "Chasing Pack: Top ML prediction already maximizes high-differential leverage."
            )

        else:  # BALANCED_ATTACK
            chosen = top_plan
            tie_applied = False
            rationale = "Balanced Attack: Selected highest baseline expected points without risk bias."

        return {
            "selected_plan": chosen,
            "tie_breaker_applied": tie_applied,
            "posture": posture,
            "margin_xp": margin_xp,
            "rationale": rationale,
        }

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

        leader = rival_squads[0]
        leader_pts = leader["total_points"]
        pts_gap = user_total_points - leader_pts

        user_elements = set(user_squad_df["element"].tolist())
        mode = self.determine_risk_posture(pts_gap, win_probability_pct=win_probability_pct)

        if mode == "DEFENDING_LEAD":
            mode_title = "Defending the Crown (Risk-Minimizing Mode)"
            rationale = (
                f"You are currently leading or well ahead in the mini-league (+{pts_gap} points ahead of 2nd place {leader['player_name']}). "
                "Your objective is to minimize variance, neutralize rival threats, and protect your lead."
            )
            tactics = [
                "Template Protection: Mirror key players owned by your immediate chasers.",
                "Safe Captaincy: Back the consensus high-floor premium captain to minimize variance.",
                "Avoid Hits: Do not take -4 hits unless forced by severe injuries.",
                "Conserve Chips: Save your chips to deploy reactively or alongside rivals' double gameweeks.",
            ]
        elif mode == "BALANCED_ATTACK":
            mode_title = "Contender Attack (Controlled Variance)"
            rationale = (
                f"You are within striking distance ({abs(pts_gap)} points behind leader {leader['player_name']}). "
                "Keep 80% of your squad anchored in elite template players while exploiting 2 high-ceiling differentials."
            )
            tactics = [
                "Target Fixture Swings: Target clubs entering favorable 4-game runs that the leader doesn't own.",
                "Selective Captain Differential: Choose an in-form alternative captain when the leader's captain faces a tough away fixture.",
                "Bank Transfers: Build up to 2-3 banked transfers to execute a sharp mini-wildcard without hits.",
            ]
        else:  # CHASING_PACK
            mode_title = "Chasing the Leader (High-Variance Differential Mode)"
            rationale = (
                f"You are trailing the leader by {abs(pts_gap)} points. Standard template moves will not close this gap. "
                "You must introduce high-ceiling differentials (P90) and diverge strategically on captaincy and chip timing."
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
            "rationale": rationale,
            "tactical_recommendations": tactics,
            "top_rivals_threat_summary": top_threats,
        }


league_strategy_advisor = LeagueStrategyAdvisor()
