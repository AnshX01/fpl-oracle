"""
Efficient Transfer Frontier Planner (Phase 8: WI-8.2).
Computes the Pareto frontier of transfer plans k in {0, 1, 2, ..., K} (where K = min(5, banked_ft + 2)),
evaluating net expected gain over the roll plan with hits explicitly priced,
option-value costs of banked free transfers, and probability of improvement.
Implements Principle P1: Minimal sufficient intervention via knee-point selection.
Zero hardcoded player names in production code.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer, MultiPeriodPlan

logger = logging.getLogger("fpl_oracle.advisor.frontier")


@dataclass
class FrontierPoint:
    transfers_count: int
    plan: MultiPeriodPlan
    moves_summary: str
    transfers_in: list[dict[str, Any]]
    transfers_out: list[dict[str, Any]]
    hits: int
    hit_cost: int
    bank_after: float
    free_transfers_next: int
    horizon_net_xp: float
    net_gain_over_roll: float
    marginal_gain: float
    prob_improvement: float
    what_is_given_up: str
    is_hit_justified: bool = True


@dataclass
class FrontierReport:
    frontier: list[FrontierPoint]
    recommended_k: int
    recommended_point: FrontierPoint
    roll_point: FrontierPoint
    knee_point_selected: int
    argmax_k: int
    max_net_gain: float
    selection_reason: str
    rho_threshold: float = 0.85
    v_ft_cost: float = 0.5


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class FrontierPlanner:
    """
    Computes the efficient frontier of transfer moves (WI-8.2).
    Enforces Principle P1: Minimal sufficient intervention.
    """

    DEFAULT_RHO = 0.85  # Knee-point selection threshold (85% of peak net gain)
    DEFAULT_V_FT = 0.50  # Option value of preserving a banked free transfer (xP)
    DEFAULT_MIN_PROB_HIT = 0.70  # Minimum P(improvement > 0) to recommend a hit

    def __init__(
        self,
        optimizer: MultiPeriodOptimizer | None = None,
        rho: float = DEFAULT_RHO,
        v_ft_cost: float = DEFAULT_V_FT,
        min_prob_hit_justified: float = DEFAULT_MIN_PROB_HIT,
    ) -> None:
        self.optimizer = optimizer or MultiPeriodOptimizer(MultiPeriodConfig())
        self.rho = rho
        self.v_ft_cost = v_ft_cost
        self.min_prob_hit_justified = min_prob_hit_justified

    def compute_frontier(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        horizon_len: int = 5,
        chips_available: list[str] | None = None,
        locked_in: list[int] | None = None,
        locked_out: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        max_k_limit: int | None = None,
    ) -> FrontierReport:
        """
        Solves the horizon problem repeatedly with week-1 transfers constrained to
        k = 0, 1, 2, ..., K where K = min(5, free_transfers + 2).
        Produces the frontier and selects the knee point.
        """
        k_max = min(5, free_transfers + 2)
        if max_k_limit is not None:
            k_max = min(k_max, max_k_limit)

        chips = chips_available or []
        # Frontier explores pure transfer strategies without forcing chips on week 1
        chips_no_w1 = [c for c in chips if c.lower() not in {"wildcard", "freehit"}]

        # 1. Solve Roll Plan (k=0)
        roll_plan = self.optimizer.optimize(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            horizon_len=horizon_len,
            chips_available=chips_no_w1,
            forced_w1_transfers=0,
            locked_in_ids=locked_in,
            locked_out_ids=locked_out,
            excluded_team_ids=excluded_team_ids,
        )

        frontier_points: list[FrontierPoint] = []
        base_obj = roll_plan.total_objective

        # Roll point (k=0)
        roll_ft_next = roll_plan.recommended_action.free_transfers_next
        roll_point = FrontierPoint(
            transfers_count=0,
            plan=roll_plan,
            moves_summary="Roll Transfer (0 moves)",
            transfers_in=[],
            transfers_out=[],
            hits=0,
            hit_cost=0,
            bank_after=roll_plan.recommended_action.bank_end,
            free_transfers_next=roll_ft_next,
            horizon_net_xp=roll_plan.horizon_net_xp,
            net_gain_over_roll=0.0,
            marginal_gain=0.0,
            prob_improvement=0.50,
            what_is_given_up="Banks free transfer (+1 FT next week); foregoes immediate upgrade to preserve optionality.",
            is_hit_justified=True,
        )
        frontier_points.append(roll_point)

        prev_net_gain = 0.0

        # 2. Solve for k = 1, ..., K
        for k in range(1, k_max + 1):
            try:
                plan_k = self.optimizer.optimize(
                    current_squad_df=current_squad_df,
                    player_pool_df=player_pool_df,
                    bank=bank,
                    free_transfers=free_transfers,
                    horizon_projections=horizon_projections,
                    current_gw=current_gw,
                    target_gw=target_gw,
                    horizon_len=horizon_len,
                    chips_available=chips_no_w1,
                    forced_w1_transfers=k,
                    locked_in_ids=locked_in,
                    locked_out_ids=locked_out,
                    excluded_team_ids=excluded_team_ids,
                )
            except Exception as e:
                logger.warning(f"Optimization for frontier k={k} failed: {e}")
                continue

            act = plan_k.recommended_action
            net_gain = round(plan_k.total_objective - base_obj, 3)
            marginal_gain = round(net_gain - prev_net_gain, 3)
            prev_net_gain = net_gain

            hits_k = act.hits
            hit_cost_k = act.hit_cost

            # Estimate P(gain > 0)
            # Under Gaussian difference: sigma_k increases with number of transfers
            # sqrt(k) * 2.81 is empirical standard error of weekly points diff
            sigma_delta = max(1.5, math.sqrt(k) * 2.0)
            z_score = net_gain / sigma_delta
            prob_improvement = round(_normal_cdf(z_score), 3)

            # Hit justification check
            hit_justified = True
            if hits_k > 0:
                hit_justified = (net_gain > 0.0) and (prob_improvement >= self.min_prob_hit_justified)

            # Moves summary
            moves_pairs = []
            for o, i in zip(act.transfers_out, act.transfers_in, strict=False):
                o_id = o.get("element")
                i_id = i.get("element")
                moves_pairs.append(f"OUT ID:{o_id} -> IN ID:{i_id}")
            moves_str = ", ".join(moves_pairs) if moves_pairs else f"{k} Transfers"

            # What is given up
            if hits_k > 0:
                trade_off = (
                    f"Spends -{hit_cost_k} pts on hits; drops banked FTs to 1 next week; "
                    f"requires +{net_gain:.2f} xP to break even."
                )
            else:
                trade_off = (
                    f"Consumes {k} free transfer(s); remaining bank £{act.bank_end:.1f}m; "
                    f"leaves {act.free_transfers_next} FT next week."
                )

            pt = FrontierPoint(
                transfers_count=k,
                plan=plan_k,
                moves_summary=moves_str,
                transfers_in=act.transfers_in,
                transfers_out=act.transfers_out,
                hits=hits_k,
                hit_cost=hit_cost_k,
                bank_after=act.bank_end,
                free_transfers_next=act.free_transfers_next,
                horizon_net_xp=plan_k.horizon_net_xp,
                net_gain_over_roll=net_gain,
                marginal_gain=marginal_gain,
                prob_improvement=prob_improvement,
                what_is_given_up=trade_off,
                is_hit_justified=hit_justified,
            )
            frontier_points.append(pt)

        # 3. Knee-Point Selection (Principle P1: Minimal sufficient intervention)
        max_pt = max(frontier_points, key=lambda p: p.net_gain_over_roll)
        max_k = max_pt.transfers_count
        max_gain = max(0.0, max_pt.net_gain_over_roll)

        threshold_gain = self.rho * max_gain

        # Filter candidate points:
        # - Must capture at least rho * max_gain
        # - If uses hits, must be strictly justified (P >= min_prob_hit and net_gain > 0)
        # - Recommended k can never exceed argmax k
        eligible_points = []
        for pt in frontier_points:
            if pt.transfers_count > max_k:
                continue
            if pt.hits > 0 and not pt.is_hit_justified:
                continue
            if pt.net_gain_over_roll >= threshold_gain:
                eligible_points.append(pt)

        if not eligible_points:
            eligible_points = [roll_point]

        # Choose the smallest k among eligible points
        recommended_pt = min(eligible_points, key=lambda p: (p.transfers_count, p.hits))

        # Check marginal gain condition:
        # If moving to recommended_pt from a smaller k has marginal gain < (v_ft_cost + hit_cost_delta),
        # prefer smaller k
        if recommended_pt.transfers_count > 0:
            for smaller_k in range(recommended_pt.transfers_count):
                smaller_pt = next((p for p in frontier_points if p.transfers_count == smaller_k), None)
                if smaller_pt is not None:
                    marginal = recommended_pt.net_gain_over_roll - smaller_pt.net_gain_over_roll
                    hits_delta = max(0, recommended_pt.hits - smaller_pt.hits)
                    hurdle = self.v_ft_cost + (4.0 * hits_delta)
                    # If marginal gain is lower than the hurdle and smaller_pt captures sufficient gain
                    if marginal < hurdle and smaller_pt.net_gain_over_roll >= (0.75 * max_gain):
                        recommended_pt = smaller_pt
                        break

        # Final safety check: never recommend hit if marginal net gain <= 0
        if recommended_pt.hits > 0 and (not recommended_pt.is_hit_justified or recommended_pt.net_gain_over_roll <= 0.0):
            recommended_pt = next(
                (p for p in frontier_points if p.hits == 0 and p.transfers_count <= free_transfers),
                roll_point,
            )

        assert recommended_pt.transfers_count <= max_k, "Recommended k cannot exceed argmax k"

        reason = (
            f"Selected k={recommended_pt.transfers_count} (Gain: +{recommended_pt.net_gain_over_roll:.2f} xP, "
            f"Hits: {recommended_pt.hits}, P(impr): {recommended_pt.prob_improvement*100:.1f}%) "
            f"capturing >= {self.rho*100:.0f}% of peak frontier gain (+{max_gain:.2f} xP at k={max_k}) "
            f"under Minimal Sufficient Intervention."
        )

        return FrontierReport(
            frontier=frontier_points,
            recommended_k=recommended_pt.transfers_count,
            recommended_point=recommended_pt,
            roll_point=roll_point,
            knee_point_selected=recommended_pt.transfers_count,
            argmax_k=max_k,
            max_net_gain=max_gain,
            selection_reason=reason,
            rho_threshold=self.rho,
            v_ft_cost=self.v_ft_cost,
        )
