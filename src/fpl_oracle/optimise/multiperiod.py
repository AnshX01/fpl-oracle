"""
Multi-Period Mixed-Integer Linear Programming (MILP) Transfer Optimizer (WI-2.1 to WI-2.5).

Formulation:
- Exact multi-gameweek horizon T = {t0, ..., t0 + H - 1} (H=1..12, default 8).
- Simultaneous joint optimization of:
  * Squad composition across all gameweeks (flow conservation)
  * Starting XI formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD)
  * Captaincy and vice-captaincy
  * Transfers in and out per gameweek
  * Free transfer banking (1 to 5 FTs, Rule R4)
  * Hit penalties (-4 points per extra transfer)
  * Chip deployments (Wildcard, Free Hit, Bench Boost, Triple Captain)
  * Team value and selling price oracle (WI-1.4)
  * Terminal banked FT value V_ft and terminal bank valuation lambda_bank
- Solvers supported: HiGHS (via highspy / pulp.HiGHS) and COIN-OR CBC (via pulp.PULP_CBC_CMD).
- Plan comparison set generation (Roll, Best 1-transfer, Best hit, Chip plans, Top-K moves).
- Scenario-based hit justification policy (WI-2.5).
"""

import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd
import pulp

from fpl_oracle.optimise.hit_policy import expected_transfer_gain
from fpl_oracle.optimise.progress import report_search_progress
from fpl_oracle.rules.scoring import (
    HIT_COST_POINTS,
    MAX_BANKED_FREE_TRANSFERS,
    SET_1_LAST_GW,
)
from fpl_oracle.rules.transfers import (
    next_free_transfers,
    sell_price,
)

logger = logging.getLogger(__name__)

# A one-week Free Hit squad is picked from the best players of that week alone.
FREE_HIT_PRUNING_LIMITS = {"DEF": 30, "MID": 30, "FWD": 20, "GKP": 10}


def best_lineup(
    squad: list[int], position: dict[int, str], xp: dict[int, float], bench_weight: float
) -> tuple[list[int], list[int], int, int | None]:
    """
    Exact (starters, bench, captain, vice) for a 15-man squad: best formation by
    (1 - bench_weight) * starters + captain, ties by total starter xP and then by id.
    The MILP's lineup variables are continuous (integral at a vertex, but a tie can leave a fractional
    split), so the reported lineup is rebuilt from the chosen squad instead of rounded from them.
    """
    ranked = {
        pos: sorted((p for p in squad if position[p] == pos), key=lambda p: (-xp.get(p, 0.0), p))
        for pos in ("GKP", "DEF", "MID", "FWD")
    }
    best: tuple[tuple[float, float], list[int]] | None = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3:
                continue
            starters = ranked["GKP"][:1] + ranked["DEF"][:d] + ranked["MID"][:m] + ranked["FWD"][:f]
            total = sum(xp.get(p, 0.0) for p in starters)
            captain = max(starters, key=lambda p: (xp.get(p, 0.0), -p))
            key = ((1.0 - bench_weight) * total + xp.get(captain, 0.0), total)
            if best is None or key > best[0]:
                best = (key, starters)
    assert best is not None
    starters = best[1]
    captain = max(starters, key=lambda p: (xp.get(p, 0.0), -p))
    rest = [p for p in starters if p != captain]
    vice = max(rest, key=lambda p: (xp.get(p, 0.0), -p)) if rest else None
    bench = sorted((p for p in squad if p not in starters), key=lambda p: (-xp.get(p, 0.0), p))
    return starters, bench, captain, vice


class MilpInfeasible(RuntimeError):
    """The MILP has no feasible plan under the given rules."""


class MilpNoSolution(MilpInfeasible):
    """The solver ran out of time before finding any feasible plan (feasibility is not disproved)."""


def _xp_map(df: pd.DataFrame) -> dict[int, float]:
    """element -> expected_points for one gameweek frame (0.0 when the column is absent)."""
    if "expected_points" not in df.columns:
        return {int(e): 0.0 for e in df["element"]}
    return {int(e): float(v) for e, v in zip(df["element"], df["expected_points"], strict=True)}


@dataclass
class MultiPeriodConfig:
    horizon_len: int = 8
    gamma: float = 0.95
    solver_name: Literal["highs", "cbc"] = "highs"
    mip_gap: float = 1e-4  # 0.01% (rel gap <= 1e-4, abs gap <= 0.05 xP on 500 xP horizon)
    time_limit_sec: int = 50  # per solve; the best incumbent and its gap are kept and reported when it is hit
    mip_heuristic_effort: float = 0.7  # HiGHS primal-heuristic share; 0.7 found ~3% better incumbents than 0.05 in 60 s
    wave1_time_limit_sec: float = 30.0  # first solve of each candidate; polishing does the rest
    polish_window: int = 4  # weeks re-optimised together in one local-search step (step = half a window)
    polish_window_time_limit_sec: float = 20.0  # 4-week windows solve to optimality in ~2-15 s
    # Owner rule: no back-to-back chips. Chip weeks must be at least this many gameweeks apart (1 = no rule).
    # With no blank/double gameweeks announced, flat projections otherwise bunch every chip into the first weeks.
    min_gameweeks_between_chips: int = 2
    # Decision rule: act only if P(true gain > 0) >= this, given the model's fitted estimation error
    # (docs/decision_model.md section 3). 0.75 = act on a 3-to-1 edge; set before looking at any live result.
    decision_confidence: float = 0.75
    w_bench_mean: float = 0.05  # flat weight on every bench player; bench order is chosen after the solve
    lambda_bank: float = 0.05  # 0.05 xP per 1.0m (£0.1m = 0.005 xP)
    chip_week_ft_rule: Literal["preserve", "preserve_plus_one"] = "preserve"
    pruning_limits: dict[str, int] = field(default_factory=lambda: {"GKP": 15, "DEF": 45, "MID": 45, "FWD": 25})
    enablers_count: int = 5
    ft_terminal_increments: tuple[float, float, float, float] = (1.0, 0.8, 0.6, 0.4)


@dataclass
class GameweekAction:
    gameweek: int
    transfers_in: list[dict[str, Any]]
    transfers_out: list[dict[str, Any]]
    transfers_count: int
    free_transfers_available: int
    free_transfers_used: int
    free_transfers_next: int
    hits: int
    hit_cost: int
    active_chip: str | None
    bank_end: float
    starters: list[int]
    captain: int
    vice_captain: int | None
    bench: list[int]
    lineup_xp: float
    captain_xp: float
    bench_xp: float
    net_xp: float
    # Fielded 15-man squad: the owned squad after this week's transfers, or the temporary FH squad.
    squad: list[int] = field(default_factory=list)


@dataclass
class MultiPeriodPlan:
    plan_type: str
    target_gw: int
    horizon_gws: list[int]
    recommended_action: GameweekAction
    trajectory: list[GameweekAction]
    total_objective: float
    horizon_net_xp: float
    horizon_hits: int
    horizon_gain_vs_roll: float
    immediate_gain: float
    hit_verdict: str
    hit_delta: float
    hit_prob_gain: float
    solve_time_sec: float
    solver_status: str
    solver_gap: float = 0.0
    is_lower_bound: bool = False
    objective_decomposition: dict[str, float] = field(default_factory=dict)
    top_k_alternatives: list[dict[str, Any]] = field(default_factory=list)
    chip_plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Integer decisions of the solve ({("x", player, gw) | (chip, gw): 0/1}), reused as a MIP start for related solves.
    integer_solution: dict[tuple[Any, ...], int] = field(default_factory=dict, repr=False)


def _rows_in_millions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transfer rows with solver prices (tenths) converted to millions, the unit every surface reads."""
    return [{**r, **{k: r[k] / 10.0 for k in ("cost", "sell_price") if k in r}} for r in rows]


class _WarmStartHiGHS(pulp.HiGHS):  # type: ignore[misc,name-defined]
    """HiGHS with a MIP start. HiGHS completes a partial assignment of the integer columns (and drops an
    infeasible one), so a good start never hurts and a feasible one guarantees an incumbent."""

    def __init__(self, start: dict[pulp.LpVariable, float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start = start

    def callSolver(self, lp: Any) -> None:  # noqa: N802 (PuLP API name)
        if self.start:
            idx = np.fromiter((v.index for v in self.start), dtype=np.int32, count=len(self.start))
            val = np.fromiter(self.start.values(), dtype=np.float64, count=len(self.start))
            lp.solverModel.setSolution(len(idx), idx, val)
        super().callSolver(lp)


class MultiPeriodOptimizer:
    """
    Mixed-Integer Linear Programming Multi-Period Optimizer for FPL (WI-2.1 to WI-2.5).
    """

    def __init__(self, config: MultiPeriodConfig | None = None) -> None:
        self.config = config or MultiPeriodConfig()

    def _get_solver(
        self, time_limit_sec: float | None = None, start: dict[pulp.LpVariable, float] | None = None
    ) -> pulp.LpSolver:
        """Selects HiGHS if available and configured, else falls back to bundled CBC."""
        limit = self.config.time_limit_sec if time_limit_sec is None else time_limit_sec
        if self.config.solver_name == "highs":
            try:
                highs = _WarmStartHiGHS(
                    start or {},
                    gapRel=self.config.mip_gap,
                    timeLimit=limit,
                    msg=False,
                    mip_heuristic_effort=self.config.mip_heuristic_effort,
                )
                if highs.available():
                    return highs
            except Exception as e:
                logger.warning(f"HiGHS solver initialization failed: {e}. Falling back to CBC.")

        return pulp.PULP_CBC_CMD(gapRel=self.config.mip_gap, timeLimit=limit, msg=False)

    def _free_hit_squad(
        self,
        week_df: pd.DataFrame,
        xp: dict[int, float],
        costs: dict[int, float],
        budget: float,
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[int],
    ) -> dict[str, Any] | None:
        """
        Best one-week squad for a Free Hit. The FH squad is picked from scratch under the team-value
        budget and is not carried to the next week, so its value depends only on that week's xP.
        Solving it separately keeps the 12-week plan MILP free of a second 15-man squad per week.
        """
        w_bench = self.config.w_bench_mean
        ids = [int(e) for e in week_df["element"] if int(e) not in locked_out]
        info = dict(
            zip(week_df["element"].astype(int), zip(week_df["position"], week_df["team"], strict=True), strict=True)
        )
        ids = [p for p in ids if info[p][1] not in excluded_teams]
        prob = pulp.LpProblem("FPL_FreeHit_Squad", pulp.LpMaximize)
        xf = {p: pulp.LpVariable(f"fx_{p}", cat=pulp.LpBinary) for p in ids}
        sf = {p: pulp.LpVariable(f"fs_{p}", cat=pulp.LpBinary) for p in ids}
        cf = {p: pulp.LpVariable(f"fc_{p}", cat=pulp.LpBinary) for p in ids}
        by_pos = {pos: [p for p in ids if info[p][0] == pos] for pos in ("GKP", "DEF", "MID", "FWD")}
        for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            prob += pulp.lpSum(xf[p] for p in by_pos[pos]) == n
        for team in {info[p][1] for p in ids}:
            prob += pulp.lpSum(xf[p] for p in ids if info[p][1] == team) <= 3
        prob += pulp.lpSum(costs[p] * xf[p] for p in ids) <= budget
        for p in ids:
            prob += sf[p] <= xf[p]
            prob += cf[p] <= sf[p]
        prob += pulp.lpSum(sf.values()) == 11
        prob += pulp.lpSum(sf[p] for p in by_pos["GKP"]) == 1
        for pos, lo, hi in (("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)):
            prob += pulp.lpSum(sf[p] for p in by_pos[pos]) >= lo
            prob += pulp.lpSum(sf[p] for p in by_pos[pos]) <= hi
        prob += pulp.lpSum(cf.values()) == 1
        for p in locked_in:
            if p in sf:
                prob += sf[p] == 1
        prob += pulp.lpSum(xp.get(p, 0.0) * ((1.0 - w_bench) * sf[p] + cf[p] + w_bench * xf[p]) for p in ids)
        if prob.solve(self._get_solver(min(self.config.time_limit_sec, 20))) != pulp.constants.LpStatusOptimal:
            return None
        chosen = [p for p in ids if (xf[p].varValue or 0.0) > 0.5]
        starters = [p for p in chosen if (sf[p].varValue or 0.0) > 0.5]
        captain = next((p for p in starters if (cf[p].varValue or 0.0) > 0.5), None)
        if len(chosen) != 15 or len(starters) != 11 or captain is None:
            return None
        bench = [p for p in chosen if p not in starters]
        vice = max((p for p in starters if p != captain), key=lambda p: xp.get(p, 0.0))
        value = (
            sum(xp.get(p, 0.0) for p in starters) + xp.get(captain, 0.0) + w_bench * sum(xp.get(p, 0.0) for p in bench)
        )
        return {
            "value": float(value),
            "squad": chosen,
            "starters": starters,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice,
        }

    def prune_candidates(
        self,
        pool_df: pd.DataFrame,
        squad_ids: set[int],
        horizon_gws: list[int],
        horizon_projections: dict[int, pd.DataFrame],
        limits: dict[str, int] | None = None,
        enablers_count: int = 5,
    ) -> pd.DataFrame:
        """
        Prunes candidate pool to top N per position by horizon-weighted xP,
        plus all owned players, plus cheapest 5 enablers per position (WI-2.3).
        """
        limits = limits or self.config.pruning_limits
        gamma = self.config.gamma

        # Compute horizon-weighted xP
        all_elements = pool_df["element"].unique()
        hw_xp: dict[int, float] = {int(e): 0.0 for e in all_elements}

        for idx, gw in enumerate(horizon_gws):
            df_gw = horizon_projections.get(gw, pool_df)
            gw_map = _xp_map(df_gw)
            disc = gamma**idx
            for eid in all_elements:
                hw_xp[int(eid)] += disc * gw_map.get(int(eid), 0.0)

        selected_eids: set[int] = set(squad_ids)

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_df = pool_df[pool_df["position"] == pos].copy()
            if pos_df.empty:
                continue

            pos_df["hw_xp"] = pos_df["element"].map(hw_xp).fillna(0.0)
            n_keep = limits.get(pos, 30)

            # Top N by horizon-weighted xP
            top_n = pos_df.sort_values(by="hw_xp", ascending=False).head(n_keep)
            selected_eids.update(top_n["element"].astype(int).tolist())

            # Cheapest enablers by xP/cost
            pos_df["eff"] = pos_df["hw_xp"] / pos_df["value"].clip(lower=38.0)
            enablers = pos_df.sort_values(by=["value", "eff"], ascending=[True, False]).head(enablers_count)
            selected_eids.update(enablers["element"].astype(int).tolist())

        pruned = pool_df[pool_df["element"].isin(selected_eids)].copy().reset_index(drop=True)
        return pruned

    def optimize(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        horizon_len: int | None = None,
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        chips_available: set[str] | list[str] | None = None,
        previous_chip: str | None = None,
        forced_w1_transfers: int | None = None,
        forced_w1_chip: str | None = None,
        forced_chip_schedule: dict[int, str] | None = None,
        min_w1_hits: int | None = None,
        forbidden_w1_moves: list[tuple[set[int], set[int]]] | None = None,
        hit_cooldown_until: int = 0,
        chips_by_set: dict[int, list[str] | set[str]] | None = None,
        time_limit_sec: float | None = None,
        fh_cache: dict[Any, Any] | None = None,
        no_chip_w1: bool = False,
        max_w1_hits: int | None = None,
        warm_start: dict[tuple[Any, ...], int] | None = None,
        fix_except_gws: list[int] | None = None,
    ) -> MultiPeriodPlan:
        """
        Solves the Multi-Period MILP formulation over horizon T.
        Supports conditional forcing for alternative plan generation.
        hit_cooldown_until: no hit in any gameweek before this one (hit-ledger guard).
        chips_by_set: chips still available per set ({1: [...], 2: [...]}); overrides chips_available.
        fh_cache: shared per-week Free Hit squad solves, reused when several plans are solved on one input.
        no_chip_w1: no chip in the first gameweek (later gameweeks stay free), without forcing a transfer count.
        max_w1_hits: cap on first-gameweek hits (0 = only free transfers this week).
        warm_start: integer_solution of a related plan used as the MIP start; default is keeping the squad.
        fix_except_gws: with warm_start, fix squads and chips of every other gameweek to warm_start (local search).
        """
        t_start = time.perf_counter()
        h_len = horizon_len or self.config.horizon_len
        horizon_gws = [target_gw + offset for offset in range(h_len) if target_gw + offset <= 38]
        if not horizon_gws:
            horizon_gws = [target_gw]
        H = len(horizon_gws)
        t0 = horizon_gws[0]

        squad_df = current_squad_df.copy()
        if "value" not in squad_df.columns and "now_cost" in squad_df.columns:
            squad_df["value"] = squad_df["now_cost"]
        if "now_cost" not in squad_df.columns and "value" in squad_df.columns:
            squad_df["now_cost"] = squad_df["value"]
        if "selling_price" not in squad_df.columns:
            squad_df["selling_price"] = squad_df["value"]
        if "purchase_price" not in squad_df.columns:
            squad_df["purchase_price"] = squad_df["value"]

        squad_ids = set(squad_df["element"].astype(int).tolist())

        # Prune candidates (tight pool for Wildcard as required by Item 3 to guarantee gap <= 1e-3)
        is_wc = (forced_w1_chip is not None and forced_w1_chip.lower() == "wildcard") or (
            forced_chip_schedule is not None and any(c.lower() == "wildcard" for c in forced_chip_schedule.values())
        )
        wc_limits = {"DEF": 25, "MID": 25, "FWD": 15, "GKP": 10} if is_wc else None
        pruned_pool = self.prune_candidates(
            player_pool_df,
            squad_ids=squad_ids,
            horizon_gws=horizon_gws,
            horizon_projections=horizon_projections,
            limits=wc_limits or self.config.pruning_limits,
        )

        players = pruned_pool["element"].astype(int).tolist()
        player_dict = {int(r["element"]): dict(r) for _, r in pruned_pool.iterrows()}

        # Locked / excluded constraints
        locked_in = set(locked_in_ids or [])
        locked_out = set(locked_out_ids or [])
        excluded_teams = set(excluded_team_ids or [])

        # Prices in tenths
        init_bank = float(bank)
        buy_costs = {p: float(player_dict[p]["value"]) for p in players}

        squad_sell_map = {
            int(r["element"]): float(sell_price(int(round(float(r["purchase_price"]))), int(round(float(r["value"])))))
            for _, r in squad_df.iterrows()
        }
        sell_vals = {p: squad_sell_map.get(p, buy_costs[p]) for p in players}

        # Positions & clubs
        gkps = [p for p in players if player_dict[p]["position"] == "GKP"]
        defs = [p for p in players if player_dict[p]["position"] == "DEF"]
        mids = [p for p in players if player_dict[p]["position"] == "MID"]
        fwds = [p for p in players if player_dict[p]["position"] == "FWD"]

        all_teams = sorted(set(player_dict[p]["team"] for p in players))
        club_map = {tm: [p for p in players if player_dict[p]["team"] == tm] for tm in all_teams}

        # Initialize PuLP Problem
        prob = pulp.LpProblem("FPL_MultiPeriod_MILP", pulp.LpMaximize)

        # Variables
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        s: dict[tuple[int, int], pulp.LpVariable] = {}
        c: dict[tuple[int, int], pulp.LpVariable] = {}
        tin: dict[tuple[int, int], pulp.LpVariable] = {}
        tout: dict[tuple[int, int], pulp.LpVariable] = {}
        b: dict[tuple[int, int], pulp.LpVariable] = {}

        is_wc_w1 = forced_w1_chip is not None and forced_w1_chip.lower() == "wildcard"

        # Only squad membership (x) and chip flags are branched on. Transfers, lineup and
        # captain are fully determined by integral x, so they stay continuous (the lineup,
        # formation and flow rows are totally unimodular); this is what keeps 12-GW chip solves tractable.
        for t in horizon_gws:
            for p in players:
                x[p, t] = pulp.LpVariable(f"x_{p}_{t}", cat=pulp.LpBinary)
                s[p, t] = pulp.LpVariable(f"s_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                c[p, t] = pulp.LpVariable(f"c_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                tin[p, t] = pulp.LpVariable(f"in_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                tout[p, t] = pulp.LpVariable(f"out_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                b[p, t] = pulp.LpVariable(f"b_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        u = {t: pulp.LpVariable(f"u_{t}", lowBound=0, upBound=15, cat=pulp.LpContinuous) for t in horizon_gws}
        h = {t: pulp.LpVariable(f"h_{t}", lowBound=0, upBound=15, cat=pulp.LpContinuous) for t in horizon_gws}
        bank_var = {t: pulp.LpVariable(f"bank_{t}", lowBound=0, cat=pulp.LpContinuous) for t in horizon_gws}
        ft_var = {
            t: pulp.LpVariable(f"ft_{t}", lowBound=0, upBound=MAX_BANKED_FREE_TRANSFERS, cat=pulp.LpInteger)
            for t in horizon_gws + [horizon_gws[-1] + 1]
        }
        z = {t: pulp.LpVariable(f"z_{t}", cat=pulp.LpBinary) for t in horizon_gws}

        # Chip variables
        wc = {t: pulp.LpVariable(f"wc_{t}", cat=pulp.LpBinary) for t in horizon_gws}
        fh = {t: pulp.LpVariable(f"fh_{t}", cat=pulp.LpBinary) for t in horizon_gws}
        bb = {t: pulp.LpVariable(f"bb_{t}", cat=pulp.LpBinary) for t in horizon_gws}
        tc = {t: pulp.LpVariable(f"tc_{t}", cat=pulp.LpBinary) for t in horizon_gws}

        # Chip product linearizations
        tcc = {
            (p, t): pulp.LpVariable(f"tcc_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
            for t in horizon_gws
            for p in players
        }
        bb_b = {
            (p, t): pulp.LpVariable(f"bb_b_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
            for t in horizon_gws
            for p in players
        }

        # Terminal FT variables
        v = [pulp.LpVariable(f"v_{k}", lowBound=0, upBound=1, cat=pulp.LpContinuous) for k in range(4)]

        # --- Constraints ---
        # Initial FT state
        prob += ft_var[t0] == int(max(0, min(MAX_BANKED_FREE_TRANSFERS, free_transfers)))

        # Total FH squad budget cap: current squad sell value + bank
        initial_team_value = sum(sell_vals[p] for p in squad_ids) + init_bank

        init_club_counts: dict[Any, int] = {}
        for p in squad_ids:
            tm = player_dict[p]["team"]
            init_club_counts[tm] = init_club_counts.get(tm, 0) + 1

        for idx, t in enumerate(horizon_gws):
            # 1. Squad structure (persistent)
            prob += pulp.lpSum(x[p, t] for p in gkps) == 2
            prob += pulp.lpSum(x[p, t] for p in defs) == 5
            prob += pulp.lpSum(x[p, t] for p in mids) == 5
            prob += pulp.lpSum(x[p, t] for p in fwds) == 3
            prob += pulp.lpSum(x[p, t] for p in players) == 15

            # Club cap <= 3 (or preexisting squad count if already >3)
            for tm, tm_players in club_map.items():
                init_c = init_club_counts.get(tm, 0)
                prob += pulp.lpSum(x[p, t] for p in tm_players) <= max(3, init_c)
                if init_c >= 3:
                    for p in tm_players:
                        if p not in squad_ids:
                            prob += tin[p, t] == 0

            # Lineup of the squad actually fielded. In a Free Hit week the owned squad is frozen and
            # fields nothing (the FH squad's value is a precomputed constant, see fh_plan below), so
            # starters, bench and captain scale with (1 - fh[t]). s + b <= x with fixed totals pins b = x - s.
            not_fh = 1 - fh[t]
            for p in players:
                prob += s[p, t] + b[p, t] <= x[p, t]
                # Per-player cap keeps the relaxation honest: a fractional fh[t] scales the whole
                # lineup instead of letting the LP drop only its weakest slots.
                prob += s[p, t] <= not_fh
                prob += b[p, t] <= not_fh
                prob += c[p, t] <= s[p, t]
                prob += tcc[p, t] <= c[p, t]
                prob += bb_b[p, t] <= b[p, t]

            # Formation: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 Starters, 1 Captain
            prob += pulp.lpSum(s[p, t] for p in players) == 11 * not_fh
            prob += pulp.lpSum(b[p, t] for p in players) == 4 * not_fh
            prob += pulp.lpSum(s[p, t] for p in gkps) == not_fh
            prob += pulp.lpSum(s[p, t] for p in defs) >= 3 * not_fh
            prob += pulp.lpSum(s[p, t] for p in defs) <= 5 * not_fh
            prob += pulp.lpSum(s[p, t] for p in mids) >= 2 * not_fh
            prob += pulp.lpSum(s[p, t] for p in mids) <= 5 * not_fh
            prob += pulp.lpSum(s[p, t] for p in fwds) >= 1 * not_fh
            prob += pulp.lpSum(s[p, t] for p in fwds) <= 3 * not_fh
            prob += pulp.lpSum(c[p, t] for p in players) == not_fh
            # Triple Captain adds exactly one extra captain multiple, on the captain
            prob += pulp.lpSum(tcc[p, t] for p in players) == tc[t]
            # Bench Boost scores exactly the four benched players
            prob += pulp.lpSum(bb_b[p, t] for p in players) == 4 * bb[t]

            if is_wc_w1 and idx == 0:
                # Dedicated Wildcard formulation for t0 (WI-8.4, Item 3)
                for p in players:
                    if p in squad_ids:
                        prob += tout[p, t] == 1 - x[p, t]
                        prob += tin[p, t] == 0
                    else:
                        prob += tin[p, t] == x[p, t]
                        prob += tout[p, t] == 0

                prob += pulp.lpSum(tin[p, t] for p in players) == u[t]
                prob += pulp.lpSum(tout[p, t] for p in players) == u[t]

                prob += (
                    pulp.lpSum(sell_vals[p] * x[p, t] for p in squad_ids)
                    + pulp.lpSum(buy_costs[p] * x[p, t] for p in players if p not in squad_ids)
                    + bank_var[t]
                    == initial_team_value
                )
                prob += h[t] == 0
                prob += wc[t] == 1
                prob += fh[t] == 0
                prob += bb[t] == 0
                prob += tc[t] == 0
                prob += z[t] == 0

                # Chip week: banked transfers are kept, same rule as every other chip week
                next_t = horizon_gws[idx + 1] if idx + 1 < H else (horizon_gws[-1] + 1)
                if self.config.chip_week_ft_rule == "preserve_plus_one":
                    prob += ft_var[next_t] <= ft_var[t] + 1
                else:
                    prob += ft_var[next_t] <= ft_var[t]
            else:
                # Flow conservation
                for p in players:
                    if idx == 0:
                        owned = 1 if p in squad_ids else 0
                        prob += x[p, t] - owned == tin[p, t] - tout[p, t]
                        # A player is either bought or sold, never both, and only owned players can be sold.
                        prob += tin[p, t] <= 1 - owned
                        prob += tout[p, t] <= owned
                    else:
                        prev_t = horizon_gws[idx - 1]
                        prob += x[p, t] - x[p, prev_t] == tin[p, t] - tout[p, t]
                        prob += tin[p, t] <= 1 - x[p, prev_t]
                        prob += tout[p, t] <= x[p, prev_t]

                    # FH freezes persistent squad moves
                    prob += tin[p, t] <= 1 - fh[t]
                    prob += tout[p, t] <= 1 - fh[t]

                prob += pulp.lpSum(tin[p, t] for p in players) == u[t]
                prob += pulp.lpSum(tout[p, t] for p in players) == u[t]

                # Budget
                if idx == 0:
                    prob += bank_var[t] == init_bank + pulp.lpSum(
                        sell_vals[p] * tout[p, t] for p in players
                    ) - pulp.lpSum(buy_costs[p] * tin[p, t] for p in players)
                else:
                    prev_t = horizon_gws[idx - 1]
                    prob += bank_var[t] == bank_var[prev_t] + pulp.lpSum(
                        sell_vals[p] * tout[p, t] for p in players
                    ) - pulp.lpSum(buy_costs[p] * tin[p, t] for p in players)

                # Hits: h[t] >= u[t] - ft[t] - 15 * wc[t]. (A Free Hit week has u = 0, so it needs no waiver;
                # leaving fh[t] out stops a fractional FH from waiving hits in the LP relaxation.)
                prob += h[t] >= u[t] - ft_var[t] - 15 * wc[t]
                if t < hit_cooldown_until:
                    # Hit-ledger guard: no paid transfers until the cooldown has passed
                    prob += h[t] == 0

                # Max 1 chip per GW
                prob += wc[t] + fh[t] + bb[t] + tc[t] <= 1

                # FT Banking State Machine (Rule R4)
                next_t = horizon_gws[idx + 1] if idx + 1 < H else (horizon_gws[-1] + 1)
                # z[t] == 1 if u[t] > ft_var[t]
                prob += u[t] - ft_var[t] <= 15 * z[t]
                # Normal week: if u <= ft, ft_next <= ft - u + 1; if u > ft, ft_next <= 1
                prob += ft_var[next_t] <= ft_var[t] - u[t] + 1 + 5 * z[t] + 5 * wc[t]
                prob += ft_var[next_t] <= 1 + 5 * (1 - z[t]) + 5 * wc[t]
                # Chip week: preserve
                if self.config.chip_week_ft_rule == "preserve_plus_one":
                    prob += ft_var[next_t] <= ft_var[t] + 1 + 5 * (1 - wc[t] - fh[t])
                else:
                    prob += ft_var[next_t] <= ft_var[t] + 5 * (1 - wc[t] - fh[t])

        # Terminal FT Concave Linearization
        t_end = horizon_gws[-1] + 1
        prob += ft_var[t_end] - 1 <= pulp.lpSum(v)

        # Chip Availability and Set 1 / Set 2 legality
        t1_gws = [t for t in horizon_gws if t <= SET_1_LAST_GW]
        t2_gws = [t for t in horizon_gws if t > SET_1_LAST_GW]

        # Check chip eligibility: None means unconstrained; [] or set() means NO chips allowed
        all_chips = {"wildcard", "freehit", "bboost", "3xc"}

        def chip_set(names: Any) -> set[str]:
            return all_chips if names is None else {str(n).lower().strip() for n in names}

        avail_by_set = {
            1: chip_set(chips_by_set.get(1) if chips_by_set else chips_available),
            2: chip_set(chips_by_set.get(2) if chips_by_set else chips_available),
        }
        for gws, avail in ((t1_gws, avail_by_set[1]), (t2_gws, avail_by_set[2])):
            if gws:
                for chip_name, chip_vars in (("wildcard", wc), ("freehit", fh), ("bboost", bb), ("3xc", tc)):
                    prob += pulp.lpSum(chip_vars[t] for t in gws) <= (1 if chip_name in avail else 0)

        spacing = self.config.min_gameweeks_between_chips
        if spacing > 1:
            for idx in range(H):
                window_gws = horizon_gws[idx : idx + spacing]
                prob += pulp.lpSum(wc[t] + fh[t] + bb[t] + tc[t] for t in window_gws) <= 1
            if previous_chip and forced_w1_chip is None:
                # A chip played last gameweek blocks chips until the spacing has passed
                for t in horizon_gws[: spacing - 1]:
                    prob += wc[t] + fh[t] + bb[t] + tc[t] == 0

        # Free Hit squads are solved once per week (depends only on that week's xP); weeks where a
        # Free Hit is not allowed are fixed to 0 so the plan never gets an FH it has no squad for.
        suppress_t0_chips = (
            forced_w1_chip is None
            and forced_chip_schedule is None
            and (no_chip_w1 or forced_w1_transfers is not None or min_w1_hits is not None)
        )
        fh_cache = fh_cache if fh_cache is not None else {}
        fh_plan: dict[int, dict[str, Any]] = {}
        for t in horizon_gws:
            fh_possible = (
                t != 1
                and not (t == t0 and suppress_t0_chips)
                and (forced_w1_chip or "").lower() != "wildcard"
                and "freehit" in (avail_by_set[1] if t <= SET_1_LAST_GW else avail_by_set[2])
            )
            fh_squad = None
            if fh_possible:
                at_t0 = t == t0
                cache_key = (
                    t,
                    initial_team_value,
                    frozenset(locked_in) if at_t0 else None,
                    frozenset(locked_out) if at_t0 else None,
                    frozenset(excluded_teams),
                    frozenset(squad_ids),
                )
                if cache_key not in fh_cache:
                    df_t = horizon_projections.get(t, player_pool_df)
                    xp_t = _xp_map(df_t)
                    week_pool = self.prune_candidates(
                        player_pool_df,
                        squad_ids=squad_ids | (locked_in if at_t0 else set()),
                        horizon_gws=[t],
                        horizon_projections=horizon_projections,
                        limits=FREE_HIT_PRUNING_LIMITS,
                    )
                    # Owned players are valued at their selling price, everyone else at market price.
                    fh_costs = {
                        int(e): float(squad_sell_map.get(int(e), cost))
                        for e, cost in zip(week_pool["element"], week_pool["value"], strict=True)
                    }
                    fh_cache[cache_key] = self._free_hit_squad(
                        week_pool,
                        xp_t,
                        fh_costs,
                        initial_team_value,
                        locked_in if at_t0 else set(),
                        locked_out if at_t0 else set(),
                        excluded_teams,
                    )
                fh_squad = fh_cache[cache_key]
            if fh_squad is None:
                prob += fh[t] == 0
            else:
                fh_plan[t] = fh_squad

        # Rule R2: neither Free Hit nor Wildcard in GW1
        if 1 in horizon_gws:
            prob += fh[1] == 0
            prob += wc[1] == 0

        # Rule R2: No FH in GW20 if FH in GW19
        if 19 in horizon_gws and 20 in horizon_gws:
            prob += fh[19] + fh[20] <= 1
        elif 20 in horizon_gws and previous_chip and previous_chip.lower() == "freehit":
            prob += fh[20] == 0

        # Apply Locked / Excluded requirements at GW t0
        # (a Free Hit at t0 honours them inside the FH squad solve instead)
        for p in locked_in:
            if p in players:
                prob += s[p, t0] + fh[t0] >= 1
        for p in locked_out:
            if p in players:
                prob += x[p, t0] == 0
        for tm in excluded_teams:
            for p in club_map.get(tm, []):
                for t in horizon_gws:
                    prob += x[p, t] == 0

        # Forced constraints for alternative plan generation
        if forced_w1_transfers is not None:
            prob += u[t0] == forced_w1_transfers
        if min_w1_hits is not None:
            prob += h[t0] >= min_w1_hits
        if max_w1_hits is not None:
            prob += h[t0] <= max_w1_hits
        if forced_chip_schedule is not None:
            for f_gw, f_chip in forced_chip_schedule.items():
                if f_gw in horizon_gws:
                    c_name = f_chip.lower().strip()
                    if c_name == "wildcard":
                        prob += wc[f_gw] == 1
                        for t in horizon_gws:
                            if t != f_gw and (
                                (t <= SET_1_LAST_GW and f_gw <= SET_1_LAST_GW)
                                or (t > SET_1_LAST_GW and f_gw > SET_1_LAST_GW)
                            ):
                                prob += wc[t] == 0
                    elif c_name == "freehit":
                        prob += fh[f_gw] == 1
                        for t in horizon_gws:
                            if t != f_gw and (
                                (t <= SET_1_LAST_GW and f_gw <= SET_1_LAST_GW)
                                or (t > SET_1_LAST_GW and f_gw > SET_1_LAST_GW)
                            ):
                                prob += fh[t] == 0
                    elif c_name == "bboost":
                        prob += bb[f_gw] == 1
                        for t in horizon_gws:
                            if t != f_gw and (
                                (t <= SET_1_LAST_GW and f_gw <= SET_1_LAST_GW)
                                or (t > SET_1_LAST_GW and f_gw > SET_1_LAST_GW)
                            ):
                                prob += bb[t] == 0
                    elif c_name == "3xc":
                        prob += tc[f_gw] == 1
                        for t in horizon_gws:
                            if t != f_gw and (
                                (t <= SET_1_LAST_GW and f_gw <= SET_1_LAST_GW)
                                or (t > SET_1_LAST_GW and f_gw > SET_1_LAST_GW)
                            ):
                                prob += tc[t] == 0
        elif forced_w1_chip is not None:
            chip_name = forced_w1_chip.lower()
            if chip_name == "wildcard":
                # Wildcard now; the other chips stay available later, like every other chip-now plan
                prob += wc[t0] == 1
                for t in horizon_gws:
                    if t != t0 and (t <= SET_1_LAST_GW) == (t0 <= SET_1_LAST_GW):
                        prob += wc[t] == 0
            elif chip_name == "freehit":
                prob += fh[t0] == 1
            elif chip_name == "bboost":
                prob += bb[t0] == 1
            elif chip_name == "3xc":
                prob += tc[t0] == 1
        elif suppress_t0_chips:
            # When evaluating normal transfer plans, suppress week 1 chips
            prob += wc[t0] == 0
            prob += fh[t0] == 0
            prob += bb[t0] == 0
            prob += tc[t0] == 0

        # No-good cuts for distinct moves
        if forbidden_w1_moves:
            for f_in, f_out in forbidden_w1_moves:
                prob += (
                    pulp.lpSum(1 - tin[p, t0] for p in f_in if p in players)
                    + pulp.lpSum(tin[p, t0] for p in players if p not in f_in)
                    + pulp.lpSum(1 - tout[p, t0] for p in f_out if p in players)
                    + pulp.lpSum(tout[p, t0] for p in players if p not in f_out)
                    >= 1
                )

        # --- Objective Function ---
        gamma = self.config.gamma
        w_bench = self.config.w_bench_mean
        v_weights = self.config.ft_terminal_increments
        obj_terms = []

        for idx, t in enumerate(horizon_gws):
            discount = gamma**idx
            df_gw = horizon_projections.get(t, player_pool_df)
            gw_xp = _xp_map(df_gw)

            starters_val = pulp.lpSum(gw_xp.get(p, 0.0) * s[p, t] for p in players)
            cap_val = pulp.lpSum(gw_xp.get(p, 0.0) * (c[p, t] + tcc[p, t]) for p in players)
            bench_val = pulp.lpSum(
                gw_xp.get(p, 0.0) * (w_bench * b[p, t] + (1.0 - w_bench) * bb_b[p, t]) for p in players
            )
            hit_pen = float(HIT_COST_POINTS) * h[t]
            fh_val = fh_plan[t]["value"] * fh[t] if t in fh_plan else 0.0

            obj_terms.append(discount * (starters_val + cap_val + bench_val + fh_val - hit_pen))

        # Terminal FT valuation
        ft_term = pulp.lpSum(w * var for w, var in zip(v_weights, v, strict=True))
        obj_terms.append((gamma**H) * ft_term)

        # Micro-incentive for maintaining maximum available banked FTs across horizon
        obj_terms.append(1e-4 * pulp.lpSum(ft_var[t] for t in horizon_gws + [horizon_gws[-1] + 1]))

        # Terminal Bank valuation
        obj_terms.append(self.config.lambda_bank * (bank_var[horizon_gws[-1]] / 10.0))

        prob += pulp.lpSum(obj_terms)

        # Solve
        chip_cols = {"wildcard": wc, "freehit": fh, "bboost": bb, "3xc": tc}
        if warm_start is None:
            # Keeping the squad every week with no chips is feasible for every unforced solve. With forced
            # week-1 transfers, swap the weakest owned players for the best affordable same-position players.
            start_squad = set(squad_ids)
            if forced_w1_transfers:
                hw: defaultdict[int, float] = defaultdict(float)
                for gw in horizon_gws:
                    for p_id, val in _xp_map(horizon_projections.get(gw, player_pool_df)).items():
                        hw[p_id] += val
                money = init_bank
                clubs = dict(init_club_counts)
                for out_p in sorted(squad_ids - locked_in, key=lambda q: hw[q])[: int(forced_w1_transfers)]:
                    pos, budget = player_dict[out_p]["position"], money + sell_vals[out_p]
                    options = [
                        q
                        for q in players
                        if q not in start_squad
                        and q not in locked_out
                        and player_dict[q]["position"] == pos
                        and buy_costs[q] <= budget
                        and clubs.get(player_dict[q]["team"], 0)
                        + (player_dict[q]["team"] != player_dict[out_p]["team"])
                        <= 3
                    ]
                    if not options:
                        continue
                    in_p = max(options, key=lambda q: hw[q])
                    start_squad = (start_squad - {out_p}) | {in_p}
                    money = budget - buy_costs[in_p]
                    clubs[player_dict[out_p]["team"]] -= 1
                    clubs[player_dict[in_p]["team"]] = clubs.get(player_dict[in_p]["team"], 0) + 1
            warm_start = {("x", p, t): int(p in start_squad) for p in players for t in horizon_gws}
            if forced_w1_chip:
                warm_start[(forced_w1_chip.lower(), t0)] = 1
        start: dict[pulp.LpVariable, float] = {}
        for key, val in warm_start.items():
            if key[0] == "x":
                var = x.get((key[1], key[2]))
            else:
                var = chip_cols.get(key[0], {}).get(key[1])
            if var is not None:
                start[var] = float(val)
        if warm_start and start:
            # Integer columns the start does not mention default to 0 (players outside its pool, other chips).
            for t in horizon_gws:
                for p in players:
                    start.setdefault(x[p, t], 0.0)
                for chip_dict in chip_cols.values():
                    start.setdefault(chip_dict[t], 0.0)

        if fix_except_gws is not None and warm_start:
            x_gws = set(horizon_gws)
            free = {id(x[p, t]) for p in players for t in fix_except_gws if t in x_gws} | {
                id(cv[t]) for cv in chip_cols.values() for t in fix_except_gws if t in cv
            }
            for var, val in start.items():
                if id(var) not in free:
                    var.lowBound = var.upBound = val

        solver = self._get_solver(time_limit_sec, start)
        solve_status = prob.solve(solver)
        solve_time = time.perf_counter() - t_start

        real_gap = 0.0
        if hasattr(prob, "solverModel") and prob.solverModel is not None:
            try:
                highs_m = prob.solverModel
                model_status_str = highs_m.modelStatusToString(highs_m.getModelStatus())
                gap_val = highs_m.getInfoValue("mip_gap")[1]
                real_gap = float(gap_val) if gap_val is not None and not np.isnan(gap_val) else 0.0
                if model_status_str.lower() == "optimal":
                    status_str = f"OPTIMAL (gap={real_gap:.4%})"
                elif "time limit" in model_status_str.lower():
                    status_str = f"TIME_LIMIT (gap={real_gap:.4%})"
                else:
                    status_str = f"{model_status_str.upper()} (gap={real_gap:.4%})"
            except Exception as ex:
                logger.debug(f"Failed extracting HiGHS solverModel stats: {ex}")
                status_str = f"{pulp.LpStatus[solve_status].upper()}"
        else:
            is_time_limit = solve_time >= ((time_limit_sec or self.config.time_limit_sec) - 1.5)
            has_feasible = any(x[p, t0].varValue is not None for p in players)
            if is_time_limit:
                status_str = "TIME_LIMIT" if has_feasible else "TIME_LIMIT (no solution)"
                real_gap = 1.0
            elif solve_status == pulp.constants.LpStatusOptimal and has_feasible:
                status_str = f"OPTIMAL (gap<={self.config.mip_gap:.4%})"
                real_gap = 0.0
            elif solve_status == pulp.constants.LpStatusNotSolved or not has_feasible:
                status_str = "TIME_LIMIT / NOT_SOLVED"
                real_gap = 1.0
            else:
                status_str = f"{pulp.LpStatus[solve_status].upper()}"

        if pulp.LpStatus[solve_status] == "Infeasible":
            raise MilpInfeasible(f"No feasible plan exists under the given rules for GW{t0}")
        if not any(var.varValue is not None for var in x.values()) or not np.isfinite(real_gap):
            raise MilpNoSolution(f"No feasible plan found within {time_limit_sec or self.config.time_limit_sec}s")

        is_lower_bound = (real_gap > 1e-3) or ("TIME_LIMIT" in status_str) or ("NOT_SOLVED" in status_str)

        # Extract Trajectory
        trajectory: list[GameweekAction] = []
        horizon_net_xp = 0.0
        horizon_hits = 0

        for _idx, t in enumerate(horizon_gws):
            t_ins = [
                {
                    "element": p,
                    "web_name": player_dict[p].get("web_name", player_dict[p].get("name", f"Player {p}")),
                    "position": player_dict[p]["position"],
                    "team": player_dict[p]["team"],
                    "cost": buy_costs[p],
                }
                for p in players
                if tin[p, t].varValue is not None and tin[p, t].varValue > 0.5
            ]
            t_outs = [
                {
                    "element": p,
                    "web_name": player_dict[p].get("web_name", player_dict[p].get("name", f"Player {p}")),
                    "position": player_dict[p]["position"],
                    "team": player_dict[p]["team"],
                    "sell_price": sell_vals[p],
                }
                for p in players
                if tout[p, t].varValue is not None and tout[p, t].varValue > 0.5
            ]

            df_gw = horizon_projections.get(t, player_pool_df)
            gw_xp = _xp_map(df_gw)

            # Chip
            gw_chip_name: str | None = None
            if wc[t].varValue is not None and wc[t].varValue > 0.5:
                gw_chip_name = "wildcard"
            elif fh[t].varValue is not None and fh[t].varValue > 0.5:
                gw_chip_name = "freehit"
            elif bb[t].varValue is not None and bb[t].varValue > 0.5:
                gw_chip_name = "bboost"
            elif tc[t].varValue is not None and tc[t].varValue > 0.5:
                gw_chip_name = "3xc"

            if gw_chip_name == "freehit":
                # The FH squad replaces the owned squad for this one gameweek only
                fh_week = fh_plan[t]
                t_starters, t_bench = list(fh_week["starters"]), list(fh_week["bench"])
                t_cap, t_vc = fh_week["captain"], fh_week["vice_captain"]
                t_squad = list(fh_week["squad"])
            else:
                t_squad = [p for p in players if (x[p, t].varValue or 0.0) > 0.5]
                t_starters, t_bench, t_cap, t_vc = best_lineup(
                    t_squad,
                    {p: player_dict[p]["position"] for p in t_squad},
                    gw_xp,
                    1.0 if gw_chip_name == "bboost" else w_bench,
                )

            t_hits = int(round(h[t].varValue or 0))
            t_u = int(round(u[t].varValue or 0))
            t_ft = int(round(ft_var[t].varValue if ft_var[t].varValue is not None else 1))
            t_ft_next, _ = next_free_transfers(
                ft=t_ft,
                transfers_made=t_u,
                chip=gw_chip_name,
            )

            l_xp = sum(gw_xp.get(p, 0.0) for p in t_starters)
            c_xp = gw_xp.get(t_cap, 0.0) * (2.0 if gw_chip_name == "3xc" else 1.0)
            b_xp = sum(gw_xp.get(p, 0.0) for p in t_bench) * (1.0 if gw_chip_name == "bboost" else w_bench)
            net_xp = l_xp + c_xp + b_xp - (t_hits * HIT_COST_POINTS)

            horizon_net_xp += net_xp
            horizon_hits += t_hits

            action = GameweekAction(
                gameweek=t,
                transfers_in=t_ins,
                transfers_out=t_outs,
                transfers_count=t_u,
                free_transfers_available=t_ft,
                free_transfers_used=min(t_u, t_ft),
                free_transfers_next=t_ft_next,
                hits=t_hits,
                hit_cost=t_hits * HIT_COST_POINTS,
                active_chip=gw_chip_name,
                bank_end=round(float(bank_var[t].varValue or 0.0), 1),
                starters=t_starters,
                captain=t_cap,
                vice_captain=t_vc,
                bench=t_bench,
                lineup_xp=round(l_xp, 2),
                captain_xp=round(c_xp, 2),
                bench_xp=round(b_xp, 2),
                net_xp=round(net_xp, 2),
                squad=sorted(t_squad),
            )
            trajectory.append(action)

        rec_action = trajectory[0]

        # Determine plan type
        if rec_action.active_chip:
            p_type = rec_action.active_chip.upper()
        elif rec_action.transfers_count == 0:
            p_type = "ROLL_TRANSFER"
        elif rec_action.transfers_count == 1:
            p_type = "1_TRANSFER"
        elif rec_action.transfers_count == 2:
            p_type = "2_TRANSFERS"
        else:
            p_type = f"{rec_action.transfers_count}_TRANSFERS"

        obj_val = float(pulp.value(prob.objective) or 0.0)

        # Objective decomposition calculation (from the extracted trajectory, so FH weeks are included)
        disc_lineup_xp = sum(gamma**i * a.lineup_xp for i, a in enumerate(trajectory))
        disc_captain_xp = sum(gamma**i * a.captain_xp for i, a in enumerate(trajectory))
        disc_bench_xp = sum(gamma**i * a.bench_xp for i, a in enumerate(trajectory))
        disc_hits_cost = sum(gamma**i * a.hit_cost for i, a in enumerate(trajectory))

        term_ft_val = (gamma**H) * sum(w * float(var.varValue or 0.0) for w, var in zip(v_weights, v, strict=True))
        micro_ft_val = 1e-4 * sum(float(ft_var[t].varValue or 0.0) for t in horizon_gws + [horizon_gws[-1] + 1])
        term_bank_val = self.config.lambda_bank * (float(bank_var[horizon_gws[-1]].varValue or 0.0) / 10.0)

        decomp = {
            "discounted_lineup_xp": round(disc_lineup_xp, 3),
            "discounted_captain_xp": round(disc_captain_xp, 3),
            "discounted_bench_xp": round(disc_bench_xp, 3),
            "discounted_hits_cost": round(disc_hits_cost, 3),
            "terminal_ft_value": round(term_ft_val, 3),
            "terminal_bank_value": round(term_bank_val, 3),
            "ft_banking_incentive": round(micro_ft_val, 4),
            "total_objective": round(obj_val, 3),
        }

        return MultiPeriodPlan(
            plan_type=p_type,
            target_gw=t0,
            horizon_gws=horizon_gws,
            recommended_action=rec_action,
            trajectory=trajectory,
            total_objective=round(obj_val, 2),
            horizon_net_xp=round(horizon_net_xp, 2),
            horizon_hits=horizon_hits,
            horizon_gain_vs_roll=0.0,
            immediate_gain=0.0,
            hit_verdict="NO_HIT" if rec_action.hits == 0 else "JUSTIFIED",
            hit_delta=0.0,
            hit_prob_gain=1.0 if rec_action.hits == 0 else 0.75,
            solve_time_sec=round(solve_time, 2),
            solver_status=status_str,
            solver_gap=real_gap,
            is_lower_bound=is_lower_bound,
            objective_decomposition=decomp,
            integer_solution={
                **{("x", p, t): int(round(x[p, t].varValue or 0.0)) for p in players for t in horizon_gws},
                **{(n, t): int(round(cv[t].varValue or 0.0)) for n, cv in chip_cols.items() for t in horizon_gws},
            },
        )

    def _polish(
        self, run: Any, extra: dict[str, Any], plan: MultiPeriodPlan, name: str, bound: float | None
    ) -> MultiPeriodPlan:
        """One fix-and-optimise sweep over overlapping windows of weeks, starting from `plan`.

        `bound` is the full solve's dual bound for this task; the reported gap stays relative to it, so a
        polished plan is never called proven unless that bound is met (None = no bound known)."""
        width = self.config.polish_window
        best = plan
        gws = plan.horizon_gws
        for i in range(0, max(1, len(gws) - width + 1) + (width // 2), max(1, width // 2)):
            window = gws[i : i + width]
            if not window:
                break
            result = run(
                f"polish:{name}:{window[0]}",
                dict(extra, warm_start=best.integer_solution, fix_except_gws=window),
                self.config.polish_window_time_limit_sec,
            )
            if isinstance(result, MultiPeriodPlan) and result.total_objective > best.total_objective + 1e-6:
                best = result
        if best is not plan or bound is None:
            gap = max(0.0, bound / best.total_objective - 1.0) if bound is not None else 1.0
            best.solver_gap = gap
            best.is_lower_bound = gap > 1e-3
            best.solver_status = f"LOCAL_SEARCH (gap<={gap:.4%} vs full-solve bound)"
            best.solve_time_sec = round(plan.solve_time_sec + best.solve_time_sec, 2)
        return best

    # ------------------------------------------------------------------
    # Plan comparison set
    # ------------------------------------------------------------------
    def _plan_to_dict(
        self,
        plan: MultiPeriodPlan,
        roll_plan: MultiPeriodPlan,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        risk_preference: str,
        hit_gws: list[int],
        maps: dict[int, dict[int, float]],
    ) -> dict[str, Any]:
        """Converts a solved MultiPeriodPlan to the plan dict used by the API, CLI, briefing and chat."""
        from fpl_oracle.optimise.lineup import lineup_optimizer

        act = plan.recommended_action
        h_gross = round(sum(a.lineup_xp + a.captain_xp + a.bench_xp for a in plan.trajectory), 2)
        h_net = round(h_gross - plan.horizon_hits * HIT_COST_POINTS, 2)
        roll_gross = sum(a.lineup_xp + a.captain_xp + a.bench_xp for a in roll_plan.trajectory)
        roll_net = round(roll_gross - roll_plan.horizon_hits * HIT_COST_POINTS, 2)
        gain_vs_roll = round(h_net - roll_net, 2)
        imm_gain = round(act.net_xp - roll_plan.recommended_action.net_xp, 2)

        # The squad that actually plays this gameweek, exactly as the solver chose it: the owned squad
        # after this week's transfers, or the temporary squad in a Free Hit week.
        fielded = set(act.squad)
        owned_ids = set(current_squad_df["element"].astype(int))
        retained = current_squad_df[current_squad_df["element"].isin(fielded)].copy()
        incoming = player_pool_df[player_pool_df["element"].isin(fielded - owned_ids)].copy()
        for col in retained.columns:
            if col not in incoming.columns:
                if col in ("purchase_price", "selling_price"):
                    incoming[col] = incoming.get("value", 50.0)
                elif col == "chance_of_playing":
                    incoming[col] = 100.0
                else:
                    incoming[col] = 0.0
        fielded_df = pd.concat([retained, incoming], ignore_index=True)
        if len(fielded_df) != 15:
            raise MilpInfeasible(f"Solver squad for GW{act.gameweek} has {len(fielded_df)} players, expected 15")
        lineup_res = lineup_optimizer.select_lineup_and_captain(
            fielded_df,
            is_triple_captain=(act.active_chip == "3xc"),
            is_bench_boost=(act.active_chip == "bboost"),
            risk_preference=risk_preference,
            simulate_captaincy=False,
        )
        cap_id = int(lineup_res["captain"]["element"])
        vc_id = int(lineup_res["vice_captain"]["element"])
        starters_list = [int(e) for e in lineup_res["starters"]["element"]]
        bench_list = [int(e) for e in lineup_res["bench"]["element"]]

        def player_rows(ids: set[int], price_key: str) -> list[dict[str, Any]]:
            rows = fielded_df if price_key == "cost" else current_squad_df
            lookup = {int(r["element"]): r for _, r in rows.iterrows()}
            out = []
            for e in sorted(ids):
                r = lookup.get(e)
                if r is None:
                    continue
                price = float(r.get("selling_price", r["value"])) if price_key == "sell_price" else float(r["value"])
                out.append(
                    {
                        "element": e,
                        "web_name": r.get("web_name", f"Player {e}"),
                        "position": r["position"],
                        "team": r["team"],
                        price_key: price,
                    }
                )
            return out

        is_fh = act.active_chip == "freehit"
        transfers_in, transfers_out = act.transfers_in, act.transfers_out
        free_hit = None
        if is_fh:
            # The FH squad is temporary: it plays this gameweek and the owned squad returns.
            transfers_in = [dict(p, temporary=True) for p in player_rows(fielded - owned_ids, "cost")]
            transfers_out = [dict(p, temporary=True) for p in player_rows(owned_ids - fielded, "sell_price")]
            free_hit = {
                "gameweek": act.gameweek,
                "squad": sorted(fielded),
                "buy": transfers_in,
                "sell": transfers_out,
                "reverts_to": sorted(owned_ids),
                "note": "Temporary squad for this gameweek only. Your current squad returns next gameweek.",
            }
        # The solver works in tenths; every surface reads the plan in millions (same unit as the beam search).
        transfers_in, transfers_out = _rows_in_millions(transfers_in), _rows_in_millions(transfers_out)
        for row in transfers_in + transfers_out:
            row["expected_points"] = round(maps.get(act.gameweek, {}).get(int(row["element"]), 0.0), 2)

        in_names = ", ".join(t.get("web_name", str(t.get("element", ""))) for t in transfers_in)
        out_names = ", ".join(t.get("web_name", str(t.get("element", ""))) for t in transfers_out)
        from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

        tail = f"(+{imm_gain:.2f} xP, Horizon: {gain_vs_roll:+.2f} xP)"
        if is_fh:
            rec_summary = (
                f"Play your Free Hit in GW{act.gameweek}: buy {in_names}, sell {out_names} for this week only. "
                f"Your current squad returns next gameweek {tail}"
            )
        elif act.active_chip:
            moves = f" with {act.transfers_count} transfer(s): {out_names} -> {in_names}" if act.transfers_count else ""
            rec_summary = f"Play your {CHIP_DISPLAY_NAMES.get(act.active_chip, act.active_chip)} in GW{act.gameweek}{moves} {tail}"
        elif act.transfers_count == 0:
            rec_summary = f"Roll transfer. Bank {act.free_transfers_next} free transfers for next gameweek."
        elif act.transfers_count == 1:
            rec_summary = f"Transfer out {out_names} -> {in_names} {tail}"
        else:
            rec_summary = f"Execute {act.transfers_count} transfers: {out_names} -> {in_names} {tail}"

        # Owner's hit rule: raw returns of the transferred players, not captain/lineup gain
        rows_by_gw = {
            g: defaultdict(lambda: {"expected_points": 0.0}, {e: {"expected_points": v} for e, v in m.items()})
            for g, m in maps.items()
        }
        raw_player_gain = (
            round(
                expected_transfer_gain(
                    [t["element"] for t in act.transfers_in],
                    [t["element"] for t in act.transfers_out],
                    rows_by_gw,
                    hit_gws,
                ),
                2,
            )
            if act.transfers_count
            else 0.0
        )

        trajectory = [
            {
                "gameweek": a.gameweek,
                "transfers_in": _rows_in_millions(a.transfers_in),
                "transfers_out": _rows_in_millions(a.transfers_out),
                "transfers_count": a.transfers_count,
                "hits": a.hits,
                "hit_cost": a.hit_cost,
                "chip": a.active_chip,
                "net_xp": a.net_xp,
                "starters": a.starters,
                "captain": a.captain,
                "vice_captain": a.vice_captain,
                "bench": a.bench,
                "elements": a.squad,
                "bank_end": a.bank_end,
                "ft_next": a.free_transfers_next,
            }
            for a in plan.trajectory
        ]
        # The first week is the one on screen: its XI and captain are the lineup shown above, never a second opinion
        trajectory[0].update(starters=starters_list, captain=cap_id, vice_captain=vc_id, bench=bench_list)

        return {
            "plan_type": plan.plan_type,
            "recommendation_summary": rec_summary,
            "summary": rec_summary,
            "transfers_count": act.transfers_count,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "hits": act.hits,
            "hit_cost": act.hit_cost,
            "free_transfers_used": act.free_transfers_used,
            "free_transfers_next": act.free_transfers_next,
            "next_banked_ft": act.free_transfers_next,
            "bank_delta": round(act.bank_end - float(bank), 1),
            "bank_remaining": act.bank_end,
            "remaining_bank": round(float(act.bank_end) / 10.0, 2),
            "expected_gain": imm_gain,
            "pure_xp_gain": imm_gain,
            "price_movement_gain": 0.0,
            "robustness_score": None,
            "is_no_regret": False,
            "p_beat_roll": None,
            "sigma_delta": None,
            "mc_delta_mean": None,
            "horizon_gain_vs_roll": gain_vs_roll,
            "horizon_gross_xp": h_gross,
            "horizon_net_xp": h_net,
            "net_expected_points": round(act.net_xp, 2),
            "horizon_hits": plan.horizon_hits,
            "gross_expected_points": round(act.lineup_xp + act.captain_xp + act.bench_xp, 2),
            "expected_player_gain": raw_player_gain,
            "chip": act.active_chip,
            "free_hit": free_hit,
            "lineup": lineup_res,
            "captain": lineup_res["captain"],
            "vice_captain": lineup_res["vice_captain"],
            "starting_xi": starters_list,
            "captain_id": cap_id,
            "vice_captain_id": vc_id,
            "bench_ids": bench_list,
            "fielded_squad": sorted(fielded),
            "total_objective": plan.total_objective,
            "objective_gain_vs_roll": round(plan.total_objective - roll_plan.total_objective, 2),
            "solver_status": plan.solver_status,
            "solver_gap": plan.solver_gap,
            "is_lower_bound": plan.is_lower_bound,
            "solve_time_sec": plan.solve_time_sec,
            "trajectory": trajectory,
        }

    def evaluate_transfer_options(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        chips_available: set[str] | list[str] | None = None,
        previous_chip: str | None = None,
        risk_preference: str = "balanced",
        mini_league_mode: str | None = None,
        mini_league_deficit: float | None = None,
        include_price_gain: bool = False,
        stability_threshold: float = 0.3,
        horizon_len: int | None = None,
        chips_by_set: dict[int, list[str] | set[str]] | None = None,
        chips_already_used: set[str] | list[str] | None = None,
        hit_cooldown_until: int = 0,
        forced_current_chip: str | None = None,
        time_limit_sec: float | None = None,
        enforce_hit_gain: bool = True,
        xp_estimation_sd: float | None = None,
        rival_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive multi-period solver producing the required comparison set (WI-2.2):
        - Roll plan (force 0 transfers in week 1, no chip)
        - Best 1-transfer plan (force 1 transfer, no chip)
        - Hold plan: best week-1 plan without a chip (chips stay available later)
        - Optimal plan (unconstrained MILP: any transfers, hits and chips)
        - Chip plans: each available chip played now, everything else optimised
        - Distinct week-1 alternatives via no-good cuts
        - Hit policy: hit cooldown, owner's raw-gain rule and simulated hit verdict (WI-2.5)
        The independent solves run in parallel. The recommendation is the best-objective plan that
        passes the hit gates; solver status and gaps are returned so a time-limited incumbent is
        never presented as proven optimal.
        """
        from concurrent.futures import ThreadPoolExecutor

        from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES
        from fpl_oracle.rules.transfers import validate_chip_legality

        limit = float(time_limit_sec if time_limit_sec is not None else self.config.time_limit_sec)
        # Wave 1 only needs a good start: a window-by-window local search then polishes each contender.
        wave1_limit = min(limit, self.config.wave1_time_limit_sec)
        wave2_limit = min(limit, 20.0)
        alt_limit = min(limit, 25.0)  # alternatives are context, not the decision
        workers = os.cpu_count() or 4

        # --- chip availability: never plan a chip that is not available in the set it would be played in
        all_chips = {"wildcard", "freehit", "bboost", "3xc"}
        target_set = 1 if target_gw <= SET_1_LAST_GW else 2
        forced = forced_current_chip.lower().strip() if forced_current_chip else None
        used = {str(c.get("name") if isinstance(c, dict) else c).lower().strip() for c in (chips_already_used or [])}
        if chips_by_set is not None:
            avail = {n: {str(c).lower().strip() for c in chips_by_set.get(n) or []} for n in (1, 2)}
            if chips_available is not None:
                avail[target_set] &= {str(c).lower().strip() for c in chips_available}
        else:
            allowed = all_chips if chips_available is None else {str(c).lower().strip() for c in chips_available}
            avail = {1: set(allowed), 2: set(allowed)}
        avail[target_set] -= used - {forced}
        if forced:
            avail[target_set].add(forced)
        avail_lists = {n: sorted(avail[n] & all_chips) for n in (1, 2)}
        chips_now = [c for c in avail_lists[target_set] if validate_chip_legality(c, target_gw, None, previous_chip)[0]]

        common = dict(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            horizon_len=horizon_len,
            locked_in_ids=locked_in_ids,
            locked_out_ids=locked_out_ids,
            excluded_team_ids=excluded_team_ids,
            chips_by_set=avail_lists,
            previous_chip=previous_chip,
            hit_cooldown_until=hit_cooldown_until,
            fh_cache={},
        )
        forced_kw = {"forced_w1_chip": forced} if forced else {}

        def run(name: str, extra: dict[str, Any], lim: float, **override: Any) -> MultiPeriodPlan | Exception:
            try:
                return self.optimize(**dict(common, **override), **extra, time_limit_sec=lim)
            except Exception as ex:
                logger.warning("MILP solve %s failed: %s", name, ex)
                return ex

        def run_parallel(jobs: dict[str, Any]) -> dict[str, Any]:
            if not jobs:
                return {}
            with ThreadPoolExecutor(max_workers=min(len(jobs), workers)) as pool:
                futures = {name: pool.submit(job) for name, job in jobs.items()}
                return {name: fut.result() for name, fut in futures.items()}

        def run_batch(batch: dict[str, dict[str, Any]], lim: float) -> dict[str, MultiPeriodPlan | Exception]:
            """Independent solves in parallel (HiGHS releases the GIL). With fewer cores than solves the
            time limit is shared so the wall clock stays near `lim`."""
            if not batch:
                return {}
            pool_size = min(len(batch), workers)
            per_solve = max(8.0, lim / math.ceil(len(batch) / pool_size))
            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                futures = {name: pool.submit(run, name, extra, per_solve) for name, extra in batch.items()}
                return {name: fut.result() for name, fut in futures.items()}

        # --- wave 1
        report_search_progress(f"Comparing transfers and chips from Gameweek {target_gw}")
        # "Roll" is the fewest moves possible: players the manager has ruled out must still be sold.
        min_moves = len(set(locked_out_ids or ()) & set(current_squad_df["element"].astype(int)))
        tasks: dict[str, dict[str, Any]] = {
            "roll": dict(forced_kw, forced_w1_transfers=min_moves),
            "plan_1": dict(forced_kw, forced_w1_transfers=min_moves + 1),
            "plan_2": dict(forced_kw, forced_w1_transfers=min_moves + 2),
            "opt": dict(forced_kw),
        }
        if not forced:
            tasks["hold"] = {"no_chip_w1": True}
            for chip in chips_now:
                tasks[f"chip:{chip}"] = {"forced_w1_chip": chip}
        solved = run_batch(tasks, wave1_limit)
        if isinstance(solved["roll"], Exception):
            raise solved["roll"]
        plans = {n: p for n, p in solved.items() if isinstance(p, MultiPeriodPlan)}

        # --- polish: fix-and-optimise sweeps. Each window of weeks is re-solved exactly with the rest of
        # the plan fixed, which closes most of the time-limited gap in a fraction of a full re-solve.
        report_search_progress("Refining the best plans week by week")

        def fits(extra: dict[str, Any], plan: MultiPeriodPlan) -> bool:
            """Whether a plan satisfies a task's week-1 restriction (so it is a valid start for that task)."""
            week1 = plan.trajectory[0]
            chip = (week1.active_chip or "").lower() or None
            if extra.get("forced_w1_chip"):
                if chip != extra["forced_w1_chip"].lower():
                    return False
            elif (extra.get("no_chip_w1") or extra.get("forced_w1_transfers") is not None) and chip:
                return False
            if extra.get("forced_w1_transfers") is not None and week1.transfers_count != extra["forced_w1_transfers"]:
                return False
            return extra.get("max_w1_hits") is None or week1.hits <= extra["max_w1_hits"]

        def polish_task(name: str) -> MultiPeriodPlan | None:
            starts = [p for p in plans.values() if fits(tasks[name], p)]
            if not starts:
                return None
            own = plans.get(name)
            bound = own.total_objective * (1.0 + own.solver_gap) if own is not None else None
            return self._polish(run, tasks[name], max(starts, key=lambda p: p.total_objective), name, bound)

        polished = run_parallel({n: (lambda n=n: polish_task(n)) for n in tasks})
        plans.update({n: p for n, p in polished.items() if isinstance(p, MultiPeriodPlan)})
        best_name = max(plans, key=lambda n: plans[n].total_objective)
        if best_name != "opt":
            # The unrestricted plan starts from the best plan found, so the superset can never lose to its subsets.
            opt_bound = (
                plans["opt"].total_objective * (1.0 + plans["opt"].solver_gap)
                if "opt" in solved and isinstance(solved["opt"], MultiPeriodPlan)
                else None
            )
            final = self._polish(run, dict(forced_kw), plans[best_name], "opt", opt_bound)
            if final.total_objective > (plans["opt"].total_objective if "opt" in plans else -math.inf):
                plans["opt"] = final
        # --- chip now vs the same chip later: the real alternative to playing it now. The best plan is shifted
        # to play its week-1 chip 1, 3 and 6 weeks later (squad kept until then, clashing chips dropped) and each
        # seed is polished as a no-chip-this-week plan, so "keep the chip" is measured against its best timing.
        best = max(plans.values(), key=lambda p: p.total_objective)
        chip_w1 = (best.trajectory[0].active_chip or "").lower()
        gws_h = best.horizon_gws
        if chip_w1 and not forced and "hold" in tasks:
            owned_ids = set(current_squad_df["element"].astype(int))
            spacing = self.config.min_gameweeks_between_chips
            hold_own = plans.get("hold")
            bound = hold_own.total_objective * (1.0 + hold_own.solver_gap) if hold_own is not None else None

            def delayed(k: int) -> MultiPeriodPlan | None:
                t_k = gws_h[k]
                seed = dict(best.integer_solution)
                for key in seed:
                    if key[0] == "x" and key[2] < t_k:
                        seed[key] = int(key[1] in owned_ids)
                    elif key[0] != "x" and key[1] < t_k + spacing:
                        seed[key] = 0
                seed[(chip_w1, t_k)] = 1
                name = f"hold_delay{k}"
                plan = run(name, {"no_chip_w1": True, "warm_start": seed}, wave2_limit)
                return self._polish(run, tasks[name], plan, name, bound) if isinstance(plan, MultiPeriodPlan) else None

            delays = [k for k in (1, 3, 6) if k < len(gws_h)]
            for k in delays:
                tasks[f"hold_delay{k}"] = {"no_chip_w1": True}
            found = run_parallel({f"hold_delay{k}": (lambda k=k: delayed(k)) for k in delays})
            plans.update({n: p for n, p in found.items() if isinstance(p, MultiPeriodPlan)})

        roll_plan = plans["roll"]
        opt_plan = plans.get("opt") or max(plans.values(), key=lambda p: p.total_objective)
        horizon_gws = list(opt_plan.horizon_gws)
        maps = {g: _xp_map(horizon_projections.get(g, player_pool_df)) for g in horizon_gws}

        by_name: dict[str, dict[str, Any]] = {}
        unique: dict[Any, dict[str, Any]] = {}

        def move_key(cand: dict[str, Any]) -> tuple[Any, frozenset[int], frozenset[int]]:
            return (
                cand["chip"],
                frozenset(t["element"] for t in cand["transfers_in"]),
                frozenset(t["element"] for t in cand["transfers_out"]),
            )

        def add_plans(new_plans: dict[str, MultiPeriodPlan]) -> None:
            for name, plan in new_plans.items():
                cand = self._plan_to_dict(
                    plan, roll_plan, current_squad_df, player_pool_df, bank, risk_preference, horizon_gws, maps
                )
                cand["source"] = name
                by_name[name] = cand
                # Dedupe by (chip, in, out): a Free Hit / Triple Captain / Bench Boost with no transfers is
                # not the same plan as Roll. The better-solved copy of an identical move is kept.
                key = move_key(cand)
                if key not in unique or cand["total_objective"] > unique[key]["total_objective"]:
                    unique[key] = cand

        report_search_progress("Checking hits and the chip options")
        add_plans(plans)
        cand_roll = unique[move_key(by_name["roll"])]
        # --- one decision rule (docs/decision_model.md section 3): act only when the gain beats the baseline by
        # more than the model's own estimation error, E[dJ] >= max(m_min, z * sigma_est(dJ)). A player's error
        # is the same in every week (correlated), errors of different players are independent.
        sigma_est = float(xp_estimation_sd or 0.0)
        z = NormalDist().inv_cdf(self.config.decision_confidence)

        def exposure(cand: dict[str, Any]) -> dict[int, float]:
            weights: dict[int, float] = defaultdict(float)
            for idx, act in enumerate(cand["trajectory"]):
                disc = self.config.gamma**idx
                for p in act["starters"]:
                    weights[int(p)] += disc
                if act.get("captain") is not None:
                    weights[int(act["captain"])] += disc * (2 if act.get("chip") == "3xc" else 1)
                if act.get("chip") == "bboost":
                    for p in act["bench"]:
                        weights[int(p)] += disc
            return weights

        def confident(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
            wa, wb = exposure(a), exposure(b)
            sigma = sigma_est * math.sqrt(sum((wa.get(p, 0.0) - wb.get(p, 0.0)) ** 2 for p in set(wa) | set(wb)))
            delta = a["total_objective"] - b["total_objective"]
            needed = max(stability_threshold, z * sigma)
            return {
                "gain": round(delta, 2),
                "uncertainty": round(sigma, 2),
                "needed": round(needed, 2),
                "passed": delta >= needed,
                "confidence": self.config.decision_confidence,
            }

        def hit_check(cand: dict[str, Any]) -> tuple[bool, str]:
            """(allowed, verdict text). A hit needs the owner's raw-gain rule AND a confident gain over the best
            plan without a hit (the -4 per hit is already inside the objective)."""
            if cand["hits"] == 0:
                return True, "No hit recommended"
            if enforce_hit_gain and cand["expected_player_gain"] + 1e-9 < cand["hit_cost"]:
                return False, (
                    f"AVOID: Hit of -{cand['hit_cost']} pts is not covered by the transferred players' own "
                    f"expected returns ({cand['expected_player_gain']:+.2f} xP)."
                )
            no_hit = [c for c in unique.values() if c["hits"] == 0 and c["chip"] == cand["chip"]]
            base = max(no_hit, key=lambda c: c["total_objective"]) if no_hit else cand_roll
            check = confident(cand, base)
            cand["decision_check"] = dict(check, versus="best plan without a hit")
            if check["passed"]:
                return True, (
                    f"JUSTIFIED: the -{cand['hit_cost']} hit gains {check['gain']:+.2f} xP over the best plan "
                    f"without it, above its {check['uncertainty']:.2f} xP uncertainty."
                )
            return False, (
                f"MARGINAL: the -{cand['hit_cost']} hit gains {check['gain']:+.2f} xP, within its "
                f"{check['uncertainty']:.2f} xP uncertainty (needs {check['needed']:.2f}). Free transfers only."
            )

        def gate_new() -> None:
            for cand in unique.values():
                if "hit_allowed" not in cand:
                    cand["hit_allowed"], cand["hit_verdict"] = hit_check(cand)

        def standings() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
            """(ranked by objective, best plan passing the hit gates, best eligible plan without a chip)."""
            ranked = sorted(unique.values(), key=lambda c: c["total_objective"], reverse=True)
            eligible = [c for c in ranked if c["hit_allowed"]]
            hold = next((c for c in eligible if c["chip"] is None), None) if not forced else None
            return ranked, eligible[0], hold

        gate_new()

        # --- wave 2: hit-free re-solves when the best plan needs a hit that the hit rule rejects, and
        # distinct week-1 alternatives (no-good cuts) in the background
        def moves_of(*names: str) -> list[tuple[set[int], set[int]]]:
            return [
                (
                    {t["element"] for t in plans[n].recommended_action.transfers_in},
                    {t["element"] for t in plans[n].recommended_action.transfers_out},
                )
                for n in names
                if n in plans
            ]

        # Two alternatives that differ from the plans already found; identical results are merged later
        alt_runs: dict[str, list[tuple[set[int], set[int]]]] = {}
        if len(player_pool_df) > 30:
            alt_runs = {"alt:0": moves_of("opt", "plan_1"), "alt:1": moves_of("opt", "plan_1", "hold", "roll")}

        ranked, top_cand, hold_cand = standings()
        extra_tasks: dict[str, dict[str, Any]] = {}
        if not ranked[0]["hit_allowed"]:
            extra_tasks["opt_nohit"] = dict(forced_kw, max_w1_hits=0)
        if not forced and next((c for c in ranked if c["chip"] is None), None) is not hold_cand:
            extra_tasks["hold_nohit"] = {"no_chip_w1": True, "max_w1_hits": 0}

        pool2 = ThreadPoolExecutor(max_workers=max(1, min(workers, len(alt_runs) + len(chips_now) + len(extra_tasks))))
        try:
            alt_futures = {
                name: pool2.submit(run, name, dict(forced_kw, forbidden_w1_moves=forbidden), alt_limit)
                for name, forbidden in alt_runs.items()
            }
            if extra_tasks:
                extra = run_batch(extra_tasks, wave2_limit)
                new_plans = {n: p for n, p in extra.items() if isinstance(p, MultiPeriodPlan)}
                plans.update(new_plans)
                add_plans(new_plans)
                gate_new()
                ranked, top_cand, hold_cand = standings()

            rec_plan = top_cand
            stability_note = None
            if not forced and rec_plan["chip"] and hold_cand is not None:
                # Playing a chip now is judged against keeping it (which includes the same chip next week)
                check = confident(rec_plan, hold_cand)
                if not check["passed"]:
                    stability_note = (
                        f"Keep the chip: playing it now gains {check['gain']:+.2f} xP over keeping it, within the "
                        f"{check['uncertainty']:.2f} xP uncertainty of the forecasts (needs {check['needed']:.2f})."
                    )
                    rejected = dict(check, chip=rec_plan["chip"], versus="keeping the chip")
                    rec_plan = hold_cand
                    rec_plan["rejected_alternative"] = rejected
                else:
                    rec_plan["decision_check"] = dict(check, versus="keeping the chip")
            if not forced and not rec_plan["chip"] and rec_plan is not cand_roll:
                check = confident(rec_plan, cand_roll)
                if not check["passed"]:
                    stability_note = (
                        f"Roll the transfer: the best move gains {check['gain']:+.2f} xP, within the "
                        f"{check['uncertainty']:.2f} xP uncertainty of the forecasts (needs {check['needed']:.2f})."
                    )
                    cand_roll["rejected_alternative"] = dict(check, versus="rolling")
                    rec_plan = cand_roll
                else:
                    rec_plan.setdefault("decision_check", dict(check, versus="rolling"))
            if stability_note:
                rec_plan["recommendation_summary"] = stability_note

            # --- mini-league tie-break (decision_model.md section 4): among plans the forecasts cannot tell
            # apart from the recommendation, take the one most likely to finish above every rival.
            if rival_context and rival_context.get("rivals"):
                from fpl_oracle.league.objective import SD_PER_P10_P90, rival_weeks, week_weights, win_probability

                positions = {
                    int(e): str(p) for e, p in zip(player_pool_df["element"], player_pool_df["position"], strict=True)
                }
                sd_maps = {}
                for g in horizon_gws:
                    frame = horizon_projections.get(g, player_pool_df)
                    spread = (
                        frame.get("p90", frame["expected_points"]) - frame.get("p10", frame["expected_points"])
                    ).clip(lower=0)
                    sd_maps[g] = dict(zip(frame["element"].astype(int), spread / SD_PER_P10_P90, strict=True))
                rival_cache: dict[int, list[dict[int, float]]] = {}
                held = rival_weeks(list(current_squad_df["element"].astype(int)), horizon_gws, maps, positions)

                def plan_weeks(cand: dict[str, Any]) -> list[dict[int, float]]:
                    return [
                        week_weights(a["starters"], a.get("captain"), a.get("chip"), a["bench"])
                        for a in cand["trajectory"]
                    ]

                def raw_points(weeks: list[dict[int, float]]) -> float:
                    return sum(
                        w * maps[g].get(p, 0.0) for g, wk in zip(horizon_gws, weeks, strict=True) for p, w in wk.items()
                    )

                best_points = max(raw_points(plan_weeks(c)) for c in unique.values())
                for cand in unique.values():
                    cand["league"] = win_probability(
                        plan_weeks(cand),
                        held,
                        best_points,
                        horizon_gws,
                        rival_context,
                        maps,
                        sd_maps,
                        positions,
                        rival_cache,
                    )
                tied = [
                    c
                    for c in unique.values()
                    if c["league"] is not None
                    and c["hit_allowed"]
                    and (c is rec_plan or not confident(rec_plan, c)["passed"])
                ]
                choice = max(tied, key=lambda c: (c["league"]["p_win"], c["total_objective"]), default=rec_plan)
                if (
                    choice is not rec_plan
                    and rec_plan.get("league")
                    and choice["league"]["p_win"] > rec_plan["league"]["p_win"] + 0.005
                ):
                    choice["league_tiebreak"] = {
                        "replaced_points_plan": rec_plan["total_objective"],
                        "points_given_up": round(rec_plan["total_objective"] - choice["total_objective"], 2),
                        "p_win_before": rec_plan["league"]["p_win"],
                        "p_win_after": choice["league"]["p_win"],
                    }
                    choice["recommendation_summary"] = (
                        f"League tie-break: as good on predicted points (within forecast uncertainty) and raises the "
                        f"chance of finishing first from {rec_plan['league']['p_win']:.0%} to {choice['league']['p_win']:.0%}."
                    )
                    rec_plan = choice
            rejected_hit = next((c for c in ranked if c["total_objective"] > top_cand["total_objective"]), None)
            rec_chip = rec_plan["chip"]

            # --- what each scheduled chip is worth: the plan re-solved without that chip
            expiry = SET_1_LAST_GW if target_gw <= SET_1_LAST_GW else 38
            scheduled = {a["chip"]: a["gameweek"] for a in rec_plan["trajectory"] if a["chip"] and a["chip"] != forced}

            def run_without(chip: str) -> MultiPeriodPlan | Exception:
                shrunk = {
                    n: [c for c in names if c != chip] if n == target_set else list(names)
                    for n, names in avail_lists.items()
                }
                # Keep this week's chip decision (it is what the card recommends) so the comparison is only
                # about the chip being measured, and the solve stays as easy as the one that found the plan.
                if forced:
                    week_one: dict[str, Any] = dict(forced_kw)
                elif rec_chip and rec_chip != chip:
                    week_one = {"forced_w1_chip": rec_chip}
                elif rec_chip is None:
                    week_one = {"no_chip_w1": True}
                else:
                    week_one = {}
                return run(f"without:{chip}", week_one, wave2_limit, chips_by_set=shrunk)

            report_search_progress("Measuring what each chip is worth")
            without_futures = {c: pool2.submit(run_without, c) for c in scheduled}
            without_results = {c: f.result() for c, f in without_futures.items()}
            alt_results = {n: f.result() for n, f in alt_futures.items()}
            alt_plans = {n: r for n, r in alt_results.items() if isinstance(r, MultiPeriodPlan)}
        finally:
            pool2.shutdown(wait=True)

        plans.update(alt_plans)
        add_plans(alt_plans)
        gate_new()
        candidate_plans = list(unique.values())

        if rejected_hit is not None and rec_plan["hits"] == 0:
            hit_verdict = f"{rejected_hit['hit_verdict']} Best hit plan not recommended."
        else:
            hit_verdict = rec_plan["hit_verdict"]

        # --- chip table: HOLD (best eligible week-1 plan without a chip) against each chip played now
        hold_obj = hold_cand["total_objective"] if hold_cand else None
        normal_plan_feasible = hold_cand is not None
        sources: list[tuple[str, str | None, dict[str, Any]]] = []
        if forced:
            sources.append((forced, forced, rec_plan))
        else:
            if hold_cand is not None:
                # What HOLD means on screen is the no-chip plan actually recommended (e.g. the Roll fallback)
                sources.append(("HOLD", None, rec_plan if rec_chip is None else hold_cand))
            sources += [(c, c, by_name[f"chip:{c}"]) for c in chips_now if f"chip:{c}" in by_name]
        rows = []
        for label, code, cand in sources:
            cand = rec_plan if code == rec_chip else unique.get(move_key(cand), cand)
            gain: float | None
            if code is None:
                gain = 0.0
            else:
                gain = round(cand["total_objective"] - hold_obj, 2) if hold_obj is not None else None
            reason = "Best plan for this week with this chip, everything else re-optimised over the horizon."
            if not normal_plan_feasible:
                reason = "No ordinary plan is possible with the active chip. Comparison gain unavailable."
            elif not cand["hit_allowed"]:
                reason = f"Needs a hit that the hit rule rejects. {cand['hit_verdict']}"
            rows.append(
                {
                    "chip": label,
                    "chip_code": code,
                    "action": "DEPLOY" if code else "HOLD",
                    "plan": cand,
                    "net_gain_vs_hold": gain,
                    "gross_gain_vs_hold": round(cand["gross_expected_points"] - hold_cand["gross_expected_points"], 2)
                    if hold_cand
                    else None,
                    "effective_trajectory_score": cand["total_objective"],
                    "is_recommended": bool(code == rec_chip),
                    "reason": reason,
                }
            )
        rows.sort(key=lambda r: (r["net_gain_vs_hold"] is None, -(r["net_gain_vs_hold"] or 0.0)))
        best_row = next((r for r in rows if r["is_recommended"]), rows[0] if rows else None)

        # --- chip timeline from the recommended trajectory: what each chip is worth, and the cost of saving it
        roadmap = []
        frontier = []
        measurement_status = "measured_through_expiry" if horizon_gws[-1] == expiry else "incomplete_window"
        for chip in [c for c in chips_now if c != forced]:
            gw = scheduled.get(chip)
            base = without_results.get(chip)
            # Any plan already found that never uses this chip is also a valid "without it" plan; the best of
            # them bounds the chip's value from above even when the dedicated re-solve ran out of time.
            unused = [p for p in plans.values() if all(a.active_chip != chip for a in p.trajectory)]
            if unused and (
                not isinstance(base, MultiPeriodPlan) or max(p.total_objective for p in unused) > base.total_objective
            ):
                base = max(unused, key=lambda p: p.total_objective)
            delta: float | None
            if gw is None:
                delta, status, base_score = 0.0, "measured", rec_plan["total_objective"]
                base_traj = rec_plan["trajectory"]
            elif isinstance(base, MultiPeriodPlan):
                delta = round(rec_plan["total_objective"] - base.total_objective, 2)
                status = "measured" if delta >= 0 else "search_inconclusive"
                base_score = base.total_objective
                base_traj = [
                    {"gameweek": a.gameweek, "chip": a.active_chip, "transfers_count": a.transfers_count}
                    for a in base.trajectory
                ]
            else:
                delta, base_score, base_traj = None, None, []
                # No legal plan without this chip vs. a search that ran out of time
                status = (
                    "search_inconclusive"
                    if isinstance(base, MilpNoSolution) or not isinstance(base, MilpInfeasible)
                    else "required_for_feasibility"
                )
            roadmap.append(
                {
                    "chip": CHIP_DISPLAY_NAMES[chip],
                    "code": chip,
                    "recommended_gw": gw,
                    "expected_gain": delta,
                    "baseline_score": base_score,
                    "baseline_solver_gap": base.solver_gap if isinstance(base, MultiPeriodPlan) else None,
                    "gain_is_upper_bound": bool(
                        gw is not None and isinstance(base, MultiPeriodPlan) and base.is_lower_bound
                    ),
                    "plan_score": rec_plan["total_objective"],
                    "baseline_trajectory": base_traj,
                    "date_comparison": [],
                    "forecast_through": expiry,
                    "status": status,
                    "reasoning": "Extra weighted points over the full plan without this chip. "
                    "Transfers and other chip dates may change; gains do not add together."
                    if gw is not None
                    else "Not used in the best plan within the horizon: no measured benefit.",
                }
            )
            if isinstance(base, MultiPeriodPlan) or gw is None:
                frontier.append(
                    {
                        "chip": chip,
                        "retained_through_gameweek": horizon_gws[-1],
                        "first_chip": base.recommended_action.active_chip
                        if isinstance(base, MultiPeriodPlan)
                        else None,
                        "retained_trajectory": base_traj,
                        "ending_bank_tenths": base.trajectory[-1].bank_end
                        if isinstance(base, MultiPeriodPlan)
                        else None,
                        "ending_free_transfers": base.trajectory[-1].free_transfers_next
                        if isinstance(base, MultiPeriodPlan)
                        else None,
                        "discounted_horizon_cost_to_preserve": delta,
                        "tail_value_break_even_at_horizon_end": round(
                            float(delta or 0.0) / self.config.gamma ** len(horizon_gws), 2
                        ),
                        "interpretation": "If this chip's later option value exceeds this threshold, the preserved-chip "
                        "trajectory may beat current-window deployment. Later value is not estimated.",
                    }
                )

        # --- honest solver report: a time-limited incumbent is never called proven
        solver_rows = dict(plans)
        solver_rows.update({f"without:{c}": r for c, r in without_results.items() if isinstance(r, MultiPeriodPlan)})
        decisive = [p for n, p in solver_rows.items() if not n.startswith(("alt:", "without:"))]
        solver = {
            "time_limit_sec": limit,
            "plans": {
                n: {
                    "status": p.solver_status,
                    "gap": round(p.solver_gap, 6),
                    "is_lower_bound": p.is_lower_bound,
                    "seconds": p.solve_time_sec,
                }
                for n, p in solver_rows.items()
            },
            "max_gap": round(max((p.solver_gap for p in decisive), default=0.0), 6),
            "global_optimum_proven": not any(p.is_lower_bound for p in decisive),
        }

        ft_option_value = 0.8
        return {
            "recommended_plan": rec_plan,
            "candidate_plans": candidate_plans,
            "hit_verdict": hit_verdict,
            "stability_note": stability_note,
            "recommended_chip": rec_chip,
            "chip_action": "DEPLOY" if rec_chip else "HOLD",
            "normal_plan_feasible": normal_plan_feasible,
            "hold_plan": rec_plan if rec_chip is None else (hold_cand or rec_plan),
            "roll_plan": cand_roll,
            "chip_comparison_table": rows,
            "best_candidate": best_row,
            "measured_chip_roadmap": roadmap,
            "chip_measurement_status": measurement_status,
            "resource_frontier": frontier,
            "decision_readiness": {
                "status": "conditional" if rec_chip else "bounded_model_candidate",
                "reasons": [
                    "Future rival actions are hypothetical, not known",
                    "No validated resource valuation beyond the supported projection window",
                    "Reassess every deadline; future steps are contingent",
                ],
                "automatic_execution": False,
            },
            "solver": solver,
            "horizon_gameweeks": horizon_gws,
            "transfer_roadmap": [
                {
                    "gameweek": act["gameweek"],
                    "planned_transfers": act["transfers_in"],
                    "planned_chip": act["chip"],
                    "projected_bank": act["bank_end"],
                    "projected_ft": act["ft_next"],
                }
                for act in rec_plan.get("trajectory", [])
            ],
            "future_hit_avoidance": {
                "greedy_horizon_hits": 0,
                "multi_gw_horizon_hits": rec_plan.get("horizon_hits", 0),
                "future_hits_avoided": 0,
                "explanation": "Multi-period MILP optimally minimizes hit penalties across the horizon.",
            },
            "ft_option_value": ft_option_value,
            "ft_option_explanation": (
                f"Option value of an extra banked free transfer is estimated at +{ft_option_value:.2f} xP "
                "providing restructuring flexibility without hit penalty."
            ),
            "target_gameweek": target_gw,
            "current_bank_millions": round(float(bank) / 10.0, 2),
            "available_free_transfers": free_transfers,
            "stability_threshold": stability_threshold,
            "include_price_gain": include_price_gain,
        }


multiperiod_optimizer = MultiPeriodOptimizer()
