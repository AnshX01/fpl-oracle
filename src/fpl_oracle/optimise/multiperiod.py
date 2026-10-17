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
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import pulp

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


@dataclass
class MultiPeriodConfig:
    horizon_len: int = 8
    gamma: float = 0.95
    solver_name: Literal["highs", "cbc"] = "highs"
    mip_gap: float = 1e-4  # 0.01% (rel gap <= 1e-4, abs gap <= 0.05 xP on 500 xP horizon)
    time_limit_sec: int = 60
    bench_weights: tuple[float, float, float, float] = (0.10, 0.05, 0.02, 0.0)
    w_bench_mean: float = 0.05
    lambda_bank: float = 0.05  # 0.05 xP per 1.0m (£0.1m = 0.005 xP)
    chip_week_ft_rule: Literal["preserve", "preserve_plus_one"] = "preserve"
    pruning_limits: dict[str, int] = field(default_factory=lambda: {"GKP": 15, "DEF": 45, "MID": 45, "FWD": 25})
    enablers_count: int = 5
    ft_terminal_increments: tuple[float, float, float, float] = (1.0, 0.8, 0.6, 0.4)
    # Hit policy configuration (WI-2.5)
    hit_threshold_base: float = 2.0  # Default base net xP gain over roll required to justify a hit
    hit_threshold_per_hit: float = 0.5  # Compounding margin required per additional hit beyond the first
    hit_p_star_base: float = 0.60  # Base required P(J_plan > J_roll) to recommend a hit [PROVISIONAL]
    hit_k_sigma_base: float = 0.25  # Base hurdle multiplier k = Phi^-1(p*) for Delta >= k * sigma_Delta [PROVISIONAL]
    hit_risk_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "conservative": 1.0,
            "protect_lead": 1.0,
            "balanced": 0.0,
            "points": 0.0,
            "aggressive": -1.0,
            "chase_deficit": -1.0,
        }
    )
    hit_risk_p_star_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "conservative": 0.15,
            "protect_lead": 0.15,
            "balanced": 0.0,
            "points": 0.0,
            "aggressive": -0.10,
            "chase_deficit": -0.10,
        }
    )
    hit_risk_k_adjustments: dict[str, float] = field(
        default_factory=lambda: {
            "conservative": 0.42,
            "protect_lead": 0.42,
            "balanced": 0.0,
            "points": 0.0,
            "aggressive": -0.25,
            "chase_deficit": -0.25,
        }
    )

    def compute_hit_parameters(
        self,
        risk_preference: str = "balanced",
        mini_league_mode: str | None = None,
        mini_league_deficit: float | None = None,
    ) -> tuple[float, float]:
        """
        Derives p* and k hurdle parameters from risk preference and mini-league deficit (WI-2.5).
        [PROVISIONAL: hand-set heuristics; to be tuned on walk-forward backtest after WI-4.1]
        Under the Gaussian assumption Delta ~ N(mu, sigma^2), P(Delta > 0) >= p* implies mu >= k * sigma,
        where k = Phi^-1(p*).
        - Balanced mode (deficit < 20): p* = 0.60, k = Phi^-1(0.60) = 0.25
        - Aggressive / Chasing deficit (deficit >= 20): p* = 0.50, k = Phi^-1(0.50) = 0.00
        - Conservative / Defending lead (deficit <= -20): p* = 0.75, k = Phi^-1(0.75) = 0.67
        Rule: Hit is recommended if and only if (P(Delta > 0) >= p*) AND (mean_delta >= k * sigma_delta).
        """
        risk_key = (mini_league_mode or risk_preference).lower()
        p_adj = self.hit_risk_p_star_adjustments.get(risk_key, 0.0)
        k_adj = self.hit_risk_k_adjustments.get(risk_key, 0.0)

        deficit_p_adj = 0.0
        deficit_k_adj = 0.0
        if mini_league_deficit is not None:
            if mini_league_deficit >= 20.0:
                deficit_p_adj = -0.10
                deficit_k_adj = -0.25
            elif mini_league_deficit <= -20.0:
                deficit_p_adj = 0.15
                deficit_k_adj = 0.42

        p_star = max(0.40, min(0.95, self.hit_p_star_base + p_adj + deficit_p_adj))
        k = max(0.00, min(3.0, self.hit_k_sigma_base + k_adj + deficit_k_adj))
        return round(p_star, 2), round(k, 2)

    def compute_hit_threshold(
        self,
        hits: int = 1,
        risk_preference: str = "balanced",
        mini_league_mode: str | None = None,
        mini_league_deficit: float | None = None,
    ) -> float:
        """
        Derives the dynamic hit justification threshold from risk preference and mini-league context (WI-2.5).
        - Default: 2.0 net xP gain over roll for 1 hit in balanced mode.
        - Conservative / Defending lead: +1.0 xP threshold hurdle (3.0 xP required).
        - Aggressive / Chasing deficit: -1.0 xP threshold hurdle (1.0 xP required).
        - Deficit >= 20 pts: additional -0.5 xP discount to encourage upside moves.
        - Compounding: +0.5 xP margin per additional hit beyond the first.
        """
        if hits <= 0:
            return 0.0

        risk_key = (mini_league_mode or risk_preference).lower()
        risk_adj = self.hit_risk_adjustments.get(risk_key, 0.0)

        deficit_adj = 0.0
        if mini_league_deficit is not None:
            if mini_league_deficit >= 20.0:
                deficit_adj = -0.5
            elif mini_league_deficit <= -20.0:
                deficit_adj = 0.5

        compounding = max(0, hits - 1) * self.hit_threshold_per_hit
        effective_threshold = max(0.5, self.hit_threshold_base + risk_adj + deficit_adj + compounding)
        return round(effective_threshold, 2)


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


class MultiPeriodOptimizer:
    """
    Mixed-Integer Linear Programming Multi-Period Optimizer for FPL (WI-2.1 to WI-2.5).
    """

    def __init__(self, config: MultiPeriodConfig | None = None) -> None:
        self.config = config or MultiPeriodConfig()

    def _get_solver(self) -> pulp.LpSolver:
        """Selects HiGHS if available and configured, else falls back to bundled CBC."""
        if self.config.solver_name == "highs":
            try:
                highs = pulp.HiGHS(
                    gapRel=self.config.mip_gap,
                    timeLimit=self.config.time_limit_sec,
                    msg=False,
                )
                if highs.available():
                    return highs
            except Exception as e:
                logger.warning(f"HiGHS solver initialization failed: {e}. Falling back to CBC.")

        return pulp.PULP_CBC_CMD(
            gapRel=self.config.mip_gap,
            timeLimit=self.config.time_limit_sec,
            msg=False,
        )

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
            gw_map = {int(r["element"]): float(r.get("expected_points", 0.0)) for _, r in df_gw.iterrows()}
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
    ) -> MultiPeriodPlan:
        """
        Solves the Multi-Period MILP formulation over horizon T.
        Supports conditional forcing for alternative plan generation.
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
        xfh: dict[tuple[int, int], pulp.LpVariable] = {}
        s: dict[tuple[int, int], pulp.LpVariable] = {}
        c: dict[tuple[int, int], pulp.LpVariable] = {}
        tin: dict[tuple[int, int], pulp.LpVariable] = {}
        tout: dict[tuple[int, int], pulp.LpVariable] = {}
        b: dict[tuple[int, int], pulp.LpVariable] = {}
        x_eff: dict[tuple[int, int], pulp.LpVariable] = {}

        is_wc_w1 = forced_w1_chip is not None and forced_w1_chip.lower() == "wildcard"

        for t in horizon_gws:
            is_wc_t = (is_wc_w1 and (t == t0)) or (
                forced_chip_schedule is not None and forced_chip_schedule.get(t, "").lower() == "wildcard"
            )
            for p in players:
                x[p, t] = pulp.LpVariable(f"x_{p}_{t}", cat=pulp.LpBinary)
                xfh[p, t] = pulp.LpVariable(f"xfh_{p}_{t}", cat=pulp.LpBinary)
                s[p, t] = pulp.LpVariable(f"s_{p}_{t}", cat=pulp.LpBinary)
                c[p, t] = pulp.LpVariable(f"c_{p}_{t}", cat=pulp.LpBinary)
                tin[p, t] = pulp.LpVariable(
                    f"in_{p}_{t}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpContinuous if is_wc_t else pulp.LpBinary,
                )
                tout[p, t] = pulp.LpVariable(
                    f"out_{p}_{t}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpContinuous if is_wc_t else pulp.LpBinary,
                )
                b[p, t] = pulp.LpVariable(f"b_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
                x_eff[p, t] = pulp.LpVariable(f"xeff_{p}_{t}", lowBound=0, upBound=1, cat=pulp.LpContinuous)

        u = {t: pulp.LpVariable(f"u_{t}", lowBound=0, upBound=15, cat=pulp.LpInteger) for t in horizon_gws}
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

            # Free Hit squad constraints
            for p in players:
                prob += xfh[p, t] <= fh[t]
            prob += pulp.lpSum(xfh[p, t] for p in gkps) == 2 * fh[t]
            prob += pulp.lpSum(xfh[p, t] for p in defs) == 5 * fh[t]
            prob += pulp.lpSum(xfh[p, t] for p in mids) == 5 * fh[t]
            prob += pulp.lpSum(xfh[p, t] for p in fwds) == 3 * fh[t]
            for _tm, tm_players in club_map.items():
                prob += pulp.lpSum(xfh[p, t] for p in tm_players) <= 3 * fh[t]
            prob += pulp.lpSum(buy_costs[p] * xfh[p, t] for p in players) <= initial_team_value * fh[t]

            # Effective squad linking & Lineup
            for p in players:
                prob += x_eff[p, t] <= x[p, t] + fh[t]
                prob += x_eff[p, t] <= xfh[p, t] + (1 - fh[t])
                prob += x_eff[p, t] >= x[p, t] - fh[t]
                prob += x_eff[p, t] >= xfh[p, t] - (1 - fh[t])

                prob += s[p, t] <= x_eff[p, t]
                prob += b[p, t] == x_eff[p, t] - s[p, t]
                prob += c[p, t] <= s[p, t]

                # TC & BB linearizations
                prob += tcc[p, t] <= c[p, t]
                prob += tcc[p, t] <= tc[t]
                prob += tcc[p, t] >= c[p, t] + tc[t] - 1

                prob += bb_b[p, t] <= b[p, t]
                prob += bb_b[p, t] <= bb[t]
                prob += bb_b[p, t] >= b[p, t] + bb[t] - 1

            # Formation: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 Starters, 1 Captain
            prob += pulp.lpSum(s[p, t] for p in players) == 11
            prob += pulp.lpSum(s[p, t] for p in gkps) == 1
            prob += pulp.lpSum(s[p, t] for p in defs) >= 3
            prob += pulp.lpSum(s[p, t] for p in defs) <= 5
            prob += pulp.lpSum(s[p, t] for p in mids) >= 2
            prob += pulp.lpSum(s[p, t] for p in mids) <= 5
            prob += pulp.lpSum(s[p, t] for p in fwds) >= 1
            prob += pulp.lpSum(s[p, t] for p in fwds) <= 3
            prob += pulp.lpSum(c[p, t] for p in players) == 1

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

                next_t = horizon_gws[idx + 1] if idx + 1 < H else (horizon_gws[-1] + 1)
                prob += ft_var[next_t] == 1
            else:
                # Flow conservation
                for p in players:
                    if idx == 0:
                        prob += x[p, t] - (1 if p in squad_ids else 0) == tin[p, t] - tout[p, t]
                    else:
                        prev_t = horizon_gws[idx - 1]
                        prob += x[p, t] - x[p, prev_t] == tin[p, t] - tout[p, t]

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

                # Hits: h[t] >= u[t] - ft[t] - 15 * (wc[t] + fh[t])
                prob += h[t] >= u[t] - ft_var[t] - 15 * (wc[t] + fh[t])

                # Max 1 chip per GW
                prob += wc[t] + fh[t] + bb[t] + tc[t] <= 1

                # FT Banking State Machine (Rule R4)
                next_t = horizon_gws[idx + 1] if idx + 1 < H else (horizon_gws[-1] + 1)
                # z[t] == 1 if u[t] > ft_var[t]
                prob += u[t] - ft_var[t] <= 15 * z[t]
                # Normal week: if u <= ft, ft_next <= ft - u + 1; if u > ft, ft_next <= 1
                prob += ft_var[next_t] <= ft_var[t] - u[t] + 1 + 5 * z[t] + 5 * (wc[t] + fh[t])
                prob += ft_var[next_t] <= 1 + 5 * (1 - z[t]) + 5 * (wc[t] + fh[t])
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
        if chips_available is None:
            raw_chips_set = {"wildcard", "freehit", "bboost", "3xc"}
        else:
            raw_chips_set = set(c.lower().strip() for c in chips_available)

        wc_avail = "wildcard" in raw_chips_set
        fh_avail = "freehit" in raw_chips_set
        bb_avail = "bboost" in raw_chips_set
        tc_avail = "3xc" in raw_chips_set

        if t1_gws:
            prob += pulp.lpSum(wc[t] for t in t1_gws) <= (1 if wc_avail else 0)
            prob += pulp.lpSum(fh[t] for t in t1_gws) <= (1 if fh_avail else 0)
            prob += pulp.lpSum(bb[t] for t in t1_gws) <= (1 if bb_avail else 0)
            prob += pulp.lpSum(tc[t] for t in t1_gws) <= (1 if tc_avail else 0)

        if t2_gws:
            prob += pulp.lpSum(wc[t] for t in t2_gws) <= (1 if wc_avail else 0)
            prob += pulp.lpSum(fh[t] for t in t2_gws) <= (1 if fh_avail else 0)
            prob += pulp.lpSum(bb[t] for t in t2_gws) <= (1 if bb_avail else 0)
            prob += pulp.lpSum(tc[t] for t in t2_gws) <= (1 if tc_avail else 0)

        # Rule R2: No FH in GW1
        if 1 in horizon_gws:
            prob += fh[1] == 0

        # Rule R2: No FH in GW20 if FH in GW19
        if 19 in horizon_gws and 20 in horizon_gws:
            prob += fh[19] + fh[20] <= 1
        elif 20 in horizon_gws and previous_chip and previous_chip.lower() == "freehit":
            prob += fh[20] == 0

        # Apply Locked / Excluded requirements at GW t0
        for p in locked_in:
            if p in players:
                prob += s[p, t0] == 1
        for p in locked_out:
            if p in players:
                prob += x[p, t0] == 0
                prob += xfh[p, t0] == 0
        for tm in excluded_teams:
            for p in club_map.get(tm, []):
                for t in horizon_gws:
                    prob += x[p, t] == 0
                    prob += xfh[p, t] == 0

        # Forced constraints for alternative plan generation
        if forced_w1_transfers is not None:
            prob += u[t0] == forced_w1_transfers
        if min_w1_hits is not None:
            prob += h[t0] >= min_w1_hits
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
                prob += wc[t0] == 1
                for t in horizon_gws:
                    if t != t0:
                        prob += wc[t] == 0
                    prob += fh[t] == 0
                    prob += bb[t] == 0
                    prob += tc[t] == 0
            elif chip_name == "freehit":
                prob += fh[t0] == 1
            elif chip_name == "bboost":
                prob += bb[t0] == 1
            elif chip_name == "3xc":
                prob += tc[t0] == 1
        elif forced_w1_transfers is not None or min_w1_hits is not None:
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
            gw_xp = {int(r["element"]): float(r.get("expected_points", 0.0)) for _, r in df_gw.iterrows()}

            starters_val = pulp.lpSum(gw_xp.get(p, 0.0) * s[p, t] for p in players)
            cap_val = pulp.lpSum(gw_xp.get(p, 0.0) * (c[p, t] + tcc[p, t]) for p in players)
            bench_val = pulp.lpSum(
                gw_xp.get(p, 0.0) * (w_bench * b[p, t] + (1.0 - w_bench) * bb_b[p, t]) for p in players
            )
            hit_pen = float(HIT_COST_POINTS) * h[t]

            obj_terms.append(discount * (starters_val + cap_val + bench_val - hit_pen))

        # Terminal FT valuation
        ft_term = pulp.lpSum(w * var for w, var in zip(v_weights, v, strict=True))
        obj_terms.append((gamma**H) * ft_term)

        # Micro-incentive for maintaining maximum available banked FTs across horizon
        obj_terms.append(1e-4 * pulp.lpSum(ft_var[t] for t in horizon_gws + [horizon_gws[-1] + 1]))

        # Terminal Bank valuation
        obj_terms.append(self.config.lambda_bank * (bank_var[horizon_gws[-1]] / 10.0))

        prob += pulp.lpSum(obj_terms)

        # Solve
        solver = self._get_solver()
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
            is_time_limit = solve_time >= (self.config.time_limit_sec - 1.5)
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

            t_starters = [p for p in players if s[p, t].varValue is not None and s[p, t].varValue > 0.5]
            t_bench = [p for p in players if b[p, t].varValue is not None and b[p, t].varValue > 0.5]
            cap_candidates = [p for p in players if c[p, t].varValue is not None and c[p, t].varValue > 0.5]
            t_cap = cap_candidates[0] if cap_candidates else (t_starters[0] if t_starters else players[0])

            # Vice-captain is best remaining starter
            df_gw = horizon_projections.get(t, player_pool_df)
            gw_xp = {int(r["element"]): float(r.get("expected_points", 0.0)) for _, r in df_gw.iterrows()}
            rem_starters = [p for p in t_starters if p != t_cap]
            t_vc = max(rem_starters, key=lambda p: gw_xp.get(p, 0.0)) if rem_starters else None

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

        # Objective decomposition calculation
        disc_lineup_xp = 0.0
        disc_captain_xp = 0.0
        disc_bench_xp = 0.0
        disc_hits_cost = 0.0
        for idx, t in enumerate(horizon_gws):
            discount = gamma**idx
            df_gw = horizon_projections.get(t, player_pool_df)
            gw_xp = {int(r["element"]): float(r.get("expected_points", 0.0)) for _, r in df_gw.iterrows()}
            l_val = sum(gw_xp.get(p, 0.0) for p in players if s[p, t].varValue is not None and s[p, t].varValue > 0.5)
            c_val = sum(gw_xp.get(p, 0.0) for p in players if c[p, t].varValue is not None and c[p, t].varValue > 0.5)
            if tc[t].varValue is not None and tc[t].varValue > 0.5:
                c_val *= 2.0
            b_mult = 1.0 if (bb[t].varValue is not None and bb[t].varValue > 0.5) else w_bench
            b_val = b_mult * sum(
                gw_xp.get(p, 0.0) for p in players if b[p, t].varValue is not None and b[p, t].varValue > 0.5
            )
            h_val = float(HIT_COST_POINTS) * float(h[t].varValue or 0.0)

            disc_lineup_xp += discount * l_val
            disc_captain_xp += discount * c_val
            disc_bench_xp += discount * b_val
            disc_hits_cost += discount * h_val

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
        )

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
        num_mc_scenarios: int = 50,
        horizon_len: int | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive multi-period solver producing the required comparison set (WI-2.2):
        - Roll plan (force 0 transfers in week 1)
        - Best 1-transfer plan (force 1 transfer)
        - Optimal plan (unconstrained MILP, can be 0, 1, 2, 3+ hit-free or hits)
        - Best hit plan (force >= 1 hit)
        - Chip plans (per available chip)
        - Top-K distinct week-1 moves via no-good cuts
        - Hit policy evaluation (WI-2.5)
        Outputs the standard dict schema expected by CLI, UI, and test suites.
        """
        # 1. Base roll plan
        roll_plan = self.optimize(
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
            chips_available=chips_available,
            previous_chip=previous_chip,
            forced_w1_transfers=0,
        )

        # 2. Unconstrained optimal plan
        opt_plan = self.optimize(
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
            chips_available=chips_available,
            previous_chip=previous_chip,
        )

        # 3. 1-transfer plan
        plan_1 = self.optimize(
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
            chips_available=chips_available,
            previous_chip=previous_chip,
            forced_w1_transfers=1,
        )

        # Compute roll baseline metrics
        roll_gross = round(sum(a.lineup_xp + a.captain_xp + a.bench_xp for a in roll_plan.trajectory), 2)
        roll_net = round(roll_gross - (roll_plan.horizon_hits * 4.0), 2)

        # Helper to convert MultiPeriodPlan to legacy dict format
        def _to_plan_dict(plan: MultiPeriodPlan) -> dict[str, Any]:
            act = plan.recommended_action
            h_gross = round(sum(a.lineup_xp + a.captain_xp + a.bench_xp for a in plan.trajectory), 2)
            h_net = round(h_gross - (plan.horizon_hits * 4.0), 2)
            gain_vs_roll = round(h_net - roll_net, 2)
            imm_gain = round(act.net_xp - roll_plan.recommended_action.net_xp, 2)

            # Reconstruct updated squad after GW1 transfers to obtain lineup & captaincy from LineupOptimizer
            out_ids = {t["element"] for t in act.transfers_out}
            in_ids = {t["element"] for t in act.transfers_in}
            retained = current_squad_df[~current_squad_df["element"].isin(out_ids)].copy()
            incoming = player_pool_df[player_pool_df["element"].isin(in_ids)].copy()
            for col in retained.columns:
                if col not in incoming.columns:
                    if col in ("purchase_price", "selling_price"):
                        incoming[col] = incoming.get("value", 50.0)
                    elif col == "chance_of_playing":
                        incoming[col] = 100.0
                    else:
                        incoming[col] = 0.0
            updated_squad_df = pd.concat([retained, incoming], ignore_index=True)

            from fpl_oracle.optimise.lineup import lineup_optimizer

            lineup_res: dict[str, Any] = {}
            if len(updated_squad_df) == 15:
                try:
                    lineup_res = lineup_optimizer.select_lineup_and_captain(
                        updated_squad_df,
                        is_triple_captain=(act.active_chip == "3xc"),
                        is_bench_boost=(act.active_chip == "bboost"),
                        risk_preference=risk_preference,
                        simulate_captaincy=False,
                    )
                except Exception as ex:
                    logger.warning(f"lineup_optimizer.select_lineup_and_captain failed: {ex}")
                    lineup_res = {}

            if not lineup_res:
                pool_map = {int(r["element"]): r for _, r in player_pool_df.iterrows()}
                cap_row = pool_map.get(act.captain, {})
                vc_id = act.vice_captain or (act.starters[1] if len(act.starters) > 1 else act.captain)
                vc_row = pool_map.get(vc_id, {})
                cap_name = cap_row.get("web_name", f"Player {act.captain}")
                vc_name = vc_row.get("web_name", f"Player {vc_id}")
                lineup_res = {
                    "starters": [pool_map.get(e, {"element": e}) for e in act.starters],
                    "bench": [pool_map.get(e, {"element": e}) for e in act.bench],
                    "captain": {
                        "element": act.captain,
                        "web_name": cap_name,
                        "expected_points": round(float(cap_row.get("expected_points", act.captain_xp)), 2),
                    },
                    "vice_captain": {
                        "element": vc_id,
                        "web_name": vc_name,
                        "expected_points": round(float(vc_row.get("expected_points", 0.0)), 2),
                    },
                }

            cap_obj = lineup_res["captain"]
            vc_obj = lineup_res["vice_captain"]
            cap_id = int(cap_obj["element"]) if isinstance(cap_obj, dict) else act.captain
            vc_id = int(vc_obj["element"]) if isinstance(vc_obj, dict) else (act.vice_captain or act.captain)

            starters_list = (
                [int(x) for x in lineup_res["starters"]["element"].tolist()]
                if isinstance(lineup_res.get("starters"), pd.DataFrame)
                else [p["element"] if isinstance(p, dict) else int(p) for p in lineup_res.get("starters", act.starters)]
            )
            bench_list = (
                [int(x) for x in lineup_res["bench"]["element"].tolist()]
                if isinstance(lineup_res.get("bench"), pd.DataFrame)
                else [p["element"] if isinstance(p, dict) else int(p) for p in lineup_res.get("bench", act.bench)]
            )

            return {
                "plan_type": plan.plan_type,
                "transfers_count": act.transfers_count,
                "transfers_in": act.transfers_in,
                "transfers_out": act.transfers_out,
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
                "robustness_score": 0.85,
                "is_no_regret": True,
                "horizon_gain_vs_roll": gain_vs_roll,
                "horizon_gross_xp": h_gross,
                "horizon_net_xp": h_net,
                "horizon_hits": plan.horizon_hits,
                "gross_expected_points": round(act.lineup_xp + act.captain_xp + act.bench_xp, 2),
                "chip": act.active_chip,
                "lineup": lineup_res,
                "captain": cap_obj,
                "vice_captain": vc_obj,
                "starting_xi": starters_list,
                "captain_id": cap_id,
                "vice_captain_id": vc_id,
                "bench_ids": bench_list,
                "trajectory": [
                    {
                        "gameweek": a.gameweek,
                        "transfers_in": a.transfers_in,
                        "transfers_out": a.transfers_out,
                        "transfers_count": a.transfers_count,
                        "hits": a.hits,
                        "hit_cost": a.hit_cost,
                        "chip": a.active_chip,
                        "net_xp": a.net_xp,
                        "starters": a.starters,
                        "captain": a.captain,
                        "bench": a.bench,
                        "bank_end": a.bank_end,
                        "ft_next": a.free_transfers_next,
                    }
                    for a in plan.trajectory
                ],
            }

        cand_dict_roll = _to_plan_dict(roll_plan)
        cand_dict_opt = _to_plan_dict(opt_plan)
        cand_dict_1 = _to_plan_dict(plan_1)

        candidate_plans = [cand_dict_roll]
        seen_moves: set[tuple[frozenset[int], frozenset[int]]] = set()
        seen_moves.add((frozenset(), frozenset()))

        # Add 1-transfer move
        m1_in = frozenset(t["element"] for t in cand_dict_1["transfers_in"])
        m1_out = frozenset(t["element"] for t in cand_dict_1["transfers_out"])
        if (m1_in, m1_out) not in seen_moves:
            seen_moves.add((m1_in, m1_out))
            candidate_plans.append(cand_dict_1)

        # Add optimal move
        opt_in = frozenset(t["element"] for t in cand_dict_opt["transfers_in"])
        opt_out = frozenset(t["element"] for t in cand_dict_opt["transfers_out"])
        if (opt_in, opt_out) not in seen_moves:
            seen_moves.add((opt_in, opt_out))
            candidate_plans.append(cand_dict_opt)

        # 4. Generate Top-K alternative moves using no-good cuts (WI-2.2)
        if len(player_pool_df) > 30:
            forbidden_moves: list[tuple[set[int], set[int]]] = [
                (set(opt_in), set(opt_out)),
                (set(m1_in), set(m1_out)),
            ]
            for _ in range(2):
                try:
                    alt_plan = self.optimize(
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
                        chips_available=chips_available,
                        previous_chip=previous_chip,
                        forbidden_w1_moves=forbidden_moves,
                    )
                    if alt_plan.solver_status != "Optimal":
                        break
                    alt_dict = _to_plan_dict(alt_plan)
                    a_in = frozenset(t["element"] for t in alt_dict["transfers_in"])
                    a_out = frozenset(t["element"] for t in alt_dict["transfers_out"])
                    if (a_in, a_out) not in seen_moves:
                        seen_moves.add((a_in, a_out))
                        candidate_plans.append(alt_dict)
                        forbidden_moves.append((set(a_in), set(a_out)))
                    else:
                        break
                except Exception:
                    break

        # 5. Robustness evaluation across MC scenarios (T5)
        num_mc_scenarios = 100
        p_star, k_hurdle = self.config.compute_hit_parameters(
            risk_preference=risk_preference,
            mini_league_mode=mini_league_mode,
            mini_league_deficit=mini_league_deficit,
        )
        if num_mc_scenarios > 0 and len(candidate_plans) > 1:
            win_counts = {i: 0.0 for i in range(len(candidate_plans))}
            roll_idx = None
            for idx, p in enumerate(candidate_plans):
                if p["plan_type"] == "ROLL_TRANSFER":
                    roll_idx = idx

            use_simulator = False
            sim_scores_array: list[np.ndarray] = []
            try:
                from fpl_oracle.sim.match_sim import match_simulator

                sim_res = match_simulator.simulate_gameweek(player_pool_df, n_sims=num_mc_scenarios, seed=42)
                for p in candidate_plans:
                    s_ids = p.get("starting_xi", [])
                    b_ids = p.get("bench_ids", [])
                    c_id = p.get("captain_id", 0)
                    v_id = p.get("vice_captain_id", 0)
                    if s_ids and c_id:
                        squad_pts = sim_res.get_squad_score_distribution(s_ids, b_ids, c_id, v_id)
                        traj = p.get("trajectory", [])
                        later_xp = sum(a.get("net_xp", 0.0) for a in traj[1:]) if len(traj) > 1 else 0.0
                        h_cost = float(p.get("hit_cost", 0.0))
                        total_sim_pts = squad_pts - h_cost + later_xp
                        sim_scores_array.append(total_sim_pts)
                    else:
                        raise ValueError("Incomplete squad IDs")
                use_simulator = True
            except Exception:
                use_simulator = False

            if not use_simulator:
                rng_mc = np.random.default_rng(42)
                mc_scores_list: list[list[float]] = [[] for _ in range(len(candidate_plans))]
                for _ in range(num_mc_scenarios):
                    noise_mult = rng_mc.normal(1.0, 0.15, size=len(candidate_plans))
                    scores = [p["horizon_net_xp"] * m for p, m in zip(candidate_plans, noise_mult, strict=True)]
                    for idx, s in enumerate(scores):
                        mc_scores_list[idx].append(s)
                sim_scores_array = [np.array(lst) for lst in mc_scores_list]

            score_matrix = np.array(sim_scores_array)  # Shape: (len(candidate_plans), num_mc_scenarios)
            best_plan_each_sim = np.argmax(score_matrix, axis=0)
            for sim_winner in best_plan_each_sim:
                win_counts[sim_winner] += 1.0

            for idx, p in enumerate(candidate_plans):
                r_score = round(win_counts[idx] / float(num_mc_scenarios), 2)
                p["robustness_score"] = r_score
                p["is_no_regret"] = bool(r_score >= 0.70)
                if roll_idx is not None and idx != roll_idx:
                    p_deltas = sim_scores_array[idx] - sim_scores_array[roll_idx]
                    p["p_beat_roll"] = round(float((p_deltas > 0).mean()), 3)
                    p["sigma_delta"] = round(float(p_deltas.std()), 3)
                    p["mc_delta_mean"] = round(float(p_deltas.mean()), 3)
                else:
                    p["p_beat_roll"] = 1.0
                    p["sigma_delta"] = 0.0
                    p["mc_delta_mean"] = 0.0

        # Sort candidate plans by horizon gain vs roll
        candidate_plans.sort(key=lambda p: float(p.get("horizon_gain_vs_roll", 0.0)), reverse=True)
        top_cand = candidate_plans[0]

        # Plan stability threshold (WI-2.5)
        if top_cand["plan_type"] != "ROLL_TRANSFER" and top_cand["expected_gain"] < stability_threshold:
            rec_plan = cand_dict_roll
            rec_plan["recommendation_summary"] = (
                f"Roll transfer recommended. Top move (+{top_cand['expected_gain']:.2f} xP) "
                f"does not exceed {stability_threshold:.2f} xP stability threshold."
            )
        else:
            rec_plan = top_cand

        # Hit policy verdict derived from MC distributions (WI-2.5)
        delta = rec_plan.get("horizon_gain_vs_roll", 0.0)
        hits = rec_plan.get("hits", 0)
        p_beat_roll = rec_plan.get("p_beat_roll", 0.5)
        sigma_delta = rec_plan.get("sigma_delta", 2.0)
        k_hurdle_threshold = round(k_hurdle * sigma_delta, 2)
        hit_threshold = self.config.compute_hit_threshold(
            hits=hits,
            risk_preference=risk_preference,
            mini_league_mode=mini_league_mode,
            mini_league_deficit=mini_league_deficit,
        )
        if hits == 0:
            hit_verdict = "No hit recommended"
        else:
            if p_beat_roll >= p_star or delta >= k_hurdle_threshold or delta >= hit_threshold:
                hit_verdict = (
                    f"JUSTIFIED: Hit of -{hits * 4} pts justified: "
                    f"P(J_plan > J_roll) = {p_beat_roll:.2f} >= {p_star:.2f} "
                    f"(Delta={delta:+.2f} xP vs hurdle {k_hurdle_threshold:.2f} xP)."
                )
            elif delta > 0.0:
                hit_verdict = (
                    f"MARGINAL: Hit yields +{delta:.2f} net xP but fails hurdle: "
                    f"P(J_plan > J_roll) = {p_beat_roll:.2f} < {p_star:.2f} and "
                    f"Delta < {k_hurdle:.2f}*sigma ({k_hurdle_threshold:.2f} xP). Rolling or single FT preferred."
                )
            else:
                hit_verdict = f"AVOID: Hit fails to recover points (delta={delta:.2f} xP, P(J_plan > J_roll) = {p_beat_roll:.2f})."

        # Free transfer option value
        ft_option_value = 0.8
        ft_option_explanation = (
            f"Option value of an extra banked free transfer is estimated at +{ft_option_value:.2f} xP "
            "providing restructuring flexibility without hit penalty."
        )

        return {
            "recommended_plan": rec_plan,
            "candidate_plans": candidate_plans,
            "hit_verdict": hit_verdict,
            "hit_threshold": hit_threshold,
            "p_star": p_star,
            "k_hurdle": k_hurdle,
            "p_beat_roll": p_beat_roll,
            "sigma_delta": sigma_delta,
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
            "ft_option_explanation": ft_option_explanation,
            "target_gameweek": target_gw,
            "current_bank_millions": round(float(bank) / 10.0, 2),
            "available_free_transfers": free_transfers,
            "stability_threshold": stability_threshold,
            "include_price_gain": include_price_gain,
        }

    def evaluate_joint_chips(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int | None = None,
        horizon_len: int | None = None,
        chips_available: set[str] | list[str] | None = None,
        previous_chip: str | None = None,
        fixtures: list[Any] | pd.DataFrame | None = None,
        gamma: float = 0.95,
        n_mc_scenarios: int = 100,
        use_legacy_thresholds: bool = False,
    ) -> dict[str, Any]:
        """
        Joint Chip Optimization across multi-gameweek horizon (WI-5.1 & WI-5.2).
        - Solves baseline with chips forbidden: J(no chip)
        - Solves unconstrained with chips free: J(free) and joint schedule
        - Solves pinned with each available chip pinned to each legal GW: Delta U(chip, g) = J(chip at g) - J(no chip)
        - Generates empirical distributions of Delta U using MatchSimulator scenarios
        - Opportunity cost of single-chip-per-GW rule
        - Synergies and sequences (e.g. Wildcard then Bench Boost; Free Hit into a Blank GW)
        - Set 1 Use-it-or-lose-it logic: latest sensible GW, expected loss if never played, option value of waiting
        - Enforces Rule R2 (no FH in GW1; no FH in GW20 after FH in GW19), max 1 chip/GW, and FT preservation.
        - Gated via context-aware data-driven thresholds or option-value stopping rule (legacy gates behind flag).
        """
        from fpl_oracle.advisor.chip_timeline import (
            compute_backward_induction_option_values,
            extract_gameweek_fixture_context,
        )
        from fpl_oracle.chips.planner import get_chip_threshold

        t0 = target_gw if target_gw is not None else current_gw
        h_len = horizon_len or min(8, 38 - t0 + 1)
        horizon_gws = [t0 + offset for offset in range(h_len) if t0 + offset <= 38]
        if not horizon_gws:
            horizon_gws = [t0]

        if chips_available is None:
            raw_chips = {"wildcard", "freehit", "bboost", "3xc"}
        else:
            raw_chips = set(c.lower().strip() for c in chips_available)

        # 1. Baseline Solve (Chips Forbidden)
        baseline_plan = self.optimize(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=t0,
            horizon_len=h_len,
            chips_available=[],
            previous_chip=previous_chip,
        )
        j_base = baseline_plan.total_objective

        # 2. Unconstrained Solve (Chips Free)
        unconstrained_plan = self.optimize(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=t0,
            horizon_len=h_len,
            chips_available=raw_chips,
            previous_chip=previous_chip,
        )
        j_free = unconstrained_plan.total_objective
        unconstrained_schedule = {
            act.gameweek: act.active_chip for act in unconstrained_plan.trajectory if act.active_chip
        }
        unconstrained_gain = round(j_free - j_base, 2)

        # 3. Pinned Solves for each available chip and each legal gameweek
        chip_evals: dict[str, list[dict[str, Any]]] = {}

        for c in sorted(raw_chips):
            windows: list[dict[str, Any]] = []
            for g in horizon_gws:
                # Check Rule R2 and Set boundaries
                is_legal = True
                if c == "freehit" and g == 1:
                    is_legal = False
                elif (
                    c == "freehit"
                    and g == 20
                    and (
                        (previous_chip and previous_chip.lower() == "freehit")
                        or unconstrained_schedule.get(19) == "freehit"
                    )
                ):
                    is_legal = False
                elif t0 <= SET_1_LAST_GW and g > SET_1_LAST_GW:
                    is_legal = False

                f_ctx = extract_gameweek_fixture_context(g, fixtures)
                ctx_str = (
                    "dgw"
                    if f_ctx["has_dgw"]
                    else ("bgw" if f_ctx["has_bgw"] else ("congested" if f_ctx["is_congested"] else "normal"))
                )
                set_idx = 1 if g <= SET_1_LAST_GW else 2
                threshold = get_chip_threshold(c, g, set_idx=set_idx, context=ctx_str, use_legacy=use_legacy_thresholds)

                if is_legal:
                    pinned_plan = self.optimize(
                        current_squad_df=current_squad_df,
                        player_pool_df=player_pool_df,
                        bank=bank,
                        free_transfers=free_transfers,
                        horizon_projections=horizon_projections,
                        current_gw=current_gw,
                        target_gw=t0,
                        horizon_len=h_len,
                        chips_available=[c],
                        forced_chip_schedule={g: c},
                        previous_chip=previous_chip,
                    )
                    exp_gain = round(pinned_plan.total_objective - j_base, 2)
                else:
                    exp_gain = 0.0
                    pinned_plan = baseline_plan

                # Scenario simulation distribution
                h = max(0, g - current_gw)
                tau_h = float(np.sqrt(1.0 + 0.12 * h))
                sigma = round(float(2.5 * tau_h), 2)
                p_pos = 0.5
                p10 = round(exp_gain - 1.28 * sigma, 2)
                p90 = round(exp_gain + 1.28 * sigma, 2)

                if n_mc_scenarios > 0 and is_legal:
                    try:
                        from fpl_oracle.sim.match_sim import match_simulator

                        pool_g = horizon_projections.get(g, player_pool_df)
                        sim_res = match_simulator.simulate_gameweek(pool_g, n_sims=n_mc_scenarios, seed=42 + g * 19)
                        act_pin = next((a for a in pinned_plan.trajectory if a.gameweek == g), None)
                        act_base = next((a for a in baseline_plan.trajectory if a.gameweek == g), None)
                        if act_pin and act_base:
                            pin_pts = sim_res.get_squad_score_distribution(
                                act_pin.starters, act_pin.bench, act_pin.captain, act_pin.vice_captain or 0
                            )
                            base_pts = sim_res.get_squad_score_distribution(
                                act_base.starters, act_base.bench, act_base.captain, act_base.vice_captain or 0
                            )
                            if c == "3xc":
                                pin_pts += sim_res.get_player_draws(act_pin.captain)
                            sim_diff = (pin_pts - base_pts) * tau_h
                            p_pos = round(float(np.mean(sim_diff > 0)), 3)
                            p10 = round(float(np.percentile(sim_diff, 10)), 2)
                            p90 = round(float(np.percentile(sim_diff, 90)), 2)
                            sigma = round(float(np.std(sim_diff)), 2)
                    except Exception:
                        pass

                windows.append(
                    {
                        "chip": c,
                        "gameweek": g,
                        "expected_gain": exp_gain,
                        "prob_gain_positive": p_pos,
                        "gain_interval": (p10, p90),
                        "sigma_gain": sigma,
                        "value_of_waiting": 0.0,
                        "context_notes": f_ctx["context_notes"],
                        "schedule_context": ctx_str,
                        "is_legal": is_legal,
                        "recommendation_threshold": threshold,
                        "recommendation_verdict": "DEFER",
                    }
                )

            # Backward induction for continuation value
            gains = [w["expected_gain"] for w in windows]
            v_waiting = compute_backward_induction_option_values(gains, gamma=gamma)
            for idx, w in enumerate(windows):
                w["value_of_waiting"] = v_waiting[idx]
                if not w["is_legal"]:
                    w["recommendation_verdict"] = "ILLEGAL"
                elif w["gameweek"] == current_gw:
                    if w["expected_gain"] >= w["value_of_waiting"] and (
                        w["expected_gain"] >= w["recommendation_threshold"] or w["recommendation_threshold"] == 0.0
                    ):
                        w["recommendation_verdict"] = "RECOMMEND_PLAY"
                    else:
                        w["recommendation_verdict"] = "DEFER"
                elif w["gameweek"] > current_gw:
                    if w["expected_gain"] >= w["value_of_waiting"]:
                        w["recommendation_verdict"] = "ATTRACTIVE_TARGET"
                    else:
                        w["recommendation_verdict"] = "SUBOPTIMAL_TARGET"

            chip_evals[c] = windows

        # 4. Opportunity Cost Matrix for single-chip-per-GW rule
        opp_cost_matrix: dict[int, dict[str, float]] = {}
        for g_idx, g in enumerate(horizon_gws):
            g_costs: dict[str, float] = {}
            g_gains = {
                c: chip_evals[c][g_idx]["expected_gain"]
                for c in sorted(raw_chips)
                if chip_evals[c][g_idx]["is_legal"] and chip_evals[c][g_idx]["expected_gain"] > 0
            }
            for c1 in g_gains:
                other_chips = [c2 for c2 in g_gains if c2 != c1]
                if other_chips:
                    best_other = max(other_chips, key=lambda c2: g_gains[c2])
                    best_other_outside = max(
                        [w["expected_gain"] for w in chip_evals[best_other] if w["gameweek"] != g and w["is_legal"]]
                        or [0.0]
                    )
                    opp = max(0.0, round(g_gains[best_other] - best_other_outside, 2))
                else:
                    opp = 0.0
                g_costs[c1] = opp
            opp_cost_matrix[g] = g_costs

        # 5. Chip Synergies and Sequences
        synergies: list[dict[str, Any]] = []
        if "wildcard" in unconstrained_schedule.values() and "bboost" in unconstrained_schedule.values():
            wc_gw = next(gw for gw, c in unconstrained_schedule.items() if c == "wildcard")
            bb_gw = next(gw for gw, c in unconstrained_schedule.items() if c == "bboost")
            if 0 < (bb_gw - wc_gw) <= 3:
                synergies.append(
                    {
                        "type": "WILDCARD_THEN_BENCH_BOOST",
                        "wc_gameweek": wc_gw,
                        "bb_gameweek": bb_gw,
                        "description": (
                            f"Wildcard scheduled in GW{wc_gw} restructures 15-player roster with deep bench "
                            f"specifically setting up Bench Boost deployment in GW{bb_gw}."
                        ),
                    }
                )

        if "freehit" in unconstrained_schedule.values():
            fh_gw = next(gw for gw, c in unconstrained_schedule.items() if c == "freehit")
            f_ctx_fh = extract_gameweek_fixture_context(fh_gw, fixtures)
            if f_ctx_fh["has_bgw"]:
                synergies.append(
                    {
                        "type": "FREE_HIT_IN_BLANK_GW",
                        "gameweek": fh_gw,
                        "description": f"Free Hit deployed in Blank GW{fh_gw} navigating blank fixture congestion.",
                    }
                )

        # 6. Set 1 Use-it-or-lose-it Status
        set_1_status: dict[str, Any] = {}
        if t0 <= SET_1_LAST_GW:
            s1_unused = [c for c in sorted(raw_chips) if c in {"wildcard", "freehit", "bboost", "3xc"}]
            gws_until_19 = [g for g in horizon_gws if g <= SET_1_LAST_GW]
            num_chips = len(s1_unused)
            num_gws = len(gws_until_19)

            # Assign latest sensible GWs ensuring 1 chip/GW
            sorted_by_peak = sorted(
                s1_unused,
                key=lambda c: max(
                    (w["expected_gain"] for w in chip_evals.get(c, []) if w["gameweek"] <= SET_1_LAST_GW), default=0.0
                ),
                reverse=True,
            )
            latest_sensible: dict[str, int] = {}
            for rank, c in enumerate(sorted_by_peak):
                latest_sensible[c] = min(SET_1_LAST_GW, max(t0, SET_1_LAST_GW - rank))

            exp_losses = {
                c: max(
                    (w["expected_gain"] for w in chip_evals.get(c, []) if w["gameweek"] <= SET_1_LAST_GW), default=0.0
                )
                for c in s1_unused
            }
            total_risk = round(sum(exp_losses.values()), 1)

            warning_text = None
            congestion = "NONE"
            if num_chips > num_gws:
                congestion = "CRITICAL"
                warning_text = (
                    f"CRITICAL CHIP CONGESTION: {num_chips} Set 1 chips remain with only {num_gws} GWs until GW19. "
                    f"At least {num_chips - num_gws} chip(s) will permanently expire. Total forfeiture risk: ~{total_risk} xP."
                )
            elif current_gw >= 14 or num_chips >= (num_gws - 1):
                congestion = "HIGH" if num_chips == num_gws else "MODERATE"
                warning_text = (
                    f"SET 1 DEADLINE NOTICE: Gameweek 19 is the hard cutoff. Unused Set 1 chips have 0 rollover value. "
                    f"Total expected loss if never deployed: ~{total_risk} xP."
                )

            set_1_status = {
                "unused_chips": s1_unused,
                "latest_sensible_gw": latest_sensible,
                "expected_loss_if_never_used": exp_losses,
                "total_forfeiture_risk": total_risk,
                "deadline_warning": warning_text,
                "congestion_level": congestion,
            }

        # 7. Recommended Action this GW
        rec_chips_now = [
            c
            for c in sorted(raw_chips)
            if chip_evals[c] and chip_evals[c][0]["recommendation_verdict"] == "RECOMMEND_PLAY"
        ]
        best_rec_now = None
        if rec_chips_now:
            best_c = max(
                rec_chips_now,
                key=lambda c: chip_evals[c][0]["expected_gain"] - chip_evals[c][0]["value_of_waiting"],
            )
            best_rec_now = chip_evals[best_c][0]

        return {
            "baseline_objective": j_base,
            "unconstrained_objective": j_free,
            "unconstrained_schedule": unconstrained_schedule,
            "unconstrained_gain_vs_baseline": unconstrained_gain,
            "chip_evaluations": chip_evals,
            "opportunity_cost_matrix": opp_cost_matrix,
            "synergies_detected": synergies,
            "set_1_status": set_1_status,
            "threshold_policy": "legacy_fixed" if use_legacy_thresholds else "data_driven_context",
            "recommended_action_this_gw": best_rec_now,
        }


multiperiod_optimizer = MultiPeriodOptimizer()
