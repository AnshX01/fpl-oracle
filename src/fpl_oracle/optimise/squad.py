"""
Squad Integer Linear Programming (MILP) Solver using PuLP.
Solves exact 15-player squad optimization subject to real FPL rules:
- 2 GKP, 5 DEF, 5 MID, 3 FWD
- Maximum 3 players per Premier League club
- Total budget constraint
- Player locking and team exclusion
- Joint Starters/Bench weighting and Talisman Anchoring
"""

from typing import Any

import pandas as pd
import pulp


class SquadOptimizer:
    def __init__(self):
        pass

    def solve_best_squad(
        self,
        player_pool_df: pd.DataFrame,
        budget: float,  # in tenths, e.g. 1000 = £100.0m
        metric_col: str = "expected_points",
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        bench_weight: float = 0.05,
        captain_mean_only: bool = False,
        require_talisman: bool = True,
        repair_elements: set[int] | None = None,
        repair_sell_prices: dict[int, int] | None = None,
        repair_free_transfers: int = 0,
        repair_hit_penalty: float = 4.0,
        repair_max_transfers: int | None = None,
        repair_min_transfers: bool = False,
    ) -> dict[str, Any]:
        """
        Solve optimal 15-man squad within budget.
        When optimizing for expected_points, uses joint Starter/Captain/Bench weighting
        to ensure high-value captaincy talismans (e.g. Haaland/Salah/Bruno) and realistic
        starting XI output.
        """
        df = player_pool_df.copy().reset_index(drop=True)
        locked_in = set(locked_in_ids or [])
        locked_out = set(locked_out_ids or [])
        excluded_teams = set(excluded_team_ids or [])

        prob = pulp.LpProblem("FPL_Squad_Optimizer", pulp.LpMaximize)

        # Column lists once, instead of thousands of row lookups while building the model.
        pos_of = df["position"].tolist()
        team_of = df["team"].tolist()
        elem_of = df["element"].tolist()
        metric_of = df[metric_col].tolist() if metric_col in df else None
        value_of = df["value"].tolist() if "value" in df else None
        unavailable_of = (
            [bool(v) for v in df["simulation_unavailable"].tolist()] if "simulation_unavailable" in df else None
        )

        use_joint = metric_col == "expected_points"

        # Decision variables
        x = {idx: pulp.LpVariable(f"sq_{idx}", cat=pulp.LpBinary) for idx in df.index}

        if use_joint:
            s = {idx: pulp.LpVariable(f"st_{idx}", cat=pulp.LpBinary) for idx in df.index}
            c = {idx: pulp.LpVariable(f"cp_{idx}", cat=pulp.LpBinary) for idx in df.index}

            # Hierarchy: Captain <= Starter <= Squad
            for idx in df.index:
                prob += s[idx] <= x[idx]
                prob += c[idx] <= s[idx]
                if (unavailable_of[idx] if unavailable_of is not None else False):
                    if repair_elements is not None and int(elem_of[idx]) in repair_elements:
                        prob += s[idx] == 0
                    else:
                        prob += x[idx] == 0

            # Objective: Starters (1.0) + Captain multiplier (1.0 + ceiling bonus) + Bench (0.05)
            cap_scores = (
                df[metric_col] if captain_mean_only else df[metric_col] + df.get("p90", df[metric_col] * 1.5) * 0.35
            )
            cap_scores_of = cap_scores.tolist()
            prob += pulp.lpSum(
                [
                    metric_of[idx] * s[idx]
                    + cap_scores_of[idx] * c[idx]
                    + metric_of[idx] * bench_weight * (x[idx] - s[idx])
                    for idx in df.index
                ]
            )

            if repair_elements is not None:
                hits = pulp.LpVariable("repair_hits", lowBound=0, cat=pulp.LpInteger)
                prob += (
                    hits
                    >= pulp.lpSum(x[i] for i in df.index if int(elem_of[i]) not in repair_elements)
                    - repair_free_transfers
                )
                prob.objective -= repair_hit_penalty * hits
                transfer_count = pulp.lpSum(x[i] for i in df.index if int(elem_of[i]) not in repair_elements)
                if repair_max_transfers is not None:
                    prob += transfer_count <= repair_max_transfers
                if repair_min_transfers:
                    # Finite lexicographic dominance: one fewer transfer beats
                    # the entire feasible scoring range, then points break ties.
                    bound = 1 + 4 * float(df[metric_col].abs().sum())
                    prob.objective -= bound * transfer_count

            # Lineup constraints
            prob += pulp.lpSum([s[idx] for idx in df.index]) == 11
            prob += pulp.lpSum([c[idx] for idx in df.index]) == 1
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "GKP"]) == 1
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "DEF"]) >= 3
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "DEF"]) <= 5
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "MID"]) >= 2
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "MID"]) <= 5
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "FWD"]) >= 1
            prob += pulp.lpSum([s[idx] for idx in df.index if pos_of[idx] == "FWD"]) <= 3

            # Talisman Anchor constraint: If pool has ultra-premiums (>= 115) and budget allows,
            # guarantee at least 1 talisman anchor in squad
            if require_talisman and budget >= 950.0:
                cost_col = "value" if "value" in df.columns else "now_cost"
                prem_indices = [idx for idx in df.index if float(df.loc[idx, cost_col]) >= 115.0]
                if prem_indices:
                    prob += pulp.lpSum([x[idx] for idx in prem_indices]) >= 1
        else:
            # Unweighted sum for naive baselines
            prob += pulp.lpSum([df.loc[idx, metric_col] * x[idx] for idx in df.index])

        # Constraint 1: Exactly 15 squad members
        prob += pulp.lpSum([x[idx] for idx in df.index]) == 15

        # Constraint 2: Squad Positional constraints
        prob += pulp.lpSum([x[idx] for idx in df.index if pos_of[idx] == "GKP"]) == 2
        prob += pulp.lpSum([x[idx] for idx in df.index if pos_of[idx] == "DEF"]) == 5
        prob += pulp.lpSum([x[idx] for idx in df.index if pos_of[idx] == "MID"]) == 5
        prob += pulp.lpSum([x[idx] for idx in df.index if pos_of[idx] == "FWD"]) == 3

        # Constraint 3: Max 3 players per Premier League club
        teams = df["team"].unique()
        for t in teams:
            team_indices = [idx for idx in df.index if team_of[idx] == t]
            prob += pulp.lpSum([x[idx] for idx in team_indices]) <= 3

        # Constraint 4: Budget limit
        if repair_elements is None:
            prob += pulp.lpSum([value_of[idx] * x[idx] for idx in df.index]) <= budget
        else:
            # Retaining an owned player consumes its actual sell value, not its
            # market price. New players consume their purchase price.
            costs = repair_sell_prices or {}
            prob += (
                pulp.lpSum(
                    (
                        costs[int(elem_of[i])]
                        if int(elem_of[i]) in repair_elements
                        else value_of[i]
                    )
                    * x[i]
                    for i in df.index
                )
                <= budget
            )

        # Constraint 5: Locked-in / Locked-out
        for idx in df.index:
            elem_id = int(elem_of[idx])
            if elem_id in locked_in:
                prob += x[idx] == 1
            if elem_id in locked_out or team_of[idx] in excluded_teams:
                prob += x[idx] == 0

        # Solve with CBC
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"Optimizer did not find optimal solution: status={pulp.LpStatus[prob.status]}")

        # Extract selected players
        selected_indices = [idx for idx in df.index if x[idx].varValue > 0.5]
        selected_df = df.loc[selected_indices].copy()

        total_cost = float(selected_df["value"].sum())
        total_xp = float(selected_df[metric_col].sum())

        result = {
            "status": "Optimal",
            "total_cost": total_cost,
            "total_expected_points": round(total_xp, 2),
            "remaining_budget": round(budget - total_cost, 1),
            "squad": selected_df,
        }

        if use_joint:
            starters_indices = [idx for idx in df.index if s[idx].varValue > 0.5]
            cap_indices = [idx for idx in df.index if c[idx].varValue > 0.5]
            result["starters"] = df.loc[starters_indices].copy()
            if cap_indices:
                result["captain_element"] = int(df.loc[cap_indices[0], "element"])

        return result


squad_optimizer = SquadOptimizer()
