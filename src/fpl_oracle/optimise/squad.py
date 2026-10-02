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
        budget: float, # in tenths, e.g. 1000 = £100.0m
        metric_col: str = "expected_points",
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        bench_weight: float = 0.05,
        require_talisman: bool = True
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

        use_joint = (metric_col == "expected_points")

        # Decision variables
        x = {idx: pulp.LpVariable(f"sq_{idx}", cat=pulp.LpBinary) for idx in df.index}

        if use_joint:
            s = {idx: pulp.LpVariable(f"st_{idx}", cat=pulp.LpBinary) for idx in df.index}
            c = {idx: pulp.LpVariable(f"cp_{idx}", cat=pulp.LpBinary) for idx in df.index}

            # Hierarchy: Captain <= Starter <= Squad
            for idx in df.index:
                prob += s[idx] <= x[idx]
                prob += c[idx] <= s[idx]

            # Objective: Starters (1.0) + Captain multiplier (1.0 + ceiling bonus) + Bench (0.05)
            cap_scores = df[metric_col] * 1.0 + df.get("p90", df[metric_col] * 1.5) * 0.35
            prob += pulp.lpSum([
                df.loc[idx, metric_col] * s[idx] +
                cap_scores.loc[idx] * c[idx] +
                df.loc[idx, metric_col] * bench_weight * (x[idx] - s[idx])
                for idx in df.index
            ])

            # Lineup constraints
            prob += pulp.lpSum([s[idx] for idx in df.index]) == 11
            prob += pulp.lpSum([c[idx] for idx in df.index]) == 1
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "GKP"]) == 1
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "DEF"]) >= 3
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "DEF"]) <= 5
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "MID"]) >= 2
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "MID"]) <= 5
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "FWD"]) >= 1
            prob += pulp.lpSum([s[idx] for idx in df.index if df.loc[idx, "position"] == "FWD"]) <= 3

            # Talisman Anchor constraint: If pool has ultra-premiums (>= 115) and budget allows,
            # guarantee at least 1 talisman anchor in squad
            if require_talisman and budget >= 950.0:
                prem_indices = [idx for idx in df.index if float(df.loc[idx, "value"]) >= 115.0]
                if prem_indices:
                    prob += pulp.lpSum([x[idx] for idx in prem_indices]) >= 1
        else:
            # Unweighted sum for naive baselines
            prob += pulp.lpSum([df.loc[idx, metric_col] * x[idx] for idx in df.index])

        # Constraint 1: Exactly 15 squad members
        prob += pulp.lpSum([x[idx] for idx in df.index]) == 15

        # Constraint 2: Squad Positional constraints
        prob += pulp.lpSum([x[idx] for idx in df.index if df.loc[idx, "position"] == "GKP"]) == 2
        prob += pulp.lpSum([x[idx] for idx in df.index if df.loc[idx, "position"] == "DEF"]) == 5
        prob += pulp.lpSum([x[idx] for idx in df.index if df.loc[idx, "position"] == "MID"]) == 5
        prob += pulp.lpSum([x[idx] for idx in df.index if df.loc[idx, "position"] == "FWD"]) == 3

        # Constraint 3: Max 3 players per Premier League club
        teams = df["team"].unique()
        for t in teams:
            team_indices = [idx for idx in df.index if df.loc[idx, "team"] == t]
            prob += pulp.lpSum([x[idx] for idx in team_indices]) <= 3

        # Constraint 4: Budget limit
        prob += pulp.lpSum([df.loc[idx, "value"] * x[idx] for idx in df.index]) <= budget

        # Constraint 5: Locked-in / Locked-out
        for idx, row in df.iterrows():
            elem_id = int(row["element"])
            if elem_id in locked_in:
                prob += x[idx] == 1
            if elem_id in locked_out or row["team"] in excluded_teams:
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
            "squad": selected_df
        }

        if use_joint:
            starters_indices = [idx for idx in df.index if s[idx].varValue > 0.5]
            cap_indices = [idx for idx in df.index if c[idx].varValue > 0.5]
            result["starters"] = df.loc[starters_indices].copy()
            if cap_indices:
                result["captain_element"] = int(df.loc[cap_indices[0], "element"])

        return result

squad_optimizer = SquadOptimizer()
