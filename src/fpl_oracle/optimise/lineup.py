"""
Lineup, Captaincy, and Formation Optimizer.
Solves the optimal starting XI, bench ordering, and captain/vice-captain
subject to valid formation rules, autosub dynamics, and P10/P50/P90 risk distributions.
"""

import math
from typing import Any

import pandas as pd
import pulp


def normal_prob_greater(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Calculates P(X1 > X2) assuming independent normal distributions using math.erf."""
    denom = math.sqrt(max(0.1, sigma1**2 + sigma2**2))
    z = (mu1 - mu2) / denom
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return round(float(prob), 3)


class LineupOptimizer:
    def __init__(self):
        pass

    def select_lineup_and_captain(
        self,
        squad_df: pd.DataFrame,
        is_triple_captain: bool = False,
        is_bench_boost: bool = False,
        risk_preference: str = "balanced",
    ) -> dict[str, Any]:
        """
        Given a 15-player squad, selects optimal starting XI, captain, vice-captain, and bench order.
        Considers P10 floor and P90 ceiling distributions for risk-aware captaincy recommendations.
        """
        if len(squad_df) != 15:
            raise ValueError(f"Squad must have exactly 15 players, got {len(squad_df)}")

        df = squad_df.copy().reset_index(drop=True)

        prob = pulp.LpProblem("FPL_Lineup_Optimizer", pulp.LpMaximize)

        # Decision variables
        starter = {i: pulp.LpVariable(f"starter_{i}", cat=pulp.LpBinary) for i in df.index}
        captain = {i: pulp.LpVariable(f"captain_{i}", cat=pulp.LpBinary) for i in df.index}

        # Captain multiplier: 2x normally, 3x for Triple Captain
        cap_multiplier = 3.0 if is_triple_captain else 2.0

        # Captain score based on risk preference:
        p90_vals = df["p90"] if "p90" in df.columns else df["expected_points"] * 1.5
        p10_vals = df["p10"] if "p10" in df.columns else df["expected_points"] * 0.4

        if risk_preference == "conservative":
            cap_scores = df["expected_points"] * 0.60 + p10_vals * 0.40
        elif risk_preference == "aggressive":
            cap_scores = df["expected_points"] * 0.40 + p90_vals * 0.60
        else:  # balanced
            cap_scores = df["expected_points"] * 0.70 + p90_vals * 0.30

        prob += pulp.lpSum(
            [
                df.loc[i, "expected_points"] * starter[i] + cap_scores.loc[i] * (cap_multiplier - 1.0) * captain[i]
                for i in df.index
            ]
        )

        # Constraint 1: Exactly 11 starters
        prob += pulp.lpSum([starter[i] for i in df.index]) == 11

        # Constraint 2: Exactly 1 goalkeeper starts
        prob += pulp.lpSum([starter[i] for i in df.index if df.loc[i, "position"] == "GKP"]) == 1

        # Constraint 3: Valid outfield formation
        # 3 to 5 Defenders
        def_starters = pulp.lpSum([starter[i] for i in df.index if df.loc[i, "position"] == "DEF"])
        prob += def_starters >= 3
        prob += def_starters <= 5

        # 2 to 5 Midfielders
        mid_starters = pulp.lpSum([starter[i] for i in df.index if df.loc[i, "position"] == "MID"])
        prob += mid_starters >= 2
        prob += mid_starters <= 5

        # 1 to 3 Forwards
        fwd_starters = pulp.lpSum([starter[i] for i in df.index if df.loc[i, "position"] == "FWD"])
        prob += fwd_starters >= 1
        prob += fwd_starters <= 3

        # Constraint 4: Exactly 1 captain, and captain must be a starter
        prob += pulp.lpSum([captain[i] for i in df.index]) == 1
        for i in df.index:
            prob += captain[i] <= starter[i]

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError("Could not find optimal starting lineup.")

        # Identify starters vs benched
        starter_indices = [i for i in df.index if starter[i].varValue > 0.5]
        bench_indices = [i for i in df.index if starter[i].varValue <= 0.5]

        starters_df = df.loc[starter_indices].copy()
        bench_df = df.loc[bench_indices].copy()

        # Identify captain
        captain_idx = [i for i in starter_indices if captain[i].varValue > 0.5][0]
        captain_row = df.loc[captain_idx]

        # Select vice-captain (highest expected points starter other than captain)
        other_starters = starters_df[starters_df["element"] != captain_row["element"]].sort_values(
            by="expected_points", ascending=False
        )
        vice_captain_row = other_starters.iloc[0]

        # Order bench:
        # Bench GK is first bench spot or reserved for GK slot
        bench_gk = bench_df[bench_df["position"] == "GKP"]
        bench_outfield = bench_df[bench_df["position"] != "GKP"].sort_values(by="expected_points", ascending=False)
        ordered_bench = pd.concat([bench_gk, bench_outfield]).reset_index(drop=True)

        # Compute formation string
        n_def = len(starters_df[starters_df["position"] == "DEF"])
        n_mid = len(starters_df[starters_df["position"] == "MID"])
        n_fwd = len(starters_df[starters_df["position"] == "FWD"])
        formation = f"{n_def}-{n_mid}-{n_fwd}"

        # Ensure p10 and p90 columns exist
        if "p10" not in starters_df.columns:
            starters_df["p10"] = starters_df["expected_points"] * 0.4
        if "p90" not in starters_df.columns:
            starters_df["p90"] = starters_df["expected_points"] * 1.8

        captain_candidates = starters_df.sort_values(by="expected_points", ascending=False).head(5).copy()

        # Compute safe pick (highest floor P10) vs ceiling pick (highest P90)
        safe_pick = starters_df.sort_values(by=["p10", "expected_points"], ascending=False).iloc[0]
        ceiling_pick = starters_df.sort_values(by=["p90", "expected_points"], ascending=False).iloc[0]

        # Probability of outscoring next best option
        cap_mu = float(captain_row["expected_points"])
        cap_p10 = float(captain_row.get("p10", cap_mu * 0.4))
        cap_p90 = float(captain_row.get("p90", cap_mu * 1.8))
        cap_sigma = max(1.0, (cap_p90 - cap_p10) / 2.56)

        next_mu = float(vice_captain_row["expected_points"])
        next_p10 = float(vice_captain_row.get("p10", next_mu * 0.4))
        next_p90 = float(vice_captain_row.get("p90", next_mu * 1.8))
        next_sigma = max(1.0, (next_p90 - next_p10) / 2.56)

        prob_outscore = normal_prob_greater(cap_mu, cap_sigma, next_mu, next_sigma)

        # Starting points and bench points
        starters_xp = float(starters_df["expected_points"].sum())
        cap_bonus_xp = float(captain_row["expected_points"] * (cap_multiplier - 1.0))
        bench_xp = float(bench_df["expected_points"].sum())

        total_lineup_xp = starters_xp + cap_bonus_xp
        if is_bench_boost:
            total_lineup_xp += bench_xp

        # Bench risk assessment
        if "chance_of_playing" in starters_df.columns:
            bench_risk_starters = starters_df[starters_df["chance_of_playing"] < 75.0]
        else:
            bench_risk_starters = starters_df.iloc[0:0]

        return {
            "formation": formation,
            "starters": starters_df,
            "bench": ordered_bench,
            "captain": {
                "element": int(captain_row["element"]),
                "web_name": captain_row["web_name"],
                "expected_points": round(float(captain_row["expected_points"]), 2),
                "p10": round(float(captain_row.get("p10", 0.0)), 2),
                "p90": round(float(captain_row.get("p90", 0.0)), 2),
                "multiplier": int(cap_multiplier),
                "prob_outscore_next": prob_outscore,
                "safe_alternative": {
                    "element": int(safe_pick["element"]),
                    "web_name": safe_pick["web_name"],
                    "floor_p10": round(float(safe_pick.get("p10", 0.0)), 2),
                    "expected_points": round(float(safe_pick["expected_points"]), 2),
                },
                "differential_alternative": {
                    "element": int(ceiling_pick["element"]),
                    "web_name": ceiling_pick["web_name"],
                    "ceiling_p90": round(float(ceiling_pick.get("p90", 0.0)), 2),
                    "expected_points": round(float(ceiling_pick["expected_points"]), 2),
                },
            },
            "vice_captain": {
                "element": int(vice_captain_row["element"]),
                "web_name": vice_captain_row["web_name"],
                "expected_points": round(float(vice_captain_row["expected_points"]), 2),
                "reasoning": f"Activates if {captain_row['web_name']} does not feature. Ranked #2 in expected points ({round(float(vice_captain_row['expected_points']), 2)} xP).",
            },
            "captain_rankings": captain_candidates[
                [
                    c
                    for c in ["element", "web_name", "position", "expected_points", "p10", "p90"]
                    if c in captain_candidates.columns
                ]
            ].to_dict(orient="records"),
            "starters_expected_points": round(starters_xp, 2),
            "bench_expected_points": round(bench_xp, 2),
            "total_gameweek_expected_points": round(total_lineup_xp, 2),
            "bench_risk_starters": bench_risk_starters[
                [c for c in ["element", "web_name", "chance_of_playing"] if c in bench_risk_starters.columns]
            ].to_dict(orient="records"),
        }


lineup_optimizer = LineupOptimizer()
