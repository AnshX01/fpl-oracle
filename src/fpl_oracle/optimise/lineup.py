"""
Lineup, Captaincy, and Formation Optimizer.
Solves the optimal starting XI, bench ordering, and captain/vice-captain
subject to valid formation rules, autosub dynamics, and P10/P50/P90 risk distributions.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import pulp

# Baseline chance any fit starter fails to play (rotation, late knock); keeps the bench ordering
# sensitive to formation legality even when nobody is flagged.
BASE_ABSENCE_RISK = 0.05
_STATUS_PLAY_PROB = {"i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0, "d": 0.5}


def play_probability(df: pd.DataFrame) -> pd.Series:
    """P(player plays) in [0, 1]. Uses chance_of_playing (else next-round); when that is missing the FPL
    status decides: i/s/u/n -> 0, d -> 0.5, anything else -> 1."""
    cop = pd.Series(np.nan, index=df.index)
    for col in ("chance_of_playing", "chance_of_playing_next_round"):
        if col in df.columns:
            cop = cop.fillna(pd.to_numeric(df[col], errors="coerce"))
    if "status" in df.columns:
        by_status = df["status"].astype(str).str.lower().map(_STATUS_PLAY_PROB).astype(float)
        cop = cop.fillna(by_status * 100.0)
    return (cop.fillna(100.0) / 100.0).clip(0.0, 1.0)


def _order_bench(bench_df: pd.DataFrame, starters_df: pd.DataFrame) -> pd.DataFrame:
    """Bench GKP first, then outfielders by expected autosub value: P(plays) x xP x sum over outfield
    starters of P(starter absent) for every swap that keeps the formation legal (xP breaks ties)."""
    bench_gk = bench_df[bench_df["position"] == "GKP"]
    outfield = bench_df[bench_df["position"] != "GKP"].copy()
    out_starters = starters_df[starters_df["position"] != "GKP"]
    absent = (1.0 - play_probability(out_starters)).clip(lower=BASE_ABSENCE_RISK)
    counts = out_starters["position"].value_counts().to_dict()
    mins = {"DEF": 3, "MID": 2, "FWD": 1}

    def legal(sub_pos: str, out_pos: str) -> bool:
        c = {k: counts.get(k, 0) for k in mins}
        c[out_pos] -= 1
        c[sub_pos] += 1
        return all(c[k] >= mins[k] for k in mins)

    play = play_probability(outfield)
    outfield["_sub_value"] = [
        float(play[i])
        * float(r["expected_points"])
        * sum(float(absent[j]) for j, s in out_starters["position"].items() if legal(r["position"], s))
        for i, r in outfield.iterrows()
    ]
    outfield = outfield.sort_values(by=["_sub_value", "expected_points"], ascending=False).drop(columns="_sub_value")
    return pd.concat([bench_gk, outfield]).reset_index(drop=True)


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
        excluded_ids: set[int] | None = None,
        simulate_captaincy: bool = True,
    ) -> dict[str, Any]:
        """
        Given a 15-player squad, selects optimal starting XI, captain, vice-captain, and bench order.
        The XI maximises expected points. Captain maximises expected points too ("balanced"/"points");
        "conservative"/"aggressive" are explicit floor/ceiling tilts for the captain only.
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

        # Captain score: expected points unless an explicit floor/ceiling mode is requested.
        p90_vals = df["p90"] if "p90" in df.columns else df["expected_points"] * 1.5
        p10_vals = df["p10"] if "p10" in df.columns else df["expected_points"] * 0.4

        if risk_preference == "conservative":
            cap_scores = df["expected_points"] * 0.60 + p10_vals * 0.40
        elif risk_preference == "aggressive":
            cap_scores = df["expected_points"] * 0.40 + p90_vals * 0.60
        else:  # balanced / points
            cap_scores = df["expected_points"]
        starter_scores = df["expected_points"]

        prob += pulp.lpSum(
            [
                starter_scores.loc[i] * starter[i] + cap_scores.loc[i] * (cap_multiplier - 1.0) * captain[i]
                for i in df.index
            ]
        )

        for i in df.index:
            if int(df.loc[i, "element"]) in (excluded_ids or set()) or bool(
                df.loc[i].get("simulation_unavailable", False)
            ):
                prob += starter[i] == 0

        if is_bench_boost:
            prob.setObjective(
                prob.objective + pulp.lpSum(df.loc[i, "expected_points"] * (1 - starter[i]) for i in df.index)
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

        # Vice-captain: best alternative if the captain does not play. Rank by xP x P(plays); a vice from the
        # captain's club shares the same kickoff, so it is a weaker hedge (5% haircut).
        others = starters_df[starters_df["element"] != captain_row["element"]].copy()
        others["_vice_score"] = others["expected_points"] * play_probability(others)
        if "team" in others.columns:
            others.loc[others["team"] == captain_row.get("team"), "_vice_score"] *= 0.95
        others = others.sort_values(by=["_vice_score", "expected_points"], ascending=False)
        vice_captain_row = others.iloc[0]
        vice_play = float(play_probability(others.iloc[:1]).iloc[0])

        ordered_bench = _order_bench(bench_df, starters_df)

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

        captain_candidates = (
            starters_df.assign(_cap_score=cap_scores.loc[starters_df.index])
            .sort_values(by=["_cap_score", "expected_points"], ascending=False)
            .head(5)
        )

        # Compute safe pick (highest floor P10) vs ceiling pick (highest P90)
        safe_pick = starters_df.sort_values(by=["p10", "expected_points"], ascending=False).iloc[0]
        ceiling_pick = starters_df.sort_values(by=["p90", "expected_points"], ascending=False).iloc[0]

        # Probability of outscoring next best option using scoreline simulator (WI-4.3) or analytical normal
        cap_id = int(captain_row["element"])
        vice_id = int(vice_captain_row["element"])
        sim_success = False
        if simulate_captaincy:
            try:
                from fpl_oracle.sim.match_sim import match_simulator

                sim_res = match_simulator.simulate_gameweek(starters_df, n_sims=1000, seed=42)
                prob_outscore = round(sim_res.get_captain_win_probability(cap_id, vice_id), 3)
                sim_success = True
            except Exception:
                sim_success = False

        if not sim_success:
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
        bench_risk_starters = starters_df[play_probability(starters_df) < 0.75]

        return {
            "formation": formation,
            "starters": starters_df,
            "bench": ordered_bench,
            "captain": {
                "element": int(captain_row["element"]),
                "web_name": captain_row["web_name"],
                "team": captain_row.get("team"),
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
                "team": vice_captain_row.get("team"),
                "expected_points": round(float(vice_captain_row["expected_points"]), 2),
                "p10": round(float(vice_captain_row.get("p10", 0.0)), 2),
                "p90": round(float(vice_captain_row.get("p90", 0.0)), 2),
                "reasoning": (
                    f"Activates if {captain_row['web_name']} does not feature. Best captain alternative: "
                    f"{round(float(vice_captain_row['expected_points']), 2)} xP, {round(vice_play * 100)}% to play."
                ),
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
