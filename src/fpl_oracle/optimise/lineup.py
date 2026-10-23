"""
Lineup, Captaincy, and Formation Optimizer.
Picks the optimal starting XI, bench ordering, and captain/vice-captain by exact enumeration of the legal
formations (ties by player id, so identical inputs give an identical lineup), then adds the P10/P90 risk read.
"""

import math
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd

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
    sub_values = []
    for i, r in outfield.iterrows():
        p_play = float(play[i])
        reliability_mult = 1.10 if p_play >= 0.99 else (0.80 if p_play <= 0.75 else 0.90)
        if "expected_minutes" in r and pd.notna(r["expected_minutes"]):
            exp_mins = float(r["expected_minutes"])
            if exp_mins >= 80.0:
                reliability_mult *= 1.05
            elif exp_mins < 60.0:
                reliability_mult *= max(0.1, exp_mins / 60.0)
        val = (
            p_play
            * float(r["expected_points"])
            * sum(float(absent[j]) for j, s in out_starters["position"].items() if legal(r["position"], s))
            * reliability_mult
        )
        sub_values.append(val)
    outfield["_sub_value"] = sub_values
    outfield = outfield.sort_values(by=["_sub_value", "expected_points", "element"], ascending=[False, False, True])
    outfield = outfield.drop(columns="_sub_value")
    return pd.concat([bench_gk, outfield]).reset_index(drop=True)


def normal_prob_greater(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Calculates P(X1 > X2) assuming independent normal distributions using math.erf."""
    denom = math.sqrt(max(0.1, sigma1**2 + sigma2**2))
    z = (mu1 - mu2) / denom
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return round(float(prob), 3)


def best_lineup(
    squad: list[int],
    position: dict[int, str],
    xp: dict[int, float],
    bench_weight: float,
    cap_score: Mapping[int, float] | None = None,
    cap_extra: float = 1.0,
    blocked: Collection[int] = (),
    starter_score: Mapping[int, float] | None = None,
) -> tuple[list[int], list[int], int, int | None]:
    """
    Exact (starters, bench, captain, vice) for a 15-man squad: best formation by
    (1 - bench_weight) * starters + cap_extra * captain score, ties by total starter xP and then by id.
    The captain is the best `cap_score` (default xP) among the starters; `blocked` players never start.
    The MILP's lineup variables are continuous (integral at a vertex, but a tie can leave a fractional
    split), so the reported lineup is rebuilt from the chosen squad instead of rounded from them.
    `starter_score` allows applying cameo risk / minutes certainty penalties when selecting the starting XI.
    """
    score = cap_score if cap_score is not None else xp
    st_score = starter_score if starter_score is not None else xp
    ranked = {
        pos: sorted((p for p in squad if position[p] == pos and p not in blocked), key=lambda p: (-st_score.get(p, 0.0), p))
        for pos in ("GKP", "DEF", "MID", "FWD")
    }

    def captain_of(starters: list[int]) -> int:
        return max(starters, key=lambda p: (score.get(p, 0.0), xp.get(p, 0.0), -p))

    best: tuple[tuple[float, float], list[int]] | None = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3 or not ranked["GKP"] or len(ranked["DEF"]) < d or len(ranked["MID"]) < m:
                continue
            if len(ranked["FWD"]) < f:
                continue
            starters = ranked["GKP"][:1] + ranked["DEF"][:d] + ranked["MID"][:m] + ranked["FWD"][:f]
            total = sum(st_score.get(p, 0.0) for p in starters)
            raw_total = sum(xp.get(p, 0.0) for p in starters)
            key = ((1.0 - bench_weight) * total + cap_extra * score.get(captain_of(starters), 0.0), raw_total)
            if best is None or key > best[0]:
                best = (key, starters)
    if best is None:
        raise RuntimeError("Could not find optimal starting lineup.")
    starters = best[1]
    captain = captain_of(starters)
    rest = [p for p in starters if p != captain]
    vice = max(rest, key=lambda p: (xp.get(p, 0.0), -p)) if rest else None
    bench = sorted((p for p in squad if p not in starters), key=lambda p: (-xp.get(p, 0.0), p))
    return starters, bench, captain, vice


def outscore_probability(captain: Mapping[str, Any], vice: Mapping[str, Any]) -> float:
    """P(captain outscores the vice-captain) from each one's xP and P10-P90 spread (normal approximation)."""

    def mu_sigma(row: Mapping[str, Any]) -> tuple[float, float]:
        mu = float(row["expected_points"])
        spread = float(row.get("p90", mu * 1.8)) - float(row.get("p10", mu * 0.4))
        return mu, max(1.0, spread / 2.56)

    return normal_prob_greater(*mu_sigma(captain), *mu_sigma(vice))


def vice_reasoning(captain_name: str, vice_xp: float, vice_play: float) -> str:
    return (
        f"Activates if {captain_name} does not feature. Best captain alternative: "
        f"{round(vice_xp, 2)} xP, {round(vice_play * 100)}% to play."
    )


class LineupOptimizer:
    def __init__(
        self,
        cameo_risk_penalty: float | None = None,
        eo_hedge_weight: float | None = None,
        differential_captain_hurdle: float | None = None,
    ):
        try:
            from fpl_oracle.config import SETTINGS
            opt_cfg = SETTINGS.get("optimizer", {})
        except Exception:
            opt_cfg = {}
        if cameo_risk_penalty is None:
            self.cameo_risk_penalty = float(opt_cfg.get("cameo_risk_penalty", 0.75))
        else:
            self.cameo_risk_penalty = float(cameo_risk_penalty)

        if eo_hedge_weight is None:
            self.eo_hedge_weight = float(opt_cfg.get("eo_hedge_weight", 0.80))
        else:
            self.eo_hedge_weight = float(eo_hedge_weight)

        if differential_captain_hurdle is None:
            self.differential_captain_hurdle = float(opt_cfg.get("differential_captain_hurdle", 1.20))
        else:
            self.differential_captain_hurdle = float(differential_captain_hurdle)

    def select_lineup_and_captain(
        self,
        squad_df: pd.DataFrame,
        is_triple_captain: bool = False,
        is_bench_boost: bool = False,
        risk_preference: str = "balanced",
        excluded_ids: set[int] | None = None,
        simulate_captaincy: bool = True,
        cameo_risk_penalty: float | None = None,
        eo_hedge_weight: float | None = None,
        differential_captain_hurdle: float | None = None,
        midweek_data: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Given a 15-player squad, selects optimal starting XI, captain, vice-captain, and bench order.
        The XI maximises expected points. Captain maximises expected points too ("balanced"/"points");
        "conservative"/"aggressive" are explicit floor/ceiling tilts for the captain only.
        """
        if len(squad_df) != 15:
            raise ValueError(f"Squad must have exactly 15 players, got {len(squad_df)}")

        df = squad_df.copy().reset_index(drop=True)

        # Midweek European & Domestic Cup Rotation Drag (Hall of Fame Feature 4)
        if midweek_data or "turnaround_hours" in df.columns:
            from fpl_oracle.optimise.turnaround import turnaround_drag_service

            df = turnaround_drag_service.apply_turnaround_drag(df, midweek_data)

        # Captain multiplier: 2x normally, 3x for Triple Captain
        cap_multiplier = 3.0 if is_triple_captain else 2.0

        # Captain score: expected points unless an explicit floor/ceiling mode is requested.
        p90_vals = df["p90"] if "p90" in df.columns else df["expected_points"] * 1.5
        p10_vals = df["p10"] if "p10" in df.columns else df["expected_points"] * 0.4

        if risk_preference == "conservative":
            cap_scores = (df["expected_points"] * 0.60 + p10_vals * 0.40).copy()
        elif risk_preference == "aggressive":
            cap_scores = (df["expected_points"] * 0.40 + p90_vals * 0.60).copy()
        else:  # balanced / points
            cap_scores = df["expected_points"].astype(float).copy()

        # Effective Ownership (EO) Hedging & Armband Downside Shield (Hall of Fame Feature 1)
        eo_values = pd.Series(0.0, index=df.index)
        if "top10k_eo" in df.columns:
            eo_values = pd.to_numeric(df["top10k_eo"], errors="coerce").fillna(0.0)
        elif "effective_ownership" in df.columns:
            eo_values = pd.to_numeric(df["effective_ownership"], errors="coerce").fillna(0.0)
        elif "selected_by_percent" in df.columns:
            own = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0.0)
            eo_values = own.apply(
                lambda x: min(190.0, float(x) * 2.2)
                if float(x) >= 35.0
                else (float(x) * 1.5 if float(x) >= 15.0 else float(x))
            )

        hedge_w = float(eo_hedge_weight if eo_hedge_weight is not None else self.eo_hedge_weight)
        hurdle_val = float(
            differential_captain_hurdle
            if differential_captain_hurdle is not None
            else self.differential_captain_hurdle
        )

        has_positive_eo = (eo_values > 0).any()
        consensus_idx = eo_values.idxmax() if len(eo_values) > 0 and has_positive_eo else None
        max_eo = float(eo_values.loc[consensus_idx]) if consensus_idx is not None else 0.0
        consensus_element = int(df.loc[consensus_idx, "element"]) if consensus_idx is not None else None
        consensus_xp = float(df.loc[consensus_idx, "expected_points"]) if consensus_idx is not None else 0.0

        eo_shield_applied = False
        shield_reason = "No consensus captain (EO >= 100%) in squad."

        if max_eo >= 100.0 and consensus_element is not None:
            if risk_preference not in ("aggressive", "chasing", "chasing_pack"):
                eo_excess = (max_eo - 100.0) / 100.0
                downside_discount = eo_excess * hedge_w

                for idx, row in df.iterrows():
                    elem = int(row["element"])
                    if elem != consensus_element:
                        cand_xp = float(row["expected_points"])
                        if cand_xp - consensus_xp < hurdle_val:
                            # Fails to clear hurdle: discount candidate score so consensus retains armband
                            cap_scores.loc[idx] -= downside_discount + 0.50


        elements = df["element"].astype(int)
        unavailable = (
            df["simulation_unavailable"].fillna(False).astype(bool) if "simulation_unavailable" in df else None
        )
        blocked = {int(e) for e in elements if e in (excluded_ids or set())}
        if unavailable is not None:
            blocked |= {int(e) for e in elements[unavailable]}
        pen = float(cameo_risk_penalty if cameo_risk_penalty is not None else self.cameo_risk_penalty)
        play_probs = play_probability(df)
        cameo_penalty = (1.0 - play_probs) * pen
        if "expected_minutes" in df.columns:
            low_mins_mask = df["expected_minutes"].notna() & (df["expected_minutes"] < 65.0)
            mins_factor = ((65.0 - df.loc[low_mins_mask, "expected_minutes"]) / 65.0).clip(0.0, 1.0)
            cameo_penalty.loc[low_mins_mask] += mins_factor * pen
        starter_scores = df["expected_points"] - cameo_penalty

        starter_ids, _, cap_id, _ = best_lineup(
            list(elements),
            dict(zip(elements, df["position"], strict=True)),
            dict(zip(elements, df["expected_points"].astype(float), strict=True)),
            1.0 if is_bench_boost else 0.0,
            dict(zip(elements, cap_scores.astype(float), strict=True)),
            cap_multiplier - 1.0,
            blocked,
            starter_score=dict(zip(elements, starter_scores.astype(float), strict=True)),
        )

        # Determine whether EO downside shield was applied or differential cleared the hurdle
        raw_xp_leader = (
            int(df.loc[df["expected_points"].astype(float).idxmax(), "element"]) if len(df) > 0 else None
        )
        if max_eo >= 100.0 and consensus_element is not None:
            if risk_preference not in ("aggressive", "chasing", "chasing_pack"):
                if raw_xp_leader != consensus_element and cap_id == consensus_element:
                    eo_shield_applied = True
                    shield_reason = (
                        f"EO Downside Shield active: consensus candidate (element {consensus_element}) has "
                        f"{max_eo:.1f}% EO. Differential (element {raw_xp_leader}) was suppressed (failed +{hurdle_val:.2f} xP hurdle)."
                    )
                elif cap_id != consensus_element:
                    eo_shield_applied = False
                    shield_reason = (
                        f"Differential cleared hurdle (+{hurdle_val:.2f} xP) against consensus candidate "
                        f"(element {consensus_element}, {max_eo:.1f}% EO)."
                    )
                else:
                    eo_shield_applied = False
                    shield_reason = f"Consensus candidate (element {consensus_element}) is natural raw xP leader ({max_eo:.1f}% EO)."
            else:
                eo_shield_applied = False
                shield_reason = "Chasing/aggressive risk preference: EO downside shield lifted to hunt differential ceiling."

        is_starter = elements.isin(starter_ids)
        starters_df = df[is_starter].copy()
        bench_df = df[~is_starter].copy()
        captain_row = df[elements == cap_id].iloc[0]

        # Vice-captain: best alternative if the captain does not play. Rank by xP x P(plays); a vice from the
        # captain's club shares the same kickoff, so it is a weaker hedge (5% haircut).
        others = starters_df[starters_df["element"] != captain_row["element"]].copy()
        others["_vice_score"] = others["expected_points"] * play_probability(others)
        if "team" in others.columns:
            others.loc[others["team"] == captain_row.get("team"), "_vice_score"] *= 0.95
        others = others.sort_values(by=["_vice_score", "expected_points", "element"], ascending=[False, False, True])
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
            .sort_values(by=["_cap_score", "expected_points", "element"], ascending=[False, False, True])
            .head(5)
        )

        # Probability of outscoring next best option using scoreline simulator (WI-4.3) or analytical normal
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
            prob_outscore = outscore_probability(captain_row, vice_captain_row)

        # Starting points and bench points
        starters_xp = float(starters_df["expected_points"].sum())
        cap_bonus_xp = float(captain_row["expected_points"] * (cap_multiplier - 1.0))
        bench_xp = float(bench_df["expected_points"].sum())

        total_lineup_xp = starters_xp + cap_bonus_xp
        if is_bench_boost:
            total_lineup_xp += bench_xp

        starters_df["cameo_penalty"] = cameo_penalty.loc[starters_df.index]
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
            },
            "vice_captain": {
                "element": int(vice_captain_row["element"]),
                "web_name": vice_captain_row["web_name"],
                "team": vice_captain_row.get("team"),
                "expected_points": round(float(vice_captain_row["expected_points"]), 2),
                "p10": round(float(vice_captain_row.get("p10", 0.0)), 2),
                "p90": round(float(vice_captain_row.get("p90", 0.0)), 2),
                "reasoning": vice_reasoning(
                    captain_row["web_name"], float(vice_captain_row["expected_points"]), vice_play
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
            "captain_eo_shield": {
                "consensus_element": consensus_element,
                "consensus_eo": round(max_eo, 1),
                "shield_applied": eo_shield_applied,
                "reason": shield_reason,
            },
        }


lineup_optimizer = LineupOptimizer()
