"""Mini-league win probability of a plan (docs/decision_model.md section 4).

Symmetric assumption: every manager improves their squad as well as our planner improves ours, so squad
quality differences are today's (both squads held), and a plan only moves the mean by how far it is below
our best plan:
  mu_r  = (my points - r's points today) + sum_horizon (my held XI xP - r's held XI xP)
          + (this plan's xP - our best plan's xP)
  var_r = sum over weeks and players of (this plan's weight - r's weight)^2 * sd^2, scaled up for the weeks
          left after the horizon at the same per-week rate
P(finish above r) = Phi(mu_r / sqrt(var_r)). The gaps to different rivals share our own players, so
cov(D_r, D_s) = sum (w_plan - w_r)(w_plan - w_s) sd^2 and P(win) = P(every gap > 0) under that joint normal,
estimated with fixed draws (the same draws for every plan, so plans are compared without sampling noise).
Weights: 1 per starter, +1 for the captain (+2 with Triple Captain), 1 per bench player with Bench Boost.
A held squad fields its best XI with the top-xP captain, except that a rival's first week keeps the captain
they picked last week while he is still in their XI (managers rarely switch a premium captain).
Shared players cancel in both mean and variance.
"""

from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np

from fpl_oracle.optimise.lineup import best_lineup

SD_PER_P10_P90 = 2.5631  # p90 - p10 of a normal distribution spans 2.563 standard deviations
JOINT_DRAWS = 20000  # standard error of P(win) about 0.3 percentage points


def week_weights(starters: list[int], captain: int | None, chip: str | None, bench: list[int]) -> dict[int, float]:
    weights = {int(p): 1.0 for p in starters}
    if captain is not None:
        weights[int(captain)] = weights.get(int(captain), 0.0) + (2.0 if chip == "3xc" else 1.0)
    if chip == "bboost":
        for p in bench:
            weights[int(p)] = weights.get(int(p), 0.0) + 1.0
    return weights


def rival_weeks(
    elements: list[int],
    gws: list[int],
    xp: dict[int, dict[int, float]],
    position: dict[int, str],
    last_captain: int | None = None,
):
    squad = [int(e) for e in elements if int(e) in position]
    weeks = []
    for index, gw in enumerate(gws):
        if len(squad) != 15:
            weeks.append({e: 1.0 for e in squad})
            continue
        starters, bench, captain, _ = best_lineup(squad, position, xp[gw], 0.0)
        if index == 0 and last_captain in starters:
            captain = last_captain
        weeks.append(week_weights(starters, captain, None, bench))
    return weeks


def league_threats(
    rivals: list[dict[str, Any]],
    gws: list[int],
    xp: dict[int, dict[int, float]],
    position: dict[int, str],
    mine: set[int],
    cache: dict[int, list[dict[int, float]]],
    top: int = 3,
) -> list[int]:
    """Players we do not own that score the most for our tracked rivals over the horizon (their weight x xP),
    i.e. the players whose good weeks would widen the gap most. Fills `cache` with each rival's weeks."""
    threat: dict[int, float] = {}
    gw0 = gws[0] if gws else None
    for index, rival in enumerate(rivals):
        if index not in cache:
            cache[index] = rival_weeks(rival["elements"], gws, xp, position, rival.get("captain"))
        for gw, week in zip(gws, cache[index], strict=True):
            for p, w in week.items():
                if p not in mine:
                    threat[p] = threat.get(p, 0.0) + w * xp[gw].get(p, 0.0)

    # Guardrail 4a: Fixture Threat Gating for Defenders
    # When generating immediate GW0 cover candidates, defenders facing an extreme difficulty matchup
    # (< 2.0 xP in GW0) do not pose an acute immediate threat in GW0 to justify an emergency buy.
    # We down-weight their immediate threat so active GW0 threats take priority.
    def threat_sort_key(p: int) -> tuple[float, float, int]:
        raw_t = threat[p]
        if gw0 is not None and position.get(p) in ("DEF", "GKP") and xp.get(gw0, {}).get(p, 0.0) < 2.0:
            eff_t = raw_t * 0.25
        else:
            eff_t = raw_t
        return (-eff_t, -raw_t, p)

    return sorted((p for p in threat if threat[p] > 0), key=threat_sort_key)[:top]


def win_probability(
    my_weeks: list[dict[int, float]],
    held_weeks: list[dict[int, float]],
    best_plan_points: float,
    gws: list[int],
    rival_context: dict[str, Any] | None,
    xp: dict[int, dict[int, float]],
    sd: dict[int, dict[int, float]],
    position: dict[int, str],
    rival_cache: dict[int, list[dict[int, float]]] | None = None,
    team_map: dict[int, int] | None = None,
    opponents_by_gw: dict[int, dict[int, int]] | None = None,
) -> dict[str, Any] | None:
    """P(finishing above every tracked rival) for one plan, or None without league data."""
    if not rival_context or not rival_context.get("rivals") or not gws:
        return None
    rivals = rival_context["rivals"]
    rest = max(0, 38 - gws[-1])
    cache = rival_cache if rival_cache is not None else {}
    effective_team_map = team_map or rival_context.get("team_map")
    effective_opponents = opponents_by_gw or rival_context.get("opponents_by_gw")
    plan_points = sum(w * xp[gw].get(p, 0.0) for gw, week in zip(gws, my_weeks, strict=True) for p, w in week.items())
    per_rival = []
    means: list[float] = []
    diffs: list[dict[tuple[int, int], float]] = []  # (gw, player) -> (my weight - r's weight) * sd
    u_weights: list[dict[tuple[int, int], float]] = []  # (gw, player) -> (my weight - r's weight)
    eff_sds: dict[tuple[int, int], float] = {}

    for index, rival in enumerate(rivals):
        if index not in cache:
            cache[index] = rival_weeks(rival["elements"], gws, xp, position, rival.get("captain"))
        mean = float(rival_context["user_points"]) - float(rival["points"]) + plan_points - best_plan_points
        diff: dict[tuple[int, int], float] = {}
        u: dict[tuple[int, int], float] = {}
        for gw, mine, held, theirs in zip(gws, my_weeks, held_weeks, cache[index], strict=True):
            mean += sum(w * xp[gw].get(p, 0.0) for p, w in held.items())
            mean -= sum(w * xp[gw].get(p, 0.0) for p, w in theirs.items())
            for p in set(mine) | set(theirs):
                eff_sd = sd[gw].get(p, 0.0)
                # Guardrail 4b: Clean sheet variance compression for mismatched defensive assets
                # In low-xP fixtures (xP < 2.0), clean sheet probability is < 15%, compressing
                # the realistic ceiling and variance of the defensive asset.
                if position.get(p) in ("DEF", "GKP") and xp[gw].get(p, 0.0) < 2.0:
                    eff_sd = min(eff_sd, max(0.5, xp[gw].get(p, 0.0) * 0.8))
                eff_sds[(gw, p)] = eff_sd
                weight_diff = mine.get(p, 0.0) - theirs.get(p, 0.0)
                d = weight_diff * eff_sd
                if d:
                    diff[(gw, p)] = d
                    u[(gw, p)] = weight_diff
        means.append(mean)
        diffs.append(diff)
        u_weights.append(u)

    scale = 1.0 + rest / len(gws)  # the weeks after the horizon at the same per-week rate

    # Covariance formulation incorporating teammate defensive stacking and opponent negative covariance
    n_rivals = len(diffs)
    cov = np.zeros((n_rivals, n_rivals), dtype=float)
    for i in range(n_rivals):
        for j in range(i, n_rivals):
            # Base orthogonal covariance term: sum_p (d_{i, p} * d_{j, p})
            base_cov = sum(v * diffs[j].get(k, 0.0) for k, v in diffs[i].items())
            cross_cov = 0.0

            if effective_team_map is not None:
                # Group active players by gameweek for cross-player covariance
                gws_active = set(gw for gw, _ in diffs[i].keys()) | set(gw for gw, _ in diffs[j].keys())
                for gw in gws_active:
                    players_i = [p for (g, p) in diffs[i].keys() if g == gw]
                    players_j = [p for (g, p) in diffs[j].keys() if g == gw]
                    for p1 in players_i:
                        u1 = u_weights[i].get((gw, p1), 0.0)
                        sd1 = eff_sds.get((gw, p1), 0.0)
                        pos1 = position.get(p1, "")
                        tm1 = effective_team_map.get(p1)
                        if not tm1 or not u1 or not sd1:
                            continue
                        for p2 in players_j:
                            if p1 == p2:
                                continue  # Handled in base_cov
                            u2 = u_weights[j].get((gw, p2), 0.0)
                            sd2 = eff_sds.get((gw, p2), 0.0)
                            pos2 = position.get(p2, "")
                            tm2 = effective_team_map.get(p2)
                            if not tm2 or not u2 or not sd2:
                                continue

                            # 1. Teammate defensive stack covariance: shared clean sheet outcome
                            if tm1 == tm2 and pos1 in ("DEF", "GKP") and pos2 in ("DEF", "GKP"):
                                rho_cs = 0.45
                                cross_cov += u1 * u2 * (rho_cs * sd1 * sd2)

                            # 2. Opposing attacker vs defender negative covariance
                            elif effective_opponents is not None:
                                opp_tm = effective_opponents.get(gw, {}).get(tm1)
                                if opp_tm == tm2:
                                    if (pos1 in ("DEF", "GKP") and pos2 in ("MID", "FWD")) or (
                                        pos1 in ("MID", "FWD") and pos2 in ("DEF", "GKP")
                                    ):
                                        rho_opp = -0.20
                                        cross_cov += u1 * u2 * (rho_opp * sd1 * sd2)

            val = scale * (base_cov + cross_cov)
            cov[i, j] = val
            cov[j, i] = val
    for mean, var, rival in zip(means, np.diag(cov), rivals, strict=True):
        if var <= 1e-12:
            p_above = 1.0 if mean > 1e-9 else (0.5 if abs(mean) <= 1e-9 else 0.0)
        else:
            p_above = NormalDist().cdf(mean / sqrt(var))
        per_rival.append(
            {"rival_points": rival["points"], "expected_margin": round(mean, 1), "p_above": round(p_above, 3)}
        )
    draws = np.random.default_rng(0).standard_normal((JOINT_DRAWS, len(means)))
    eigval, eigvec = np.linalg.eigh(cov)
    root = (eigvec * np.sqrt(np.clip(eigval, 0.0, None))) @ eigvec.T  # the symmetric root is unique
    gaps = np.array(means) + draws @ root
    p_all = float((gaps > 0).all(axis=1).mean())
    return {"p_win": round(p_all, 3), "rivals": per_rival, "weeks_after_horizon": rest}
