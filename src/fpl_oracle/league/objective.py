"""Mini-league win probability of a plan (docs/decision_model.md section 4).

Symmetric assumption: every manager improves their squad as well as our planner improves ours, so squad
quality differences are today's (both squads held), and a plan only moves the mean by how far it is below
our best plan:
  mu_r  = (my points - r's points today) + sum_horizon (my held XI xP - r's held XI xP)
          + (this plan's xP - our best plan's xP)
  var_r = sum over weeks and players of (this plan's weight - r's weight)^2 * sd^2, scaled up for the weeks
          left after the horizon at the same per-week rate
P(finish above r) = Phi(mu_r / sqrt(var_r)); P(win) ~ product over the tracked rivals.
Weights: 1 per starter, +1 for the captain (+2 with Triple Captain), 1 per bench player with Bench Boost.
A held squad fields its best XI with the top-xP captain. Shared players cancel in both mean and variance.
"""

from math import sqrt
from statistics import NormalDist
from typing import Any

SD_PER_P10_P90 = 2.5631  # p90 - p10 of a normal distribution spans 2.563 standard deviations


def week_weights(starters: list[int], captain: int | None, chip: str | None, bench: list[int]) -> dict[int, float]:
    weights = {int(p): 1.0 for p in starters}
    if captain is not None:
        weights[int(captain)] = weights.get(int(captain), 0.0) + (2.0 if chip == "3xc" else 1.0)
    if chip == "bboost":
        for p in bench:
            weights[int(p)] = weights.get(int(p), 0.0) + 1.0
    return weights


def rival_weeks(elements: list[int], gws: list[int], xp: dict[int, dict[int, float]], position: dict[int, str]):
    from fpl_oracle.optimise.multiperiod import best_lineup

    squad = [int(e) for e in elements if int(e) in position]
    weeks = []
    for gw in gws:
        if len(squad) != 15:
            weeks.append({e: 1.0 for e in squad})
            continue
        starters, bench, captain, _ = best_lineup(squad, position, xp[gw], 0.0)
        weeks.append(week_weights(starters, captain, None, bench))
    return weeks


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
) -> dict[str, Any] | None:
    """P(finishing above every tracked rival) for one plan, or None without league data."""
    if not rival_context or not rival_context.get("rivals") or not gws:
        return None
    rivals = rival_context["rivals"]
    rest = max(0, 38 - gws[-1])
    cache = rival_cache if rival_cache is not None else {}
    plan_points = sum(w * xp[gw].get(p, 0.0) for gw, week in zip(gws, my_weeks, strict=True) for p, w in week.items())
    per_rival = []
    p_all = 1.0
    for index, rival in enumerate(rivals):
        if index not in cache:
            cache[index] = rival_weeks(rival["elements"], gws, xp, position)
        mean = float(rival_context["user_points"]) - float(rival["points"]) + plan_points - best_plan_points
        var = 0.0
        for gw, mine, held, theirs in zip(gws, my_weeks, held_weeks, cache[index], strict=True):
            mean += sum(w * xp[gw].get(p, 0.0) for p, w in held.items())
            mean -= sum(w * xp[gw].get(p, 0.0) for p, w in theirs.items())
            for p in set(mine) | set(theirs):
                diff = mine.get(p, 0.0) - theirs.get(p, 0.0)
                var += (diff * sd[gw].get(p, 0.0)) ** 2
        var += var / len(gws) * rest
        p_above = NormalDist().cdf(mean / sqrt(var)) if var > 0 else float(mean > 0)
        per_rival.append(
            {"rival_points": rival["points"], "expected_margin": round(mean, 1), "p_above": round(p_above, 3)}
        )
        p_all *= p_above
    return {"p_win": round(p_all, 3), "rivals": per_rival, "weeks_after_horizon": rest}
