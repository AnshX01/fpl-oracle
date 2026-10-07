"""Bounded balanced rival exposure utility from observed rosters, not captains."""

from math import tanh


def select_balanced_sequence(states, rival_context, maps, max_xp_loss=0.5):
    best = max(states, key=lambda s: s["score"])
    rivals = (rival_context or {}).get("rivals", [])
    if not rivals:
        return best, dict(status="unavailable", applied=False, max_xp_loss=max_xp_loss)
    user_points = float(rival_context["user_points"])
    from fpl_oracle.league.scenarios import evolve_rival

    evolved = {index: evolve_rival(rival, maps, "one_transfer_per_week") for index, rival in enumerate(rivals)}

    def utility(state):
        value = 0.0
        for index, rival in enumerate(rivals):
            # Positive deficit gently prefers differential exposure. A lead
            # gently prefers coverage. Never change the captain to guess theirs.
            pressure = tanh((float(rival["points"]) - user_points) / 25.0)
            for step, row in enumerate(state["history"]):
                observed = set(rival["elements"])
                possible = set(evolved[index][step]["elements"]) if evolved[index] else observed
                pmap = maps[row["gameweek"]]
                for e in row["elements"]:
                    player = pmap[e]
                    xp = float(player.get("expected_points", 0))
                    spread = max(0.0, float(player.get("p90", xp)) - float(player.get("p10", xp)))
                    # Minimax exposure across observed-hold and legal evolving
                    # rival stress cases. No invented likelihood weights.
                    scenarios = [
                        pressure * (1 if e not in roster else -1) * min(spread, 8.0) for roster in (observed, possible)
                    ]
                    value += min(scenarios)
        return value / len(rivals)

    close = [s for s in states if best["score"] - s["score"] <= max_xp_loss + 1e-9]
    chosen = max(close, key=lambda s: (utility(s), s["score"]))
    return chosen, dict(
        status="observed_and_hypothetical_evolving_roster_scenarios",
        applied=chosen is not best,
        max_xp_loss=max_xp_loss,
        actual_xp_loss=round(best["score"] - chosen["score"], 4),
        rival_count=len(rivals),
        upcoming_rival_captains_assumed=False,
        utility="bounded_robust_gap_weighted_exposure",
        rival_evolution="legal_same_position_non_price_increasing_one_transfer_per_week_stress_case",
        behaviour_calibrated=False,
        winner_probability=False,
    )
