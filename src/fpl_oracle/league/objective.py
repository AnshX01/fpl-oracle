"""Bounded balanced rival exposure utility from observed rosters, not captains."""

from math import tanh


def select_balanced_sequence(states, rival_context, maps, max_xp_loss=0.5):
    best = max(states, key=lambda s: s["score"])
    rivals = (rival_context or {}).get("rivals", [])
    if not rivals:
        return best, dict(status="unavailable", applied=False, max_xp_loss=max_xp_loss)
    user_points = float(rival_context["user_points"])

    def utility(state):
        value = 0.0
        for rival in rivals:
            # Positive deficit gently prefers differential exposure. A lead
            # gently prefers coverage. Never change the captain to guess theirs.
            pressure = tanh((float(rival["points"]) - user_points) / 25.0)
            owned = set(rival["elements"])
            for row in state["history"]:
                pmap = maps[row["gameweek"]]
                for e in row["elements"]:
                    player = pmap[e]
                    xp = float(player.get("expected_points", 0))
                    spread = max(0.0, float(player.get("p90", xp)) - float(player.get("p10", xp)))
                    value += pressure * (1 if e not in owned else -1) * min(spread, 8.0)
        return value / len(rivals)

    close = [s for s in states if best["score"] - s["score"] <= max_xp_loss + 1e-9]
    chosen = max(close, key=lambda s: (utility(s), s["score"]))
    return chosen, dict(
        status="observed_prior_gw_scenario",
        applied=chosen is not best,
        max_xp_loss=max_xp_loss,
        actual_xp_loss=round(best["score"] - chosen["score"], 4),
        rival_count=len(rivals),
        upcoming_rival_captains_assumed=False,
        utility="bounded_gap_weighted_roster_exposure",
        winner_probability=False,
    )
