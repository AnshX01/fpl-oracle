"""Owner's hit rule: raw transferred-player returns, not captain/lineup gain."""


def expected_transfer_gain(ins, outs, maps, gameweeks):
    return sum(
        sum(float(maps[g][e]["expected_points"]) for e in ins) - sum(float(maps[g][e]["expected_points"]) for e in outs)
        for g in gameweeks
    )


def settle_hit(ins, outs, points, cost):
    needed = set(ins) | set(outs)
    if not needed.issubset(points):
        return {"status": "unknown", "reason": "Player points missing"}
    gain = sum(points[e] for e in ins) - sum(points[e] for e in outs)
    return {"status": "success" if gain >= cost else "failed", "realised_gain": gain, "hit_cost": cost}


def cooldown_until(records):
    # Six following gameweeks are blocked after a settled failed hit.
    return max((r["gameweek"] + 7 for r in records if r.get("status") == "failed"), default=0)
