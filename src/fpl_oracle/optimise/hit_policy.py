"""Owner's hit rule: raw transferred-player returns, not captain/lineup gain."""


def expected_transfer_gain(ins, outs, maps, gameweeks):
    return sum(
        sum(float(maps[g][e]["expected_points"]) for e in ins) - sum(float(maps[g][e]["expected_points"]) for e in outs)
        for g in gameweeks
    )


def settle_hit(ins, outs, points, cost, horizon_completed: bool = True):
    if points and isinstance(next(iter(points.values())), dict):
        # Multi-gameweek nested points mapping {gw: {element: pts}}
        needed = set(ins) | set(outs)
        all_pts_elements = set().union(*(gw_p.keys() for gw_p in points.values()))
        if not needed.issubset(all_pts_elements):
            return {"status": "unknown", "reason": "Player points missing in multi-gameweek data"}
        gain = sum(
            sum(float(gw_pts.get(e, 0.0)) for e in ins) - sum(float(gw_pts.get(e, 0.0)) for e in outs)
            for gw_pts in points.values()
        )
        if gain >= cost:
            return {"status": "success", "realised_gain": round(gain, 2), "hit_cost": cost}
        if not horizon_completed:
            return {"status": "evaluating", "realised_gain": round(gain, 2), "hit_cost": cost}
        return {"status": "failed", "realised_gain": round(gain, 2), "hit_cost": cost}

    needed = set(ins) | set(outs)
    if not needed.issubset(points):
        return {"status": "unknown", "reason": "Player points missing"}
    gain = sum(points[e] for e in ins) - sum(points[e] for e in outs)
    if gain >= cost:
        return {"status": "success", "realised_gain": gain, "hit_cost": cost}
    if not horizon_completed:
        return {"status": "evaluating", "realised_gain": gain, "hit_cost": cost}
    return {"status": "failed", "realised_gain": gain, "hit_cost": cost}


def cooldown_until(records):
    # Six following gameweeks are blocked after a settled failed hit.
    return max((r["gameweek"] + 7 for r in records if r.get("status") == "failed"), default=0)
