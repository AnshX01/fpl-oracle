"""Replan legal transfers and other chips for each possible date of one chip."""


def compare_dates(search, kwargs, chip, baseline, gameweeks):
    rows = []
    for gw in gameweeks:
        states = search(**kwargs, forced_chip_schedule={gw: chip})
        if not states:
            rows.append(dict(gameweek=gw, status="infeasible", expected_gain=None))
            continue
        best = max(states, key=lambda state: state["score"])
        gain = None if baseline is None else round(best["score"] - baseline["score"], 2)
        rows.append(
            dict(
                gameweek=gw,
                status="required_for_feasibility" if baseline is None else "measured",
                expected_gain=gain,
                plan_score=round(best["score"], 2),
                baseline_score=None if baseline is None else round(baseline["score"], 2),
                trajectory=best["history"],
            )
        )
    return rows
