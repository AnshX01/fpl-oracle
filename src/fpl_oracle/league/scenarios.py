"""Legal hypothetical rival evolutions. Scenarios are not inferred future intent."""

import numpy as np

from fpl_oracle.optimise.sequential import evaluate_lineup
from fpl_oracle.optimise.squad import squad_optimizer


def evolve_rival(rival, maps, mode):
    elements = set(rival["elements"])
    path = []
    for gw, pool in sorted(maps.items()):
        if not elements.issubset(pool):
            return None
        if mode == "one_transfer_per_week":
            # No rival bank or buying basis is assumed. Only same-position,
            # non-price-increasing replacements with legal club counts are used.
            choices = []
            for outgoing in sorted(elements):
                counts = {}
                for e in elements - {outgoing}:
                    team = pool[e]["team"]
                    counts[team] = counts.get(team, 0) + 1
                for incoming, row in pool.items():
                    if incoming in elements or row["position"] != pool[outgoing]["position"]:
                        continue
                    if row["value"] > pool[outgoing]["value"] or counts.get(row["team"], 0) >= 3:
                        continue
                    gain = float(row["expected_points"]) - float(pool[outgoing]["expected_points"])
                    if gain > 0:
                        choices.append((gain, -incoming, outgoing, incoming))
            if choices:
                _, _, outgoing, incoming = max(choices)
                elements = elements - {outgoing} | {incoming}
        elif mode == "restructure_at_start" and not path:
            # An optimistic WC-like threat envelope, not known chip availability.
            import pandas as pd

            budget = sum(float(pool[e]["value"]) for e in elements)
            try:
                solved = squad_optimizer.solve_best_squad(pd.DataFrame(pool.values()), budget)
                elements = set(int(e) for e in solved["squad"]["element"])
            except (ValueError, RuntimeError):
                return None
        gross, captain, _ = evaluate_lineup(elements, pool)
        path.append(dict(gameweek=gw, elements=sorted(elements), captain=captain, gross_xp=gross))
    return path


def compare_plan_scenarios(states, rival_context, maps, samples=512):
    rivals = (rival_context or {}).get("rivals", [])
    if not rivals:
        return dict(status="unavailable", reason="No current relevant rival observations", scenarios=[])
    scenario_rows = []
    rng = np.random.default_rng(42)
    from fpl_oracle.league.montecarlo import monte_carlo_simulator

    draws = {}
    for gw, pool in sorted(maps.items()):
        # One shared player score and fixture/team clean-sheet event per trial.
        # This fixes independent-Normal stress draws, but is still not empirical
        # calibration of future manager decisions or season championship odds.
        draws[gw] = {e: np.empty(samples) for e in pool}
        for trial in range(samples):
            outcome = monte_carlo_simulator.draw_gameweek(pool, rng)
            for e, points in outcome.items():
                draws[gw][e][trial] = points

    path_cache = {}

    def simulated_path(path):
        key = tuple((r["gameweek"], tuple(r["elements"]), r.get("chip"), r.get("hit_cost", 0)) for r in path)
        if key in path_cache:
            return path_cache[key]
        score = np.zeros(samples)
        for row in path:
            gw = row["gameweek"]
            pool = maps[gw]
            elements = row["elements"]
            # Reconstruct the selected legal XI rather than counting all 15.
            import pandas as pd

            from fpl_oracle.optimise.lineup import lineup_optimizer

            line = lineup_optimizer.select_lineup_and_captain(
                pd.DataFrame([pool[e] for e in elements]),
                is_triple_captain=row.get("chip") == "3xc",
                is_bench_boost=row.get("chip") == "bboost",
            )
            ids = list(line["starters"]["element"])
            if row.get("chip") == "bboost":
                ids += list(line["bench"]["element"])
            for e in ids:
                score += draws[gw][int(e)]
            score += draws[gw][int(line["captain"]["element"])] * (2 if row.get("chip") == "3xc" else 1)
            score -= row.get("hit_cost", 0)
        path_cache[key] = score
        return score

    modes = ["hold_roster", "one_transfer_per_week", "restructure_at_start"]
    for mode in modes:
        paths = [(r, evolve_rival(r, maps, mode)) for r in rivals]
        if any(path is None for _, path in paths):
            continue
        opponent = np.maximum.reduce([float(r["points"]) + simulated_path(path) for r, path in paths])
        options = []
        for state in states:
            total = float(rival_context["user_points"]) + simulated_path(state["history"])
            options.append(
                dict(
                    first_chip=state["first_chip"],
                    uncalibrated_stress_beat_rate_pct=round(float(np.mean(total > opponent)) * 100, 1),
                )
            )
        scenario_rows.append(
            dict(
                mode=mode,
                assumptions="Hypothetical, not predicted rival intent; future bank/FT/chip availability unverified",
                options=options,
            )
        )
    return dict(
        status="hypothetical_stress_test",
        calibrated=False,
        championship_probability=False,
        samples=samples,
        scenarios=scenario_rows,
        source="latest_released_rival_rosters",
        unresolved="Hidden upcoming choices, score distributions and season-tail resources are not empirically calibrated",
    )
