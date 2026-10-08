"""Paired per-week chip uplift on the owned squad, not a season calendar."""

import pandas as pd

from fpl_oracle.optimise.lineup import lineup_optimizer


def measured_windows(squad, projections):
    rows = []
    owned = set(squad.element)
    for gw, pool in sorted(projections.items()):
        if pool.empty or not owned.issubset(set(pool.element)):
            continue
        frame = pool[pool.element.isin(owned)].copy()
        if len(frame) != 15 or not pd.to_numeric(frame.expected_points, errors="coerce").notna().all():
            continue
        base = lineup_optimizer.select_lineup_and_captain(frame, risk_preference="points")
        for code, display, flag in [
            ("3xc", "Triple Captain", "is_triple_captain"),
            ("bboost", "Bench Boost", "is_bench_boost"),
        ]:
            with_chip = lineup_optimizer.select_lineup_and_captain(frame, risk_preference="points", **{flag: True})
            rows.append(
                dict(
                    chip=code,
                    chip_name=display,
                    gameweek=gw,
                    baseline_points=base["total_gameweek_expected_points"],
                    chip_points=with_chip["total_gameweek_expected_points"],
                    expected_gain=round(
                        with_chip["total_gameweek_expected_points"] - base["total_gameweek_expected_points"], 2
                    ),
                    squad_scope="current_owned_squad",
                    recommended=False,
                )
            )
    return rows
