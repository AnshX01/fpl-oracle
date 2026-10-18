"""What a manager learns by waiting (docs/decision_model.md section 3).

Playing a chip now commits to a squad on today's forecasts; keeping it lets a later chip react to what happens
first. Today's forecasts already price each player's chance of missing a week, but not which players will be
missing: absences come in spells. Each scenario draws, per player, a path of available / absent weeks whose
long-run share matches the planner's own chance of playing, with spell lengths as measured from past seasons.
Only the next REVEAL_WEEKS are realised (that is what can be known before a later chip); later weeks keep the
deterministic forecast. Form changes are not simulated, so this understates the value of waiting.
"""

import numpy as np
import pandas as pd

# Absences of regular starters (started the previous three) by length 1..9 and 10+ matches, 2023-24 to 2026-27;
# a 10+ spell averages 16 matches. Weekly start rate 8.6%.
SPELL_COUNTS = np.array([717, 179, 116, 86, 66, 38, 29, 22, 16, 87], dtype=float)
SPELL_LENGTHS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 16])
MEAN_SPELL = float(SPELL_LENGTHS @ SPELL_COUNTS / SPELL_COUNTS.sum())
REVEAL_WEEKS = 3  # about one mean absence: injuries known before a chip a few weeks out
MIN_P_PLAY = 0.3  # the spells are measured on regular starters; less likely players keep their forecast
SCENARIOS = 10


def availability_path(p_play: float, weeks: int, rng: np.random.Generator) -> np.ndarray:
    """Available (True) / absent weeks with long-run share p_play: absent spells drawn from SPELL_COUNTS, available
    spells geometric with the start rate that makes the absent share 1 - p_play. The path starts in the long-run
    state: absent with probability 1 - p_play, for the remaining part of a spell (P(r weeks left) ~ P(length >= r))."""
    p = float(np.clip(p_play, MIN_P_PLAY, 0.99))
    start_rate = (1.0 - p) / (MEAN_SPELL * p)
    path = np.empty(weeks + int(SPELL_LENGTHS.max()), dtype=bool)
    t = 0
    if rng.random() >= p:
        left = np.arange(1, int(SPELL_LENGTHS.max()) + 1)
        weight = np.array([SPELL_COUNTS[SPELL_LENGTHS >= r].sum() for r in left])
        run = int(rng.choice(left, p=weight / weight.sum()))
        path[:run] = False
        t = run
    while t < weeks:
        run = int(rng.geometric(start_rate))
        path[t : t + run] = True
        t += run
        run = int(rng.choice(SPELL_LENGTHS, p=SPELL_COUNTS / SPELL_COUNTS.sum()))
        path[t : t + run] = False
        t += run
    return path[:weeks]


def scenario_projections(
    horizon_projections: dict[int, pd.DataFrame], target_gw: int, rng: np.random.Generator
) -> dict[int, pd.DataFrame]:
    """Projections with one draw of who is missing over the REVEAL_WEEKS after target_gw: an available player keeps
    expected_points / p_play (his points when he plays), an absent one scores 0; on average this is the forecast."""
    weeks = sorted(g for g in horizon_projections if g > target_gw)[:REVEAL_WEEKS]
    if not weeks or "p_play" not in horizon_projections[weeks[0]]:
        return horizon_projections
    first = horizon_projections[weeks[0]]
    p_first = dict(zip(first["element"].astype(int), first["p_play"].astype(float), strict=True))
    paths = {e: availability_path(p, len(weeks), rng) for e, p in p_first.items() if p >= MIN_P_PLAY}
    out = dict(horizon_projections)
    for i, gw in enumerate(weeks):
        frame = horizon_projections[gw].copy()
        factor = np.array(
            [paths[e][i] / p_first[e] if e in paths else 1.0 for e in frame["element"].astype(int)], dtype=float
        )
        frame["expected_points"] = frame["expected_points"].to_numpy(float) * factor
        out[gw] = frame
    return out
