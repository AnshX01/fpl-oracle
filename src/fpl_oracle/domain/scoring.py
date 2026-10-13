"""
Deterministic Match Scoring Engine & Exact Rule Implementation for FPL.
Computes deterministic points from actual match event counts according to verified 2026/27 rules.
Also provides expected points calculation under discrete count approximations for thresholded stats.
"""

import math
from typing import Any

from fpl_oracle.rules.scoring import (
    APPEARANCE_GE60,
    APPEARANCE_LT60,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFCON_POINTS,
    GOAL_POINTS,
    GOALS_CONCEDED_PER_MINUS1,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    RED_CARD_POINTS,
    SAVES_PER_PLUS1,
    YELLOW_CARD_POINTS,
)


def calculate_match_points(
    position: str,  # "GKP", "DEF", "MID", "FWD"
    minutes: int,
    goals_scored: int = 0,
    assists: int = 0,
    clean_sheet: bool = False,
    goals_conceded: int = 0,
    saves: int = 0,
    penalties_saved: int = 0,
    penalties_missed: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    own_goals: int = 0,
    defensive_contribution: bool = False,
    bonus: int = 0,
    multiplier: int = 1,
) -> int:
    """
    Exact deterministic scoring rule implementation.
    Allows negative total scores (e.g., played <60m, scored own goal, yellow card = 1 - 2 - 1 = -2).
    """
    pos = position.upper()
    pts = 0

    # 1. Appearance
    if minutes >= 60:
        pts += APPEARANCE_GE60
    elif minutes > 0:
        pts += APPEARANCE_LT60

    # 2. Goals scored
    pts += goals_scored * GOAL_POINTS.get(pos, 4)

    # 3. Assists (3 pts for all positions)
    pts += assists * ASSIST_POINTS

    # 4. Clean Sheet (only if played >= 60 minutes)
    if clean_sheet and minutes >= 60:
        pts += CLEAN_SHEET_POINTS.get(pos, 0)

    # 5. Goals Conceded (-1 per 2 goals for GKP and DEF)
    if pos in GOALS_CONCEDED_PER_MINUS1:
        pts -= goals_conceded // GOALS_CONCEDED_PER_MINUS1[pos]

    # 6. Saves (1 pt per 3 saves for GKP)
    if pos == "GKP":
        pts += saves // SAVES_PER_PLUS1

    # 7. Penalties
    pts += penalties_saved * PENALTY_SAVE_POINTS
    pts += penalties_missed * PENALTY_MISS_POINTS

    # 8. Disciplinary & Own Goals
    pts += yellow_cards * YELLOW_CARD_POINTS
    pts += red_cards * RED_CARD_POINTS
    pts += own_goals * OWN_GOAL_POINTS

    # 9. Defensive Contribution (DefCon +2 for DEF, MID, FWD)
    if defensive_contribution and pos in ("DEF", "MID", "FWD"):
        pts += DEFCON_POINTS

    # 10. Bonus
    pts += bonus

    return pts * multiplier


def expected_floor_div(rate: float, divisor: int, max_k: int | None = None, tol: float = 1e-12) -> float:
    """
    Compute E[floor(K / divisor)] where K ~ Poisson(rate).
    Accounts for the Jensen inequality gap between floor(E[K]/divisor) and E[floor(K/divisor)].
    Uses dynamic upper bound when max_k is None, summing until residual mass < tol.
    """
    if rate <= 0.0 or divisor <= 0:
        return 0.0

    cap = max(100, int(rate + 10 * math.sqrt(rate) + 20)) if max_k is None else max_k
    prob_k = math.exp(-rate)
    cdf = prob_k
    exp_val = 0.0
    k = 0
    while k <= cap:
        if k > 0:
            prob_k = prob_k * (rate / k)
            cdf += prob_k
        exp_val += (k // divisor) * prob_k
        if max_k is None and k > rate and (1.0 - cdf) < tol:
            break
        k += 1
    return exp_val


def calculate_expected_fixture_points(
    position: str,
    p_play: float,
    p_min60: float,
    p_starts: float,
    expected_minutes: float,
    expected_goals: float,
    expected_assists: float,
    p_clean_sheet: float,
    expected_goals_conceded: float,
    expected_saves: float,
    p_defcon: float,
    expected_card_deduction: float,
    expected_bonus: float,
    expected_own_goals: float = 0.0,
    expected_penalties_saved: float = 0.0,
    expected_penalties_missed: float = 0.0,
    multiplier: int = 1,
) -> dict[str, Any]:
    """
    Probabilistic expectation of points for a single match fixture.
    Enforces appearance coherence and non-clipped expectation.
    """
    pos = position.upper()

    # Coherence adjustments:
    # If p_play == 0, expected points must be strictly 0.0
    if p_play <= 1e-4 or expected_minutes <= 1e-4:
        return {
            "expected_points": 0.0,
            "appearance_pts": 0.0,
            "attacking_pts": 0.0,
            "defending_pts": 0.0,
            "defcon_pts": 0.0,
            "bonus_pts": 0.0,
            "deductions": 0.0,
        }

    # Appearance points: 2 for >=60m, 1 for <60m
    p_sub = max(0.0, p_play - p_min60)
    app_pts = p_min60 * 2.0 + p_sub * 1.0

    # Attacking returns
    goal_rate = float(GOAL_POINTS.get(pos, 4))
    goal_pts = expected_goals * goal_rate
    assist_pts = expected_assists * float(ASSIST_POINTS)

    # Clean sheet: requires 60+ minutes
    cs_rate = float(CLEAN_SHEET_POINTS.get(pos, 0))
    cs_pts = p_clean_sheet * p_min60 * cs_rate

    # Goals conceded deduction (-1 per 2 goals for GKP/DEF)
    # Using Poisson-based expectation for floor(goals / 2)
    if pos in ("GKP", "DEF"):
        gc_deduction = expected_floor_div(expected_goals_conceded, 2)
    else:
        gc_deduction = 0.0

    # Saves (1 per 3 saves for GKP)
    if pos == "GKP":
        save_pts = expected_floor_div(expected_saves, 3)
    else:
        save_pts = 0.0

    # Penalties
    pen_pts = expected_penalties_saved * 5.0 - expected_penalties_missed * 2.0

    # DefCon (+2 for DEF, MID, FWD)
    if pos in ("DEF", "MID", "FWD"):
        defcon_pts = p_defcon * p_play * 2.0
    else:
        defcon_pts = 0.0

    # Disciplinary & own goals
    card_deduction = expected_card_deduction + (expected_own_goals * 2.0)

    # Bonus
    bonus_pts = expected_bonus

    # Fixture total
    total_xp = (
        app_pts
        + goal_pts
        + assist_pts
        + cs_pts
        - gc_deduction
        + save_pts
        + pen_pts
        + defcon_pts
        - card_deduction
        + bonus_pts
    ) * multiplier

    return {
        "expected_points": round(total_xp, 3),
        "appearance_pts": round(app_pts * multiplier, 3),
        "attacking_pts": round((goal_pts + assist_pts) * multiplier, 3),
        "defending_pts": round((cs_pts - gc_deduction + save_pts) * multiplier, 3),
        "defcon_pts": round(defcon_pts * multiplier, 3),
        "bonus_pts": round(bonus_pts * multiplier, 3),
        "deductions": round(card_deduction * multiplier, 3),
    }
