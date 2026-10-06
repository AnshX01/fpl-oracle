"""
Deterministic Match Scoring Engine & Exact Rule Implementation for FPL.
Computes deterministic points from actual match event counts according to verified 2026/27 rules.
Also provides expected points calculation under discrete count approximations for thresholded stats.
"""

import math
from typing import Any


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
        pts += 2
    elif minutes > 0:
        pts += 1

    # 2. Goals scored
    if pos == "GKP":
        pts += goals_scored * 10
    elif pos == "DEF":
        pts += goals_scored * 6
    elif pos == "MID":
        pts += goals_scored * 5
    elif pos == "FWD":
        pts += goals_scored * 4

    # 3. Assists (3 pts for all positions)
    pts += assists * 3

    # 4. Clean Sheet (only if played >= 60 minutes)
    if clean_sheet and minutes >= 60:
        if pos in ("GKP", "DEF"):
            pts += 4
        elif pos == "MID":
            pts += 1

    # 5. Goals Conceded (-1 per 2 goals for GKP and DEF)
    if pos in ("GKP", "DEF"):
        pts -= (goals_conceded // 2)

    # 6. Saves (1 pt per 3 saves for GKP)
    if pos == "GKP":
        pts += (saves // 3)

    # 7. Penalties
    pts += penalties_saved * 5
    pts -= penalties_missed * 2

    # 8. Disciplinary & Own Goals
    pts -= yellow_cards * 1
    pts -= red_cards * 3
    pts -= own_goals * 2

    # 9. Defensive Contribution (DefCon +2 for DEF, MID, FWD)
    if defensive_contribution and pos in ("DEF", "MID", "FWD"):
        pts += 2

    # 10. Bonus
    pts += bonus

    return pts * multiplier


def expected_floor_div(rate: float, divisor: int, max_k: int = 20) -> float:
    """
    Compute E[floor(K / divisor)] where K ~ Poisson(rate).
    Accounts for the Jensen inequality gap between floor(E[K]/divisor) and E[floor(K/divisor)].
    """
    if rate <= 0.0:
        return 0.0

    exp_val = 0.0
    for k in range(max_k + 1):
        prob_k = (math.exp(-rate) * (rate**k)) / math.factorial(k)
        exp_val += (k // divisor) * prob_k
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
    goal_rate = {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}.get(pos, 5.0)
    goal_pts = expected_goals * goal_rate
    assist_pts = expected_assists * 3.0

    # Clean sheet: requires 60+ minutes
    cs_rate = {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}.get(pos, 0.0)
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
