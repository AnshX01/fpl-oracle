"""
Centralised FPL Scoring Rules Module.
Single source of truth for all scoring constants, point rates, and event multipliers.
Verified against official Premier League 2026/27 rules and FPL API configurations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleEntry:
    value: Any
    source_url: str = "https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627"
    verified_on: str = "2026-10-10"


# 1. Appearance
APPEARANCE_LT60: int = 1
APPEARANCE_GE60: int = 2

# 2. Goals Scored by Position (Official Rule: GKP=6, DEF=6, MID=5, FWD=4)
GOAL_POINTS: dict[str, int] = {
    "GKP": 6,
    "DEF": 6,
    "MID": 5,
    "FWD": 4,
}

# 3. Assists
ASSIST_POINTS: int = 3

# 4. Clean Sheet (requires >= 60 minutes played)
CLEAN_SHEET_POINTS: dict[str, int] = {
    "GKP": 4,
    "DEF": 4,
    "MID": 1,
    "FWD": 0,
}

# 5. Goals Conceded Deduction (-1 per 2 goals for GKP & DEF)
GOALS_CONCEDED_PER_MINUS1: dict[str, int] = {
    "GKP": 2,
    "DEF": 2,
}

# 6. Saves (+1 per 3 saves for GKP)
SAVES_PER_PLUS1: int = 3

# 7. Penalties
PENALTY_SAVE_POINTS: int = 5
PENALTY_MISS_POINTS: int = -2
PENALTY_MISSED_POINTS: int = PENALTY_MISS_POINTS

# 8. Discipline & Own Goals
YELLOW_CARD_POINTS: int = -1
RED_CARD_POINTS: int = -3
OWN_GOAL_POINTS: int = -2

# 9. Bonus
BONUS_POINTS: tuple[int, int, int] = (3, 2, 1)

# 10. Defensive Contribution (DefCon) 2026/27
DEFCON_POINTS: int = 2
DEFCON_CAP: int = 2
DEFCON_THRESHOLDS: dict[str, int | None] = {
    "DEF": 10,   # >=10 CBIT (clearances, blocks, interceptions, tackles)
    "MID": 12,   # >=12 CBIRT (+ recoveries)
    "FWD": 12,   # >=12 CBIRT (+ recoveries)
    "GKP": None, # Ineligible
}

# Transfer & Chip Constants
MAX_BANKED_FREE_TRANSFERS: int = 5
HIT_COST_POINTS: int = 4
SET_1_LAST_GW: int = 19
FREE_HIT_FORBIDDEN_GWS: tuple[int, ...] = (1,)
MAX_CHIPS_PER_GW: int = 1


def calculate_match_points(
    position: str,
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
    defensive_contribution_actions: int | float = 0,
    bonus: int = 0,
    multiplier: int = 1,
    season: str = "2026-27",
    rules_version: str | None = None,
) -> int:
    """
    Deterministic FPL match points calculator enforcing verified official rules.
    Accurately supports historical rules eras via season parameter or explicit rules_version:
    - '2023-standard': Seasons 2023-24 and 2024-25 without DefCon rule.
    - '2025-defcon': Seasons 2025-26 and 2026-27 with DefCon rule active.
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

    # 3. Assists
    pts += assists * ASSIST_POINTS

    # 4. Clean Sheet (requires >= 60 mins)
    if clean_sheet and minutes >= 60:
        pts += CLEAN_SHEET_POINTS.get(pos, 0)

    # 5. Goals Conceded (-1 per 2 goals for GKP & DEF)
    if pos in GOALS_CONCEDED_PER_MINUS1:
        pts -= goals_conceded // GOALS_CONCEDED_PER_MINUS1[pos]

    # 6. Saves (GKP only)
    if pos == "GKP":
        pts += saves // SAVES_PER_PLUS1

    # 7. Penalties
    pts += penalties_saved * PENALTY_SAVE_POINTS
    pts += penalties_missed * PENALTY_MISS_POINTS

    # 8. Discipline & Own Goals
    pts += yellow_cards * YELLOW_CARD_POINTS
    pts += red_cards * RED_CARD_POINTS
    pts += own_goals * OWN_GOAL_POINTS

    # 9. Defensive Contribution (2025/26 and 2026/27 rules)
    # Triggered when actions reach pos threshold: DEF >= 10, MID/FWD >= 12
    has_defcon = (
        (rules_version == "2025-defcon")
        if rules_version is not None
        else (season in ("2025-26", "2026-27"))
    )
    if has_defcon and pos != "GKP":
        thresh = DEFCON_THRESHOLDS.get(pos)
        if thresh is not None and defensive_contribution_actions >= thresh:
            pts += DEFCON_POINTS

    # 10. Bonus
    pts += bonus

    return pts * multiplier
