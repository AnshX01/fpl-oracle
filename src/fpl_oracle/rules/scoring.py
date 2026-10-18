"""
Centralised FPL Scoring Rules Module.
Single source of truth for all scoring constants, point rates, and event multipliers.
Verified against official Premier League 2026/27 rules and FPL API configurations.
"""

# 2. Goals Scored by Position (Official Rule, live game_config.scoring: GKP=10, DEF=6, MID=5, FWD=4)
GOAL_POINTS: dict[str, int] = {
    "GKP": 10,
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

# 7. Penalties
PENALTY_SAVE_POINTS: int = 5
PENALTY_MISS_POINTS: int = -2
PENALTY_MISSED_POINTS: int = PENALTY_MISS_POINTS

# 8. Discipline & Own Goals
OWN_GOAL_POINTS: int = -2

# 10. Defensive Contribution (DefCon) 2026/27
DEFCON_POINTS: int = 2
DEFCON_THRESHOLDS: dict[str, int | None] = {
    "DEF": 10,  # >=10 CBIT (clearances, blocks, interceptions, tackles)
    "MID": 12,  # >=12 CBIRT (+ recoveries)
    "FWD": 12,  # >=12 CBIRT (+ recoveries)
    "GKP": None,  # Ineligible
}

# Transfer & Chip Constants
MAX_BANKED_FREE_TRANSFERS: int = 5
HIT_COST_POINTS: int = 4
SET_1_LAST_GW: int = 19
