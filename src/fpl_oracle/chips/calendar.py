"""
Gameweek Calendar & Blank/Double Gameweek Detector.
Analyzes fixtures across the 38 gameweeks to detect blanks, doubles, and reschedulings.
"""

from collections import defaultdict
from typing import Any

from fpl_oracle.api.models import BootstrapStatic, Fixture


class FixtureCalendar:
    def __init__(self):
        pass

    def analyze_calendar(self, fixtures: list[Fixture], bootstrap: BootstrapStatic) -> dict[str, Any]:
        """
        Analyze all fixtures to detect Blank Gameweeks (BGW) and Double Gameweeks (DGW).
        """
        team_names = {t.id: t.name for t in bootstrap.teams}
        team_short = {t.id: t.short_name for t in bootstrap.teams}

        # Count fixtures per team per gameweek: (gw, team_id) -> list of fixture_ids
        team_gw_fixtures: dict[tuple[int, int], list[Fixture]] = defaultdict(list)
        gw_fixtures_count: dict[int, int] = defaultdict(int)

        for f in fixtures:
            if f.event is not None:
                team_gw_fixtures[(f.event, f.team_h)].append(f)
                team_gw_fixtures[(f.event, f.team_a)].append(f)
                gw_fixtures_count[f.event] += 1

        blank_gameweeks = []
        double_gameweeks = []
        regular_gameweeks = []

        for gw in range(1, 39):
            dgw_teams = []
            bgw_teams = []

            for t_id in team_names:
                count = len(team_gw_fixtures[(gw, t_id)])
                if count > 1:
                    dgw_teams.append(
                        {
                            "team_id": t_id,
                            "team_name": team_names[t_id],
                            "team_short": team_short[t_id],
                            "fixture_count": count,
                        }
                    )
                elif count == 0:
                    bgw_teams.append({"team_id": t_id, "team_name": team_names[t_id], "team_short": team_short[t_id]})

            if dgw_teams:
                double_gameweeks.append(
                    {
                        "gameweek": gw,
                        "type": "DOUBLE_GAMEWEEK",
                        "teams_with_doubles": dgw_teams,
                        "description": f"Double Gameweek {gw}: {len(dgw_teams)} teams playing twice.",
                    }
                )
            if bgw_teams:
                blank_gameweeks.append(
                    {
                        "gameweek": gw,
                        "type": "BLANK_GAMEWEEK",
                        "blanking_teams": bgw_teams,
                        "description": f"Blank Gameweek {gw}: {len(bgw_teams)} teams have no fixture.",
                    }
                )
            if not dgw_teams and not bgw_teams:
                regular_gameweeks.append(gw)

        return {
            "blank_gameweeks": blank_gameweeks,
            "double_gameweeks": double_gameweeks,
            "regular_gameweeks_count": len(regular_gameweeks),
            "total_fixtures": len(fixtures),
        }


fixture_calendar = FixtureCalendar()
