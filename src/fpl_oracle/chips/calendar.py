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

        unscheduled_teams: set[int] = set()
        for f in fixtures:
            if f.event is not None:
                team_gw_fixtures[(f.event, f.team_h)].append(f)
                team_gw_fixtures[(f.event, f.team_a)].append(f)
                gw_fixtures_count[f.event] += 1
            else:
                # Postponed / not yet rescheduled: that team's missing week may only be a placeholder
                unscheduled_teams.update((f.team_h, f.team_a))

        blank_gameweeks = []
        double_gameweeks = []
        regular_gameweeks = []

        for gw in range(1, 39):
            dgw_teams = []
            bgw_teams = []
            uncertain_teams = []

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
                    # A team with an unscheduled fixture may only be waiting for its date, so it is
                    # not counted as a blank for chip timing (planner reads blank_gameweeks)
                    if t_id in unscheduled_teams:
                        uncertain_teams.append(t_id)
                    else:
                        bgw_teams.append(
                            {"team_id": t_id, "team_name": team_names[t_id], "team_short": team_short[t_id]}
                        )

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
                        "uncertain_teams": uncertain_teams,
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

    PROSPECTIVE_CUP_MILESTONES = {
        26: {
            "type": "PROSPECTIVE_BLANK",
            "competition": "EFL Cup Final",
            "clash_probability": 0.90,
            "estimated_blanking_teams": 4,
            "description": "EFL Cup Final weekend: prospective blank for final participants and their opponents (typically 2 fixtures postponed).",
        },
        29: {
            "type": "PROSPECTIVE_BLANK",
            "competition": "FA Cup Quarter-Finals",
            "clash_probability": 0.85,
            "estimated_blanking_teams": 8,
            "description": "FA Cup Quarter-Finals: historically the season's major Blank Gameweek (4-6 fixtures postponed). Ideal Free Hit or 3-5 FT accumulation window.",
        },
        34: {
            "type": "PROSPECTIVE_DOUBLE",
            "competition": "FA Cup Semi-Finals / Rescheduling Window",
            "clash_probability": 0.85,
            "estimated_doubling_teams": 6,
            "description": "Major Double Gameweek: rescheduled fixtures from GW26/GW29 typically moved here. Prime Bench Boost or Triple Captain deployment target.",
        },
        37: {
            "type": "PROSPECTIVE_DOUBLE",
            "competition": "Late Season Rescheduling Window",
            "clash_probability": 0.80,
            "estimated_doubling_teams": 4,
            "description": "Late Double Gameweek: secondary rescheduling buffer for remaining postponed fixtures ahead of GW38.",
        },
    }

    def project_prospective_blank_and_double_gameweeks(
        self,
        current_gw: int = 1,
        confirmed_calendar: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ben Crellin Playbook: projects cup-clash Blank and Double Gameweeks months ahead of official FPL
        announcements based on fixed FA Cup / EFL Cup calendar milestones.
        """
        confirmed_bgw_gws = set()
        confirmed_dgw_gws = set()
        if confirmed_calendar:
            confirmed_bgw_gws = {b["gameweek"] for b in confirmed_calendar.get("blank_gameweeks", [])}
            confirmed_dgw_gws = {d["gameweek"] for d in confirmed_calendar.get("double_gameweeks", [])}

        prospective_blanks = []
        prospective_doubles = []

        for gw, info in sorted(self.PROSPECTIVE_CUP_MILESTONES.items()):
            if gw < current_gw:
                continue
            if info["type"] == "PROSPECTIVE_BLANK" and gw not in confirmed_bgw_gws:
                prospective_blanks.append(dict(info, gameweek=gw))
            elif info["type"] == "PROSPECTIVE_DOUBLE" and gw not in confirmed_dgw_gws:
                prospective_doubles.append(dict(info, gameweek=gw))

        return {
            "prospective_blank_gameweeks": prospective_blanks,
            "prospective_double_gameweeks": prospective_doubles,
            "bgw_target_gw": 29 if 29 >= current_gw else (next((b["gameweek"] for b in prospective_blanks), None)),
            "dgw_target_gw": 34 if 34 >= current_gw else (next((d["gameweek"] for d in prospective_doubles), None)),
        }


fixture_calendar = FixtureCalendar()
