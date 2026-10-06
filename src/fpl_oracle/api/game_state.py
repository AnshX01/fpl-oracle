"""
Gameweek State Machine for FPL Oracle.
Derives live gameweek status:
- PRE_DEADLINE: Upcoming deadline, matches haven't started
- LIVE: Matches currently underway
- BONUS_PENDING: Matches finished for the day/round, awaiting bonus finalisation
- FINISHED: Round completed, bonus and league tables finalised
- BETWEEN_GWS: Rest interval between finished GW and next GW pre-deadline

Also detects Blank Gameweeks (BGW), Double Gameweeks (DGW), and postponed fixtures.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GameweekPhase(StrEnum):
    PRE_DEADLINE = "PRE_DEADLINE"
    LIVE = "LIVE"
    BONUS_PENDING = "BONUS_PENDING"
    FINISHED = "FINISHED"
    BETWEEN_GWS = "BETWEEN_GWS"


class GameState(BaseModel):
    season: str = "2026/27"
    current_gw: int | None = None
    next_gw: int | None = None
    phase: GameweekPhase = GameweekPhase.PRE_DEADLINE
    deadline_time: str | None = None
    seconds_to_deadline: float | None = None
    is_live: bool = False
    bonus_added: bool = False
    leagues_updated: bool = False
    blank_gws: list[int] = Field(default_factory=list)
    double_gws: list[int] = Field(default_factory=list)
    postponed_fixtures_count: int = 0
    data_as_of: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    stale: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class GameStateManager:
    """Derives and caches the shared GameState object."""

    def __init__(self):
        self._cached_state: GameState | None = None
        self._cached_at: float | None = None

    async def get_game_state(self, force_refresh: bool = False) -> GameState:
        import time

        now = time.time()
        # Cache for 60 seconds unless forced
        if not force_refresh and self._cached_state is not None and self._cached_at and (now - self._cached_at < 60):
            return self._cached_state

        from fpl_oracle.api.fpl_client import fpl_client

        bootstrap, is_stale = await fpl_client.get_bootstrap_static(force_refresh=force_refresh)
        fixtures, fix_stale = await fpl_client.get_fixtures(force_refresh=force_refresh)

        # Safe event-status check
        event_status_raw: dict[str, Any] = {}
        try:
            event_status_raw, _ = await fpl_client.get_event_status()
        except Exception:
            event_status_raw = {"status": [], "leagues": "Updated"}

        curr_gw = None
        next_gw = None
        next_event = None

        for ev in bootstrap.events:
            if ev.is_current:
                curr_gw = ev.id
            if ev.is_next:
                next_gw = ev.id
                next_event = ev

        if next_gw is None and curr_gw is not None and curr_gw < 38:
            next_gw = curr_gw + 1
            for ev in bootstrap.events:
                if ev.id == next_gw:
                    next_event = ev
                    break

        now_utc = datetime.now(UTC)
        deadline_iso = None
        seconds_to_deadline = None

        if next_event and next_event.deadline_time:
            deadline_iso = next_event.deadline_time
            try:
                # Parse ISO deadline string
                clean_time = next_event.deadline_time.replace("Z", "+00:00")
                dl_dt = datetime.fromisoformat(clean_time)
                seconds_to_deadline = max(0.0, (dl_dt - now_utc).total_seconds())
            except Exception:
                pass

        # Check live match state for curr_gw
        is_live = False
        curr_fixtures = [f for f in fixtures if f.event == curr_gw]
        if curr_fixtures:
            live_count = sum(1 for f in curr_fixtures if f.started and not f.finished)
            if live_count > 0:
                is_live = True

        # Check bonus and league status
        bonus_added = False
        leagues_updated = False
        status_items = event_status_raw.get("status", [])
        if status_items:
            # Check if all status items have bonus added and points finalised
            all_bonus = all(item.get("bonus_added") is True for item in status_items)
            all_points = all(item.get("points") == "r" for item in status_items)
            bonus_added = all_bonus and all_points
        if event_status_raw.get("leagues") == "Updated":
            leagues_updated = True

        # Determine GameweekPhase
        phase = GameweekPhase.PRE_DEADLINE
        if is_live:
            phase = GameweekPhase.LIVE
        elif curr_gw and curr_fixtures and all(f.finished for f in curr_fixtures):
            if not bonus_added or not leagues_updated:
                phase = GameweekPhase.BONUS_PENDING
            else:
                if seconds_to_deadline and seconds_to_deadline > 86400 * 2:
                    phase = GameweekPhase.BETWEEN_GWS
                else:
                    phase = GameweekPhase.PRE_DEADLINE
        elif seconds_to_deadline is not None and seconds_to_deadline > 0:
            phase = GameweekPhase.PRE_DEADLINE
        else:
            phase = GameweekPhase.BETWEEN_GWS

        # Blank & Double Gameweek detection across all fixtures
        gw_team_counts: dict[int, dict[int, int]] = {}
        postponed_count = 0

        for f in fixtures:
            if f.event is None:
                postponed_count += 1
                continue
            if f.event not in gw_team_counts:
                gw_team_counts[f.event] = {}
            gw_team_counts[f.event][f.team_h] = gw_team_counts[f.event].get(f.team_h, 0) + 1
            gw_team_counts[f.event][f.team_a] = gw_team_counts[f.event].get(f.team_a, 0) + 1

        blank_gws = []
        double_gws = []
        for gw in range(1, 39):
            teams_in_gw = gw_team_counts.get(gw, {})
            # If fewer than 20 teams play or any team plays 0
            if len(teams_in_gw) < 20 and gw >= (curr_gw or 1):
                blank_gws.append(gw)
            # If any team plays 2 or more fixtures
            if any(count >= 2 for count in teams_in_gw.values()):
                double_gws.append(gw)

        state = GameState(
            season="2026/27",
            current_gw=curr_gw,
            next_gw=next_gw,
            phase=phase,
            deadline_time=deadline_iso,
            seconds_to_deadline=round(seconds_to_deadline, 1) if seconds_to_deadline is not None else None,
            is_live=is_live,
            bonus_added=bonus_added,
            leagues_updated=leagues_updated,
            blank_gws=sorted(list(set(blank_gws))),
            double_gws=sorted(list(set(double_gws))),
            postponed_fixtures_count=postponed_count,
            data_as_of=now_utc.isoformat(),
            stale=is_stale or fix_stale,
            details={
                "curr_fixtures_count": len(curr_fixtures),
                "event_status_leagues": event_status_raw.get("leagues", "Unknown"),
            },
        )

        self._cached_state = state
        self._cached_at = now
        return state


game_state_manager = GameStateManager()
