"""
Domain Time Service for FPL Oracle.
Resolves active season, current gameweek, first still-actionable gameweek,
timezone-aware deadlines, gameweek phase, and clock abstraction for deterministic testing.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class GameweekPhase(StrEnum):
    PRE_DEADLINE = "PRE_DEADLINE"
    LIVE = "LIVE"
    BONUS_PENDING = "BONUS_PENDING"
    FINISHED = "FINISHED"
    BETWEEN_GWS = "BETWEEN_GWS"
    SEASON_OVER = "SEASON_OVER"


class Clock(Protocol):
    def now_utc(self) -> datetime:
        ...


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    def __init__(self, frozen_time: datetime):
        if frozen_time.tzinfo is None:
            self._time = frozen_time.replace(tzinfo=UTC)
        else:
            self._time = frozen_time.astimezone(UTC)

    def now_utc(self) -> datetime:
        return self._time

    def set_time(self, new_time: datetime):
        if new_time.tzinfo is None:
            self._time = new_time.replace(tzinfo=UTC)
        else:
            self._time = new_time.astimezone(UTC)


class TimeContext(BaseModel):
    season: str = "2026/27"
    current_gw: int | None = None
    next_actionable_gw: int | None = None
    phase: GameweekPhase = GameweekPhase.PRE_DEADLINE
    deadline_utc: str | None = None
    seconds_to_deadline: float | None = None
    is_live: bool = False
    bonus_added: bool = False
    leagues_updated: bool = False
    is_season_over: bool = False
    blank_gws: list[int] = Field(default_factory=list)
    double_gws: list[int] = Field(default_factory=list)
    postponed_fixtures_count: int = 0
    data_as_of_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TimeService:
    def __init__(self, clock: Clock | None = None):
        self.clock: Clock = clock or SystemClock()

    def resolve_time_context(
        self,
        bootstrap: Any,
        fixtures: list[Any],
        event_status_raw: dict[str, Any] | None = None,
    ) -> TimeContext:
        """
        Pure deterministic resolution of gameweek timing and phases from FPL API objects.
        """
        now = self.clock.now_utc()
        events = getattr(bootstrap, "events", []) if bootstrap else []
        event_status = event_status_raw or {"status": [], "leagues": "Updated"}

        curr_gw: int | None = None
        next_gw: int | None = None
        next_event = None

        for ev in events:
            if getattr(ev, "is_current", False):
                curr_gw = getattr(ev, "id", None)
            if getattr(ev, "is_next", False):
                next_gw = getattr(ev, "id", None)
                next_event = ev

        # If next_gw is not explicitly flagged by FPL API, inspect event progression
        if next_gw is None:
            if curr_gw is not None and curr_gw < 38:
                next_candidate = curr_gw + 1
                for ev in events:
                    if getattr(ev, "id", None) == next_candidate:
                        next_gw = next_candidate
                        next_event = ev
                        break
            elif curr_gw == 38:
                # End of season check
                gw38_fixtures = [f for f in fixtures if getattr(f, "event", None) == 38]
                if gw38_fixtures and all(getattr(f, "finished", False) for f in gw38_fixtures):
                    return TimeContext(
                        season="2026/27",
                        current_gw=38,
                        next_actionable_gw=None,
                        phase=GameweekPhase.SEASON_OVER,
                        deadline_utc=None,
                        seconds_to_deadline=None,
                        is_live=False,
                        bonus_added=True,
                        leagues_updated=True,
                        is_season_over=True,
                        data_as_of_utc=now.isoformat(),
                    )

        deadline_iso = None
        seconds_to_deadline = None

        if next_event and getattr(next_event, "deadline_time", None):
            dl_str = next_event.deadline_time
            deadline_iso = dl_str
            try:
                clean_time = dl_str.replace("Z", "+00:00")
                dl_dt = datetime.fromisoformat(clean_time)
                if dl_dt.tzinfo is None:
                    dl_dt = dl_dt.replace(tzinfo=UTC)
                else:
                    dl_dt = dl_dt.astimezone(UTC)
                seconds_to_deadline = (dl_dt - now).total_seconds()
            except Exception:
                pass

        # Live fixture check
        curr_fixtures = [f for f in fixtures if getattr(f, "event", None) == curr_gw]
        is_live = False
        if curr_fixtures:
            live_count = sum(1 for f in curr_fixtures if getattr(f, "started", False) and not getattr(f, "finished", False))
            if live_count > 0:
                is_live = True

        # Bonus & League finalization
        status_items = event_status.get("status", [])
        bonus_added = False
        leagues_updated = event_status.get("leagues") == "Updated"
        if status_items:
            # Must have at least one status item for curr_gw to avoid vacuous truth
            curr_items = [s for s in status_items if s.get("event") == curr_gw]
            if curr_items:
                bonus_added = all(s.get("bonus_added") is True and s.get("points") == "r" for s in curr_items)
            else:
                bonus_added = all(s.get("bonus_added") is True and s.get("points") == "r" for s in status_items)

        # GameweekPhase derivation
        phase = GameweekPhase.PRE_DEADLINE
        if is_live:
            phase = GameweekPhase.LIVE
        elif seconds_to_deadline is not None and seconds_to_deadline <= 0:
            phase = GameweekPhase.LIVE
        elif curr_gw and curr_fixtures and all(getattr(f, "finished", False) for f in curr_fixtures):
            if not bonus_added or not leagues_updated:
                phase = GameweekPhase.BONUS_PENDING
            else:
                if seconds_to_deadline is not None and seconds_to_deadline > 86400 * 2:
                    phase = GameweekPhase.BETWEEN_GWS
                else:
                    phase = GameweekPhase.PRE_DEADLINE
        elif seconds_to_deadline is not None and seconds_to_deadline > 0:
            phase = GameweekPhase.PRE_DEADLINE
        else:
            phase = GameweekPhase.BETWEEN_GWS

        # Blank & Double GW detection
        gw_team_counts: dict[int, dict[int, int]] = {}
        postponed_count = 0
        for f in fixtures:
            ev_id = getattr(f, "event", None)
            if ev_id is None:
                postponed_count += 1
                continue
            th = getattr(f, "team_h", None)
            ta = getattr(f, "team_a", None)
            if ev_id not in gw_team_counts:
                gw_team_counts[ev_id] = {}
            if th:
                gw_team_counts[ev_id][th] = gw_team_counts[ev_id].get(th, 0) + 1
            if ta:
                gw_team_counts[ev_id][ta] = gw_team_counts[ev_id].get(ta, 0) + 1

        blank_gws: list[int] = []
        double_gws: list[int] = []
        for gw in range(1, 39):
            teams_in_gw = gw_team_counts.get(gw, {})
            if len(teams_in_gw) < 20 and gw >= (curr_gw or 1):
                blank_gws.append(gw)
            if any(cnt >= 2 for cnt in teams_in_gw.values()):
                double_gws.append(gw)

        return TimeContext(
            season="2026/27",
            current_gw=curr_gw,
            next_actionable_gw=next_gw,
            phase=phase,
            deadline_utc=deadline_iso,
            seconds_to_deadline=round(seconds_to_deadline, 1) if seconds_to_deadline is not None else None,
            is_live=is_live,
            bonus_added=bonus_added,
            leagues_updated=leagues_updated,
            is_season_over=False,
            blank_gws=sorted(list(set(blank_gws))),
            double_gws=sorted(list(set(double_gws))),
            postponed_fixtures_count=postponed_count,
            data_as_of_utc=now.isoformat(),
        )


time_service = TimeService()
