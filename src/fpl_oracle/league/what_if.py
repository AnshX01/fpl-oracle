"""
Live Matchday 'What-If' Scenario Engine.
Allows managers to test hypothetical match outcomes (goals, assists, clean sheets, bonus points)
and instantaneously observe the projected impact on live scores, rival point deltas,
mini-league rank movement, and title gaps.
Zero hardcoded player names in production code.
"""

from dataclasses import dataclass, field
from typing import Any

from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.league.live_matchday import LiveMatchdaySnapshot


@dataclass(frozen=True)
class HypotheticalEvent:
    element_id: int
    goals: int = 0
    assists: int = 0
    clean_sheets: int = 0
    bonus: int = 0
    goals_conceded: int = 0
    yellow_cards: int = 0
    red_cards: int = 0


@dataclass
class ManagerWhatIfResult:
    manager_id: int
    manager_name: str
    team_name: str
    previous_rank: int
    new_rank: int
    rank_delta: int
    original_points: int
    simulated_points: int
    points_delta: int
    is_user: bool = False


@dataclass
class WhatIfScenarioResult:
    gameweek: int
    league_id: int
    league_name: str
    user_manager_id: int
    user_rank_before: int
    user_rank_after: int
    user_rank_delta: int  # e.g. +1 if moved up from 2 to 1
    user_points_delta: int
    gap_to_leader_before: int
    gap_to_leader_after: int
    summary_narrative: str
    standings: list[ManagerWhatIfResult]
    hypothetical_events: list[dict[str, Any]] = field(default_factory=list)


def calculate_points_from_events(
    position: str,
    event: HypotheticalEvent,
) -> int:
    """
    Computes standard FPL points from match actions based on player position.
    """
    pos = position.upper()
    points = 0

    # Goals
    if pos in ("GKP", "DEF"):
        points += event.goals * 6
    elif pos == "MID":
        points += event.goals * 5
    elif pos == "FWD":
        points += event.goals * 4

    # Assists (all positions 3 pts)
    points += event.assists * 3

    # Clean sheets
    if pos in ("GKP", "DEF"):
        points += event.clean_sheets * 4
    elif pos == "MID":
        points += event.clean_sheets * 1

    # Bonus points
    points += event.bonus

    # Deductions
    points -= event.yellow_cards * 1
    points -= event.red_cards * 3
    if pos in ("GKP", "DEF") and event.goals_conceded >= 2:
        points -= event.goals_conceded // 2

    return points


class WhatIfEngine:
    """Simulates real-time score and rank deltas from hypothetical player events."""

    def simulate_events(
        self,
        snapshot: LiveMatchdaySnapshot,
        events: list[HypotheticalEvent],
        bootstrap: BootstrapStatic,
    ) -> WhatIfScenarioResult:
        if not snapshot or not snapshot.standings:
            raise ValueError("LiveMatchdaySnapshot contains no standings.")

        elem_map = {e.id: e for e in bootstrap.elements}
        pos_map = {t.id: t.singular_name_short for t in getattr(bootstrap, "element_types", [])}

        # Index event points by element_id
        event_points_by_elem: dict[int, int] = {}
        for ev in events:
            elem = elem_map.get(ev.element_id)
            if not elem:
                continue
            pos = pos_map.get(elem.element_type, "MID")
            pts = calculate_points_from_events(pos, ev)
            event_points_by_elem[ev.element_id] = pts

        # Score adjustments per manager
        # If user squad is present in snapshot, match starters
        manager_deltas: dict[int, int] = {}

        # If full snapshot squads are available, compute exact multiplier-weighted gains
        for row in snapshot.standings:
            mid = row.manager_id
            delta = 0

            # If this is user and we have user_squad object
            if mid == snapshot.user_manager_id and snapshot.user_squad:
                for s in snapshot.user_squad.starters:
                    if s.element_id in event_points_by_elem:
                        base = event_points_by_elem[s.element_id]
                        mult = 2 if s.element_id == snapshot.user_squad.effective_captain_id else 1
                        if (
                            snapshot.user_squad.active_chip in ("3xc", "triple_captain")
                            and s.element_id == snapshot.user_squad.effective_captain_id
                        ):
                            mult = 3
                        delta += base * mult
            else:
                # If captain matches an event player
                for ev_id, base in event_points_by_elem.items():
                    elem = elem_map.get(ev_id)
                    elem_name = elem.web_name if elem else ""
                    if row.captain_name and elem_name and row.captain_name.lower() in elem_name.lower():
                        mult = 3 if row.active_chip in ("3xc", "triple_captain") else 2
                        delta += base * mult

            manager_deltas[mid] = delta

        # Re-compute total points and sort
        simulated_rows: list[ManagerWhatIfResult] = []
        for row in snapshot.standings:
            mid = row.manager_id
            orig_pts = row.live_total_points
            delta = manager_deltas.get(mid, 0)
            sim_pts = orig_pts + delta
            is_u = bool(mid == snapshot.user_manager_id)

            simulated_rows.append(
                ManagerWhatIfResult(
                    manager_id=mid,
                    manager_name=row.manager_name,
                    team_name=row.team_name,
                    previous_rank=row.live_rank,
                    new_rank=row.live_rank,  # will update after sort
                    rank_delta=0,
                    original_points=orig_pts,
                    simulated_points=sim_pts,
                    points_delta=delta,
                    is_user=is_u,
                )
            )

        # Sort descending by simulated total points, preserving previous rank as tiebreak
        simulated_rows.sort(key=lambda r: (-r.simulated_points, r.previous_rank))

        # Assign new ranks
        user_before = snapshot.user_live_rank or 1
        user_after = user_before
        user_delta_pts = 0

        for idx, r in enumerate(simulated_rows, start=1):
            r.new_rank = idx
            r.rank_delta = r.previous_rank - r.new_rank  # positive means moved up (e.g. 2 -> 1 is +1)
            if r.is_user:
                user_after = r.new_rank
                user_delta_pts = r.points_delta

        user_rank_change = user_before - user_after
        orig_leader_pts = simulated_rows[0].original_points if simulated_rows else 0
        user_orig_pts = snapshot.user_live_total or 0
        gap_before = max(0, orig_leader_pts - user_orig_pts)

        new_leader_pts = simulated_rows[0].simulated_points if simulated_rows else 0
        user_sim_pts = user_orig_pts + user_delta_pts
        gap_after = max(0, new_leader_pts - user_sim_pts)

        # Narrative generation
        if user_after < user_before:
            narrative = (
                f"Hypothetical event gains you +{user_delta_pts} points, elevating your rank from "
                f"#{user_before} to #{user_after} in {snapshot.league_name}! "
                f"Gap to 1st place reduced from {gap_before} to {gap_after} pts."
            )
        elif user_after > user_before:
            narrative = f"Rival event gains push you from #{user_before} to #{user_after} in {snapshot.league_name}."
        else:
            narrative = (
                f"Net score change of +{user_delta_pts} points maintains your position at #{user_before} "
                f"in {snapshot.league_name} (gap to leader: {gap_after} pts)."
            )

        return WhatIfScenarioResult(
            gameweek=snapshot.gameweek,
            league_id=snapshot.league_id,
            league_name=snapshot.league_name,
            user_manager_id=snapshot.user_manager_id or 0,
            user_rank_before=user_before,
            user_rank_after=user_after,
            user_rank_delta=user_rank_change,
            user_points_delta=user_delta_pts,
            gap_to_leader_before=gap_before,
            gap_to_leader_after=gap_after,
            summary_narrative=narrative,
            standings=simulated_rows,
            hypothetical_events=[
                {
                    "element": ev.element_id,
                    "goals": ev.goals,
                    "assists": ev.assists,
                    "clean_sheets": ev.clean_sheets,
                    "bonus": ev.bonus,
                }
                for ev in events
            ],
        )


what_if_engine = WhatIfEngine()
