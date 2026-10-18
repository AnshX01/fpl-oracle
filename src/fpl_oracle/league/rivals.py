"""
Rival Intelligence & League Ownership Analyzer.

Features:
- Rival selection: all managers above + managers below who can still finish above us (reach_window) (R1)
- Full pagination support through mini-league standings (R1)
- Published rival details: Effective Ownership (EO), net exposure vs user squad, differentials (R2)
- 2026/27 chip tracking: remaining chips, used chips, and Set 1 GW19 expiration rules (R2)
- Observed vs estimated picks distinction: marks upcoming deadline picks unknown (R3)
"""

import asyncio
import logging
import math
from collections import defaultdict
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.config import RIVAL_POINTS_WINDOW

logger = logging.getLogger("fpl_oracle.league.rivals")


# Spread of the points gap between two managers per remaining gameweek. It matches the planner's own
# win-probability variance (docs/decision_model.md section 4: GW8 2026-27, 71 points over 31 weeks).
GAP_SD_PER_GW = 13.0
REACH_Z = 1.645  # a manager below us is tracked while they have at least a 5% chance to finish above us


def reach_window(target_gw: int) -> int:
    """Points behind us that a manager can still make up with 5% probability over the weeks left."""
    weeks_left = max(1, 38 - int(target_gw) + 1)
    return math.ceil(REACH_Z * GAP_SD_PER_GW * math.sqrt(weeks_left))


def get_rival_set(
    standings: list[dict[str, Any]],
    user_manager_id: int | None,
    points_window: int | None = None,
) -> tuple[list[dict[str, Any]], str, int | None]:
    """
    Unified single source of truth for proximity rival selection (Requirement G8):
    - Rivals = every manager ranked above user + every manager within points_window below.
    - If user is first, the set is only those within points_window below, even if empty
      (then status 'leader, no close chasers'), NEVER forced chasers.
    - If manager is unknown or not found in standings, returns explicit 'unavailable' status.
    """
    if points_window is None:
        points_window = RIVAL_POINTS_WINDOW

    if not standings:
        return [], "EMPTY_STANDINGS", None

    if user_manager_id is None:
        return [], "unavailable", None

    # Find user entry
    user_idx = None
    for idx, entry in enumerate(standings):
        if entry.get("entry") == user_manager_id:
            user_idx = idx
            break

    if user_idx is None:
        return [], "unavailable", None

    user_entry = standings[user_idx]
    user_rank = user_entry.get("rank", user_idx + 1)
    user_pts = float(user_entry.get("total", 0))

    # All managers above user (no cap)
    above = [s for s in standings[:user_idx] if s.get("entry") != user_manager_id]
    # Managers below user within exact points window (no cap)
    below = [
        s
        for s in standings[user_idx + 1 :]
        if s.get("entry") != user_manager_id and (user_pts - float(s.get("total", 0))) <= points_window
    ]

    # If user is in 1st place, select ONLY chasers within points_window (plus any manager tied on rank 1,
    # who is listed in `above`). If none, return status "leader, no close chasers", NEVER forced 10 chasers.
    if user_rank == 1:
        chasers = above + below
        if not chasers:
            return [], "leader, no close chasers", user_rank
        return chasers, "PROXIMITY_WINDOW", user_rank

    selected = above + below
    if not selected:
        return [], "no_rivals_in_window", user_rank

    return selected, "PROXIMITY_WINDOW", user_rank


def attach_squad_overlap(standings, rivals, user_squad):
    """Observed released squad similarity, not multiplier-weighted effective ownership."""
    by_id = {r["entry_id"]: r for r in rivals}
    owned = {p.element for p in user_squad}
    rows = []
    for original in standings:
        row = dict(original)
        rival = by_id.get(row.get("entry"))
        observed = {p["element"] for p in rival.get("squad", [])} if rival else set()
        if len(owned) == 15 and len(observed) == 15:
            shared = len(owned & observed)
            row.update(
                squad_overlap_pct=round(shared / 15 * 100, 1),
                shared_players=shared,
                compared_squad_size=15,
                overlap_gameweek=rival.get("observed_gameweek"),
            )
        rows.append(row)
    return rows


class RivalAnalyzer:
    def __init__(self):
        pass

    def select_proximity_rivals(
        self,
        standings: list[dict[str, Any]],
        user_manager_id: int | None,
        points_window: int = RIVAL_POINTS_WINDOW,
        max_rivals: int | None = None,
    ) -> tuple[list[dict[str, Any]], str, int | None]:
        """
        Selects rivals using get_rival_set.
        """
        selected, mode, rank = get_rival_set(
            standings=standings,
            user_manager_id=user_manager_id,
            points_window=points_window,
        )
        if max_rivals is not None and len(selected) > max_rivals:
            selected = selected[:max_rivals]
        return selected, mode, rank

    def calculate_chips_status(self, chips_used_list: list[dict[str, Any]], current_gw: int) -> dict[str, Any]:
        """
        Remaining chips for the set currently in play (set 1: GW1-19, set 2: GW20-38). A chip burned in the
        other set does not count, and an unused set 1 chip is gone after GW19.
        """
        in_set_1 = current_gw <= 19
        aliases = {"bboost": "bench_boost", "3xc": "triple_captain"}
        used_now = {
            aliases.get(c.get("name", ""), c.get("name", ""))
            for c in chips_used_list
            if (int(c.get("event") or 0) <= 19) == in_set_1
        }
        wc_set1_used = any(c.get("name") == "wildcard" and int(c.get("event") or 0) <= 19 for c in chips_used_list)

        remaining = []
        for chip in ("wildcard", "freehit", "bench_boost", "triple_captain"):
            if chip not in used_now:
                remaining.append(f"wildcard_set{1 if in_set_1 else 2}" if chip == "wildcard" else chip)

        return {
            "chips_used": chips_used_list,
            "chips_remaining": remaining,
            "set1_wildcard_expired": (not wc_set1_used) and (current_gw > 19),
        }

    async def analyze_rivals(
        self,
        standings: list[dict[str, Any]],
        user_manager_id: int | None,
        current_gw: int,
        bootstrap: BootstrapStatic,
        max_rivals_to_inspect: int | None = None,
        user_squad_df: pd.DataFrame | None = None,
        points_window: int | None = None,
    ) -> dict[str, Any]:
        """
        Extract squad compositions for proximity rivals and compute effective ownership & exposure.
        """
        elem_map = {e.id: e for e in bootstrap.elements}
        active_window = points_window or RIVAL_POINTS_WINDOW or reach_window(current_gw)

        selected_entries, selection_mode, user_rank = self.select_proximity_rivals(
            standings=standings,
            user_manager_id=user_manager_id,
            points_window=active_window,
            max_rivals=max_rivals_to_inspect,
        )

        rival_squads = []
        stale_flags = []
        failed_entries: list[int] = []
        player_multipliers: dict[int, float] = defaultdict(float)
        player_started_counts: dict[int, int] = defaultdict(int)
        player_captained_counts: dict[int, int] = defaultdict(int)

        sem = asyncio.Semaphore(5)

        # Use official released-event state, not caller-specific current/target
        # arithmetic. After GW5 is finalized but GW6 has not started, every
        # surface must inspect GW5, never GW4 or private GW6 picks.
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        released = []
        for event in bootstrap.events:
            try:
                deadline = datetime.fromisoformat(event.deadline_time.replace("Z", "+00:00"))
                if event.finished or deadline <= now:
                    released.append(event.id)
            except (ValueError, TypeError):
                if event.finished:
                    released.append(event.id)
        eval_gw = max(released) if released else max(1, current_gw - 1)
        target_gw = next((e.id for e in bootstrap.events if e.is_next), current_gw)

        async def fetch_rival_details(entry_dict):
            entry_id = entry_dict["entry"]
            async with sem:
                try:
                    picks_resp, picks_stale = await fpl_client.get_manager_picks(entry_id, eval_gw)
                    observed_gw = eval_gw
                    observed_chip = getattr(picks_resp, "active_chip", None)
                    if observed_chip == "freehit" and eval_gw > 1:
                        # A Free Hit squad reverts next week: the squad that carries forward is last week's.
                        picks_resp, picks_stale = await fpl_client.get_manager_picks(entry_id, eval_gw - 1)
                        observed_gw = eval_gw - 1
                    history_resp, history_stale = await fpl_client.get_manager_history(entry_id)
                    stale_flags.append(picks_stale or history_stale)

                    squad_elements = []
                    cap_elem = None
                    for p in picks_resp.picks:
                        elem_id = p.element
                        mult = p.multiplier
                        is_cap = bool(p.is_captain)
                        is_start = mult > 0 and p.position <= 11  # Bench Boost gives bench players a multiplier
                        if is_cap:
                            cap_elem = elem_id
                            player_captained_counts[elem_id] += 1
                        if is_start:
                            player_started_counts[elem_id] += 1

                        player_multipliers[elem_id] += mult
                        elem_obj = elem_map.get(elem_id)
                        squad_elements.append(
                            {
                                "element": elem_id,
                                "web_name": elem_obj.web_name if elem_obj else f"Player {elem_id}",
                                "position": p.position,
                                "multiplier": mult,
                                "is_captain": is_cap,
                                "is_starter": is_start,
                            }
                        )

                    burned_chips = [
                        {"name": getattr(c, "name", ""), "event": getattr(c, "event", 0)}
                        for c in (getattr(history_resp, "chips", []) if history_resp else [])
                    ]
                    chip_status = self.calculate_chips_status(burned_chips, current_gw)

                    rival_squads.append(
                        {
                            "entry_id": entry_id,
                            "player_name": entry_dict.get("player_name", ""),
                            "entry_name": entry_dict.get("entry_name", ""),
                            "rank": entry_dict.get("rank", 0),
                            "total_points": entry_dict.get("total", 0),
                            "captain_element": cap_elem,
                            "chips_used": [c["name"] for c in burned_chips],
                            "chips_remaining": chip_status["chips_remaining"],
                            "set1_wildcard_expired": chip_status["set1_wildcard_expired"],
                            "picks_status": "OBSERVED_PRIOR_GW",
                            "observed_gameweek": observed_gw,
                            "released_gameweek_chip": observed_chip,
                            "upcoming_transfers_known": False,
                            "transfers_visibility_note": (
                                f"Rival transfers for Gameweek {target_gw} are private until the deadline. "
                                f"Picks reflect confirmed lineups from Gameweek {observed_gw}."
                            ),
                            "squad": squad_elements,
                        }
                    )
                except Exception as ex:
                    failed_entries.append(entry_id)
                    logger.warning("Could not fetch rival %s: %s", entry_id, ex)

        tasks = [fetch_rival_details(e) for e in selected_entries]
        if tasks:
            await asyncio.gather(*tasks)

        n_rivals = len(rival_squads)
        if n_rivals == 0:
            return {
                "rivals_analyzed_count": 0,
                "rivals_requested": len(selected_entries),
                "rivals_failed": failed_entries,
                "rival_selection_mode": selection_mode,
                "rival_points_window": active_window,
                "user_rank_in_league": user_rank,
                "rival_squads": [],
                "league_effective_ownership": [],
                "template_players": [],
                "differential_players": [],
            }

        # Build user ownership lookup if user squad provided
        user_eo_map: dict[int, float] = defaultdict(float)
        if user_squad_df is not None and not user_squad_df.empty:
            for _, r in user_squad_df.iterrows():
                eid = int(r["element"])
                is_start = bool(r.get("is_starter", True))
                is_cap = bool(r.get("is_captain", False))
                if is_start:
                    user_eo_map[eid] = 200.0 if is_cap else 100.0

        # Calculate Effective Ownership (EO) % and Net Exposure
        eo_records = []
        for elem_id, total_mult in player_multipliers.items():
            elem_obj = elem_map.get(elem_id)
            if not elem_obj:
                continue
            eo_pct = round((total_mult / n_rivals) * 100.0, 1)
            started_pct = round((player_started_counts[elem_id] / n_rivals) * 100.0, 1)
            captained_pct = round((player_captained_counts[elem_id] / n_rivals) * 100.0, 1)
            user_eo = user_eo_map.get(elem_id, 0.0)
            net_exposure = round(eo_pct - user_eo, 1)

            eo_records.append(
                {
                    "element": elem_id,
                    "web_name": elem_obj.web_name,
                    "team": elem_obj.team,
                    "effective_ownership": eo_pct,
                    "started_percent": started_pct,
                    "captained_percent": captained_pct,
                    "user_ownership": user_eo,
                    "net_exposure": net_exposure,
                    "total_multiplier": total_mult,
                }
            )

        eo_df = (
            pd.DataFrame(eo_records).sort_values(by="effective_ownership", ascending=False)
            if eo_records
            else pd.DataFrame()
        )

        template = eo_df[eo_df["effective_ownership"] >= 50.0].to_dict(orient="records") if not eo_df.empty else []
        differentials = (
            eo_df[(eo_df["effective_ownership"] > 0) & (eo_df["effective_ownership"] <= 20.0)]
            .head(10)
            .to_dict(orient="records")
            if not eo_df.empty
            else []
        )

        # Sort rival squads by rank
        rival_squads.sort(key=lambda x: x["rank"])

        return {
            "rivals_analyzed_count": n_rivals,
            "rivals_requested": len(selected_entries),
            "rivals_failed": failed_entries,
            "rival_selection_mode": selection_mode,
            "rival_points_window": active_window,
            "user_rank_in_league": user_rank,
            "rival_squads": rival_squads,
            "is_stale": any(stale_flags),
            "league_effective_ownership": eo_df.to_dict(orient="records") if not eo_df.empty else [],
            "template_players": template,
            "differential_players": differentials,
        }


rival_analyzer = RivalAnalyzer()
