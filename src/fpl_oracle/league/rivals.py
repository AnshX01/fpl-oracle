"""
Rival Intelligence & League Ownership Analyzer.

Features:
- Proximity-based rival selection (all managers above + managers within RIVAL_POINTS_WINDOW below) (R1)
- Full pagination support through mini-league standings (R1)
- Published rival details: Effective Ownership (EO), net exposure vs user squad, differentials (R2)
- 2026/27 chip tracking: remaining chips, used chips, and Set 1 GW19 expiration rules (R2)
- Observed vs estimated picks distinction: marks upcoming deadline picks unknown (R3)
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import BootstrapStatic
from fpl_oracle.config import RIVAL_POINTS_WINDOW

logger = logging.getLogger("fpl_oracle.league.rivals")


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

    # If user is in 1st place, select ONLY chasers within points_window.
    # If none in window, return status "leader, no close chasers", NEVER forced 10 chasers.
    if user_rank == 1:
        if not below:
            return [], "leader, no close chasers", user_rank
        return below, "PROXIMITY_WINDOW", user_rank

    selected = above + below
    if not selected:
        return [], "no_rivals_in_window", user_rank

    return selected, "PROXIMITY_WINDOW", user_rank


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
        Computes remaining, used, and expired chips according to 2026/27 rules.
        """
        used_names = [c.get("name", "") for c in chips_used_list]

        # 2026/27 rules
        wc1_used = "wildcard" in used_names
        wc1_expired = (not wc1_used) and (current_gw > 19)

        remaining = []
        if not wc1_used and current_gw <= 19:
            remaining.append("wildcard_set1")
        if current_gw >= 20:
            remaining.append("wildcard_set2")

        if "freehit" not in used_names:
            remaining.append("freehit")
        if "bboost" not in used_names and "bench_boost" not in used_names:
            remaining.append("bench_boost")
        if "3xc" not in used_names and "triple_captain" not in used_names:
            remaining.append("triple_captain")

        return {
            "chips_used": chips_used_list,
            "chips_remaining": remaining,
            "set1_wildcard_expired": wc1_expired,
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
        active_window = points_window if points_window is not None else RIVAL_POINTS_WINDOW

        selected_entries, selection_mode, user_rank = self.select_proximity_rivals(
            standings=standings,
            user_manager_id=user_manager_id,
            points_window=active_window,
            max_rivals=max_rivals_to_inspect,
        )

        rival_squads = []
        stale_flags = []
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
                    history_resp, history_stale = await fpl_client.get_manager_history(entry_id)
                    stale_flags.append(picks_stale or history_stale)

                    squad_elements = []
                    cap_elem = None
                    for p in picks_resp.picks:
                        elem_id = p.element
                        mult = p.multiplier
                        is_cap = bool(p.is_captain)
                        is_start = bool(mult > 0)
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
                            "observed_gameweek": eval_gw,
                            "upcoming_transfers_known": False,
                            "transfers_visibility_note": (
                                f"Rival transfers for Gameweek {target_gw} are private until the deadline. "
                                f"Picks reflect confirmed lineups from Gameweek {eval_gw}."
                            ),
                            "squad": squad_elements,
                        }
                    )
                except Exception as ex:
                    logger.debug(f"Could not fetch rival {entry_id}: {ex}")

        tasks = [fetch_rival_details(e) for e in selected_entries]
        if tasks:
            await asyncio.gather(*tasks)

        n_rivals = len(rival_squads)
        if n_rivals == 0:
            return {
                "rivals_analyzed_count": 0,
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
            "rival_selection_mode": selection_mode,
            "rival_points_window": RIVAL_POINTS_WINDOW,
            "user_rank_in_league": user_rank,
            "rival_squads": rival_squads,
            "is_stale": any(stale_flags),
            "league_effective_ownership": eo_df.to_dict(orient="records") if not eo_df.empty else [],
            "template_players": template,
            "differential_players": differentials,
        }


rival_analyzer = RivalAnalyzer()
