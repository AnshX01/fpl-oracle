"""
Domain Model & Service for EffectiveManagerState.
Single source of truth for personal squad, bank, prices, free transfers, chips,
lineups, and manual overrides across all endpoints, tools, and optimizers.
"""

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("fpl_oracle.domain.manager_state")


class ManagerMode(StrEnum):
    REAL = "real"
    MANUAL = "manual"
    DEMO = "demo"


class PriceProvenance(StrEnum):
    CONFIRMED_TRANSFER = "confirmed_transfer"
    INITIAL_SQUAD = "initial_squad"
    MANUAL_OVERRIDE = "manual_override"
    MARKET_ESTIMATE = "market_estimate"


class PlayerSquadState(BaseModel):
    element: int
    web_name: str
    team: int
    position: str  # "GKP", "DEF", "MID", "FWD"
    now_cost: int  # Tenths (e.g., 55 = £5.5m)
    purchase_price: int  # Tenths
    selling_price: int  # Tenths
    price_provenance: PriceProvenance = PriceProvenance.MARKET_ESTIMATE
    is_starter: bool = True
    is_captain: bool = False
    is_vice_captain: bool = False
    multiplier: int = 1
    bench_order: int = 0  # 0 for starters, 1-4 for bench
    chance_of_playing: float | None = 100.0
    status: str = "a"
    news: str = ""
    expected_points: float = 0.0


class PublishedLineup(BaseModel):
    starters: list[int] = Field(default_factory=list)
    bench: list[int] = Field(default_factory=list)
    captain: int | None = None
    vice_captain: int | None = None
    active_chip: str | None = None


class EffectiveManagerState(BaseModel):
    manager_id: int | None = None
    target_league_id: int | None = None
    mode: ManagerMode = ManagerMode.REAL
    manager_name: str = "Manager"
    team_name: str = "My Squad"
    overall_points: int = 0
    overall_rank: int | None = None

    # 15 players
    squad: list[PlayerSquadState] = Field(default_factory=list)

    # Lineups
    published_lineup: PublishedLineup = Field(default_factory=PublishedLineup)
    recommended_lineup: dict[str, Any] | None = None

    # Canonical financial & transfer state (in tenths: 15 = £1.5m)
    bank_tenths: int = 0
    bank_source: str = "fpl_api"  # "fpl_api", "manual_override", "default"
    has_bank_override: bool = False

    free_transfers: int = 1
    ft_source: str = "fpl_api_calculated"  # "fpl_api_calculated", "manual_override", "default"
    has_ft_override: bool = False

    transfers_made_current_gw: int = 0
    hit_cost_rate: int = 4

    # Chips
    chips_used: list[dict[str, Any]] = Field(default_factory=list)
    chips_remaining_set_1: list[str] = Field(default_factory=list)
    chips_remaining_set_2: list[str] = Field(default_factory=list)
    active_chip: str | None = None

    # Gameweek context
    current_gw: int = 1
    target_gw: int = 1
    deadline_utc: str | None = None

    # Provenance & Freshness
    source_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    is_stale: bool = False
    confidence: str = "high"  # "high", "estimated", "manual"
    error_message: str | None = None
    confirmation_required: bool = False
    team_confirmed: bool = False
    confirmation_as_of: str | None = None
    confirmation_comparison: str | None = None

    @property
    def bank_millions(self) -> float:
        return round(self.bank_tenths / 10.0, 1)

    @property
    def total_team_value_tenths(self) -> int:
        return sum(p.selling_price for p in self.squad) + self.bank_tenths

    @property
    def total_team_value_millions(self) -> float:
        return round(self.total_team_value_tenths / 10.0, 1)

    def to_squad_dataframe(self) -> pd.DataFrame:
        """Converts squad to pandas DataFrame format required by optimizers."""
        if not self.squad:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "element": p.element,
                    "web_name": p.web_name,
                    "team": p.team,
                    "position": p.position,
                    "value": p.now_cost,
                    "now_cost": p.now_cost,
                    "purchase_price": p.purchase_price,
                    "selling_price": p.selling_price,
                    "selling_price_millions": round(p.selling_price / 10.0, 1),
                    "is_starter": p.is_starter,
                    "is_captain": p.is_captain,
                    "is_vice_captain": p.is_vice_captain,
                    "multiplier": p.multiplier,
                    "bench_order": p.bench_order,
                    "chance_of_playing": p.chance_of_playing,
                    "status": p.status,
                    "news": p.news,
                    "expected_points": p.expected_points,
                }
                for p in self.squad
            ]
        )


class ManagerStateService:
    @staticmethod
    def calculate_selling_price(purchase_price: int, now_cost: int) -> int:
        """
        Official FPL Selling Price rule:
        retains 50% of price rise rounded down.
        selling_price = purchase_price + floor((now_cost - purchase_price) / 2) if now_cost > purchase_price else now_cost
        """
        if now_cost > purchase_price:
            profit = now_cost - purchase_price
            return purchase_price + (profit // 2)
        return now_cost

    @staticmethod
    def calculate_banked_free_transfers(history_current: list[Any], chips_history: list[Any] | None = None) -> int:
        """
        Replay free transfer banking from Gameweek 1 through current round.
        Rules verified for 2026/27:
        - Initial deadline unlimited; one FT is granted for the following gameweek.
        - Wildcard and Free Hit preserve banked FTs unchanged for the next round.
        - Normal gameweek: remaining = max(0, banked - transfers_made), next_banked = min(5, remaining + 1).
        """
        if not history_current:
            return 1

        entries = sorted(
            history_current, key=lambda e: getattr(e, "event", 0) if not isinstance(e, dict) else e.get("event", 0)
        )
        chips_used_map = {}
        if chips_history:
            for c in chips_history:
                ev = getattr(c, "event", 0) if not isinstance(c, dict) else c.get("event", 0)
                nm = getattr(c, "name", "") if not isinstance(c, dict) else c.get("name", "")
                chips_used_map[ev] = nm.lower()

        banked = 1
        for index, entry in enumerate(entries):
            gw = getattr(entry, "event", 0) if not isinstance(entry, dict) else entry.get("event", 0)
            transfers_made = (
                getattr(entry, "event_transfers", 0) if not isinstance(entry, dict) else entry.get("event_transfers", 0)
            )
            active_chip = chips_used_map.get(gw, "")

            if index == 0:
                # Initial deadline is unlimited; the following week starts at one FT.
                banked = 1
            elif active_chip in ("wildcard", "freehit"):
                banked = banked
            else:
                if transfers_made <= banked:
                    remaining = banked - transfers_made
                else:
                    remaining = 0
                banked = min(5, remaining + 1)

        return max(1, min(5, banked))

    @classmethod
    def resolve_prices_for_squad(
        cls,
        elements_ids: list[int],
        bootstrap_elements: list[Any],
        transfers_history: list[Any] | None = None,
    ) -> dict[int, tuple[int, int, PriceProvenance]]:
        """
        Reconstruct purchase price and calculate selling price with provenance.
        Returns map: element_id -> (purchase_price_tenths, selling_price_tenths, provenance).
        """
        elem_map = {e.id: e for e in bootstrap_elements}
        result = {}

        # Organize transfers in by element_id, sorted by time/event descending
        transfers_by_elem: dict[int, list[Any]] = {}
        if transfers_history:
            for t in transfers_history:
                el_in = getattr(t, "element_in", None) if not isinstance(t, dict) else t.get("element_in")
                if el_in is not None:
                    transfers_by_elem.setdefault(el_in, []).append(t)

        for _el_in, t_list in transfers_by_elem.items():
            t_list.sort(
                key=lambda item: (
                    getattr(item, "event", 0) if not isinstance(item, dict) else item.get("event", 0),
                    getattr(item, "time", "") if not isinstance(item, dict) else item.get("time", ""),
                ),
                reverse=True,
            )

        for elem_id in elements_ids:
            elem = elem_map.get(elem_id)
            if not elem:
                result[elem_id] = (50, 50, PriceProvenance.MARKET_ESTIMATE)
                continue

            now_cost = getattr(elem, "now_cost", 50)

            # Case A: Element found in transfers in
            elem_transfers = transfers_by_elem.get(elem_id)
            if elem_transfers and len(elem_transfers) > 0:
                most_recent = elem_transfers[0]
                cost = (
                    getattr(most_recent, "element_in_cost", None)
                    if not isinstance(most_recent, dict)
                    else most_recent.get("element_in_cost")
                )
                if cost is not None and cost > 0:
                    purchase_price = int(cost)
                    selling_price = cls.calculate_selling_price(purchase_price, now_cost)
                    result[elem_id] = (purchase_price, selling_price, PriceProvenance.CONFIRMED_TRANSFER)
                    continue

            # Case B: Initial Squad player (no transfer in recorded, used from GW1)
            cost_change_start = getattr(elem, "cost_change_start", 0)
            initial_cost = now_cost - cost_change_start
            if initial_cost > 0:
                purchase_price = int(initial_cost)
                selling_price = cls.calculate_selling_price(purchase_price, now_cost)
                result[elem_id] = (purchase_price, selling_price, PriceProvenance.INITIAL_SQUAD)
            else:
                purchase_price = now_cost
                selling_price = now_cost
                result[elem_id] = (purchase_price, selling_price, PriceProvenance.MARKET_ESTIMATE)

        return result

    @classmethod
    def build_effective_state(
        cls,
        profile_data: Any,
        bootstrap: Any,
        fixtures: list[Any],
        manager_entry: Any | None = None,
        manager_picks: Any | None = None,
        manager_history: Any | None = None,
        manager_transfers: list[Any] | None = None,
        current_gw: int = 1,
        target_gw: int = 1,
        deadline_utc: str | None = None,
        is_stale: bool = False,
    ) -> EffectiveManagerState:
        """
        Builds the unified EffectiveManagerState domain object.
        Preserves manual overrides, validates squads, avoids silent demo replacements.
        """
        pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        elem_map = {e.id: e for e in getattr(bootstrap, "elements", [])}

        manager_id = getattr(profile_data, "manager_id", None)
        target_league_id = getattr(profile_data, "target_league_id", None)
        mode = ManagerMode.REAL

        # Bank resolution
        # User profile bank is stored in millions (e.g. 1.5). In tenths it is 15.
        profile_bank_val = getattr(profile_data, "bank", None)
        if getattr(profile_data, "bank_override_enabled", None) is False:
            profile_bank_val = None
        # FT resolution
        profile_ft_val = getattr(profile_data, "free_transfers", None)
        if getattr(profile_data, "ft_override_enabled", None) is False:
            profile_ft_val = None

        bank_tenths = 0
        bank_source = "default"
        has_bank_override = False

        free_transfers = 1
        ft_source = "default"
        has_ft_override = False

        manager_name = "Manager"
        team_name = "My FPL Squad"
        overall_points = 0
        overall_rank = None

        squad_ids: list[int] = []
        picks_list: list[Any] = []
        published_lineup = PublishedLineup()
        chips_used_list: list[dict[str, Any]] = []

        # 1. Attempt Real Manager Path
        if manager_id and manager_picks and hasattr(manager_picks, "picks"):
            mode = ManagerMode.REAL
            if manager_entry:
                fn = getattr(manager_entry, "player_first_name", "")
                ln = getattr(manager_entry, "player_last_name", "")
                manager_name = f"{fn} {ln}".strip() or "Manager"
                team_name = getattr(manager_entry, "name", "My FPL Squad")
                overall_points = getattr(manager_entry, "summary_overall_points", 0) or 0
                overall_rank = getattr(manager_entry, "summary_overall_rank", None)

            # Bank from picks entry_history
            if manager_picks.entry_history:
                bank_tenths = getattr(manager_picks.entry_history, "bank", 0) or 0
                bank_source = "fpl_api"

            # Replay FTs
            if manager_history and hasattr(manager_history, "current"):
                chips_hist = getattr(manager_history, "chips", [])
                free_transfers = cls.calculate_banked_free_transfers(manager_history.current, chips_hist)
                ft_source = "fpl_api_calculated"
                for c in chips_hist:
                    chips_used_list.append(
                        {
                            "name": getattr(c, "name", ""),
                            "event": getattr(c, "event", 0),
                            "time": getattr(c, "time", None),
                        }
                    )

            picks_list = manager_picks.picks
            squad_ids = [p.element for p in picks_list]

            # Populate published lineup
            published_lineup.active_chip = getattr(manager_picks, "active_chip", None)
            starters_published = []
            bench_published = []
            for p in picks_list:
                pos_slot = getattr(p, "position", 1)
                el = getattr(p, "element", 0)
                is_cap = getattr(p, "is_captain", False)
                is_vc = getattr(p, "is_vice_captain", False)
                if is_cap:
                    published_lineup.captain = el
                if is_vc:
                    published_lineup.vice_captain = el
                if pos_slot <= 11:
                    starters_published.append(el)
                else:
                    bench_published.append(el)
            published_lineup.starters = starters_published
            published_lineup.bench = bench_published

        # 2. Attempt Manual Squad Path if real manager is not configured or failed
        elif not manager_id and getattr(profile_data, "manual_squad", None):
            mode = ManagerMode.MANUAL
            man_raw = getattr(profile_data, "manual_squad", None)
            parsed_ids = json.loads(man_raw) if isinstance(man_raw, str) else (man_raw or [])
            if isinstance(parsed_ids, list) and len(parsed_ids) == 15:
                squad_ids = parsed_ids
                team_name = "Custom / Pasted Squad"

        # 3. Handle Explicit Overrides (distinguish 0.0 / 0 from None!)
        if profile_bank_val is not None:
            # User set explicit bank override
            bank_tenths = int(round(float(profile_bank_val) * 10.0))
            bank_source = "manual_override"
            has_bank_override = True

        if profile_ft_val is not None and profile_ft_val >= 0:
            free_transfers = int(profile_ft_val)
            ft_source = "manual_override"
            has_ft_override = True

        # Resolve prices
        prices_map = cls.resolve_prices_for_squad(squad_ids, getattr(bootstrap, "elements", []), manager_transfers)

        # Build PlayerSquadState list
        squad_players: list[PlayerSquadState] = []
        picks_pos_map = {p.element: p for p in picks_list} if picks_list else {}

        for elem_id in squad_ids:
            elem = elem_map.get(elem_id)
            if not elem:
                continue

            purchase_p, selling_p, prov = prices_map.get(
                elem_id, (elem.now_cost, elem.now_cost, PriceProvenance.MARKET_ESTIMATE)
            )
            pick_meta = picks_pos_map.get(elem_id)

            is_starter = True
            is_cap = False
            is_vc = False
            multiplier = 1
            bench_order = 0

            if pick_meta:
                slot = getattr(pick_meta, "position", 1)
                is_starter = slot <= 11
                is_cap = getattr(pick_meta, "is_captain", False)
                is_vc = getattr(pick_meta, "is_vice_captain", False)
                multiplier = getattr(pick_meta, "multiplier", 1)
                bench_order = 0 if is_starter else (slot - 11)

            cop = None
            if elem.status in ("i", "s"):
                cop = 0.0
            elif elem.chance_of_playing_next_round is not None:
                cop = float(elem.chance_of_playing_next_round)
            else:
                cop = None

            squad_players.append(
                PlayerSquadState(
                    element=elem.id,
                    web_name=elem.web_name,
                    team=elem.team,
                    position=pos_map.get(elem.element_type, "MID"),
                    now_cost=int(elem.now_cost),
                    purchase_price=purchase_p,
                    selling_price=selling_p,
                    price_provenance=prov,
                    is_starter=is_starter,
                    is_captain=is_cap,
                    is_vice_captain=is_vc,
                    multiplier=multiplier,
                    bench_order=bench_order,
                    chance_of_playing=cop,
                    status=elem.status or "a",
                    news=elem.news or "",
                )
            )

        # Chip sets calculation
        all_chips = ["wildcard", "freehit", "3xc", "bboost"]
        rem_set_1 = [
            c
            for c in all_chips
            if c not in [c_item.get("name", "").lower() for c_item in chips_used_list if c_item.get("event", 0) <= 19]
        ]
        rem_set_2 = [
            c
            for c in all_chips
            if c not in [c_item.get("name", "").lower() for c_item in chips_used_list if c_item.get("event", 0) > 19]
        ]

        # If squad is incomplete (e.g. no manager ID and no manual squad yet)
        if len(squad_players) < 15:
            mode = ManagerMode.DEMO
            confidence = "estimated"
        else:
            confidence = "manual" if mode == ManagerMode.MANUAL else "high"

        return EffectiveManagerState(
            manager_id=manager_id,
            target_league_id=target_league_id,
            mode=mode,
            manager_name=manager_name,
            team_name=team_name,
            overall_points=overall_points,
            overall_rank=overall_rank,
            squad=squad_players,
            published_lineup=published_lineup,
            bank_tenths=bank_tenths,
            bank_source=bank_source,
            has_bank_override=has_bank_override,
            free_transfers=free_transfers,
            ft_source=ft_source,
            has_ft_override=has_ft_override,
            chips_used=chips_used_list,
            chips_remaining_set_1=rem_set_1,
            chips_remaining_set_2=rem_set_2,
            active_chip=published_lineup.active_chip,
            current_gw=current_gw,
            target_gw=target_gw,
            deadline_utc=deadline_utc,
            source_timestamp=datetime.now(UTC).isoformat(),
            is_stale=is_stale,
            confidence=confidence,
        )

    async def get_current_state(self, force_refresh: bool = False) -> EffectiveManagerState:
        """Fetch upstream API data and stored profile, returning canonical EffectiveManagerState."""
        from fpl_oracle.api.fpl_client import fpl_client
        from fpl_oracle.data.store import data_store

        profile = data_store.get_profile()
        boot, is_stale = await fpl_client.get_bootstrap_static(force_refresh=force_refresh)
        fixtures, fixtures_stale = await fpl_client.get_fixtures(force_refresh=force_refresh)
        is_stale = is_stale or fixtures_stale
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()

        curr_gw = curr_gw or 1
        target_gw = next_gw or (curr_gw + 1 if curr_gw < 38 else 38)

        manager_entry = None
        manager_picks = None
        manager_history = None
        manager_transfers = None

        if profile.manager_id:
            import asyncio

            calls = [
                fpl_client.get_manager_entry(profile.manager_id, force_refresh=force_refresh),
                fpl_client.get_manager_picks(profile.manager_id, curr_gw, force_refresh=force_refresh),
                fpl_client.get_manager_history(profile.manager_id, force_refresh=force_refresh),
                fpl_client.get_manager_transfers(profile.manager_id, force_refresh=force_refresh),
            ]
            results = await asyncio.gather(*calls, return_exceptions=True)
            values: list[Any] = []
            for result in results:
                if isinstance(result, BaseException):
                    values.append(None)
                    is_stale = True
                else:
                    value, stale = result
                    values.append(value)
                    is_stale = is_stale or stale
            manager_entry, manager_picks, manager_history, manager_transfers = values

        deadline = None
        for ev in getattr(boot, "events", []):
            if ev.id == target_gw:
                deadline = ev.deadline_time
                break

        state = self.build_effective_state(
            profile_data=profile,
            bootstrap=boot,
            fixtures=fixtures,
            manager_entry=manager_entry,
            manager_picks=manager_picks,
            manager_history=manager_history,
            manager_transfers=manager_transfers,
            current_gw=curr_gw,
            target_gw=target_gw,
            deadline_utc=deadline,
            is_stale=is_stale,
        )

        keys = ["bootstrap-static", "fixtures:all"]
        if profile.manager_id:
            keys += [
                f"entry:{profile.manager_id}",
                f"entry:{profile.manager_id}:event:{curr_gw}:picks",
                f"entry:{profile.manager_id}:history",
                f"entry:{profile.manager_id}:transfers",
            ]
        timestamps = [fpl_client._cache_timestamps.get(key) for key in keys]
        known = [stamp for stamp in timestamps if stamp is not None]
        if known:
            state.source_timestamp = min(known).isoformat()
        if len(known) != len(timestamps):
            state.is_stale = True
            state.error_message = "Data age unknown. Refresh before using advice."
        if profile.manager_id and manager_picks is None:
            state.error_message = "Official published squad unavailable; saved manual squad was not substituted"
        from fpl_oracle.domain.team_confirmation import team_confirmation

        return team_confirmation.apply(state, boot)


manager_state_service = ManagerStateService()
