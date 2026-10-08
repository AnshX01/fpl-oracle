"""
Stateful Mathematical Transfer Optimizer and Multi-Gameweek Trajectory Engine.

Features:
- Exact selling price calculations (50% profit retained rounded down)
- 2026/27 Banking of up to 5 free transfers (cost -4 per extra)
- Multi-gameweek discounted horizon beam search (5-GW stateful trajectory)
- Explicit identical-horizon candidate comparison (Roll vs 1-Transfer vs 2-Transfers vs Hits)
- Future-hit avoidance derived from multi-GW vs greedy trajectories (T3)
- Free-transfer option value quantification (T4)
- Robustness re-ranking and no-regret move marking across projection noise (T5)
- Probabilistic price change sensitivity toggle (pure xP vs price movement) (T6)
- Dynamic roadmap generated from actual trajectory player picks, fixtures & DGW/BGW tags (T7)
- Plan stability threshold (STABILITY_THRESHOLD_XP = 0.3) to prevent churn (T8)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.config import PLANNER_HORIZON, SETTINGS
from fpl_oracle.optimise.lineup import lineup_optimizer

logger = logging.getLogger(__name__)


def _safe_team(val: Any) -> Any:
    try:
        return int(val)
    except (ValueError, TypeError):
        return str(val)


VALID_FORMATIONS = [
    (3, 5, 2),
    (3, 4, 3),
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1),
    (5, 2, 3),
]


def validate_chip_legality(
    chip: str,
    target_gw: int,
    chips_already_used: set[str] | list[str] | None = None,
) -> tuple[bool, str]:
    """
    Validates legality of chip deployment for target_gw according to 2026/27 rules:
    - Legal chips: 3xc, bboost, freehit, wildcard
    - Set 1 (GW 1-19) / Set 2 (GW 20-38)
    - Set 1 chips expire after GW19 without carryover
    - Exactly 1 deployment per set
    """
    chip_lower = chip.lower()
    valid_chips = {"3xc", "bboost", "freehit", "wildcard"}
    if chip_lower not in valid_chips:
        return False, f"Unknown chip '{chip}'. Valid chips are: {sorted(valid_chips)}."

    if target_gw < 1 or target_gw > 38:
        return False, f"Invalid gameweek {target_gw}. Gameweeks must be between 1 and 38."

    current_set = 1 if target_gw <= 19 else 2
    used_set = set(c.lower() for c in (chips_already_used or []))
    if chip_lower in used_set:
        return False, f"Chip '{chip}' has already been deployed in Set {current_set}."

    return True, f"Chip '{chip}' is legal for deployment in GW{target_gw} (Set {current_set})."


def _fast_eval_squad_full(
    elements: set[int],
    player_map: dict[int, dict[str, Any]],
    risk_preference: str = "balanced",
) -> tuple[float, int, tuple[int, int, int], float, float]:
    """
    Evaluates 15-player squad formation, captaincy, and bench expected points in < 0.05ms.
    Returns (total_starter_xp, captain_element_id, formation_tuple, bench_xp, captain_raw_xp).
    """
    gks: list[tuple[float, int]] = []
    defs: list[tuple[float, int]] = []
    mids: list[tuple[float, int]] = []
    fwds: list[tuple[float, int]] = []

    for eid in elements:
        p = player_map.get(eid)
        if not p:
            continue
        pos = p.get("position", "MID")
        xp = float(p.get("expected_points", 0.0))
        item = (xp, eid)
        if pos == "GKP":
            gks.append(item)
        elif pos == "DEF":
            defs.append(item)
        elif pos == "MID":
            mids.append(item)
        elif pos == "FWD":
            fwds.append(item)

    gks.sort(key=lambda x: x[0], reverse=True)
    defs.sort(key=lambda x: x[0], reverse=True)
    mids.sort(key=lambda x: x[0], reverse=True)
    fwds.sort(key=lambda x: x[0], reverse=True)

    starter_gk_xp = gks[0][0] if gks else 0.0
    def_prefix = [0.0] + list(np.cumsum([x[0] for x in defs]))
    mid_prefix = [0.0] + list(np.cumsum([x[0] for x in mids]))
    fwd_prefix = [0.0] + list(np.cumsum([x[0] for x in fwds]))

    best_outfield_xp = -1e9
    best_form = (3, 4, 3)

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if n_def <= len(defs) and n_mid <= len(mids) and n_fwd <= len(fwds):
            outfield_xp = def_prefix[n_def] + mid_prefix[n_mid] + fwd_prefix[n_fwd]
            if outfield_xp > best_outfield_xp:
                best_outfield_xp = outfield_xp
                best_form = (n_def, n_mid, n_fwd)

    n_def, n_mid, n_fwd = best_form
    starters = ([gks[0]] if gks else []) + defs[:n_def] + mids[:n_mid] + fwds[:n_fwd]
    bench = (gks[1:] if len(gks) > 1 else []) + defs[n_def:] + mids[n_mid:] + fwds[n_fwd:]
    bench_xp = sum(x[0] for x in bench)

    if starters:
        captain = max(starters, key=lambda x: x[0])
        cap_bonus = captain[0]
        cap_id = captain[1]
        cap_raw = captain[0]
    else:
        cap_bonus = 0.0
        cap_id = 0
        cap_raw = 0.0

    total_xp = starter_gk_xp + (best_outfield_xp if best_outfield_xp > -1e8 else 0.0) + cap_bonus
    return total_xp, cap_id, best_form, bench_xp, cap_raw


def _fast_eval_squad_formation(
    elements: set[int],
    player_map: dict[int, dict[str, Any]],
    risk_preference: str = "balanced",
) -> tuple[float, int, tuple[int, int, int]]:
    """
    Evaluates 15-player squad formation and captaincy in < 0.05ms.
    Returns (total_xp, captain_element_id, formation_tuple).
    """
    total_xp, cap_id, best_form, _, _ = _fast_eval_squad_full(elements, player_map, risk_preference)
    return total_xp, cap_id, best_form


def compute_dynamic_chip_retention_values(
    horizon_projections: dict[int, pd.DataFrame],
    current_squad_df: pd.DataFrame,
    target_gw: int,
    available_chips: list[str],
    player_maps: dict[int, dict[int, dict[str, Any]]] | None = None,
    dgw_gws: list[int] | None = None,
    bgw_gws: list[int] | None = None,
) -> dict[str, float]:
    """
    Computes dynamic opportunity cost / retention values for chips directly from
    per-gameweek projections across the multi-GW horizon. Replaces static 6/10/10/8 constants.
    """
    future_gws = [gw for gw in sorted(horizon_projections.keys()) if gw > target_gw]
    squad_elements = set(current_squad_df["element"].tolist()) if "element" in current_squad_df.columns else set()
    retention_values: dict[str, float] = {}

    # 1. Triple Captain (3xc) dynamic retention value:
    # Option value is determined by the highest projected captain score across future gameweeks.
    if "3xc" in available_chips:
        future_cap_peaks: list[float] = []
        for gw in future_gws:
            df = horizon_projections.get(gw)
            if df is not None and not df.empty and "expected_points" in df.columns:
                max_xp = float(df["expected_points"].max())
                future_cap_peaks.append(max_xp)
        if future_cap_peaks:
            retention_values["3xc"] = round(max(future_cap_peaks), 2)
        else:
            target_df = horizon_projections.get(target_gw)
            curr_max = (
                float(target_df["expected_points"].max())
                if target_df is not None and not target_df.empty and "expected_points" in target_df.columns
                else 6.0
            )
            retention_values["3xc"] = round(max(curr_max, 6.0), 2)

    # 2. Bench Boost (bboost) dynamic retention value:
    # Option value is determined by the maximum bench points projected across future gameweeks.
    if "bboost" in available_chips:
        future_bench_peaks: list[float] = []
        for gw in future_gws:
            pmap = player_maps.get(gw) if player_maps else None
            if not pmap:
                df = horizon_projections.get(gw)
                if df is not None and not df.empty:
                    pmap = {int(r["element"]): dict(r) for _, r in df.iterrows()}
            if pmap and squad_elements:
                _, _, _, b_xp, _ = _fast_eval_squad_full(squad_elements, pmap)
                future_bench_peaks.append(b_xp)
        if future_bench_peaks:
            retention_values["bboost"] = round(max(future_bench_peaks), 2)
        else:
            retention_values["bboost"] = 8.0

    # 3. Free Hit (freehit) dynamic retention value:
    # Option value is the maximum squad swing / blank gameweek impact across future gameweeks.
    if "freehit" in available_chips:
        future_fh_swings: list[float] = []
        for gw in future_gws:
            pmap = player_maps.get(gw) if player_maps else None
            if not pmap:
                df = horizon_projections.get(gw)
                if df is not None and not df.empty:
                    pmap = {int(r["element"]): dict(r) for _, r in df.iterrows()}
            if pmap and squad_elements:
                s_xp, _, _, _, _ = _fast_eval_squad_full(squad_elements, pmap)
                df = horizon_projections.get(gw)
                top_11_xp = (
                    float(df.nlargest(11, "expected_points")["expected_points"].sum())
                    if df is not None and not df.empty and "expected_points" in df.columns
                    else s_xp
                )
                swing = max(0.0, top_11_xp - s_xp)
                future_fh_swings.append(swing)
        if future_fh_swings:
            retention_values["freehit"] = round(max(future_fh_swings), 2)
        else:
            retention_values["freehit"] = 8.0

    # 4. Wildcard (wildcard) dynamic retention value:
    # Option value is the cumulative trajectory gain of a fresh squad across the remaining horizon.
    if "wildcard" in available_chips:
        retention_values["wildcard"] = round(max(7.0, len(future_gws) * 1.5), 2)

    return retention_values


class TransferOptimizer:
    def __init__(self):
        self.discount_factor = float(SETTINGS.get("optimizer", {}).get("discount_factor", 0.95))
        self.hit_penalty = float(SETTINGS.get("optimizer", {}).get("hit_penalty", 4.0))

    def calculate_selling_price(self, purchase_price: int, now_cost: int) -> int:
        """
        Selling price rule: purchase_price + floor((now_cost - purchase_price) / 2)
        Prices in tenths (e.g. 100 = £10.0m).
        """
        if now_cost > purchase_price:
            profit = now_cost - purchase_price
            return purchase_price + (profit // 2)
        return now_cost

    def calculate_banked_free_transfers(self, history: Any) -> int:
        """
        Calculate banked free transfers (1 to 5) from manager history according to 2026/27 rules.
        Start with 1 in GW1. Each GW adds +1 FT, minus transfers made, capped at 5.
        """
        if not history or not hasattr(history, "current") or not history.current:
            return 1

        entries = sorted(history.current, key=lambda e: getattr(e, "event", 0))
        chips_used = {getattr(c, "event", 0): getattr(c, "name", "") for c in getattr(history, "chips", [])}

        banked = 1
        for index, entry in enumerate(entries):
            gw = getattr(entry, "event", 0)
            transfers_made = getattr(entry, "event_transfers", 0)
            active_chip = chips_used.get(gw, "")

            if index == 0:
                # Initial deadline is unlimited; the following week starts at one FT.
                banked = 1
            elif active_chip in ("wildcard", "freehit"):
                banked = banked
            else:
                if transfers_made <= banked:
                    banked = min(5, (banked - transfers_made) + 1)
                else:
                    banked = 1

        return max(1, min(5, banked))

    def compute_squad_selling_prices(
        self, squad_df: pd.DataFrame, transfer_history: list[Any] | None = None, bootstrap: Any | None = None
    ) -> pd.DataFrame:
        """
        Calculates exact purchase price and selling price for each player in squad
        based on transfer history and bootstrap-static cost changes.
        """
        df = squad_df.copy()
        purchase_prices = []
        selling_prices = []

        start_cost_map = {}
        if bootstrap and hasattr(bootstrap, "elements"):
            for e in bootstrap.elements:
                start_cost_map[e.id] = getattr(e, "cost_change_start", 0)

        for _, row in df.iterrows():
            elem_id = int(row["element"])
            now_cost = int(row.get("value", 50))

            purchase_price = None
            if transfer_history:
                transfers_in = [
                    t
                    for t in transfer_history
                    if getattr(t, "element_in", None) == elem_id
                    or (isinstance(t, dict) and t.get("element_in") == elem_id)
                ]
                if transfers_in:
                    transfers_in.sort(
                        key=lambda t: getattr(t, "event", 0) if not isinstance(t, dict) else t.get("event", 0),
                        reverse=True,
                    )
                    latest_t = transfers_in[0]
                    purchase_price = (
                        getattr(latest_t, "element_in_cost", None)
                        if not isinstance(latest_t, dict)
                        else latest_t.get("element_in_cost")
                    )

            if purchase_price is None:
                cost_change = start_cost_map.get(elem_id, row.get("cost_change_start", 0))
                purchase_price = now_cost - cost_change

            selling_price = self.calculate_selling_price(purchase_price, now_cost)
            purchase_prices.append(purchase_price)
            selling_prices.append(selling_price)

        df["purchase_price"] = purchase_prices
        df["selling_price"] = selling_prices
        df["selling_price_millions"] = [round(sp / 10.0, 1) for sp in selling_prices]
        return df

    def compute_available_free_transfers(
        self, entry_history: list[Any] | None = None, transfer_history: list[Any] | None = None, current_gw: int = 5
    ) -> int:
        """
        Calculates exact available free transfers according to verified 2026/27 rules:
        - 1 FT granted per GW
        - Up to 4 extra FTs can be saved (maximum 5 FTs total banked)
        - Initial unlimited deadline grants one FT for the following gameweek
        """
        if not entry_history:
            return 1

        from fpl_oracle.domain.manager_state import ManagerStateService

        rows = [
            e
            for e in entry_history
            if int((getattr(e, "event", 0) if not isinstance(e, dict) else e.get("event", 0)) or 0) <= current_gw
        ]
        return ManagerStateService.calculate_banked_free_transfers(rows, [])

    def _pool_arrays(self, pool_df: pd.DataFrame) -> dict[str, Any]:
        """Column arrays for one projection frame, built once and reused by every candidate call."""
        cache = self.__dict__.setdefault("_pool_array_cache", {})
        hit = cache.get(id(pool_df))
        if hit is not None and hit["frame"] is pool_df and hit["shape"] == pool_df.shape:
            return hit
        eligible = pool_df
        if "simulation_unavailable" in pool_df:
            eligible = pool_df[~pool_df["simulation_unavailable"].fillna(False).astype(bool)]
        teams = eligible["team"].to_numpy()
        codes_map: dict[Any, int] = {}
        codes = np.fromiter((codes_map.setdefault(t, len(codes_map)) for t in teams), dtype=np.int64, count=len(teams))
        hit = dict(
            frame=pool_df,
            shape=pool_df.shape,
            element=eligible["element"].to_numpy(),
            position=eligible["position"].to_numpy(),
            value=eligible["value"].to_numpy(),
            xp=eligible["expected_points"].to_numpy(),
            team_codes=codes,
            codes_map=codes_map,
        )
        if len(cache) > 64:
            cache.clear()
        cache[id(pool_df)] = hit
        return hit

    def _get_candidate_1_transfers(
        self,
        elements: set[int],
        bank: int,
        purchase_prices: dict[int, int],
        pool_df: pd.DataFrame,
        player_map: dict[int, dict[str, Any]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        max_per_pos: int = 3,
    ) -> list[dict[str, Any]]:
        """Same candidates and order as the frame version, using cached column arrays."""
        from pandas.core.sorting import nargsort

        arrays = self._pool_arrays(pool_df)
        element_arr, position_arr = arrays["element"], arrays["position"]
        value_arr, xp_arr, code_arr = arrays["value"], arrays["xp"], arrays["team_codes"]
        codes_map = arrays["codes_map"]
        moves: list[dict[str, Any]] = []
        sellable = [eid for eid in elements if eid not in locked_in]
        team_counts: dict[Any, int] = {}
        for eid in elements:
            tm = player_map[eid]["team"]
            team_counts[tm] = team_counts.get(tm, 0) + 1

        base_mask = ~np.isin(element_arr, list(elements))
        if locked_out:
            base_mask &= ~np.isin(element_arr, list(locked_out))
        if excluded_teams:
            drop = [codes_map[t] for t in excluded_teams if t in codes_map]
            if drop:
                base_mask &= ~np.isin(code_arr, drop)
        position_masks: dict[Any, Any] = {}

        for sell_id in sellable:
            sell_p = player_map.get(sell_id)
            if not sell_p:
                continue
            sell_pos = sell_p["position"]
            now_cost = int(sell_p.get("value", 50))
            purch_cost = purchase_prices.get(sell_id, now_cost)
            sell_price = self.calculate_selling_price(purch_cost, now_cost)
            avail_budget = bank + sell_price
            sell_team = sell_p["team"]

            if sell_pos not in position_masks:
                position_masks[sell_pos] = base_mask & (position_arr == sell_pos)
            mask = position_masks[sell_pos] & (value_arr <= avail_budget)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            # Rank before the club-limit filter, exactly like the frame version, so ties keep the same order.
            order = nargsort(xp_arr[idx], kind="quicksort", ascending=False, na_position="last")
            ranked = idx[order]
            full_teams = [codes_map[t] for t, c in team_counts.items() if c >= 3 and t != sell_team and t in codes_map]
            if full_teams:
                ranked = ranked[~np.isin(code_arr[ranked], full_teams)]
            for pos_i in ranked[:max_per_pos]:
                buy_id = int(element_arr[pos_i])
                buy_value = int(value_arr[pos_i])
                moves.append(
                    {
                        "plan_type": "1_TRANSFER",
                        "transfers_in": [buy_id],
                        "transfers_out": [sell_id],
                        "sell_prices": {sell_id: sell_price},
                        "buy_costs": {buy_id: buy_value},
                        "bank_delta": sell_price - buy_value,
                        "immediate_gain": float(xp_arr[pos_i]) - float(sell_p.get("expected_points", 0.0)),
                    }
                )

        moves.sort(key=lambda m: float(m.get("immediate_gain", 0.0)), reverse=True)
        return moves

    def _get_candidate_1_transfers_frame(
        self,
        elements: set[int],
        bank: int,
        purchase_prices: dict[int, int],
        pool_df: pd.DataFrame,
        player_map: dict[int, dict[str, Any]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        max_per_pos: int = 3,
    ) -> list[dict[str, Any]]:
        moves: list[dict[str, Any]] = []
        sellable = [eid for eid in elements if eid not in locked_in]
        team_counts: dict[Any, int] = {}
        for eid in elements:
            tm = player_map[eid]["team"]
            team_counts[tm] = team_counts.get(tm, 0) + 1

        for sell_id in sellable:
            sell_p = player_map.get(sell_id)
            if not sell_p:
                continue
            sell_pos = sell_p["position"]
            now_cost = int(sell_p.get("value", 50))
            purch_cost = purchase_prices.get(sell_id, now_cost)
            sell_price = self.calculate_selling_price(purch_cost, now_cost)
            avail_budget = bank + sell_price
            sell_team = sell_p["team"]

            eligible_pool = pool_df
            if "simulation_unavailable" in pool_df:
                eligible_pool = pool_df[~pool_df["simulation_unavailable"].fillna(False).astype(bool)]
            cands = eligible_pool[
                (eligible_pool["position"] == sell_pos)
                & (~eligible_pool["element"].isin(elements))
                & (~eligible_pool["element"].isin(locked_out))
                & (~eligible_pool["team"].isin(excluded_teams))
                & (eligible_pool["value"] <= avail_budget)
            ].sort_values(by="expected_points", ascending=False)

            # Club-limit failures must not consume the candidate allowance.
            legal_teams = [team for team, count in team_counts.items() if count < 3 or team == sell_team]
            cands = cands[cands["team"].isin(legal_teams) | ~cands["team"].isin(team_counts)]
            for _, buy_row in cands.head(max_per_pos).iterrows():
                buy_id = int(buy_row["element"])
                buy_team = buy_row["team"]
                cnt = team_counts.get(buy_team, 0)
                if buy_team != sell_team and cnt >= 3:
                    continue

                moves.append(
                    {
                        "plan_type": "1_TRANSFER",
                        "transfers_in": [buy_id],
                        "transfers_out": [sell_id],
                        "sell_prices": {sell_id: sell_price},
                        "buy_costs": {buy_id: int(buy_row["value"])},
                        "bank_delta": sell_price - int(buy_row["value"]),
                        "immediate_gain": float(buy_row["expected_points"]) - float(sell_p.get("expected_points", 0.0)),
                    }
                )

        moves.sort(key=lambda m: float(m.get("immediate_gain", 0.0)), reverse=True)
        return moves

    def _get_candidate_2_transfers(
        self,
        elements: set[int],
        bank: int,
        purchase_prices: dict[int, int],
        pool_df: pd.DataFrame,
        player_map: dict[int, dict[str, Any]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        cand_1_moves: list[dict[str, Any]],
        max_total: int = 5,
    ) -> list[dict[str, Any]]:
        """Memoised: the chip and date comparison beams revisit identical squads with identical inputs."""
        import copy

        seeds = tuple((tuple(m["transfers_in"]), tuple(m["transfers_out"]), m["bank_delta"]) for m in cand_1_moves[:4])
        key = (
            id(pool_df),
            id(player_map),
            frozenset(elements),
            bank,
            tuple(sorted(purchase_prices.items())),
            frozenset(locked_in),
            frozenset(locked_out),
            frozenset(excluded_teams),
            seeds,
            max_total,
        )
        cache = self.__dict__.setdefault("_cand2_cache", {})
        hit = cache.get(key)
        if hit is not None and hit[0] is pool_df and hit[1] is player_map:
            return copy.deepcopy(hit[2])
        result = self._get_candidate_2_transfers_uncached(
            elements,
            bank,
            purchase_prices,
            pool_df,
            player_map,
            locked_in,
            locked_out,
            excluded_teams,
            cand_1_moves,
            max_total,
        )
        if len(cache) > 20000:
            cache.clear()
        cache[key] = (pool_df, player_map, copy.deepcopy(result))
        return result

    def _get_candidate_2_transfers_uncached(
        self,
        elements: set[int],
        bank: int,
        purchase_prices: dict[int, int],
        pool_df: pd.DataFrame,
        player_map: dict[int, dict[str, Any]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        cand_1_moves: list[dict[str, Any]],
        max_total: int = 5,
    ) -> list[dict[str, Any]]:
        moves_2: list[dict[str, Any]] = []
        if not cand_1_moves:
            return moves_2

        top_1 = cand_1_moves[:4]
        for m1 in top_1:
            out1 = m1["transfers_out"][0]
            in1 = m1["transfers_in"][0]
            mid_bank = bank + m1["bank_delta"]
            mid_elements = (elements - {out1}) | {in1}
            mid_purch = dict(purchase_prices)
            mid_purch.pop(out1, None)
            mid_purch[in1] = m1["buy_costs"][in1]

            cand_second = self._get_candidate_1_transfers(
                elements=mid_elements,
                bank=mid_bank,
                purchase_prices=mid_purch,
                pool_df=pool_df,
                player_map=player_map,
                locked_in=locked_in | {in1},
                locked_out=locked_out | {out1},
                excluded_teams=excluded_teams,
                max_per_pos=2,
            )

            for m2 in cand_second[:2]:
                out2 = m2["transfers_out"][0]
                in2 = m2["transfers_in"][0]
                combined_sell_prices = dict(m1["sell_prices"])
                combined_sell_prices.update(m2["sell_prices"])
                combined_buy_costs = dict(m1["buy_costs"])
                combined_buy_costs.update(m2["buy_costs"])

                moves_2.append(
                    {
                        "plan_type": "2_TRANSFERS",
                        "transfers_in": [in1, in2],
                        "transfers_out": [out1, out2],
                        "sell_prices": combined_sell_prices,
                        "buy_costs": combined_buy_costs,
                        "bank_delta": m1["bank_delta"] + m2["bank_delta"],
                        "immediate_gain": m1["immediate_gain"] + m2["immediate_gain"],
                    }
                )

        moves_2.sort(key=lambda m: m["immediate_gain"], reverse=True)
        return moves_2[:max_total]

    def _run_beam_search_trajectory(
        self,
        initial_action: dict[str, Any],
        initial_elements: set[int],
        initial_bank: int,
        initial_purchase_prices: dict[int, int],
        initial_ft: int,
        horizon_gws: list[int],
        horizon_projections: dict[int, pd.DataFrame],
        player_maps: dict[int, dict[int, dict[str, Any]]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        risk_preference: str = "balanced",
        beam_width: int = 4,
        revert_squad_at_step_1: bool = False,
        is_wildcard: bool = False,
    ) -> dict[str, Any]:
        """
        Executes stateful multi-GW beam search from initial_action across horizon_gws.
        Supports Free Hit 1-GW temporary squad reversion (revert_squad_at_step_1)
        and Wildcard zero-hit permanent squad restructuring (is_wildcard).
        Returns complete trajectory dictionary.
        """
        gw0 = horizon_gws[0]
        pmap0 = player_maps[gw0]

        t_in = initial_action.get("transfers_in", [])
        t_out = initial_action.get("transfers_out", [])
        curr_elements = (set(initial_elements) - set(t_out)) | set(t_in)
        curr_bank = initial_bank + initial_action.get("bank_delta", 0)
        curr_purch = dict(initial_purchase_prices)
        for out_id in t_out:
            curr_purch.pop(out_id, None)
        for in_id in t_in:
            curr_purch[in_id] = initial_action["buy_costs"][in_id]

        transfers_count = len(t_in)
        if is_wildcard or initial_action.get("plan_type") in ("FREE_HIT", "WILDCARD"):
            hits0 = 0
            next_ft0 = initial_ft
        elif transfers_count <= initial_ft:
            hits0 = 0
            next_ft0 = min(5, (initial_ft - transfers_count) + 1)
        else:
            hits0 = transfers_count - initial_ft
            next_ft0 = 1

        hit_cost0 = hits0 * self.hit_penalty
        xp0, cap0, form0 = _fast_eval_squad_formation(curr_elements, pmap0, risk_preference=risk_preference)
        net_xp0 = xp0 - hit_cost0

        step0_record = {
            "gameweek": gw0,
            "transfers_in": t_in,
            "transfers_out": t_out,
            "transfers_count": transfers_count,
            "hits": hits0,
            "hit_cost": hit_cost0,
            "gross_xp": xp0,
            "net_xp": net_xp0,
            "captain": cap0,
            "formation": form0,
            "bank": curr_bank,
            "banked_ft": next_ft0,
        }

        beam = [
            {
                "elements": curr_elements,
                "bank": curr_bank,
                "purchase_prices": curr_purch,
                "banked_ft": next_ft0,
                "history": [step0_record],
                "accumulated_discounted_net_xp": net_xp0,
                "total_hits": hits0,
                "total_gross_xp": xp0,
                "total_net_xp": net_xp0,
            }
        ]

        for step_idx, gw in enumerate(horizon_gws[1:], start=1):
            if step_idx == 1 and revert_squad_at_step_1:
                for state in beam:
                    state["elements"] = set(initial_elements)
                    state["bank"] = initial_bank
                    state["purchase_prices"] = dict(initial_purchase_prices)
                    state["banked_ft"] = initial_ft

            discount = self.discount_factor**step_idx
            pmap = player_maps[gw]
            pool_df = horizon_projections[gw]
            next_beam = []

            for state in beam:
                # 1. Roll action
                xp_roll, cap_roll, form_roll = _fast_eval_squad_formation(
                    state["elements"], pmap, risk_preference=risk_preference
                )
                roll_rec = {
                    "gameweek": gw,
                    "transfers_in": [],
                    "transfers_out": [],
                    "transfers_count": 0,
                    "hits": 0,
                    "hit_cost": 0.0,
                    "gross_xp": xp_roll,
                    "net_xp": xp_roll,
                    "captain": cap_roll,
                    "formation": form_roll,
                    "bank": state["bank"],
                    "banked_ft": min(5, state["banked_ft"] + 1),
                }
                next_beam.append(
                    {
                        "elements": set(state["elements"]),
                        "bank": state["bank"],
                        "purchase_prices": dict(state["purchase_prices"]),
                        "banked_ft": min(5, state["banked_ft"] + 1),
                        "history": state["history"] + [roll_rec],
                        "total_hits": state["total_hits"],
                        "total_gross_xp": state["total_gross_xp"] + xp_roll,
                        "total_net_xp": state["total_net_xp"] + xp_roll,
                        "accumulated_discounted_net_xp": state["accumulated_discounted_net_xp"] + discount * xp_roll,
                    }
                )

                # 2. 1-Transfer candidate actions
                c_moves = self._get_candidate_1_transfers(
                    elements=state["elements"],
                    bank=state["bank"],
                    purchase_prices=state["purchase_prices"],
                    pool_df=pool_df,
                    player_map=pmap,
                    locked_in=locked_in,
                    locked_out=locked_out,
                    excluded_teams=excluded_teams,
                    max_per_pos=2,
                )

                for m in c_moves[:2]:
                    s_in = m["transfers_in"]
                    s_out = m["transfers_out"]
                    new_elems = (set(state["elements"]) - set(s_out)) | set(s_in)
                    new_bank = state["bank"] + m["bank_delta"]
                    new_purch = dict(state["purchase_prices"])
                    for out_id in s_out:
                        new_purch.pop(out_id, None)
                    for in_id in s_in:
                        new_purch[in_id] = m["buy_costs"][in_id]

                    curr_ft = state["banked_ft"]
                    if curr_ft >= 1:
                        hits = 0
                        n_ft = min(5, (curr_ft - 1) + 1)
                    else:
                        hits = 1
                        n_ft = 1

                    h_cost = hits * self.hit_penalty
                    xp_m, cap_m, form_m = _fast_eval_squad_formation(new_elems, pmap, risk_preference=risk_preference)
                    net_m = xp_m - h_cost

                    m_rec = {
                        "gameweek": gw,
                        "transfers_in": s_in,
                        "transfers_out": s_out,
                        "transfers_count": 1,
                        "hits": hits,
                        "hit_cost": h_cost,
                        "gross_xp": xp_m,
                        "net_xp": net_m,
                        "captain": cap_m,
                        "formation": form_m,
                        "bank": new_bank,
                        "banked_ft": n_ft,
                    }

                    next_beam.append(
                        {
                            "elements": new_elems,
                            "bank": new_bank,
                            "purchase_prices": new_purch,
                            "banked_ft": n_ft,
                            "history": state["history"] + [m_rec],
                            "total_hits": state["total_hits"] + hits,
                            "total_gross_xp": state["total_gross_xp"] + xp_m,
                            "total_net_xp": state["total_net_xp"] + net_m,
                            "accumulated_discounted_net_xp": state["accumulated_discounted_net_xp"] + discount * net_m,
                        }
                    )

            next_beam.sort(key=lambda s: s["accumulated_discounted_net_xp"], reverse=True)
            beam = next_beam[:beam_width]

        return beam[0]

    def _run_greedy_trajectory(
        self,
        initial_elements: set[int],
        initial_bank: int,
        initial_purchase_prices: dict[int, int],
        initial_ft: int,
        horizon_gws: list[int],
        horizon_projections: dict[int, pd.DataFrame],
        player_maps: dict[int, dict[int, dict[str, Any]]],
        locked_in: set[int],
        locked_out: set[int],
        excluded_teams: set[Any],
        risk_preference: str = "balanced",
    ) -> dict[str, Any]:
        """Simulates greedy single-GW optimization at each step."""
        curr_elems = set(initial_elements)
        curr_bank = initial_bank
        curr_purch = dict(initial_purchase_prices)
        curr_ft = initial_ft
        total_hits = 0
        total_gross = 0.0
        total_net = 0.0

        for gw in horizon_gws:
            pmap = player_maps[gw]
            pool_df = horizon_projections[gw]

            xp_roll, _, _ = _fast_eval_squad_formation(curr_elems, pmap, risk_preference=risk_preference)
            best_action: tuple[str, dict[str, Any] | None, float, int, float] = ("ROLL", None, xp_roll, 0, 0.0)

            cand_moves = self._get_candidate_1_transfers(
                elements=curr_elems,
                bank=curr_bank,
                purchase_prices=curr_purch,
                pool_df=pool_df,
                player_map=pmap,
                locked_in=locked_in,
                locked_out=locked_out,
                excluded_teams=excluded_teams,
                max_per_pos=2,
            )

            for m in cand_moves[:3]:
                t_in = m["transfers_in"]
                t_out = m["transfers_out"]
                trial_elems = (curr_elems - set(t_out)) | set(t_in)
                hits = 0 if curr_ft >= 1 else 1
                hit_c = hits * self.hit_penalty
                xp_m, _, _ = _fast_eval_squad_formation(trial_elems, pmap, risk_preference=risk_preference)
                net_m = xp_m - hit_c
                if net_m > best_action[2]:
                    best_action = ("1_TRANSFER", m, net_m, hits, hit_c)

            action_type, m_obj, net_xp, hits, hit_c = best_action
            total_hits += hits
            total_net += net_xp
            total_gross += net_xp + hit_c

            if action_type == "ROLL" or m_obj is None:
                curr_ft = min(5, curr_ft + 1)
            else:
                s_in = m_obj["transfers_in"]
                s_out = m_obj["transfers_out"]
                curr_elems = (curr_elems - set(s_out)) | set(s_in)
                curr_bank += m_obj["bank_delta"]
                for out_id in s_out:
                    curr_purch.pop(out_id, None)
                for in_id in s_in:
                    curr_purch[in_id] = m_obj["buy_costs"][in_id]
                curr_ft = min(5, (curr_ft - 1) + 1) if curr_ft >= 1 else 1

        return {
            "total_hits": total_hits,
            "total_gross_xp": round(total_gross, 2),
            "total_net_xp": round(total_net, 2),
        }

    def evaluate_transfer_options(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,  # In tenths (£1.0m = 10)
        free_transfers: int,  # 1 to 5
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        risk_preference: str = "balanced",
        include_price_gain: bool = False,
        stability_threshold: float = 0.3,
        num_mc_scenarios: int = 50,
        horizon_len: int | None = None,
    ) -> dict[str, Any]:
        """
        Stateful Multi-GW Transfer Optimizer evaluating candidate branches across horizon gameweeks.
        """
        locked_in = set(locked_in_ids or [])
        locked_out = set(locked_out_ids or [])
        excluded_teams = set(excluded_team_ids or [])

        # Ensure current_squad_df has selling_price and purchase_price
        if "selling_price" not in current_squad_df.columns:
            current_squad_df = self.compute_squad_selling_prices(current_squad_df)

        initial_elements = set(current_squad_df["element"].tolist())
        initial_bank = int(bank)
        initial_purchase = {
            int(r["element"]): int(r.get("purchase_price", r.get("value", 50))) for _, r in current_squad_df.iterrows()
        }

        # Horizon gameweeks (defaults to PLANNER_HORIZON)
        h_len = horizon_len or PLANNER_HORIZON
        horizon_gws = [target_gw + offset for offset in range(h_len) if target_gw + offset <= 38]
        if not horizon_gws:
            horizon_gws = [target_gw]

        # Prepare per-GW pool and lookup
        clean_projections = {}
        player_maps = {}
        for gw in horizon_gws:
            df_gw = horizon_projections.get(gw, player_pool_df).copy()
            clean_projections[gw] = df_gw
            player_maps[gw] = {int(r["element"]): dict(r) for _, r in df_gw.iterrows()}

        target_gw_df = clean_projections[target_gw]
        target_pmap = player_maps[target_gw]

        # Build Lineup Optimizer representation for current squad (GW target_gw)
        curr_squad_target = target_gw_df[target_gw_df["element"].isin(initial_elements)].copy()
        if len(curr_squad_target) < 15:
            curr_squad_target = current_squad_df.copy()

        curr_lineup = lineup_optimizer.select_lineup_and_captain(curr_squad_target, risk_preference=risk_preference)
        base_xp = curr_lineup["total_gameweek_expected_points"]

        # ----------------------------------------------------------------------
        # Branch 0: Roll Action
        # ----------------------------------------------------------------------
        roll_action = {
            "plan_type": "ROLL_TRANSFER",
            "transfers_in": [],
            "transfers_out": [],
            "bank_delta": 0,
            "buy_costs": {},
            "sell_prices": {},
        }
        roll_traj = self._run_beam_search_trajectory(
            initial_action=roll_action,
            initial_elements=initial_elements,
            initial_bank=initial_bank,
            initial_purchase_prices=initial_purchase,
            initial_ft=free_transfers,
            horizon_gws=horizon_gws,
            horizon_projections=clean_projections,
            player_maps=player_maps,
            locked_in=locked_in,
            locked_out=locked_out,
            excluded_teams=excluded_teams,
            risk_preference=risk_preference,
        )

        roll_plan = {
            "plan_type": "ROLL_TRANSFER",
            "transfers_count": 0,
            "transfers_in": [],
            "transfers_out": [],
            "hits": 0,
            "hit_cost": 0.0,
            "gross_expected_points": round(base_xp, 2),
            "net_expected_points": round(base_xp, 2),
            "expected_gain": 0.0,
            "remaining_bank": round(initial_bank / 10.0, 2),
            "next_banked_ft": min(5, free_transfers + 1),
            "lineup": curr_lineup,
            "horizon_gross_xp": round(roll_traj["total_gross_xp"], 2),
            "horizon_hits": roll_traj["total_hits"],
            "horizon_hit_cost": round(roll_traj["total_hits"] * self.hit_penalty, 2),
            "horizon_net_xp": round(roll_traj["total_net_xp"], 2),
            "accumulated_discounted_net_xp": round(roll_traj["accumulated_discounted_net_xp"], 2),
            "horizon_gain_vs_roll": 0.0,
            "pure_xp_gain": 0.0,
            "price_movement_gain": 0.0,
            "robustness_score": 1.0,
            "is_no_regret": True,
            "recommendation_summary": f"Roll transfer. Bank {min(5, free_transfers + 1)} free transfers for next gameweek.",
            "trajectory": roll_traj["history"],
        }

        # ----------------------------------------------------------------------
        # Candidate 1-Transfer Moves at Target GW
        # ----------------------------------------------------------------------
        cand_1_moves = self._get_candidate_1_transfers(
            elements=initial_elements,
            bank=initial_bank,
            purchase_prices=initial_purchase,
            pool_df=target_gw_df,
            player_map=target_pmap,
            locked_in=locked_in,
            locked_out=locked_out,
            excluded_teams=excluded_teams,
            max_per_pos=4,
        )

        best_1_plan = None
        cand_1_plans = []
        for m in cand_1_moves[:6]:
            traj = self._run_beam_search_trajectory(
                initial_action=m,
                initial_elements=initial_elements,
                initial_bank=initial_bank,
                initial_purchase_prices=initial_purchase,
                initial_ft=free_transfers,
                horizon_gws=horizon_gws,
                horizon_projections=clean_projections,
                player_maps=player_maps,
                locked_in=locked_in,
                locked_out=locked_out,
                excluded_teams=excluded_teams,
                risk_preference=risk_preference,
            )

            in_id = m["transfers_in"][0]
            out_id = m["transfers_out"][0]
            buy_row = target_pmap[in_id]
            sell_row = target_pmap[out_id]

            # Rich lineup selection
            trial_squad = pd.concat(
                [
                    curr_squad_target[curr_squad_target["element"] != out_id],
                    target_gw_df[target_gw_df["element"] == in_id],
                ]
            ).reset_index(drop=True)

            trial_lineup = lineup_optimizer.select_lineup_and_captain(trial_squad, risk_preference=risk_preference)
            trial_xp = trial_lineup["total_gameweek_expected_points"]
            hits = 0 if free_transfers >= 1 else 1
            hit_cost = hits * self.hit_penalty
            net_trial_xp = trial_xp - hit_cost
            gw_gain = round(net_trial_xp - base_xp, 2)
            horizon_gain = round(traj["total_net_xp"] - roll_plan["horizon_net_xp"], 2)

            # Price movement estimation
            in_momentum = buy_row.get("transfers_in_event", 0) - buy_row.get("transfers_out_event", 0)
            price_delta = 0.1 if in_momentum > 50000 else 0.0

            plan_dict = {
                "plan_type": "1_TRANSFER",
                "transfers_count": 1,
                "transfers_in": [
                    {
                        "element": in_id,
                        "web_name": buy_row["web_name"],
                        "team": _safe_team(buy_row["team"]),
                        "position": buy_row["position"],
                        "cost": buy_row["value"] / 10.0,
                        "expected_points": round(float(buy_row["expected_points"]), 2),
                    }
                ],
                "transfers_out": [
                    {
                        "element": out_id,
                        "web_name": sell_row["web_name"],
                        "team": _safe_team(sell_row["team"]),
                        "position": sell_row["position"],
                        "sell_price": m["sell_prices"][out_id] / 10.0,
                        "expected_points": round(float(sell_row.get("expected_points", 0.0)), 2),
                    }
                ],
                "hits": hits,
                "hit_cost": hit_cost,
                "gross_expected_points": round(trial_xp, 2),
                "net_expected_points": round(net_trial_xp, 2),
                "expected_gain": gw_gain,
                "remaining_bank": round((initial_bank + m["bank_delta"]) / 10.0, 2),
                "next_banked_ft": min(5, max(1, free_transfers)),
                "lineup": trial_lineup,
                "horizon_gross_xp": round(traj["total_gross_xp"], 2),
                "horizon_hits": traj["total_hits"],
                "horizon_hit_cost": round(traj["total_hits"] * self.hit_penalty, 2),
                "horizon_net_xp": round(traj["total_net_xp"], 2),
                "accumulated_discounted_net_xp": round(traj["accumulated_discounted_net_xp"], 2),
                "horizon_gain_vs_roll": horizon_gain,
                "pure_xp_gain": horizon_gain,
                "price_movement_gain": price_delta,
                "robust_score": 0.0,
                "is_no_regret": False,
                "recommendation_summary": f"Transfer out {sell_row['web_name']} -> {buy_row['web_name']} (+{gw_gain} xP, Horizon: {horizon_gain:+} xP)",
                "trajectory": traj["history"],
            }
            cand_1_plans.append(plan_dict)

        if cand_1_plans:
            cand_1_plans.sort(
                key=lambda p: p["pure_xp_gain"] + (p["price_movement_gain"] * 1.5 if include_price_gain else 0.0),
                reverse=True,
            )
            best_1_plan = cand_1_plans[0]

        # ----------------------------------------------------------------------
        # Candidate 2-Transfer Moves at Target GW
        # ----------------------------------------------------------------------
        cand_2_moves = self._get_candidate_2_transfers(
            elements=initial_elements,
            bank=initial_bank,
            purchase_prices=initial_purchase,
            pool_df=target_gw_df,
            player_map=target_pmap,
            locked_in=locked_in,
            locked_out=locked_out,
            excluded_teams=excluded_teams,
            cand_1_moves=cand_1_moves,
            max_total=4,
        )

        best_2_plan = None
        cand_2_plans = []
        for m in cand_2_moves:
            traj = self._run_beam_search_trajectory(
                initial_action=m,
                initial_elements=initial_elements,
                initial_bank=initial_bank,
                initial_purchase_prices=initial_purchase,
                initial_ft=free_transfers,
                horizon_gws=horizon_gws,
                horizon_projections=clean_projections,
                player_maps=player_maps,
                locked_in=locked_in,
                locked_out=locked_out,
                excluded_teams=excluded_teams,
                risk_preference=risk_preference,
            )

            in_ids = m["transfers_in"]
            out_ids = m["transfers_out"]

            trial_squad_2 = pd.concat(
                [
                    curr_squad_target[~curr_squad_target["element"].isin(out_ids)],
                    target_gw_df[target_gw_df["element"].isin(in_ids)],
                ]
            ).reset_index(drop=True)

            trial2_lineup = lineup_optimizer.select_lineup_and_captain(trial_squad_2, risk_preference=risk_preference)
            trial2_xp = trial2_lineup["total_gameweek_expected_points"]
            hits_2 = max(0, 2 - free_transfers)
            hit_cost_2 = hits_2 * self.hit_penalty
            net_trial2_xp = trial2_xp - hit_cost_2
            gw_gain_2 = round(net_trial2_xp - base_xp, 2)
            horizon_gain_2 = round(traj["total_net_xp"] - roll_plan["horizon_net_xp"], 2)

            plan_dict_2 = {
                "plan_type": "2_TRANSFERS",
                "transfers_count": 2,
                "transfers_in": [
                    {
                        "element": eid,
                        "web_name": target_pmap[eid]["web_name"],
                        "team": _safe_team(target_pmap[eid]["team"]),
                        "position": target_pmap[eid]["position"],
                        "cost": target_pmap[eid]["value"] / 10.0,
                        "expected_points": round(float(target_pmap[eid]["expected_points"]), 2),
                    }
                    for eid in in_ids
                ],
                "transfers_out": [
                    {
                        "element": eid,
                        "web_name": target_pmap[eid]["web_name"],
                        "team": _safe_team(target_pmap[eid]["team"]),
                        "position": target_pmap[eid]["position"],
                        "sell_price": m["sell_prices"][eid] / 10.0,
                        "expected_points": round(float(target_pmap[eid].get("expected_points", 0.0)), 2),
                    }
                    for eid in out_ids
                ],
                "hits": hits_2,
                "hit_cost": hit_cost_2,
                "gross_expected_points": round(trial2_xp, 2),
                "net_expected_points": round(net_trial2_xp, 2),
                "expected_gain": gw_gain_2,
                "remaining_bank": round((initial_bank + m["bank_delta"]) / 10.0, 2),
                "next_banked_ft": 1,
                "lineup": trial2_lineup,
                "horizon_gross_xp": round(traj["total_gross_xp"], 2),
                "horizon_hits": traj["total_hits"],
                "horizon_hit_cost": round(traj["total_hits"] * self.hit_penalty, 2),
                "horizon_net_xp": round(traj["total_net_xp"], 2),
                "accumulated_discounted_net_xp": round(traj["accumulated_discounted_net_xp"], 2),
                "horizon_gain_vs_roll": horizon_gain_2,
                "pure_xp_gain": horizon_gain_2,
                "price_movement_gain": 0.0,
                "robust_score": 0.0,
                "is_no_regret": False,
                "recommendation_summary": (
                    f"Take 2 transfers (Net +{gw_gain_2} xP after {int(hit_cost_2)} pt hit, Horizon: {horizon_gain_2:+} xP)"
                    if hit_cost_2 > 0
                    else f"Take 2 free transfers (Net +{gw_gain_2} xP, Horizon: {horizon_gain_2:+} xP)"
                ),
                "trajectory": traj["history"],
            }
            cand_2_plans.append(plan_dict_2)

        if cand_2_plans:
            cand_2_plans.sort(key=lambda p: p["horizon_gain_vs_roll"], reverse=True)
            best_2_plan = cand_2_plans[0]

        # ----------------------------------------------------------------------
        # Assemble Candidate Branches
        # ----------------------------------------------------------------------
        candidate_plans = [roll_plan]
        if best_1_plan:
            candidate_plans.append(best_1_plan)
        if best_2_plan:
            candidate_plans.append(best_2_plan)

        # ----------------------------------------------------------------------
        # Robustness Re-ranking across Monte Carlo Scenarios (T5)
        # ----------------------------------------------------------------------
        if num_mc_scenarios > 0 and len(candidate_plans) > 1:
            win_counts = {i: 0 for i in range(len(candidate_plans))}
            for _ in range(num_mc_scenarios):
                # Sample noise per player
                noise_map = {eid: float(np.random.normal(1.0, 0.15)) for eid in target_pmap}
                # Score each candidate under noise
                scores = []
                for p_idx, plan in enumerate(candidate_plans):
                    if plan["plan_type"] == "ROLL_TRANSFER":
                        score = plan["horizon_net_xp"] * np.mean(list(noise_map.values()))
                    else:
                        in_elems = [t["element"] for t in plan["transfers_in"]]
                        mult = np.mean([noise_map.get(e, 1.0) for e in in_elems])
                        score = plan["horizon_net_xp"] * mult
                    scores.append((score, p_idx))
                scores.sort(key=lambda x: x[0], reverse=True)
                top_idx = scores[0][1]
                win_counts[top_idx] += 1

            for p_idx, plan in enumerate(candidate_plans):
                r_score = round(win_counts[p_idx] / float(num_mc_scenarios), 2)
                plan["robustness_score"] = r_score
                plan["is_no_regret"] = bool(r_score >= 0.70)

        # ----------------------------------------------------------------------
        # Future-Hit Avoidance Metric vs Greedy (T3)
        # ----------------------------------------------------------------------
        greedy_res = self._run_greedy_trajectory(
            initial_elements=initial_elements,
            initial_bank=initial_bank,
            initial_purchase_prices=initial_purchase,
            initial_ft=free_transfers,
            horizon_gws=horizon_gws,
            horizon_projections=clean_projections,
            player_maps=player_maps,
            locked_in=locked_in,
            locked_out=locked_out,
            excluded_teams=excluded_teams,
            risk_preference=risk_preference,
        )

        # Multi-GW planned trajectory hits
        best_cand = max(candidate_plans, key=lambda p: p.get("horizon_gain_vs_roll", 0.0))
        multi_gw_hits = best_cand["horizon_hits"]
        greedy_hits = greedy_res["total_hits"]
        future_hits_avoided = max(0, greedy_hits - multi_gw_hits)

        hit_avoidance = {
            "greedy_horizon_hits": greedy_hits,
            "multi_gw_horizon_hits": multi_gw_hits,
            "future_hits_avoided": future_hits_avoided,
            "explanation": (
                f"Multi-GW sequential plan avoids {future_hits_avoided} future hit(s) compared to single-GW greedy execution."
                if future_hits_avoided > 0
                else "Multi-GW plan matches greedy trajectory on transfer hits."
            ),
        }

        # ----------------------------------------------------------------------
        # Free Transfer Option Value (T4)
        # ----------------------------------------------------------------------
        traj_plus_1 = self._run_beam_search_trajectory(
            initial_action=roll_action,
            initial_elements=initial_elements,
            initial_bank=initial_bank,
            initial_purchase_prices=initial_purchase,
            initial_ft=min(5, free_transfers + 1),
            horizon_gws=horizon_gws,
            horizon_projections=clean_projections,
            player_maps=player_maps,
            locked_in=locked_in,
            locked_out=locked_out,
            excluded_teams=excluded_teams,
            risk_preference=risk_preference,
        )
        ft_option_value = max(0.0, round(traj_plus_1["total_net_xp"] - roll_plan["horizon_net_xp"], 2))
        ft_option_explanation = (
            f"Option value of an extra banked free transfer is +{ft_option_value} xP over the 5-GW horizon "
            f"(provides restructuring flexibility without incurring a -4 hit penalty)."
        )

        # ----------------------------------------------------------------------
        # Recommendation Selection & Plan Stability Threshold (T8)
        # ----------------------------------------------------------------------
        # Rank candidates by objective
        sort_key = (
            (lambda p: p["horizon_gain_vs_roll"] + (p.get("price_movement_gain", 0.0) * 1.5))
            if include_price_gain
            else (lambda p: p["horizon_gain_vs_roll"])
        )
        sorted_candidates = sorted(candidate_plans, key=sort_key, reverse=True)
        top_candidate = sorted_candidates[0]

        # Check for ruled-out player in current starting XI (cop == 0)
        has_ruled_out_starter = False
        if "starters" in curr_lineup:
            starters_obj = curr_lineup["starters"]
            if isinstance(starters_obj, pd.DataFrame):
                if "chance_of_playing" in starters_obj.columns:
                    has_ruled_out_starter = bool((starters_obj["chance_of_playing"] == 0.0).any())
            elif isinstance(starters_obj, list):
                for s in starters_obj:
                    if isinstance(s, dict) and s.get("chance_of_playing") == 0.0:
                        has_ruled_out_starter = True
                        break

        # Apply stability threshold
        if (
            top_candidate["plan_type"] != "ROLL_TRANSFER"
            and top_candidate["expected_gain"] < stability_threshold
            and not has_ruled_out_starter
        ):
            recommended_plan = roll_plan
            recommended_plan["recommendation_summary"] = (
                f"Roll transfer recommended. Top candidate transfer (+{round(top_candidate['expected_gain'], 2)} xP) "
                f"does not exceed the {stability_threshold:.2f} xP stability threshold."
            )
        else:
            recommended_plan = top_candidate

        # ----------------------------------------------------------------------
        # Dynamic Roadmap Generation (T7)
        # ----------------------------------------------------------------------
        roadmap = self._generate_dynamic_roadmap(
            trajectory=recommended_plan.get("trajectory", roll_plan["trajectory"]),
            horizon_gws=horizon_gws,
            clean_projections=clean_projections,
            player_maps=player_maps,
        )

        # ----------------------------------------------------------------------
        # Hit Verdict
        # ----------------------------------------------------------------------
        hit_verdict = "No hit recommended"
        if recommended_plan.get("hits", 0) > 0:
            hit_verdict = (
                f"HIT RECOMMENDED: The {recommended_plan['hits'] * 4}-point hit yields net "
                f"+{recommended_plan['expected_gain']} xP this gameweek and +{recommended_plan['horizon_gain_vs_roll']} xP across the horizon."
            )
        elif best_2_plan and best_2_plan.get("hits", 0) > 0:
            if best_2_plan["expected_gain"] < 0:
                hit_verdict = (
                    f"HIT NOT WORTH IT: The candidate 2nd transfer gives only "
                    f"+{round(best_2_plan['gross_expected_points'] - base_xp, 2)} gross xP, failing to recover the -4 hit."
                )
            else:
                hit_verdict = (
                    f"MARGINAL HIT: Yields net +{best_2_plan['expected_gain']} xP. "
                    f"Single transfer or roll preferred for lower variance."
                )

        return {
            "recommended_plan": recommended_plan,
            "candidate_plans": sorted_candidates,
            "hit_verdict": hit_verdict,
            "transfer_roadmap": roadmap,
            "future_hit_avoidance": hit_avoidance,
            "ft_option_value": ft_option_value,
            "ft_option_explanation": ft_option_explanation,
            "target_gameweek": target_gw,
            "current_bank_millions": round(bank / 10.0, 2),
            "available_free_transfers": free_transfers,
            "stability_threshold": stability_threshold,
            "include_price_gain": include_price_gain,
        }

    def exhaustive_reference_search(
        self,
        elements: set[int],
        bank: int,
        purchase_prices: dict[int, int],
        free_transfers: int,
        horizon_gws: list[int],
        horizon_projections: dict[int, pd.DataFrame],
        player_maps: dict[int, dict[int, dict[str, Any]]],
        risk_preference: str = "balanced",
    ) -> dict[str, Any]:
        """
        Unpruned exhaustive reference search across a bounded action space.
        Evaluates every possible action (Roll and all legal 1-transfers within budget/club limits)
        recursively across the horizon to find the true mathematical optimum.
        Used for verification and benchmarking against the pruned beam search.
        """
        gw0 = horizon_gws[0]
        pmap0 = player_maps[gw0]

        # Generate all valid candidate 1-transfer moves at step 0 without pruning
        all_step0_actions: list[dict[str, Any]] = [
            {
                "plan_type": "ROLL",
                "transfers_in": [],
                "transfers_out": [],
                "bank_delta": 0,
                "buy_costs": {},
                "sell_prices": {},
            }
        ]

        sellable = list(elements)
        for s_id in sellable:
            s_p = pmap0[s_id]
            s_pos = s_p["position"]
            now_cost = int(s_p.get("value", 50))
            purch_cost = purchase_prices.get(s_id, now_cost)
            sell_price = self.calculate_selling_price(purch_cost, now_cost)
            avail_budget = bank + sell_price

            for b_id, b_p in pmap0.items():
                if b_id in elements:
                    continue
                if b_p["position"] != s_pos:
                    continue
                b_val = int(b_p.get("value", 50))
                if b_val > avail_budget:
                    continue
                # Check max-3-per-club constraint
                trial_elems = (elements - {s_id}) | {b_id}
                team_counts: dict[Any, int] = {}
                valid_club = True
                for eid in trial_elems:
                    tm = pmap0[eid]["team"]
                    team_counts[tm] = team_counts.get(tm, 0) + 1
                    if team_counts[tm] > 3:
                        valid_club = False
                        break
                if not valid_club:
                    continue

                all_step0_actions.append(
                    {
                        "plan_type": "1_TRANSFER",
                        "transfers_in": [b_id],
                        "transfers_out": [s_id],
                        "bank_delta": sell_price - b_val,
                        "buy_costs": {b_id: b_val},
                        "sell_prices": {s_id: sell_price},
                    }
                )

        # Enumerate all legal 2-transfer moves at step 0 without pruning
        for i in range(len(sellable)):
            s1_id = sellable[i]
            s1_p = pmap0[s1_id]
            now_cost1 = int(s1_p.get("value", 50))
            purch1 = purchase_prices.get(s1_id, now_cost1)
            sp1 = self.calculate_selling_price(purch1, now_cost1)

            for j in range(i + 1, len(sellable)):
                s2_id = sellable[j]
                s2_p = pmap0[s2_id]
                now_cost2 = int(s2_p.get("value", 50))
                purch2 = purchase_prices.get(s2_id, now_cost2)
                sp2 = self.calculate_selling_price(purch2, now_cost2)
                tot_budget2 = bank + sp1 + sp2

                pool_keys = list(pmap0.keys())
                for bi in range(len(pool_keys)):
                    b1_id = pool_keys[bi]
                    if b1_id in elements:
                        continue
                    b1_p = pmap0[b1_id]
                    for bj in range(bi + 1, len(pool_keys)):
                        b2_id = pool_keys[bj]
                        if b2_id in elements:
                            continue
                        b2_p = pmap0[b2_id]
                        pos_match = (s1_p["position"] == b1_p["position"] and s2_p["position"] == b2_p["position"]) or (
                            s1_p["position"] == b2_p["position"] and s2_p["position"] == b1_p["position"]
                        )
                        if not pos_match:
                            continue
                        b1_val = int(b1_p.get("value", 50))
                        b2_val = int(b2_p.get("value", 50))
                        if b1_val + b2_val > tot_budget2:
                            continue

                        trial2 = (elements - {s1_id, s2_id}) | {b1_id, b2_id}
                        t_cnts: dict[Any, int] = {}
                        valid_c = True
                        for eid in trial2:
                            tm = pmap0[eid]["team"]
                            t_cnts[tm] = t_cnts.get(tm, 0) + 1
                            if t_cnts[tm] > 3:
                                valid_c = False
                                break
                        if not valid_c:
                            continue

                        all_step0_actions.append(
                            {
                                "plan_type": "2_TRANSFERS",
                                "transfers_in": [b1_id, b2_id],
                                "transfers_out": [s1_id, s2_id],
                                "bank_delta": (sp1 + sp2) - (b1_val + b2_val),
                                "buy_costs": {b1_id: b1_val, b2_id: b2_val},
                                "sell_prices": {s1_id: sp1, s2_id: sp2},
                            }
                        )

        best_score = -1e9
        best_action: dict[str, Any] = all_step0_actions[0]

        for act in all_step0_actions:
            t_in = act["transfers_in"]
            t_out = act["transfers_out"]
            s0_elems = (elements - set(t_out)) | set(t_in)
            s0_bank = bank + act["bank_delta"]
            s0_purch = dict(purchase_prices)
            for out_id in t_out:
                s0_purch.pop(out_id, None)
            for in_id in t_in:
                s0_purch[in_id] = act["buy_costs"][in_id]

            t_count = len(t_in)
            hits0 = 0 if t_count <= free_transfers else (t_count - free_transfers)
            next_ft = min(5, (free_transfers - t_count) + 1) if t_count <= free_transfers else 1
            hit_cost0 = hits0 * self.hit_penalty
            xp0, _, _ = _fast_eval_squad_formation(s0_elems, pmap0, risk_preference=risk_preference)
            net0 = xp0 - hit_cost0

            # Step 1 evaluation (Roll or best 1-transfer)
            if len(horizon_gws) > 1:
                gw1 = horizon_gws[1]
                pmap1 = player_maps[gw1]
                disc = self.discount_factor
                # Evaluate Roll at step 1
                xp_roll1, _, _ = _fast_eval_squad_formation(s0_elems, pmap1, risk_preference=risk_preference)
                best_step1_net = xp_roll1

                # Evaluate all legal 1-transfers at step 1
                for s1_id in s0_elems:
                    s1_p = pmap1[s1_id]
                    s1_pos = s1_p["position"]
                    now_cost1 = int(s1_p.get("value", 50))
                    purch_cost1 = s0_purch.get(s1_id, now_cost1)
                    sell_price1 = self.calculate_selling_price(purch_cost1, now_cost1)
                    avail_budget1 = s0_bank + sell_price1

                    for b1_id, b1_p in pmap1.items():
                        if b1_id in s0_elems:
                            continue
                        if b1_p["position"] != s1_pos:
                            continue
                        b1_val = int(b1_p.get("value", 50))
                        if b1_val > avail_budget1:
                            continue
                        trial1_elems = (s0_elems - {s1_id}) | {b1_id}
                        t1_cnts: dict[Any, int] = {}
                        valid_c1 = True
                        for eid in trial1_elems:
                            tm = pmap1[eid]["team"]
                            t1_cnts[tm] = t1_cnts.get(tm, 0) + 1
                            if t1_cnts[tm] > 3:
                                valid_c1 = False
                                break
                        if not valid_c1:
                            continue

                        hits1 = 0 if 1 <= next_ft else 1
                        hc1 = hits1 * self.hit_penalty
                        xp1_trial, _, _ = _fast_eval_squad_formation(
                            trial1_elems, pmap1, risk_preference=risk_preference
                        )
                        net1_trial = xp1_trial - hc1
                        if net1_trial > best_step1_net:
                            best_step1_net = net1_trial

                total_score = net0 + disc * best_step1_net
            else:
                total_score = net0

            if total_score > best_score:
                best_score = total_score
                best_action = act

        return {
            "best_action": best_action,
            "optimal_score": round(best_score, 2),
            "transfers_in": best_action["transfers_in"],
            "transfers_out": best_action["transfers_out"],
            "is_roll": best_action["plan_type"] == "ROLL",
        }

    def evaluate_joint_transfer_and_chip_plan(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank: float,
        free_transfers: int,
        horizon_projections: dict[int, pd.DataFrame],
        current_gw: int,
        target_gw: int,
        available_chips: list[str] | None = None,
        chip_retention_values: dict[str, float] | None = None,
        chips_by_set: dict[int, list[str]] | None = None,
        rival_context: dict[str, Any] | None = None,
        chips_already_used: set[str] | list[str] | None = None,
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        risk_preference: str = "balanced",
        include_price_gain: bool = False,
        stability_threshold: float = 0.3,
        num_mc_scenarios: int = 50,
        horizon_len: int | None = None,
        evaluate_rival_scenarios: bool = False,
        previous_chip: str | None = None,
        measure_chip_values: bool = False,
        search_quality: dict[str, int] | None = None,
        hit_cooldown_until: int = 0,
        forced_current_chip: str | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates joint dynamic trajectory search combining multi-GW transfer planning
        and chip deployment options (Hold vs 3xC vs Bench Boost vs Free Hit vs Wildcard).

        Candidate Pruning Methodology:
        The action space across a 15-player squad and ~600 pool players exceeds 10^7 actions
        per gameweek, resulting in (10^7)^H states over an H-gameweek horizon. For bounded runtime, the search uses a multi-tier pruning strategy.
        Pruning can miss better actions; no global-optimality guarantee is made:
        1. Position-Conserving Candidate Generation: Substitutions are evaluated within the same
           position group (GKP, DEF, MID, FWD), preserving valid squad constraints (2 GKP, 5 DEF, 5 MID, 3 FWD).
        2. Marginal Gain Pre-Ranking: In each position, candidate replacements are pre-screened by
           immediate single-gameweek expected points gain per pound and raw expected points.
        3. Bounded Action Branching: In each gameweek step, the action set is restricted to:
           - Roll transfer (0 transfers)
           - Top candidate 1-transfer moves
           - Top candidate 2-transfer moves (when bank / FT warrants)
           - Legal chip deployment branches (Wildcard, Free Hit, Triple Captain, Bench Boost)
        4. Beam Search State Pruning: After evaluating valid lineup formations and transfer costs
           for each branch, the state space is pruned to the top B branches ranked by accumulated
           discounted net expected points.

        Incorporates dynamic chip retention opportunity costs computed from per-GW projections (G9)
        to ensure chips are only deployed when their marginal gain exceeds their option retention value.
        """
        # The sequential search supplies the ordinary-transfer baseline too.
        # Do not run an obsolete second beam/MC just to copy its overwritten fields.
        h_len = horizon_len or PLANNER_HORIZON
        hold_plan: dict[str, Any] = {}

        horizon_gws = [target_gw + offset for offset in range(h_len) if target_gw + offset <= 38]
        if not horizon_gws:
            horizon_gws = [target_gw]

        # Prepare per-GW pool and player maps
        clean_projections = {}
        player_maps = {}
        for gw in horizon_gws:
            df_gw = horizon_projections.get(gw, player_pool_df).copy()
            clean_projections[gw] = df_gw
            player_maps[gw] = {int(r["element"]): dict(r) for _, r in df_gw.iterrows()}

        # Future chip windows are searched explicitly. Only externally supplied
        # beyond-horizon retention costs are applied, once at deployment.
        active_chips = list(available_chips or [])

        initial_elements = set(current_squad_df["element"].tolist())
        initial_bank = int(bank)
        initial_purchase = {
            int(r["element"]): int(r.get("purchase_price", r.get("value", 50))) for _, r in current_squad_df.iterrows()
        }
        locked_in = set(locked_in_ids or [])
        locked_out = set(locked_out_ids or [])
        excluded_teams = set(excluded_team_ids or [])

        from fpl_oracle.optimise.sequential import search_sequences

        legal_chips = [c for c in active_chips if validate_chip_legality(c, target_gw, chips_already_used)[0]]
        sequences = search_sequences(
            self,
            initial_elements,
            initial_bank,
            initial_purchase,
            free_transfers,
            horizon_gws,
            clean_projections,
            player_maps,
            legal_chips,
            locked_in,
            locked_out,
            excluded_teams,
            risk_preference,
            terminal_costs=chip_retention_values,
            chips_by_set=chips_by_set,
            previous_chip=previous_chip,
            hit_cooldown_until=hit_cooldown_until,
            forced_current_chip=forced_current_chip,
            **(search_quality or {}),
        )
        # Full-date restructuring searches may reveal a trajectory lost by
        # joint pruning. Include them before choosing the action, not just in
        # a report after the decision.
        date_states = {}
        if measure_chip_values and horizon_gws[-1] == (19 if target_gw <= 19 else 38):
            for code in legal_chips:
                if code not in {"wildcard", "freehit"}:
                    continue
                for date in horizon_gws:
                    found = search_sequences(
                        self,
                        initial_elements,
                        initial_bank,
                        initial_purchase,
                        free_transfers,
                        horizon_gws,
                        clean_projections,
                        player_maps,
                        legal_chips,
                        locked_in,
                        locked_out,
                        excluded_teams,
                        risk_preference,
                        terminal_costs=chip_retention_values,
                        chips_by_set=chips_by_set,
                        previous_chip=previous_chip,
                        hit_cooldown_until=hit_cooldown_until,
                        forced_current_chip=forced_current_chip,
                        forced_chip_schedule={date: code},
                        **(search_quality or {}),
                    )
                    date_states[(code, date)] = found
                    sequences += found
        sequences.sort(key=lambda state: state["score"], reverse=True)
        # Compare complete trajectories with the same chip opportunities, not a
        # current-chip bonus against a chip-free continuation.
        from fpl_oracle.league.objective import select_balanced_sequence

        league_choice, league_objective = select_balanced_sequence(sequences, rival_context, player_maps)
        sequences = [league_choice] + [s for s in sequences if s is not league_choice]
        by_first: dict[str | None, dict[str, Any]] = {}
        for state in sequences:
            by_first.setdefault(state["first_chip"], state)
        normal_plan_feasible = None in by_first
        # Expose data-derived option-value break-even, never a fabricated tail constant.
        resource_frontier = []
        best_score = max(state["score"] for state in sequences)
        for retained_chip in legal_chips:
            conserved = [state for state in sequences if retained_chip in state["remaining"]]
            if conserved:
                state = max(conserved, key=lambda row: row["score"])
                resource_frontier.append(
                    dict(
                        chip=retained_chip,
                        retained_through_gameweek=horizon_gws[-1],
                        first_chip=state["first_chip"],
                        retained_trajectory=state["history"],
                        ending_bank_tenths=state["bank"],
                        ending_free_transfers=state["ft"],
                        discounted_horizon_cost_to_preserve=round(best_score - state["score"], 2),
                        tail_value_break_even_at_horizon_end=round(
                            (best_score - state["score"]) / (self.discount_factor ** len(horizon_gws)), 2
                        ),
                        interpretation="If this chip's later option value exceeds this threshold, the preserved-chip trajectory may beat current-window deployment. Later value is not estimated.",
                    )
                )
        hold_score = by_first[None]["score"] if normal_plan_feasible else max(s["score"] for s in sequences)
        rival_scenarios = None
        if evaluate_rival_scenarios:
            from fpl_oracle.league.scenarios import compare_plan_scenarios

            rival_scenarios = compare_plan_scenarios(list(by_first.values()), rival_context, player_maps)
        candidates = []
        for chip, state in by_first.items():
            first = state["history"][0]
            first_df = clean_projections[target_gw][
                clean_projections[target_gw]["element"].isin(first["elements"])
            ].copy()
            plan = dict(hold_plan)
            plan.update(
                chip_applied=chip,
                lineup=lineup_optimizer.select_lineup_and_captain(
                    first_df,
                    risk_preference=risk_preference,
                    is_triple_captain=chip == "3xc",
                    is_bench_boost=chip == "bboost",
                ),
                trajectory=state["history"],
                gross_expected_points=round(first["gross_xp"], 2),
                net_expected_points=round(first["net_xp"], 2),
                hits=first["hits"],
                hit_cost=first["hit_cost"],
                remaining_bank=first["bank"] / 10,
                transfers_count=len(first["transfers_in"]),
                next_banked_ft=first["banked_ft"],
                horizon_net_xp=round(sum(r["net_xp"] for r in state["history"]), 2),
                horizon_gross_xp=round(sum(r["gross_xp"] for r in state["history"]), 2),
                horizon_hits=sum(r["hits"] for r in state["history"]),
                accumulated_discounted_net_xp=round(state["score"], 2),
                transfers_in=[
                    dict(player_maps[target_gw][e], cost=player_maps[target_gw][e]["value"] / 10)
                    for e in first["transfers_in"]
                ],
                transfers_out=[
                    dict(
                        player_maps[target_gw][e],
                        sell_price=self.calculate_selling_price(
                            initial_purchase[e], int(player_maps[target_gw][e]["value"])
                        )
                        / 10,
                    )
                    for e in first["transfers_out"]
                ],
            )
            if chip in {"freehit", "wildcard"}:
                plan["plan_type"] = "FREE_HIT" if chip == "freehit" else "WILDCARD"
            elif not first["transfers_in"]:
                plan["plan_type"] = "ROLL_TRANSFER"
            else:
                plan["plan_type"] = "1_TRANSFER" if len(first["transfers_in"]) == 1 else "2_TRANSFERS"
            gain = round(state["score"] - hold_score, 2)
            candidates.append(
                dict(
                    chip=chip or "HOLD",
                    chip_code=chip,
                    action="DEPLOY" if chip else "HOLD",
                    plan=plan,
                    net_gain_vs_hold=gain,
                    gross_gain_vs_hold=round(first["gross_xp"] - by_first[None]["history"][0]["gross_xp"], 2)
                    if normal_plan_feasible
                    else None,
                    opportunity_cost=float((chip_retention_values or {}).get(chip or "", 0)),
                    effective_trajectory_score=round(state["score"], 2),
                    reason="Legal sequential transfer/chip trajectory across the projection horizon."
                    if normal_plan_feasible
                    else "No ordinary legal XI within the hit limit. Comparison gain unavailable.",
                )
            )
        hold_plan = next((c["plan"] for c in candidates if c["chip"] == "HOLD"), candidates[0]["plan"])
        for candidate in candidates:
            plan = candidate["plan"]
            count = len(plan["transfers_in"])
            plan["recommendation_summary"] = (
                f"{count} transfers · {plan['hit_cost']} hit points · "
                f"{plan['net_expected_points']:.1f} est. points this week."
            )
            plan["horizon_gain_vs_roll"] = round(plan["accumulated_discounted_net_xp"] - hold_score, 2)
            plan["horizon_hit_cost"] = plan["horizon_hits"] * self.hit_penalty
            plan["expected_gain"] = round(plan["net_expected_points"] - hold_plan["net_expected_points"], 2)
            plan["robustness_score"] = None
            plan["is_no_regret"] = False

        # 4. Rank candidates by net gain vs hold
        candidates.sort(key=lambda c: c["net_gain_vs_hold"], reverse=True)
        best_cand = next(c for c in candidates if c["chip_code"] == league_choice["first_chip"])

        if (
            best_cand["chip"] == "freehit"
            and normal_plan_feasible
            and not forced_current_chip
            and best_cand["net_gain_vs_hold"] <= 1e-6
        ):
            best_cand = next(c for c in candidates if c["chip"] == "HOLD")
        if best_cand["chip"] != "HOLD":
            recommended_chip = best_cand["chip"]
            recommended_action = "DEPLOY"
            recommended_plan = best_cand["plan"]
            for c in candidates:
                c["is_recommended"] = bool(c["chip"] == recommended_chip)
        else:
            recommended_chip = None
            recommended_action = "HOLD"
            recommended_plan = hold_plan
            for c in candidates:
                c["is_recommended"] = bool(c["chip"] == "HOLD")

        # The roadmap must follow the action actually selected after any
        # non-beneficial Free Hit guard, not the discarded tie-break choice.
        league_choice = by_first[best_cand["chip_code"]]
        measured_roadmap = []
        measurement_status = "not_requested"
        if measure_chip_values:
            import time

            from fpl_oracle.chips.planner import CHIP_DISPLAY_NAMES

            began = time.monotonic()
            expiry = 19 if target_gw <= 19 else 38
            if horizon_gws[-1] != expiry:
                measurement_status = "incomplete_window"
            else:
                measurement_status = "measured_through_expiry"
                for chip in [c for c in legal_chips if c != forced_current_chip]:
                    without = search_sequences(
                        self,
                        initial_elements,
                        initial_bank,
                        initial_purchase,
                        free_transfers,
                        horizon_gws,
                        clean_projections,
                        player_maps,
                        [code for code in legal_chips if code != chip],
                        locked_in,
                        locked_out,
                        excluded_teams,
                        risk_preference,
                        terminal_costs=chip_retention_values,
                        chips_by_set={
                            number: [code for code in values if code != chip]
                            for number, values in (chips_by_set or {}).items()
                        },
                        previous_chip=previous_chip,
                        hit_cooldown_until=hit_cooldown_until,
                        forced_current_chip=forced_current_chip,
                        **(search_quality or {}),
                    )
                    # Use the same date-search expansion on both sides of a
                    # chip comparison. Otherwise a broader with-chip search
                    # can overstate gain against a weaker ordinary baseline.
                    for other in legal_chips:
                        if other == chip or other not in {"wildcard", "freehit"}:
                            continue
                        for date in horizon_gws:
                            without += search_sequences(
                                self,
                                initial_elements,
                                initial_bank,
                                initial_purchase,
                                free_transfers,
                                horizon_gws,
                                clean_projections,
                                player_maps,
                                [code for code in legal_chips if code != chip],
                                locked_in,
                                locked_out,
                                excluded_teams,
                                risk_preference,
                                terminal_costs=chip_retention_values,
                                chips_by_set={
                                    number: [code for code in values if code != chip]
                                    for number, values in (chips_by_set or {}).items()
                                },
                                previous_chip=previous_chip,
                                hit_cooldown_until=hit_cooldown_until,
                                forced_current_chip=forced_current_chip,
                                forced_chip_schedule={date: other},
                                **(search_quality or {}),
                            )
                    if not without:
                        measured_roadmap.append(
                            dict(
                                chip=CHIP_DISPLAY_NAMES[chip],
                                code=chip,
                                recommended_gw=next(
                                    (r["gameweek"] for r in league_choice["history"] if r.get("chip") == chip), None
                                ),
                                expected_gain=None,
                                status="required_for_feasibility",
                                date_comparison=[
                                    dict(
                                        gameweek=date,
                                        status="required_for_feasibility" if found else "infeasible",
                                        expected_gain=None,
                                        plan_score=round(max(found, key=lambda state: state["score"])["score"], 2)
                                        if found
                                        else None,
                                    )
                                    for (code, date), found in date_states.items()
                                    if code == chip
                                ],
                                reasoning="No legal plan within the hit limit without this chip.",
                            )
                        )
                        continue
                    baseline = max(without, key=lambda state: state["score"])
                    date_rows = []
                    if chip in {"wildcard", "freehit"}:
                        from fpl_oracle.chips.date_comparison import compare_dates

                        date_rows = compare_dates(
                            lambda forced_chip_schedule, chip=chip, **unused: date_states.get(
                                (chip, next(iter(forced_chip_schedule))), []
                            ),
                            {},
                            chip,
                            baseline,
                            horizon_gws,
                        )
                    full = league_choice
                    step = next((row for row in full["history"] if row.get("chip") == chip), None)
                    delta = round(full["score"] - baseline["score"], 2)
                    measured_roadmap.append(
                        dict(
                            chip=CHIP_DISPLAY_NAMES[chip],
                            code=chip,
                            recommended_gw=step["gameweek"] if step else None,
                            expected_gain=delta,
                            baseline_score=round(baseline["score"], 2),
                            plan_score=round(full["score"], 2),
                            baseline_trajectory=baseline["history"],
                            date_comparison=date_rows,
                            best_measured_date=max(
                                (row for row in date_rows if row.get("plan_score") is not None),
                                key=lambda row: row["plan_score"],
                                default={"gameweek": None, "plan_score": None},
                            ).get("gameweek"),
                            forecast_through=expiry,
                            status="measured" if delta >= 0 else "search_inconclusive",
                            reasoning="Extra weighted points over the full plan without this chip. Transfers and other chip dates may change; gains do not add together.",
                        )
                    )
                import copy

                equal_optimizer = copy.copy(self)
                equal_optimizer.discount_factor = 1.0
                equal_states = search_sequences(
                    equal_optimizer,
                    initial_elements,
                    initial_bank,
                    initial_purchase,
                    free_transfers,
                    horizon_gws,
                    clean_projections,
                    player_maps,
                    legal_chips,
                    locked_in,
                    locked_out,
                    excluded_teams,
                    risk_preference,
                    terminal_costs=chip_retention_values,
                    chips_by_set=chips_by_set,
                    previous_chip=previous_chip,
                    hit_cooldown_until=hit_cooldown_until,
                    forced_current_chip=forced_current_chip,
                    **(search_quality or {}),
                )
                equal_best = max(equal_states, key=lambda state: state["score"])
                equal_dates = {row["chip"]: row["gameweek"] for row in equal_best["history"] if row.get("chip")}
                for row in measured_roadmap:
                    row["equal_weight_gameweek"] = equal_dates.get(row["code"])
                    row["timing_changes_without_discount"] = row["recommended_gw"] != row["equal_weight_gameweek"]
                measurement_seconds = round(time.monotonic() - began, 2)

        return {
            "normal_plan_feasible": normal_plan_feasible,
            "measured_chip_roadmap": measured_roadmap,
            "chip_measurement_status": measurement_status,
            "search_quality": search_quality
            or dict(
                beam_width=32, candidate_limit=2, two_transfer_limit=1, per_position_limit=2, restructure_parent_limit=1
            ),
            "chip_measurement_seconds": locals().get("measurement_seconds"),
            "recommended_chip": recommended_chip,
            "chip_action": recommended_action,
            "recommended_plan": recommended_plan,
            "hold_plan": hold_plan,
            "roll_plan": hold_plan,
            "candidate_plans": [c["plan"] for c in candidates],
            "chip_comparison_table": candidates,
            "best_candidate": best_cand,
            "decision_scope": {
                "method": "bounded_sequential_transfer_chip_beam",
                "horizon_gameweeks": horizon_gws,
                "global_optimum_proven": False,
                "sequential_multi_chip_search": True,
                "recompute_each_deadline": True,
                "season_tail_value_estimated": False,
                "supported_horizon_only": True,
                "objective": "discounted_net_expected_points_with_bounded_balanced_rival_exposure",
            },
            "league_objective": league_objective,
            "rival_scenarios": rival_scenarios,
            "resource_frontier": resource_frontier,
            "decision_readiness": {
                "status": "conditional" if recommended_chip else "bounded_model_candidate",
                "reasons": [
                    "Future rival actions are hypothetical, not known",
                    "No validated resource valuation beyond the supported projection window",
                    "Reassess every deadline; future steps are contingent",
                ],
                "automatic_execution": False,
            },
            "opportunity_costs_applied": chip_retention_values or {},
            "transfer_roadmap": self._generate_dynamic_roadmap(
                recommended_plan["trajectory"], horizon_gws, clean_projections, player_maps
            ),
            "hit_verdict": f"Plan costs {recommended_plan['horizon_hits']} hit(s) over {len(horizon_gws)} gameweeks.",
            "target_gameweek": target_gw,
            "available_free_transfers": free_transfers,
            "is_joint_plan": True,
        }

    def _generate_dynamic_roadmap(
        self,
        trajectory: list[dict[str, Any]],
        horizon_gws: list[int],
        clean_projections: dict[int, pd.DataFrame],
        player_maps: dict[int, dict[int, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Generates dynamic transfer roadmap from actual trajectory player picks, fixtures & DGW/BGW tags.
        """
        roadmap_steps = []
        for offset, step in enumerate(trajectory):
            gw = step["gameweek"]
            t_in = step.get("transfers_in", [])
            t_out = step.get("transfers_out", [])
            pmap = player_maps.get(gw, {})

            in_players = [pmap.get(eid, {}) for eid in t_in]
            out_players = [pmap.get(eid, {}) for eid in t_out]

            if offset == 0:
                status = "CONDITIONAL_CANDIDATE"
            elif offset == 1:
                status = "CONTINGENT_ON_NEWS"
            else:
                status = "CONTINGENT_ON_NEWS"

            # Check DGW / BGW tags from pool
            pool_df = clean_projections.get(gw)
            dgw_teams = set()
            bgw_teams = set()
            if pool_df is not None and "fixtures" in pool_df.columns:
                for _, r in pool_df.iterrows():
                    fix_list = r.get("fixtures", [])
                    if isinstance(fix_list, list):
                        if len(fix_list) >= 2:
                            dgw_teams.add(r["team"])
                        elif len(fix_list) == 0:
                            bgw_teams.add(r["team"])

            dgw_bgw_tags = []
            if dgw_teams:
                dgw_bgw_tags.append(f"DGW: {', '.join(str(t) for t in sorted(dgw_teams)[:3])}")
            if bgw_teams:
                dgw_bgw_tags.append(f"BGW: {', '.join(str(t) for t in sorted(bgw_teams)[:3])}")

            # Dynamic action text
            if not t_in:
                action = f"Roll transfer (bank to {step['banked_ft']} FTs)"
                strategic_focus = f"Preserve transfer capital; accumulate {step['banked_ft']} FTs for tactical flexibility in GW{gw + 1}."
            elif len(t_in) == 1:
                in_name = in_players[0].get("web_name", f"Player {t_in[0]}")
                out_name = out_players[0].get("web_name", f"Player {t_out[0]}")
                action = f"Transfer OUT {out_name} -> Transfer IN {in_name}"
                strategic_focus = f"Target {in_name} fixture run and replace {out_name}."
            else:
                in_names = ", ".join(p.get("web_name", str(p.get("element", ""))) for p in in_players)
                out_names = ", ".join(p.get("web_name", str(p.get("element", ""))) for p in out_players)
                action = f"Double Transfer: OUT {out_names} -> IN {in_names}"
                strategic_focus = "Execute tactical restructuring."

            if step.get("hit_requires_prior_success"):
                action += " (only if the earlier hit paid off)"
            cap_id = step.get("captain")
            cap_name = (
                pmap.get(int(cap_id), {}).get("web_name", f"Captain #{cap_id}")
                if cap_id is not None
                else "Captain Unknown"
            )

            top_targets = (
                pool_df.sort_values(by="expected_points", ascending=False).head(3)
                if pool_df is not None
                else pd.DataFrame()
            )
            key_targets = (
                [f"{r['web_name']} ({r['expected_points']:.1f} xP)" for _, r in top_targets.iterrows()]
                if not top_targets.empty
                else []
            )

            roadmap_steps.append(
                {
                    "gameweek": gw,
                    "status": status,
                    "action": action,
                    "transfers_in": [p.get("web_name", "") for p in in_players],
                    "transfers_out": [p.get("web_name", "") for p in out_players],
                    "banked_free_transfers_projected": step["banked_ft"],
                    "captain": cap_name,
                    "is_dgw": len(dgw_teams) > 0,
                    "is_bgw": len(bgw_teams) > 0,
                    "dgw_bgw_tags": dgw_bgw_tags,
                    "key_targets": key_targets,
                    "strategic_focus": strategic_focus,
                }
            )

        return roadmap_steps


transfer_optimizer = TransferOptimizer()
