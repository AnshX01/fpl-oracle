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

from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.config import SETTINGS
from fpl_oracle.optimise.lineup import lineup_optimizer


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


def _fast_eval_squad_formation(
    elements: set[int],
    player_map: dict[int, dict[str, Any]],
    risk_preference: str = "balanced",
) -> tuple[float, int, tuple[int, int, int]]:
    """
    Evaluates 15-player squad formation and captaincy in < 0.05ms.
    Returns (total_xp, captain_element_id, formation_tuple).
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
    if starters:
        captain = max(starters, key=lambda x: x[0])
        cap_bonus = captain[0]
        cap_id = captain[1]
    else:
        cap_bonus = 0.0
        cap_id = 0

    total_xp = starter_gk_xp + (best_outfield_xp if best_outfield_xp > -1e8 else 0.0) + cap_bonus
    return total_xp, cap_id, best_form


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
        for entry in entries:
            gw = getattr(entry, "event", 0)
            transfers_made = getattr(entry, "event_transfers", 0)
            active_chip = chips_used.get(gw, "")

            if active_chip in ("wildcard", "freehit"):
                banked = min(5, banked + 1)
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
        - Starts at 1 in GW1
        """
        if not entry_history:
            return 1

        banked = 1
        for gw in range(1, current_gw + 1):
            gw_entry = next(
                (
                    e
                    for e in entry_history
                    if (getattr(e, "event", None) or (isinstance(e, dict) and e.get("event"))) == gw
                ),
                None,
            )
            transfers_made = 0
            if gw_entry:
                transfers_made = (
                    getattr(gw_entry, "event_transfers", 0)
                    if not isinstance(gw_entry, dict)
                    else gw_entry.get("event_transfers", 0)
                )

            remaining = max(0, banked - transfers_made)
            banked = min(5, remaining + 1)

        return max(1, min(5, banked))

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

            cands = pool_df[
                (pool_df["position"] == sell_pos)
                & (~pool_df["element"].isin(elements))
                & (~pool_df["element"].isin(locked_out))
                & (~pool_df["team"].isin(excluded_teams))
                & (pool_df["value"] <= avail_budget)
            ].sort_values(by="expected_points", ascending=False)

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
    ) -> dict[str, Any]:
        """
        Executes stateful multi-GW beam search from initial_action across horizon_gws.
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
        if transfers_count <= initial_ft:
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
    ) -> dict[str, Any]:
        """
        Stateful Multi-GW Transfer Optimizer evaluating candidate branches across a 5-GW horizon.
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

        # Horizon gameweeks (up to 5 GWs)
        horizon_gws = [target_gw + offset for offset in range(5) if target_gw + offset <= 38]
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
        locked_in_ids: list[int] | None = None,
        locked_out_ids: list[int] | None = None,
        excluded_team_ids: list[int] | None = None,
        risk_preference: str = "balanced",
        include_price_gain: bool = False,
        stability_threshold: float = 0.3,
        num_mc_scenarios: int = 50,
    ) -> dict[str, Any]:
        """
        Evaluates joint dynamic trajectory search combining multi-GW transfer planning
        and chip deployment options (Hold vs 3xC vs Bench Boost vs Free Hit vs Wildcard).
        Incorporates explicit chip retention opportunity cost to ensure chips are only
        deployed when their marginal gain exceeds their option retention value.
        """
        # 1. Base transfer optimization (HOLD case)
        base_res = self.evaluate_transfer_options(
            current_squad_df=current_squad_df,
            player_pool_df=player_pool_df,
            bank=bank,
            free_transfers=free_transfers,
            horizon_projections=horizon_projections,
            current_gw=current_gw,
            target_gw=target_gw,
            locked_in_ids=locked_in_ids,
            locked_out_ids=locked_out_ids,
            excluded_team_ids=excluded_team_ids,
            risk_preference=risk_preference,
            include_price_gain=include_price_gain,
            stability_threshold=stability_threshold,
            num_mc_scenarios=num_mc_scenarios,
        )
        hold_plan = dict(base_res["recommended_plan"])
        hold_discounted_score = float(
            hold_plan.get("accumulated_discounted_net_xp") or hold_plan.get("horizon_net_xp", 0.0)
        )

        # 2. Chip retention opportunity cost definitions
        default_retention = {
            "3xc": 6.0,  # Opportunity cost of burning TC outside a Double Gameweek
            "bboost": 10.0,  # Opportunity cost of burning BB outside a full-squad DGW
            "freehit": 10.0,  # Opportunity cost of burning FH outside a major blank/double GW
            "wildcard": 8.0,  # Opportunity cost of burning WC outside an emergency/major swing
        }
        retention_costs = dict(default_retention)
        if chip_retention_values:
            retention_costs.update(chip_retention_values)

        active_chips = list(available_chips or [])

        # 3. Evaluate candidate trajectories
        candidates: list[dict[str, Any]] = [
            {
                "chip": "HOLD",
                "chip_code": None,
                "action": "HOLD",
                "gross_gain_vs_hold": 0.0,
                "opportunity_cost": 0.0,
                "net_gain_vs_hold": 0.0,
                "effective_trajectory_score": round(hold_discounted_score, 2),
                "plan": hold_plan,
                "reason": "Hold chips for higher expected value windows (e.g. Double Gameweeks).",
            }
        ]

        if "selling_price" in current_squad_df.columns:
            squad_selling_total = float(current_squad_df["selling_price"].sum())
        elif "value" in current_squad_df.columns:
            squad_selling_total = float(current_squad_df["value"].sum())
        else:
            squad_selling_total = 1000.0
        total_budget = bank + squad_selling_total

        # Evaluate 3xc
        if "3xc" in active_chips:
            cap_info = hold_plan.get("lineup", {}).get("captain", {})
            cap_xp = (
                float(cap_info.get("expected_points", 0.0))
                if isinstance(cap_info, dict)
                else float(getattr(cap_info, "expected_points", 0.0))
            )
            gross_tc = round(cap_xp, 2)
            cost_tc = round(retention_costs.get("3xc", 6.0), 2)
            net_tc = round(gross_tc - cost_tc, 2)
            score_tc = round(hold_discounted_score + net_tc, 2)

            tc_plan = dict(hold_plan)
            tc_plan["chip_applied"] = "3xc"
            tc_plan["gross_expected_points"] = round(hold_plan["gross_expected_points"] + gross_tc, 2)
            tc_plan["net_expected_points"] = round(hold_plan["net_expected_points"] + gross_tc, 2)

            cap_name = (
                cap_info.get("web_name", "Captain")
                if isinstance(cap_info, dict)
                else getattr(cap_info, "web_name", "Captain")
            )
            candidates.append(
                {
                    "chip": "3xc",
                    "chip_code": "3xc",
                    "action": "DEPLOY",
                    "gross_gain_vs_hold": gross_tc,
                    "opportunity_cost": cost_tc,
                    "net_gain_vs_hold": net_tc,
                    "effective_trajectory_score": score_tc,
                    "plan": tc_plan,
                    "reason": f"Triple Captain on {cap_name} gains +{gross_tc} xP (retention cost: {cost_tc} xP, net: {net_tc:+} xP).",
                }
            )

        # Evaluate Bench Boost
        if "bboost" in active_chips:
            bench_obj = hold_plan.get("lineup", {}).get("bench")
            if isinstance(bench_obj, pd.DataFrame):
                bench_xp = (
                    float(bench_obj["expected_points"].sum())
                    if not bench_obj.empty and "expected_points" in bench_obj.columns
                    else 0.0
                )
            elif isinstance(bench_obj, list):
                bench_xp = sum(float(b.get("expected_points", 0.0)) for b in bench_obj)
            else:
                bench_xp = 0.0
            gross_bb = round(bench_xp, 2)
            cost_bb = round(retention_costs.get("bboost", 10.0), 2)
            net_bb = round(gross_bb - cost_bb, 2)
            score_bb = round(hold_discounted_score + net_bb, 2)

            bb_plan = dict(hold_plan)
            bb_plan["chip_applied"] = "bboost"
            bb_plan["gross_expected_points"] = round(hold_plan["gross_expected_points"] + gross_bb, 2)
            bb_plan["net_expected_points"] = round(hold_plan["net_expected_points"] + gross_bb, 2)

            candidates.append(
                {
                    "chip": "bboost",
                    "chip_code": "bboost",
                    "action": "DEPLOY",
                    "gross_gain_vs_hold": gross_bb,
                    "opportunity_cost": cost_bb,
                    "net_gain_vs_hold": net_bb,
                    "effective_trajectory_score": score_bb,
                    "plan": bb_plan,
                    "reason": f"Bench Boost contributes bench points (+{gross_bb} xP, retention cost: {cost_bb} xP, net: {net_bb:+} xP).",
                }
            )

        # Evaluate Free Hit
        if "freehit" in active_chips:
            try:
                from fpl_oracle.optimise.squad import squad_optimizer

                fh_solve = squad_optimizer.solve_best_squad(
                    player_pool_df=player_pool_df,
                    budget=total_budget,
                    metric_col="expected_points",
                )
                fh_squad = fh_solve["squad"]
                fh_lineup = lineup_optimizer.select_lineup_and_captain(fh_squad, risk_preference=risk_preference)
                fh_gross = float(fh_lineup["total_gameweek_expected_points"])
                hold_gross = float(hold_plan["gross_expected_points"])
                gross_fh = round(max(0.0, fh_gross - hold_gross), 2)
                cost_fh = round(retention_costs.get("freehit", 10.0), 2)
                net_fh = round(gross_fh - cost_fh, 2)
                score_fh = round(hold_discounted_score + net_fh, 2)

                fh_plan = dict(hold_plan)
                fh_plan["chip_applied"] = "freehit"
                fh_plan["plan_type"] = "FREE_HIT"
                fh_plan["lineup"] = fh_lineup
                fh_plan["gross_expected_points"] = round(fh_gross, 2)
                fh_plan["net_expected_points"] = round(fh_gross, 2)
                fh_plan["hits"] = 0
                fh_plan["hit_cost"] = 0.0

                candidates.append(
                    {
                        "chip": "freehit",
                        "chip_code": "freehit",
                        "action": "DEPLOY",
                        "gross_gain_vs_hold": gross_fh,
                        "opportunity_cost": cost_fh,
                        "net_gain_vs_hold": net_fh,
                        "effective_trajectory_score": score_fh,
                        "plan": fh_plan,
                        "reason": f"Free Hit single-gameweek restructure yields +{gross_fh} xP (retention cost: {cost_fh} xP, net: {net_fh:+} xP).",
                    }
                )
            except Exception:
                pass

        # Evaluate Wildcard
        if "wildcard" in active_chips:
            try:
                from fpl_oracle.optimise.squad import squad_optimizer

                wc_solve = squad_optimizer.solve_best_squad(
                    player_pool_df=player_pool_df,
                    budget=total_budget,
                    metric_col="expected_points",
                )
                wc_squad = wc_solve["squad"]
                wc_lineup = lineup_optimizer.select_lineup_and_captain(wc_squad, risk_preference=risk_preference)
                wc_step0_gross = float(wc_lineup["total_gameweek_expected_points"])

                horizon_gws = [target_gw + offset for offset in range(5) if target_gw + offset <= 38]
                wc_multi_net = 0.0
                for idx, h_gw in enumerate(horizon_gws):
                    disc = self.discount_factor**idx
                    gw_df = horizon_projections.get(h_gw, player_pool_df)
                    sub_df = gw_df[gw_df["element"].isin(wc_squad["element"])].copy()
                    if len(sub_df) < 15:
                        sub_df = wc_squad.copy()
                    sub_lineup = lineup_optimizer.select_lineup_and_captain(sub_df, risk_preference=risk_preference)
                    wc_multi_net += disc * float(sub_lineup["total_gameweek_expected_points"])

                gross_wc = round(max(0.0, wc_multi_net - hold_discounted_score), 2)
                cost_wc = round(retention_costs.get("wildcard", 8.0), 2)
                net_wc = round(gross_wc - cost_wc, 2)
                score_wc = round(hold_discounted_score + net_wc, 2)

                wc_plan = dict(hold_plan)
                wc_plan["chip_applied"] = "wildcard"
                wc_plan["plan_type"] = "WILDCARD"
                wc_plan["lineup"] = wc_lineup
                wc_plan["gross_expected_points"] = round(wc_step0_gross, 2)
                wc_plan["net_expected_points"] = round(wc_step0_gross, 2)
                wc_plan["hits"] = 0
                wc_plan["hit_cost"] = 0.0

                candidates.append(
                    {
                        "chip": "wildcard",
                        "chip_code": "wildcard",
                        "action": "DEPLOY",
                        "gross_gain_vs_hold": gross_wc,
                        "opportunity_cost": cost_wc,
                        "net_gain_vs_hold": net_wc,
                        "effective_trajectory_score": score_wc,
                        "plan": wc_plan,
                        "reason": f"Wildcard multi-GW restructure yields +{gross_wc} trajectory xP (retention cost: {cost_wc} xP, net: {net_wc:+} xP).",
                    }
                )
            except Exception:
                pass

        # 4. Rank candidates by net gain vs hold
        candidates.sort(key=lambda c: c["net_gain_vs_hold"], reverse=True)
        best_cand = candidates[0]

        if best_cand["chip"] != "HOLD" and best_cand["net_gain_vs_hold"] > 0.0:
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

        return {
            "recommended_chip": recommended_chip,
            "chip_action": recommended_action,
            "recommended_plan": recommended_plan,
            "hold_plan": hold_plan,
            "chip_comparison_table": candidates,
            "best_candidate": best_cand,
            "opportunity_costs_applied": retention_costs,
            "transfer_roadmap": base_res.get("transfer_roadmap", []),
            "hit_verdict": base_res.get("hit_verdict", ""),
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
                status = "FIRM"
            elif offset == 1:
                status = "PROBABLE"
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
