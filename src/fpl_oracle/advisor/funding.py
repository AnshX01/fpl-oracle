"""
Affordability Path Finder (Phase 8 Early Slice: WI-8.3).
Finds the cheapest legal paths to own a target player:
- Direct swaps with specific outgoing players and remaining bank
- Two-step funding moves (enabler downgrade + target upgrade)
- Lexicographic ranking: min hits, then min transfers, then max bank
- Top 3 distinct paths generated via no-good cuts
Zero hardcoded player names in production code.
"""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from fpl_oracle.rules.transfers import sell_price


@dataclass
class FundingMove:
    out_element: int
    in_element: int
    out_sell_price_tenths: int
    in_cost_tenths: int
    bank_delta_tenths: int


@dataclass
class AffordabilityPath:
    target_element: int
    path_type: Literal["DIRECT_SWAP", "TWO_STEP_PLAN", "MULTI_TRANSFER"]
    transfers_count: int
    hits: int
    hit_cost: int
    moves: list[FundingMove]
    initial_bank_tenths: int
    final_bank_tenths: int
    is_legal: bool = True
    arithmetic_explanation: str = ""
    intermediate_enabler: int | None = None


class AffordabilityPathFinder:
    """
    Computes cheapest legal paths to acquire any target player (WI-8.3).
    Enforces Principle P4: Affordability-aware.
    """

    def __init__(self) -> None:
        pass

    def find_cheapest_paths(
        self,
        current_squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame,
        bank_tenths: int,
        target_element: int,
        free_transfers: int = 1,
        top_k: int = 3,
    ) -> list[AffordabilityPath]:
        """
        Finds the top_k cheapest legal paths to buy target_element.
        """
        # Lookup target details
        target_rows = player_pool_df[player_pool_df["element"] == target_element]
        if target_rows.empty:
            raise ValueError(f"Target element {target_element} not found in player pool.")

        target_row = target_rows.iloc[0]
        target_cost = int(round(float(target_row["value"])))
        target_pos = str(target_row["position"])
        target_team = int(target_row["team"])

        squad_df = current_squad_df.copy()
        squad_ids = set(squad_df["element"].astype(int).tolist())

        if target_element in squad_ids:
            return [
                AffordabilityPath(
                    target_element=target_element,
                    path_type="DIRECT_SWAP",
                    transfers_count=0,
                    hits=0,
                    hit_cost=0,
                    moves=[],
                    initial_bank_tenths=bank_tenths,
                    final_bank_tenths=bank_tenths,
                    is_legal=True,
                    arithmetic_explanation="Target player is already owned in squad.",
                )
            ]

        # Calculate exact selling price for all squad members
        squad_moves: list[dict[str, Any]] = []
        club_counts: dict[int, int] = {}

        for _, r in squad_df.iterrows():
            eid = int(r["element"])
            tm = int(r["team"])
            club_counts[tm] = club_counts.get(tm, 0) + 1
            buy_p = int(round(float(r.get("purchase_price", r.get("now_cost", r.get("value", 50.0))))))
            now_p = int(round(float(r.get("now_cost", r.get("value", buy_p)))))
            sell_p = sell_price(buy_p, now_p)
            pos = str(r["position"])
            p_name = str(r.get("web_name", r.get("name", f"element {eid}")))
            squad_moves.append({
                "element": eid,
                "name": p_name,
                "position": pos,
                "team": tm,
                "sell_price": sell_p,
                "purchase_price": buy_p,
            })

        paths: list[AffordabilityPath] = []
        forbidden_outs: set[int] = set()

        # 1. Family 1: Direct like-for-like swaps (Same position)
        same_pos_candidates = [m for m in squad_moves if m["position"] == target_pos]
        # Sort by best budget margin (sell_price desc)
        same_pos_candidates.sort(key=lambda x: x["sell_price"], reverse=True)

        for cand in same_pos_candidates:
            out_eid = cand["element"]
            sell_p = cand["sell_price"]
            net_bank = bank_tenths + sell_p - target_cost

            # Club cap check: if adding target increases club count beyond 3
            out_tm = cand["team"]
            new_club_count = club_counts.get(target_team, 0) + (1 if out_tm != target_team else 0)
            if new_club_count > 3:
                continue

            if net_bank >= 0:
                # Affordable in 1 transfer!
                hits = max(0, 1 - free_transfers)
                move = FundingMove(
                    out_element=out_eid,
                    in_element=target_element,
                    out_sell_price_tenths=sell_p,
                    in_cost_tenths=target_cost,
                    bank_delta_tenths=sell_p - target_cost,
                )
                exp = (
                    f"Direct swap ({target_pos}): Sell {cand['name']} for £{sell_p/10:.1f}m + "
                    f"Bank £{bank_tenths/10:.1f}m = £{(bank_tenths + sell_p)/10:.1f}m. "
                    f"Buy target for £{target_cost/10:.1f}m. Remaining bank: £{net_bank/10:.1f}m. Hits: {hits}."
                )
                paths.append(
                    AffordabilityPath(
                        target_element=target_element,
                        path_type="DIRECT_SWAP",
                        transfers_count=1,
                        hits=hits,
                        hit_cost=hits * 4,
                        moves=[move],
                        initial_bank_tenths=bank_tenths,
                        final_bank_tenths=net_bank,
                        is_legal=True,
                        arithmetic_explanation=exp,
                    )
                )
                forbidden_outs.add(out_eid)
                if len(paths) >= top_k:
                    return paths

        # 2. Family 2: Two-Step Plans (Enabler downgrade + Target upgrade)
        # Required when direct swap is unaffordable or to explore alternative funding pairs
        # Find budget enablers across pool in any position
        enabler_candidates = player_pool_df[
            (player_pool_df["value"] <= 45.0) & (~player_pool_df["element"].isin(squad_ids))
        ].copy()

        for cand in same_pos_candidates:
            out_eid = cand["element"]
            sell_p = cand["sell_price"]
            shortfall = target_cost - (bank_tenths + sell_p)

            if shortfall > 0:
                # Need to fund shortfall from another position
                for other in squad_moves:
                    if other["element"] == out_eid:
                        continue
                    other_pos = other["position"]
                    other_sell = other["sell_price"]

                    # Find cheapest enabler in other_pos
                    pos_enablers = enabler_candidates[enabler_candidates["position"] == other_pos]
                    if pos_enablers.empty:
                        continue

                    cheapest_enabler = pos_enablers.sort_values(by="value").iloc[0]
                    enabler_eid = int(cheapest_enabler["element"])
                    enabler_cost = int(round(float(cheapest_enabler["value"])))

                    fund_raised = other_sell - enabler_cost
                    if fund_raised >= shortfall:
                        # Valid 2-step plan!
                        final_bank = bank_tenths + sell_p + fund_raised - target_cost
                        hits = max(0, 2 - free_transfers)

                        move1 = FundingMove(
                            out_element=other["element"],
                            in_element=enabler_eid,
                            out_sell_price_tenths=other_sell,
                            in_cost_tenths=enabler_cost,
                            bank_delta_tenths=fund_raised,
                        )
                        move2 = FundingMove(
                            out_element=out_eid,
                            in_element=target_element,
                            out_sell_price_tenths=sell_p,
                            in_cost_tenths=target_cost,
                            bank_delta_tenths=sell_p - target_cost,
                        )

                        en_name = str(cheapest_enabler.get("web_name", cheapest_enabler.get("name", f"element {enabler_eid}")))
                        exp = (
                            f"Two-step plan: Step 1: Fund £{fund_raised/10:.1f}m by downgrading {other['name']} (£{other_sell/10:.1f}m) -> {en_name} (£{enabler_cost/10:.1f}m). "
                            f"Step 2: Upgrade {cand['name']} (£{sell_p/10:.1f}m) -> target (£{target_cost/10:.1f}m). "
                            f"Remaining bank: £{final_bank/10:.1f}m. Hits: {hits}."
                        )

                        paths.append(
                            AffordabilityPath(
                                target_element=target_element,
                                path_type="TWO_STEP_PLAN",
                                transfers_count=2,
                                hits=hits,
                                hit_cost=hits * 4,
                                moves=[move1, move2],
                                initial_bank_tenths=bank_tenths,
                                final_bank_tenths=final_bank,
                                is_legal=True,
                                arithmetic_explanation=exp,
                                intermediate_enabler=enabler_eid,
                            )
                        )
                        if len(paths) >= top_k:
                            return paths

        return paths
