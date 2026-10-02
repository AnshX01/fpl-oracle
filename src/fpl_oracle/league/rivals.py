"""
Rival Intelligence & League Ownership Analyzer.
Computes Effective Ownership (EO), detects template vs differentials,
and monitors rival squad compositions and chip usage.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.models import BootstrapStatic

logger = logging.getLogger("fpl_oracle.league.rivals")

class RivalAnalyzer:
    def __init__(self):
        pass

    async def analyze_rivals(
        self,
        standings: list[dict[str, Any]],
        user_manager_id: int | None,
        current_gw: int,
        bootstrap: BootstrapStatic,
        max_rivals_to_inspect: int = 8
    ) -> dict[str, Any]:
        """
        Extract squad compositions for top rivals and compute effective ownership.
        """
        elem_map = {e.id: e for e in bootstrap.elements}
        top_entries = standings[:max_rivals_to_inspect]

        rival_squads = []
        player_multipliers: dict[int, float] = defaultdict(float)
        rival_chips_burned: dict[int, list[str]] = defaultdict(list)

        sem = asyncio.Semaphore(5)

        async def fetch_rival_details(entry_dict):
            entry_id = entry_dict["entry"]
            async with sem:
                try:
                    picks_resp, _ = await fpl_client.get_manager_picks(entry_id, current_gw)
                    history_resp, _ = await fpl_client.get_manager_history(entry_id)

                    squad_elements = []
                    cap_elem = None
                    for p in picks_resp.picks:
                        elem_id = p.element
                        mult = p.multiplier
                        if p.is_captain:
                            cap_elem = elem_id
                        # Effective ownership multiplier (starters 1x, captain 2x, bench 0x)
                        player_multipliers[elem_id] += mult
                        elem_obj = elem_map.get(elem_id)
                        squad_elements.append({
                            "element": elem_id,
                            "web_name": elem_obj.web_name if elem_obj else f"Player {elem_id}",
                            "position": p.position,
                            "multiplier": mult,
                            "is_captain": p.is_captain,
                            "is_starter": mult > 0
                        })

                    burned_chips = [c.name for c in history_resp.chips] if history_resp else []
                    rival_chips_burned[entry_id] = burned_chips

                    rival_squads.append({
                        "entry_id": entry_id,
                        "player_name": entry_dict.get("player_name", ""),
                        "entry_name": entry_dict.get("entry_name", ""),
                        "rank": entry_dict.get("rank", 0),
                        "total_points": entry_dict.get("total", 0),
                        "captain_element": cap_elem,
                        "chips_used": burned_chips,
                        "squad": squad_elements
                    })
                except Exception as ex:
                    logger.debug(f"Could not fetch rival {entry_id}: {ex}")

        tasks = [fetch_rival_details(e) for e in top_entries]
        await asyncio.gather(*tasks)

        n_rivals = max(1, len(rival_squads))

        # Calculate Effective Ownership (EO) %
        eo_records = []
        for elem_id, total_mult in player_multipliers.items():
            elem_obj = elem_map.get(elem_id)
            if not elem_obj:
                continue
            eo_pct = round((total_mult / n_rivals) * 100.0, 1)
            eo_records.append({
                "element": elem_id,
                "web_name": elem_obj.web_name,
                "team": elem_obj.team,
                "effective_ownership": eo_pct,
                "total_multiplier": total_mult
            })

        eo_df = pd.DataFrame(eo_records).sort_values(by="effective_ownership", ascending=False)

        # Classify into Template vs Differentials
        template = eo_df[eo_df["effective_ownership"] >= 50.0].to_dict(orient="records")
        differentials = eo_df[(eo_df["effective_ownership"] > 0) & (eo_df["effective_ownership"] <= 20.0)].head(10).to_dict(orient="records")

        # Sort rival squads by rank
        rival_squads.sort(key=lambda x: x["rank"])

        return {
            "rivals_analyzed_count": n_rivals,
            "rival_squads": rival_squads,
            "league_effective_ownership": eo_df.to_dict(orient="records"),
            "template_players": template,
            "differential_players": differentials
        }

rival_analyzer = RivalAnalyzer()
