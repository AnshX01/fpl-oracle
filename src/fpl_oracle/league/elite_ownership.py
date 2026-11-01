"""
Top-10k Elite Effective Ownership (EO) Engine.
Eliminates dead-team / casual distortion by calculating calibrated Effective Ownership (EO)
across the active top-tier manager cohort (Top 10k).

Key Metrics:
- Elite EO (%) = Starting XI Ownership (%) + Captaincy (%) + 2 * Triple Captaincy (%)
- Ghost Distortion Delta (%) = Overall Ownership (%) - Elite EO (%)
- Asset Classification:
    * ELITE_CORE (Elite EO >= 60%)
    * ELITE_DIFFERENTIAL (Elite EO <= 20%, xP >= 4.5)
    * ZOMBIE_TRAP (Overall >= 20%, Elite EO <= 8%)
    * CASUAL_POPULAR (Overall >> Elite EO)
    * NEUTRAL
- Differential Leverage Index = Projected xP / max(1.0, Elite EO)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic


@dataclass(frozen=True)
class EliteAssetEO:
    element: int
    web_name: str
    position: str
    team: int
    overall_ownership_pct: float
    elite_ownership_pct: float
    elite_captain_pct: float
    elite_effective_ownership_pct: float
    ghost_distortion_delta: float
    classification: str
    differential_leverage_index: float


class EliteOwnershipEngine:
    """
    Calculates and models effective ownership distributions across active elite managers,
    identifying zombie traps and high-leverage differentials.
    """

    def compute_elite_eo_from_squads(
        self,
        manager_squads: list[dict[str, Any]],
        bootstrap: BootstrapStatic,
        projections_df: pd.DataFrame | None = None,
    ) -> dict[int, EliteAssetEO]:
        """
        Computes exact empirical Elite EO from a sample cohort of manager squads (e.g. Top 10k).
        manager_squads: list of dicts with keys:
            - 'squad': list of elements (starters or all 15)
            - 'captain': element id (int)
            - 'is_triple_captain': bool (optional)
        """
        if not manager_squads or not bootstrap.elements:
            return {}

        n_managers = len(manager_squads)
        elem_map = {e.id: e for e in bootstrap.elements}
        pos_map = {t.id: t.singular_name_short for t in getattr(bootstrap, "element_types", [])}

        xp_map: dict[int, float] = {}
        if projections_df is not None and not projections_df.empty:
            if "element" in projections_df.columns and "expected_points" in projections_df.columns:
                xp_map = dict(
                    zip(
                        projections_df["element"].astype(int),
                        projections_df["expected_points"].astype(float),
                        strict=False,
                    )
                )

        # Counters
        starter_counts: dict[int, int] = {}
        captain_counts: dict[int, int] = {}
        triple_captain_counts: dict[int, int] = {}

        for m in manager_squads:
            squad = m.get("squad", [])
            cap = m.get("captain")
            is_3xc = bool(m.get("is_triple_captain", False))

            for item in squad:
                eid = int(item["element"] if isinstance(item, dict) else item)
                starter_counts[eid] = starter_counts.get(eid, 0) + 1

            if cap is not None:
                cap_id = int(cap)
                captain_counts[cap_id] = captain_counts.get(cap_id, 0) + 1
                if is_3xc:
                    triple_captain_counts[cap_id] = triple_captain_counts.get(cap_id, 0) + 1

        results: dict[int, EliteAssetEO] = {}

        for eid, elem in elem_map.items():
            s_count = starter_counts.get(eid, 0)
            c_count = captain_counts.get(eid, 0)
            tc_count = triple_captain_counts.get(eid, 0)

            # Proportions
            s_pct = round((s_count / n_managers) * 100.0, 1)
            c_pct = round((c_count / n_managers) * 100.0, 1)
            tc_pct = round((tc_count / n_managers) * 100.0, 1)

            # Effective ownership formula
            eo_pct = round(s_pct + c_pct + (2.0 * tc_pct), 1)

            # Overall game ownership
            overall_pct = float(getattr(elem, "selected_by_percent", 0.0) or 0.0)

            # Ghost distortion: positive means casuals own him far more than elites
            distortion = round(overall_pct - eo_pct, 1)

            xp = xp_map.get(eid, float(getattr(elem, "ep_next", 0.0) or 0.0))
            leverage = round(xp / max(1.0, eo_pct), 3)

            # Classification
            if eo_pct >= 60.0:
                classification = "ELITE_CORE"
            elif distortion >= 15.0 and eo_pct <= 10.0:
                classification = "ZOMBIE_TRAP"
            elif eo_pct <= 25.0 and xp >= 4.5:
                classification = "ELITE_DIFFERENTIAL"
            elif distortion >= 10.0:
                classification = "CASUAL_POPULAR"
            else:
                classification = "NEUTRAL"

            pos_name = pos_map.get(elem.element_type, "MID")

            results[eid] = EliteAssetEO(
                element=eid,
                web_name=elem.web_name,
                position=pos_name,
                team=elem.team,
                overall_ownership_pct=overall_pct,
                elite_ownership_pct=s_pct,
                elite_captain_pct=c_pct,
                elite_effective_ownership_pct=eo_pct,
                ghost_distortion_delta=distortion,
                classification=classification,
                differential_leverage_index=leverage,
            )

        return results

    def model_calibrated_elite_eo(
        self,
        bootstrap: BootstrapStatic,
        projections_df: pd.DataFrame,
    ) -> dict[int, EliteAssetEO]:
        """
        Bayesian calibrated model projecting Top-10k Effective Ownership when raw top-10k scrape
        is not available in real-time.
        Combines expected points, transfer momentum, and price tiers using empirical logistic scaling.
        """
        if not bootstrap or not bootstrap.elements or projections_df.empty:
            return {}

        elem_map = {e.id: e for e in bootstrap.elements}
        pos_map = {t.id: t.singular_name_short for t in getattr(bootstrap, "element_types", [])}

        xp_map = dict(
            zip(projections_df["element"].astype(int), projections_df["expected_points"].astype(float), strict=False)
        )
        results: dict[int, EliteAssetEO] = {}

        # Top captain candidates are high-xP premiums
        max_xp = max(xp_map.values()) if xp_map else 6.0

        for eid, elem in elem_map.items():
            overall_pct = float(getattr(elem, "selected_by_percent", 0.0) or 0.0)
            xp = xp_map.get(eid, float(getattr(elem, "ep_next", 0.0) or 0.0))
            transfers_in_event = int(getattr(elem, "transfers_in_event", 0) or 0)
            transfers_out_event = int(getattr(elem, "transfers_out_event", 0) or 0)
            net_transfers = transfers_in_event - transfers_out_event

            # Elite ownership responds much more aggressively to current xP and recent transfers
            # than casual ownership, which decays slowly.
            # Base logistic curve for elite starter ownership
            z = 0.55 * (xp - 3.8) + 0.00002 * net_transfers
            elite_starter_prob = 1.0 / (1.0 + np.exp(-z))
            elite_s_pct = round(float(np.clip(elite_starter_prob * 95.0, 0.1, 99.0)), 1)

            # Elite captaincy heavily concentrates on top 1-2 xP assets
            if xp >= max_xp - 0.3 and xp >= 5.5:
                # Top candidate
                captain_pct = round(float(np.clip((xp - (max_xp - 0.5)) * 40.0, 10.0, 85.0)), 1)
            else:
                captain_pct = round(float(np.clip((xp / max(1.0, max_xp)) ** 4 * 10.0, 0.0, 5.0)), 1)

            eo_pct = round(elite_s_pct + captain_pct, 1)
            distortion = round(overall_pct - eo_pct, 1)
            leverage = round(xp / max(1.0, eo_pct), 3)

            if eo_pct >= 60.0:
                classification = "ELITE_CORE"
            elif distortion >= 15.0 and eo_pct <= 10.0:
                classification = "ZOMBIE_TRAP"
            elif eo_pct <= 25.0 and xp >= 4.5:
                classification = "ELITE_DIFFERENTIAL"
            elif distortion >= 10.0:
                classification = "CASUAL_POPULAR"
            else:
                classification = "NEUTRAL"

            pos_name = pos_map.get(elem.element_type, "MID")

            results[eid] = EliteAssetEO(
                element=eid,
                web_name=elem.web_name,
                position=pos_name,
                team=elem.team,
                overall_ownership_pct=overall_pct,
                elite_ownership_pct=elite_s_pct,
                elite_captain_pct=captain_pct,
                elite_effective_ownership_pct=eo_pct,
                ghost_distortion_delta=distortion,
                classification=classification,
                differential_leverage_index=leverage,
            )

        return results


elite_ownership_engine = EliteOwnershipEngine()
