"""
Interactive Multi-Gameweek Visual Transfer Trajectory Board Engine (Pillar 2 / Pillar 4).
Extracts, structures, and visualizes the complete multi-period MILP transfer roadmap,
bank balance progression, free transfer velocity, and starting XI evolution across
the entire planning horizon (e.g. GW9 -> GW16).
Zero hardcoded player names in production code.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic


@dataclass
class TrajectoryPlayerCard:
    element: int
    web_name: str
    position: str
    team_id: int
    team_short: str
    price_millions: float
    expected_points: float
    is_captain: bool = False
    is_vice_captain: bool = False


@dataclass
class TrajectoryGameweekStep:
    gameweek: int
    bank_millions: float
    free_transfers_available: int
    free_transfers_next: int
    hit_cost: int
    active_chip: str | None
    chip_display_name: str | None
    transfers_in: list[TrajectoryPlayerCard] = field(default_factory=list)
    transfers_out: list[TrajectoryPlayerCard] = field(default_factory=list)
    starters: list[TrajectoryPlayerCard] = field(default_factory=list)
    bench: list[TrajectoryPlayerCard] = field(default_factory=list)
    captain: TrajectoryPlayerCard | None = None
    vice_captain: TrajectoryPlayerCard | None = None
    predicted_points: float = 0.0
    cumulative_points: float = 0.0
    formation: str = "4-4-2"


@dataclass
class TrajectoryBoard:
    start_gw: int
    end_gw: int
    total_horizon_weeks: int
    total_cumulative_xp: float
    total_hits_cost: int
    steps: list[TrajectoryGameweekStep] = field(default_factory=list)


class TrajectoryBoardEngine:
    """Builds an interactive multi-gameweek transfer roadmap from solved plans."""

    def build_trajectory_board(
        self,
        plan_dict: dict[str, Any],
        bootstrap: BootstrapStatic,
        horizon_projections: dict[int, pd.DataFrame],
    ) -> TrajectoryBoard:
        elem_map = {e.id: e for e in bootstrap.elements}
        team_map = {t.id: t.short_name for t in getattr(bootstrap, "teams", [])}
        pos_map = {t.id: t.singular_name_short for t in getattr(bootstrap, "element_types", [])}

        plan = plan_dict.get("recommended_plan") or plan_dict
        trajectory = plan.get("trajectory", [])
        if not trajectory:
            return TrajectoryBoard(
                start_gw=1, end_gw=1, total_horizon_weeks=0, total_cumulative_xp=0.0, total_hits_cost=0, steps=[]
            )

        steps: list[TrajectoryGameweekStep] = []
        cumulative_xp = 0.0
        total_hits = 0

        for act in trajectory:
            gw = int(act.get("gameweek", 1))
            gw_df = horizon_projections.get(gw, pd.DataFrame())
            xp_map: dict[int, float] = {}
            if not gw_df.empty and "element" in gw_df.columns and "expected_points" in gw_df.columns:
                xp_map = dict(zip(gw_df["element"].astype(int), gw_df["expected_points"].astype(float), strict=False))

            def make_card(
                eid: int, is_cap: bool = False, is_vc: bool = False, xps: dict[int, float] = xp_map
            ) -> TrajectoryPlayerCard:
                elem = elem_map.get(eid)
                w_name = elem.web_name if elem else f"Player {eid}"
                p_name = pos_map.get(elem.element_type, "MID") if elem else "MID"
                t_short = team_map.get(elem.team, "PL") if elem else "PL"
                cost = float(getattr(elem, "now_cost", 50) or 50) / 10.0
                xp = xps.get(eid, float(getattr(elem, "ep_next", 0.0) or 0.0))
                return TrajectoryPlayerCard(
                    element=eid,
                    web_name=w_name,
                    position=p_name,
                    team_id=elem.team if elem else 1,
                    team_short=t_short,
                    price_millions=cost,
                    expected_points=round(xp, 2),
                    is_captain=is_cap,
                    is_vice_captain=is_vc,
                )

            # Cap / VC
            cap_id = act.get("captain")
            vc_id = act.get("vice_captain")

            starters = [
                make_card(int(eid), is_cap=(eid == cap_id), is_vc=(eid == vc_id)) for eid in act.get("starters", [])
            ]
            bench = [make_card(int(eid)) for eid in act.get("bench", [])]

            # In / Out
            ins = [make_card(int(t["element"] if isinstance(t, dict) else t)) for t in act.get("transfers_in", [])]
            outs = [make_card(int(t["element"] if isinstance(t, dict) else t)) for t in act.get("transfers_out", [])]

            # Formation
            def_cnt = sum(1 for s in starters if s.position == "DEF")
            mid_cnt = sum(1 for s in starters if s.position == "MID")
            fwd_cnt = sum(1 for s in starters if s.position == "FWD")
            formation = f"{def_cnt}-{mid_cnt}-{fwd_cnt}"

            # Projected points this GW
            gw_pts = sum(s.expected_points for s in starters)
            if cap_id:
                gw_pts += xp_map.get(int(cap_id), 0.0)  # Double captain
                if act.get("active_chip") in ("3xc", "triple_captain"):
                    gw_pts += xp_map.get(int(cap_id), 0.0)  # Triple captain

            hits = int(act.get("hit_cost", 0) or 0)
            net_gw_pts = max(0.0, gw_pts - hits)
            cumulative_xp += net_gw_pts
            total_hits += hits

            bank_val = float(act.get("bank_end", act.get("bank", 0)) or 0)
            if bank_val > 20.0:  # tenths
                bank_val = bank_val / 10.0

            step = TrajectoryGameweekStep(
                gameweek=gw,
                bank_millions=round(bank_val, 2),
                free_transfers_available=int(act.get("free_transfers_available", 1)),
                free_transfers_next=int(act.get("free_transfers_next", 1)),
                hit_cost=hits,
                active_chip=act.get("active_chip"),
                chip_display_name=act.get("active_chip", "").replace("_", " ").title()
                if act.get("active_chip")
                else None,
                transfers_in=ins,
                transfers_out=outs,
                starters=starters,
                bench=bench,
                captain=make_card(int(cap_id), is_cap=True) if cap_id else None,
                vice_captain=make_card(int(vc_id), is_vc=True) if vc_id else None,
                predicted_points=round(net_gw_pts, 2),
                cumulative_points=round(cumulative_xp, 2),
                formation=formation,
            )
            steps.append(step)

        start_gw = steps[0].gameweek if steps else 1
        end_gw = steps[-1].gameweek if steps else 1

        return TrajectoryBoard(
            start_gw=start_gw,
            end_gw=end_gw,
            total_horizon_weeks=len(steps),
            total_cumulative_xp=round(cumulative_xp, 2),
            total_hits_cost=total_hits,
            steps=steps,
        )


trajectory_board_engine = TrajectoryBoardEngine()
