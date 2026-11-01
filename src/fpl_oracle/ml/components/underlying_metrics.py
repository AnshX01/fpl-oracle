"""
Underlying Event Metrics & Spatial Threat Model (Pillar 3).
Calculates Expected Threat (xT), box touch intensity, and finishing regression to separate
sustainable shot-volume generation from ephemeral overperformance streaks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PlayerUnderlyingProfile:
    element: int
    web_name: str
    expected_threat_xt: float
    box_touch_intensity: float
    set_piece_threat_share: float
    finishing_sustainability_factor: float
    adjusted_xg_per_90: float
    adjusted_xa_per_90: float


class UnderlyingMetricsEngine:
    """
    Computes advanced event-level underlying metrics and applies Bayesian shrinkage
    to prevent chasing unsustainable finishing luck while identifying volume creators.
    """

    def __init__(self, shrinkage_weight: float = 0.35):
        """
        shrinkage_weight: Weight applied towards league-average conversion (1.0).
        Higher shrinkage regresses extreme conversion streaks faster.
        """
        self.shrinkage = shrinkage_weight

    def calculate_underlying_profile(
        self,
        element: int,
        web_name: str,
        position: str,
        raw_xg_per_90: float,
        raw_xa_per_90: float,
        ict_threat: float = 0.0,
        ict_creativity: float = 0.0,
        actual_goals: int | None = None,
        actual_assists: int | None = None,
        minutes_played: int = 0,
    ) -> PlayerUnderlyingProfile:
        """
        Derives xT, box touch intensity, and regressed sustainable output.
        """
        mins_90s = max(1.0, minutes_played / 90.0)
        actual_g_per_90 = (actual_goals / mins_90s) if actual_goals is not None else None

        # 1. Expected Threat (xT) proxy from ICT creativity, threat, and progressive actions
        # In modern football analytics, xT measures progressive possession advancement into dangerous zones
        raw_xt = (ict_creativity * 0.005 + ict_threat * 0.003) / mins_90s
        xt = round(float(np.clip(raw_xt, 0.05, 0.95)), 3)

        # 2. Box touch intensity (penalty area penetration proxy)
        raw_box_touches = (ict_threat / mins_90s) * 0.12
        box_touches = round(float(np.clip(raw_box_touches, 0.2, 12.0)), 2)

        # 3. Set-piece threat share (higher for DEF/MID with high aerial threat or direct free-kick role)
        if position == "DEF":
            set_piece_share = round(
                float(np.clip(0.40 * (raw_xg_per_90 / max(0.05, raw_xg_per_90 + 0.1)), 0.1, 0.7)), 2
            )
        elif position == "MID":
            set_piece_share = round(
                float(np.clip(0.20 * (ict_creativity / max(10.0, ict_creativity + 20.0)), 0.05, 0.4)), 2
            )
        else:
            set_piece_share = 0.10

        # 4. Bayesian Finishing Sustainability Factor
        # Regresses actual_g / xG towards 1.0 (mean reversion). If actual goals are unobserved, neutral baseline 1.0 applies.
        if actual_g_per_90 is not None:
            raw_conversion_ratio = actual_g_per_90 / max(0.1, raw_xg_per_90)
            regressed_conversion = (1.0 - self.shrinkage) * np.clip(
                raw_conversion_ratio, 0.6, 1.8
            ) + self.shrinkage * 1.0
        else:
            regressed_conversion = 1.0
        sustainability = round(float(regressed_conversion), 2)

        # 5. Adjusted Sustainable Rates
        # Blends raw xG with volume indicators (box touch density and xT)
        vol_boost = 1.0 + 0.05 * np.clip((box_touches - 3.0) / 3.0, -0.3, 0.3)
        adj_xg = round(float(raw_xg_per_90 * vol_boost * (0.85 + 0.15 * sustainability)), 3)

        creative_boost = 1.0 + 0.08 * np.clip((xt - 0.25) / 0.25, -0.2, 0.4)
        adj_xa = round(float(raw_xa_per_90 * creative_boost), 3)

        return PlayerUnderlyingProfile(
            element=element,
            web_name=web_name,
            expected_threat_xt=xt,
            box_touch_intensity=box_touches,
            set_piece_threat_share=set_piece_share,
            finishing_sustainability_factor=sustainability,
            adjusted_xg_per_90=adj_xg,
            adjusted_xa_per_90=adj_xa,
        )

    def enhance_projections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends underlying spatial threat signals to player projections DataFrame.
        """
        if df.empty or "expected_points" not in df.columns:
            return df

        enhanced = df.copy()
        xt_list = []
        touch_list = []
        sustain_list = []

        for _, row in enhanced.iterrows():
            eid = int(row.get("element", 0))
            name = str(row.get("web_name", f"Player {eid}"))
            pos = str(row.get("position", "MID"))
            goal_divisor = 10.0 if pos == "GKP" else (6.0 if pos == "DEF" else (5.0 if pos == "MID" else 4.0))
            raw_xg_pts = row.get("exp_goals_pts", 0.0)
            raw_xa_pts = row.get("exp_assists_pts", 0.0)
            raw_xg = float(np.nan_to_num(raw_xg_pts if pd.notna(raw_xg_pts) else 0.0, nan=0.0)) / goal_divisor
            raw_xa = float(np.nan_to_num(raw_xa_pts if pd.notna(raw_xa_pts) else 0.0, nan=0.0)) / 3.0
            threat = float(row.get("threat", 15.0) or 15.0)
            creativity = float(row.get("creativity", 15.0) or 15.0)
            mins = int(row.get("minutes", 90) or 90)

            act_goals = (
                int(row["goals_scored"])
                if ("goals_scored" in row and pd.notna(row["goals_scored"]))
                else (int(row["actual_goals"]) if ("actual_goals" in row and pd.notna(row["actual_goals"])) else None)
            )
            act_assists = (
                int(row["assists"])
                if ("assists" in row and pd.notna(row["assists"]))
                else (
                    int(row["actual_assists"])
                    if ("actual_assists" in row and pd.notna(row["actual_assists"]))
                    else None
                )
            )
            prof = self.calculate_underlying_profile(
                element=eid,
                web_name=name,
                position=pos,
                raw_xg_per_90=max(0.01, raw_xg),
                raw_xa_per_90=max(0.01, raw_xa),
                ict_threat=threat,
                ict_creativity=creativity,
                actual_goals=act_goals,
                actual_assists=act_assists,
                minutes_played=mins,
            )
            xt_list.append(prof.expected_threat_xt)
            touch_list.append(prof.box_touch_intensity)
            sustain_list.append(prof.finishing_sustainability_factor)

        enhanced["expected_threat_xt"] = xt_list
        enhanced["box_touch_intensity"] = touch_list
        enhanced["finishing_sustainability"] = sustain_list
        return enhanced

    def summarize_squad_underlying(
        self,
        squad_df: pd.DataFrame,
        projections_df: pd.DataFrame,
        bootstrap: Any = None,
    ) -> dict[str, Any]:
        """
        Summarizes squad spatial Expected Threat (xT) and Bayesian finishing sustainability.
        """
        if squad_df.empty:
            return {"xt_leaders": [], "bayesian_finishing_profile": []}

        squad_eids = set(squad_df["element"])
        proj_map: dict[int, Any] = {}
        if not projections_df.empty and "element" in projections_df.columns:
            for _, r in projections_df[projections_df["element"].isin(squad_eids)].iterrows():
                proj_map[int(r["element"])] = r

        profiles: list[PlayerUnderlyingProfile] = []
        for _, s in squad_df.iterrows():
            eid = int(s["element"])
            name = str(s.get("web_name", f"Player {eid}"))
            pos = str(s.get("position", "MID"))
            proj_row = proj_map.get(eid, {})

            raw_xg_pts = proj_row.get("exp_goals_pts", 0.0) if hasattr(proj_row, "get") else 0.0
            raw_xa_pts = proj_row.get("exp_assists_pts", 0.0) if hasattr(proj_row, "get") else 0.0
            goal_divisor = 10.0 if pos == "GKP" else (6.0 if pos == "DEF" else (5.0 if pos == "MID" else 4.0))

            raw_xg = float(np.nan_to_num(raw_xg_pts or 0.0, nan=0.0)) / goal_divisor
            raw_xa = float(np.nan_to_num(raw_xa_pts or 0.0, nan=0.0)) / 3.0
            threat = float(proj_row.get("threat", 15.0) or 15.0) if hasattr(proj_row, "get") else 15.0
            creativity = float(proj_row.get("creativity", 15.0) or 15.0) if hasattr(proj_row, "get") else 15.0
            mins = int(proj_row.get("minutes", 90) or 90) if hasattr(proj_row, "get") else 90

            act_goals = None
            if hasattr(proj_row, "get"):
                if "goals_scored" in proj_row and pd.notna(proj_row["goals_scored"]):
                    act_goals = int(proj_row["goals_scored"])
                elif "actual_goals" in proj_row and pd.notna(proj_row["actual_goals"]):
                    act_goals = int(proj_row["actual_goals"])
            if act_goals is None and hasattr(s, "get"):
                if "goals_scored" in s and pd.notna(s["goals_scored"]):
                    act_goals = int(s["goals_scored"])
                elif "actual_goals" in s and pd.notna(s["actual_goals"]):
                    act_goals = int(s["actual_goals"])

            act_assists = None
            if hasattr(proj_row, "get"):
                if "assists" in proj_row and pd.notna(proj_row["assists"]):
                    act_assists = int(proj_row["assists"])
                elif "actual_assists" in proj_row and pd.notna(proj_row["actual_assists"]):
                    act_assists = int(proj_row["actual_assists"])
            if act_assists is None and hasattr(s, "get"):
                if "assists" in s and pd.notna(s["assists"]):
                    act_assists = int(s["assists"])
                elif "actual_assists" in s and pd.notna(s["actual_assists"]):
                    act_assists = int(s["actual_assists"])
            prof = self.calculate_underlying_profile(
                element=eid,
                web_name=name,
                position=pos,
                raw_xg_per_90=max(0.01, raw_xg),
                raw_xa_per_90=max(0.01, raw_xa),
                ict_threat=threat,
                ict_creativity=creativity,
                actual_goals=act_goals,
                actual_assists=act_assists,
                minutes_played=mins,
            )
            profiles.append(prof)

        # 1. xT Leaders: Top 5 in squad by expected threat
        sorted_by_xt = sorted(profiles, key=lambda p: p.expected_threat_xt, reverse=True)
        xt_leaders = [
            {
                "element": p.element,
                "web_name": p.web_name,
                "expected_threat_xt": p.expected_threat_xt,
                "box_touch_intensity": p.box_touch_intensity,
                "set_piece_threat_share": p.set_piece_threat_share,
            }
            for p in sorted_by_xt[:5]
        ]

        # 2. Bayesian Finishing Profile: Categorized by regression sustainability
        finishing_profile = []
        for p in profiles:
            cat = (
                "SUSTAINABLE"
                if 0.85 <= p.finishing_sustainability_factor <= 1.15
                else (
                    "OVERPERFORMING_STREAK"
                    if p.finishing_sustainability_factor > 1.15
                    else "UNDERPERFORMING_REVERSION_UP"
                )
            )
            finishing_profile.append(
                {
                    "element": p.element,
                    "web_name": p.web_name,
                    "sustainability_factor": p.finishing_sustainability_factor,
                    "adjusted_xg_per_90": p.adjusted_xg_per_90,
                    "adjusted_xa_per_90": p.adjusted_xa_per_90,
                    "category": cat,
                }
            )

        return {
            "xt_leaders": xt_leaders,
            "bayesian_finishing_profile": finishing_profile,
        }


underlying_metrics_engine = UnderlyingMetricsEngine()
