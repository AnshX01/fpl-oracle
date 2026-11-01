"""
Midweek European & Domestic Cup Quick-Turnaround Minutes Discount Engine.
Quantifies rotation and fatigue risk when outfield players play 70+ minutes in midweek
European (UCL, UEL, UECL) or domestic cup (Carabao Cup, FA Cup) fixtures with short turnaround (<72 hours).
Zero hardcoded player names in production code.
"""

from typing import Any

import pandas as pd

from fpl_oracle.config import SETTINGS


class TurnaroundDragService:
    """
    Applies quick-turnaround fatigue discounts to expected minutes and expected points
    for outfield players following midweek fixtures.
    """

    def __init__(
        self,
        extreme_crunch_hours: float = 48.0,
        high_crunch_hours: float = 72.0,
        extreme_discount: float | None = None,
        high_discount: float | None = None,
    ):
        opt_cfg = SETTINGS.get("optimizer", {})
        self.extreme_crunch_hours = extreme_crunch_hours
        self.high_crunch_hours = high_crunch_hours

        # Load default discounts
        base_drag = float(opt_cfg.get("turnaround_drag", 0.85))
        self.extreme_discount = float(extreme_discount if extreme_discount is not None else (base_drag - 0.05))
        self.high_discount = float(high_discount if high_discount is not None else base_drag)

    def calculate_multiplier(
        self,
        turnaround_hours: float,
        position: str,
        midweek_minutes: float = 90.0,
    ) -> float:
        """
        Calculates expected minutes multiplier based on turnaround and position.
        Goalkeepers (GKP) are exempt from fatigue rotation.
        """
        if position.upper() == "GKP":
            return 1.0

        if midweek_minutes < 45.0:
            return 1.0  # Negligible fatigue from sub appearance

        hours = float(turnaround_hours)
        if hours < self.extreme_crunch_hours:
            # Extreme turnaround crunch (e.g. Thursday 20:00 -> Saturday 12:30, <48h)
            if midweek_minutes >= 70.0:
                return self.extreme_discount
            return (self.extreme_discount + 1.0) / 2.0
        elif hours < self.high_crunch_hours:
            # High turnaround crunch (e.g. Wednesday 20:00 -> Saturday 12:30, <72h)
            if midweek_minutes >= 70.0:
                return self.high_discount
            return (self.high_discount + 1.0) / 2.0

        return 1.0

    def apply_turnaround_drag(
        self,
        df: pd.DataFrame,
        midweek_data: dict[int, dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """
        Applies turnaround minutes discount and scales expected points accordingly.

        Parameters:
            df: Squad or player pool DataFrame containing element, position, expected_minutes, expected_points.
            midweek_data: Optional mapping of element_id -> {"midweek_minutes": float, "turnaround_hours": float}.
        """
        if df.empty:
            return df.copy()

        res = df.copy()
        if "midweek_fatigue_risk" not in res.columns:
            res["midweek_fatigue_risk"] = False

        has_cols = "turnaround_hours" in res.columns
        if not has_cols and not midweek_data:
            return res

        # Prevent double-counting only when explicitly flagged
        already_applied = bool(
            res.get("turnaround_drag_applied", False).all()
            if isinstance(res.get("turnaround_drag_applied"), pd.Series)
            else res.get("turnaround_drag_applied", False)
        )
        if already_applied:
            return res

        for idx, row in res.iterrows():
            elem_id = int(row["element"])
            pos = str(row.get("position", "MID")).upper()

            # Retrieve turnaround hours and midweek minutes
            t_hours = None
            m_mins = None

            if midweek_data and elem_id in midweek_data:
                entry = midweek_data[elem_id]
                t_hours = float(entry.get("turnaround_hours", 999.0))
                m_mins = float(entry.get("midweek_minutes", 90.0))
            elif has_cols and pd.notna(row["turnaround_hours"]):
                t_hours = float(row["turnaround_hours"])
                if "midweek_minutes" in row and pd.notna(row["midweek_minutes"]):
                    m_mins = float(row["midweek_minutes"])

            # Rested players or negligible minutes (< 45) face zero turnaround drag
            is_rested = (m_mins is not None and m_mins < 45.0) or (
                "congestion_fatigue_factor" in res.columns
                and pd.notna(row.get("congestion_fatigue_factor"))
                and float(row["congestion_fatigue_factor"]) >= 1.0
                and "midweek_minutes" not in row
            )
            if is_rested:
                if "congestion_fatigue_factor" in res.columns:
                    cfg = float(res.loc[idx, "congestion_fatigue_factor"])
                    if 0.0 < cfg < 1.0:
                        if "expected_points" in res.columns:
                            res.loc[idx, "expected_points"] = round(float(res.loc[idx, "expected_points"]) / cfg, 2)
                        if "expected_minutes" in res.columns:
                            res.loc[idx, "expected_minutes"] = round(float(res.loc[idx, "expected_minutes"]) / cfg, 1)
                        res.loc[idx, "congestion_fatigue_factor"] = 1.0
                res.loc[idx, "midweek_fatigue_risk"] = False
                continue

            # Prevent double-counting: if congestion was already applied with explicit midweek minutes
            if (
                "congestion_fatigue_factor" in res.columns
                and "midweek_minutes" in row
                and pd.notna(row["midweek_minutes"])
            ):
                res.loc[idx, "midweek_fatigue_risk"] = True
                continue

            effective_mins = m_mins if m_mins is not None else 90.0
            if t_hours is not None:
                mult = self.calculate_multiplier(t_hours, pos, effective_mins)
                if mult < 1.0:
                    res.loc[idx, "midweek_fatigue_risk"] = True
                    # Discount expected minutes if available
                    if "expected_minutes" in res.columns and pd.notna(res.loc[idx, "expected_minutes"]):
                        orig_mins = float(res.loc[idx, "expected_minutes"])
                        new_mins = round(orig_mins * mult, 1)
                        res.loc[idx, "expected_minutes"] = new_mins
                        # Proportionally scale expected points
                        if "expected_points" in res.columns and orig_mins > 0:
                            orig_xp = float(res.loc[idx, "expected_points"])
                            res.loc[idx, "expected_points"] = round(orig_xp * (new_mins / orig_mins), 2)
                    elif "expected_points" in res.columns:
                        orig_xp = float(res.loc[idx, "expected_points"])
                        res.loc[idx, "expected_points"] = round(orig_xp * mult, 2)

        res["turnaround_drag_applied"] = True
        return res


turnaround_drag_service = TurnaroundDragService()
