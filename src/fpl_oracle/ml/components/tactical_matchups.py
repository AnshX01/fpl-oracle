"""
Tactical Archetype Matchup Engine (Peak Frontier Pillar 1).
Conditions player expected goal involvement (xGI) and clean sheet expectations
on opponent tactical styles (High-Press High Defensive Line vs Compact Low Block vs Balanced Transition).
Zero hardcoded player names in production code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TacticalArchetype:
    name: str
    description: str
    forward_transition_xg_mult: float
    creative_box_xa_mult: float
    set_piece_aerial_xg_mult: float
    defending_clean_sheet_mult: float


class TacticalMatchupEngine:
    """
    Applies tactical style conditioning to player projections based on
    opponent defensive line height and pressing intensity.
    """

    ARCHETYPES: dict[str, TacticalArchetype] = {
        "HIGH_PRESS_HIGH_LINE": TacticalArchetype(
            name="HIGH_PRESS_HIGH_LINE",
            description="Aggressive high line and intense counter-press. Vulnerable to pace in behind.",
            forward_transition_xg_mult=1.10,  # +10% xG for transition forwards exploiting depth
            creative_box_xa_mult=0.96,  # Slightly compressed passing windows in midfield
            set_piece_aerial_xg_mult=0.95,
            defending_clean_sheet_mult=0.94,  # High-turnover match increases defensive vulnerability
        ),
        "COMPACT_LOW_BLOCK": TacticalArchetype(
            name="COMPACT_LOW_BLOCK",
            description="Deep, compact defensive shape denying space behind. Vulnerable to box crosses and set pieces.",
            forward_transition_xg_mult=0.92,  # -8% xG: Space in behind is congested
            creative_box_xa_mult=1.08,  # +8% xA: High sustained possession and cross volume
            set_piece_aerial_xg_mult=1.12,  # +12% xG on set-piece headers for aerial center-backs
            defending_clean_sheet_mult=1.04,  # Attacking opponent creates fewer explosive counter chances
        ),
        "BALANCED_TRANSITION": TacticalArchetype(
            name="BALANCED_TRANSITION",
            description="Standard mid-block transition dynamics with balanced territorial split.",
            forward_transition_xg_mult=1.00,
            creative_box_xa_mult=1.00,
            set_piece_aerial_xg_mult=1.00,
            defending_clean_sheet_mult=1.00,
        ),
    }

    # Archetype profile by team short name
    HIGH_LINE_TEAMS: frozenset[str] = frozenset({"MCI", "LIV", "ARS", "TOT", "BHA"})
    LOW_BLOCK_TEAMS: frozenset[str] = frozenset({"EVE", "NFO", "IPS", "HUL", "COV", "WOL"})

    def classify_team_archetype(self, team_short: str, opponent_strength: float = 3.0) -> TacticalArchetype:
        """
        Dynamically classifies opponent defensive archetype.
        """
        t_clean = str(team_short).strip().upper()
        if t_clean in self.HIGH_LINE_TEAMS:
            return self.ARCHETYPES["HIGH_PRESS_HIGH_LINE"]
        elif t_clean in self.LOW_BLOCK_TEAMS:
            return self.ARCHETYPES["COMPACT_LOW_BLOCK"]
        elif opponent_strength >= 4.0:
            return self.ARCHETYPES["HIGH_PRESS_HIGH_LINE"]
        elif opponent_strength <= 2.0:
            return self.ARCHETYPES["COMPACT_LOW_BLOCK"]
        else:
            return self.ARCHETYPES["BALANCED_TRANSITION"]

    def compute_matchup_modifiers(
        self,
        opponent_short: str,
        position: str,
        xt_creativity: float = 0.20,
        set_piece_share: float = 0.10,
    ) -> dict[str, float]:
        """
        Derives xG, xA, and clean sheet multipliers for a specific matchup.
        """
        arch = self.classify_team_archetype(opponent_short)
        pos = str(position).strip().upper()

        # 1. Attacking Multiplier
        if pos == "FWD":
            xg_mult = arch.forward_transition_xg_mult
            xa_mult = 1.00
        elif pos == "MID":
            # Creative playmakers (high xT) benefit more against low blocks
            if xt_creativity >= 0.25:
                xa_mult = arch.creative_box_xa_mult
                xg_mult = 1.00 + (arch.forward_transition_xg_mult - 1.00) * 0.5
            else:
                xa_mult = 1.00
                xg_mult = arch.forward_transition_xg_mult
        elif pos == "DEF":
            # Aerial set-piece defenders benefit against low blocks
            if set_piece_share >= 0.25:
                xg_mult = arch.set_piece_aerial_xg_mult
            else:
                xg_mult = 1.00
            xa_mult = 1.00
        else:  # GKP
            xg_mult = 1.00
            xa_mult = 1.00

        return {
            "xg_mult": round(xg_mult, 3),
            "xa_mult": round(xa_mult, 3),
            "cs_mult": round(arch.defending_clean_sheet_mult, 3),
            "archetype": arch.name,
        }

    def enhance_projections(
        self,
        df: pd.DataFrame,
        team_short_map: dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """
        Applies tactical matchup modifiers across a gameweek projection DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()
        if "opponent_team" not in df.columns and "opponent_short" not in df.columns:
            return df

        xg_mods = []
        archetypes = []

        for _, row in df.iterrows():
            opp_val = row.get("opponent_team")
            opp_short = (
                team_short_map.get(int(opp_val), "")
                if team_short_map and pd.notna(opp_val) and int(opp_val) in team_short_map
                else str(row.get("opponent_short", ""))
            )
            pos = str(row.get("position", "MID"))
            xt = float(row.get("expected_threat_xt", 0.20))
            sp_share = float(row.get("set_piece_threat_share", 0.10))

            mods = self.compute_matchup_modifiers(opp_short, pos, xt_creativity=xt, set_piece_share=sp_share)
            archetypes.append(mods["archetype"])

            # Attacking net factor
            if pos in ("FWD", "MID"):
                net_att = 0.65 * mods["xg_mult"] + 0.35 * mods["xa_mult"]
            elif pos in ("DEF", "GKP"):
                net_att = mods["cs_mult"]
            else:
                net_att = 1.00

            xg_mods.append(net_att)

        mod_series = pd.Series(xg_mods, index=df.index)

        # Scale specific sub-components if present to preserve scoring physics
        has_subcomponents = all(c in df.columns for c in ("exp_goals_pts", "exp_assists_pts", "exp_cs_pts"))
        if has_subcomponents:
            for idx_r, (_, row) in enumerate(df.iterrows()):
                pos_r = str(row.get("position", "MID"))
                opp_v = row.get("opponent_team")
                opp_s = (
                    team_short_map.get(int(opp_v), "")
                    if (team_short_map and pd.notna(opp_v) and int(opp_v) in team_short_map)
                    else str(row.get("opponent_short", ""))
                )
                xt_r = float(row.get("expected_threat_xt", 0.20))
                sp_r = float(row.get("set_piece_threat_share", 0.10))
                m_r = self.compute_matchup_modifiers(opp_s, pos_r, xt_creativity=xt_r, set_piece_share=sp_r)

                df.loc[df.index[idx_r], "exp_goals_pts"] = round(
                    float(df.loc[df.index[idx_r], "exp_goals_pts"]) * m_r["xg_mult"], 2
                )
                df.loc[df.index[idx_r], "exp_assists_pts"] = round(
                    float(df.loc[df.index[idx_r], "exp_assists_pts"]) * m_r["xa_mult"], 2
                )
                df.loc[df.index[idx_r], "exp_cs_pts"] = round(
                    float(df.loc[df.index[idx_r], "exp_cs_pts"]) * m_r["cs_mult"], 2
                )
                if "exp_gc_deduction" in df.columns:
                    cs_div = max(0.5, m_r["cs_mult"])
                    df.loc[df.index[idx_r], "exp_gc_deduction"] = round(
                        float(df.loc[df.index[idx_r], "exp_gc_deduction"]) / cs_div, 2
                    )

            comp_cols = [
                "exp_appearance",
                "exp_goals_pts",
                "exp_assists_pts",
                "exp_cs_pts",
                "exp_saves_pts",
                "exp_pen_saves_pts",
                "exp_defcon_pts",
                "exp_bonus_pts",
                "exp_blend_adj",
            ]
            ded_cols = ["exp_gc_deduction", "exp_card_deduction", "exp_og_deduction", "exp_pen_miss_deduction"]
            pts = sum(df[c] for c in comp_cols if c in df.columns) - sum(df[c] for c in ded_cols if c in df.columns)
            if "congestion_fatigue_factor" in df.columns:
                pts = pts * df["congestion_fatigue_factor"]
            df["expected_points"] = pts.round(2)
        elif "expected_points" in df.columns:
            df["expected_points"] = (df["expected_points"] * mod_series).round(2)

        # Ensure quantile monotonicity and distribution coherence
        if "p10" in df.columns and "p50" in df.columns and "p90" in df.columns:
            p10 = df["p10"].to_numpy(dtype=float)
            p50 = df["p50"].to_numpy(dtype=float)
            p90 = df["p90"].to_numpy(dtype=float)
            p50 = np.maximum(p50, p10)
            p90 = np.maximum(p90, p50)
            df["p50"] = np.round(p50, 2)
            df["p90"] = np.round(p90, 2)

        df["tactical_archetype"] = archetypes
        df["tactical_matchup_factor"] = mod_series
        return df


tactical_matchup_engine = TacticalMatchupEngine()
