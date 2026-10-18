"""
Squad-Health Diagnosis (Phase 8 Early Slice: WI-8.1).
Evaluates manager squad state into KEEP / WATCH / REPLACE classifications with machine-readable reasons,
counts functional starters, and computes structural deficiency score.
All thresholds fitted from backtest or flagged as placeholders.
Zero hard-coded player names in production code.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from fpl_oracle.optimise.lineup import play_probability


@dataclass
class PlayerHealthStatus:
    element: int
    position: str
    status: Literal["KEEP", "WATCH", "REPLACE"]
    reasons: list[str]
    functional_starter: bool
    expected_points: float
    chance_of_playing: float
    deficiency_loss: float = 0.0


@dataclass
class SquadHealthReport:
    player_diagnoses: dict[int, PlayerHealthStatus]
    functional_starters_count: int
    structural_deficiency_score: float
    classification: Literal["HEALTHY_CORE_TARGETED_FIXES", "STRUCTURAL_OVERHAUL_REQUIRED"]
    healthy_core_ids: list[int] = field(default_factory=list)
    watch_ids: list[int] = field(default_factory=list)
    replace_ids: list[int] = field(default_factory=list)
    thresholds_used: dict[str, Any] = field(default_factory=dict)


class SquadHealthDiagnosis:
    """
    Diagnoses squad health prior to recommending transfer or chip plans (WI-8.1).
    Enforces Principle P3: Diagnose before prescribing.
    """

    # Thresholds (flagged as placeholders per spec until backtest calibration in WI-4.1)
    DEFAULT_MIN_PLAY_PROB = 0.75  # [PLACEHOLDER]
    DEFAULT_POSITION_FLOORS = {  # [PLACEHOLDER]
        "GKP": 2.5,
        "DEF": 3.0,
        "MID": 3.5,
        "FWD": 3.5,
    }
    DEFAULT_POSITION_BENCHMARKS = {  # Top replacement tier benchmark
        "GKP": 4.5,
        "DEF": 4.5,
        "MID": 6.0,
        "FWD": 6.0,
    }
    MIN_FUNCTIONAL_STARTERS_THRESHOLD = 9  # Emergency overhaul allowed only if functional starters < 9
    STRUCTURAL_DEFICIENCY_THRESHOLD = 15.0  # Or total deficiency > 15.0 xP

    def __init__(
        self,
        min_play_prob: float = DEFAULT_MIN_PLAY_PROB,
        position_floors: dict[str, float] | None = None,
        position_benchmarks: dict[str, float] | None = None,
        min_functional_starters: int = MIN_FUNCTIONAL_STARTERS_THRESHOLD,
        deficiency_threshold: float = STRUCTURAL_DEFICIENCY_THRESHOLD,
    ) -> None:
        self.min_play_prob = min_play_prob
        self.position_floors = position_floors or self.DEFAULT_POSITION_FLOORS
        self.position_benchmarks = position_benchmarks or self.DEFAULT_POSITION_BENCHMARKS
        self.min_functional_starters = min_functional_starters
        self.deficiency_threshold = deficiency_threshold

    def diagnose_squad(
        self,
        squad_df: pd.DataFrame,
        player_pool_df: pd.DataFrame | None = None,
    ) -> SquadHealthReport:
        """
        Diagnoses each player in squad_df.
        squad_df must contain columns: element, position, expected_points (or now_cost, etc.).
        """
        diagnoses: dict[int, PlayerHealthStatus] = {}

        # 1. Build positional pools for benchmark replacement calculations
        benchmarks = dict(self.position_benchmarks)
        if player_pool_df is not None and not player_pool_df.empty:
            for pos in ["GKP", "DEF", "MID", "FWD"]:
                sub = player_pool_df[player_pool_df["position"] == pos]
                if not sub.empty and "expected_points" in sub.columns:
                    top_vals = sub["expected_points"].nlargest(5).tolist()
                    if top_vals:
                        benchmarks[pos] = float(top_vals[0])

        # 2. Diagnose individual players (missing chance_of_playing falls back to FPL status: i/s/u/n -> 0, d -> 50)
        play_pct = play_probability(squad_df) * 100.0
        for n, (_, row) in enumerate(squad_df.iterrows()):
            eid = int(row["element"])
            pos = str(row.get("position", "MID"))
            xp = float(row.get("expected_points", 0.0))

            chance_val = float(play_pct.iloc[n])

            news = str(row.get("news", "") or "").strip()
            reasons: list[str] = []
            status: Literal["KEEP", "WATCH", "REPLACE"] = "KEEP"

            # Reason 1: Availability / Injury flag
            if (
                chance_val == 0.0
                or "loan" in news.lower()
                or "suspended" in news.lower()
                or "ruled out" in news.lower()
            ):
                status = "REPLACE"
                reasons.append(f"UNAVAILABLE: {news if news else '0% chance of playing'}")
            elif chance_val <= 25.0:
                status = "REPLACE"
                reasons.append(f"MAJOR_INJURY_RISK: {chance_val:.0f}% chance ({news if news else 'doubtful'})")
            elif chance_val < 100.0 or (news and "doubt" in news.lower()):
                status = "WATCH"
                reasons.append(
                    f"AVAILABILITY_DOUBT: {chance_val:.0f}% chance of playing ({news if news else 'flagged'})"
                )

            # Reason 2: Minutes risk / Low projection
            pos_floor = self.position_floors.get(pos, 3.0)
            if xp == 0.0:
                if status != "REPLACE":
                    status = "REPLACE"
                reasons.append("ZERO_PROJECTION: player projected for 0.0 xP")
            elif xp < pos_floor:
                if status == "KEEP":
                    status = "WATCH"
                reasons.append(f"SUB_FLOOR_XP: {xp:.2f} xP below position floor of {pos_floor:.1f}")

            # Reason 3: Price pressure (selling value risk)
            buy_p = float(row.get("purchase_price", row.get("value", 0.0)))
            curr_p = float(row.get("now_cost", row.get("value", 0.0)))
            if curr_p < buy_p:
                reasons.append(f"PRICE_FALL: value dropped by £{(buy_p - curr_p) / 10:.1f}m since purchase")

            # Reason 4: Form / Fixture deficit
            pos_bench = benchmarks.get(pos, 5.0)
            deficiency = max(0.0, pos_bench - xp * chance_val / 100.0)  # availability-adjusted
            if deficiency > 2.5:
                reasons.append(
                    f"FIXTURE_DEFICIT: {deficiency:.2f} xP gap to position elite benchmark ({pos_bench:.2f})"
                )

            # Functional starter definition: P(play) >= min_prob and xP >= position_floor
            is_functional = (chance_val >= (self.min_play_prob * 100.0)) and (xp >= pos_floor)

            diagnoses[eid] = PlayerHealthStatus(
                element=eid,
                position=pos,
                status=status,
                reasons=reasons,
                functional_starter=is_functional,
                expected_points=xp,
                chance_of_playing=chance_val,
                deficiency_loss=round(deficiency, 2),
            )

        # 3. Determine best functional starting XI
        # Formation rules: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD = 11
        # Sort each position by xP desc
        gkps = [d for d in diagnoses.values() if d.position == "GKP"]
        defs = [d for d in diagnoses.values() if d.position == "DEF"]
        mids = [d for d in diagnoses.values() if d.position == "MID"]
        fwds = [d for d in diagnoses.values() if d.position == "FWD"]

        def avail_xp(d: PlayerHealthStatus) -> float:
            return d.expected_points * d.chance_of_playing / 100.0

        for group in (gkps, defs, mids, fwds):
            group.sort(key=avail_xp, reverse=True)

        # Base formation 1-3-4-3 or best 11
        starters: list[PlayerHealthStatus] = []
        if gkps:
            starters.append(gkps[0])
        starters.extend(defs[:3])
        starters.extend(mids[:2])
        starters.extend(fwds[:1])

        # Remaining outfield spots (5 spots)
        pool = defs[3:5] + mids[2:5] + fwds[1:3]
        pool.sort(key=avail_xp, reverse=True)
        starters.extend(pool[:5])

        functional_count = sum(1 for s in starters if s.functional_starter)

        # 4. Structural deficiency score: points lost on deficient starter slots vs benchmark
        structural_deficiency = sum(
            s.deficiency_loss for s in starters if not s.functional_starter or s.status == "REPLACE"
        )

        # 5. Overall squad classification
        if functional_count >= self.min_functional_starters and structural_deficiency <= self.deficiency_threshold:
            classification: Literal["HEALTHY_CORE_TARGETED_FIXES", "STRUCTURAL_OVERHAUL_REQUIRED"] = (
                "HEALTHY_CORE_TARGETED_FIXES"
            )
        else:
            classification = "STRUCTURAL_OVERHAUL_REQUIRED"

        healthy_core = [d.element for d in diagnoses.values() if d.status == "KEEP"]
        watch_list = [d.element for d in diagnoses.values() if d.status == "WATCH"]
        replace_list = [d.element for d in diagnoses.values() if d.status == "REPLACE"]

        return SquadHealthReport(
            player_diagnoses=diagnoses,
            functional_starters_count=functional_count,
            structural_deficiency_score=round(structural_deficiency, 2),
            classification=classification,
            healthy_core_ids=healthy_core,
            watch_ids=watch_list,
            replace_ids=replace_list,
            thresholds_used={
                "status": "[PLACEHOLDER - TO BE FITTED ON BACKTEST]",
                "min_play_prob": self.min_play_prob,
                "position_floors": self.position_floors,
                "position_benchmarks": benchmarks,
                "min_functional_starters": self.min_functional_starters,
                "deficiency_threshold": self.deficiency_threshold,
            },
        )
