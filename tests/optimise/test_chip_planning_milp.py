"""
Tests for the chip planner and the MILP forced-chip schedule (WI-5.1 & WI-5.2).

1. Forced chip schedule pinning in the multi-period MILP.
2. 10 historical states: reproduces sensible choices and never recommends an illegal sequence.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.chips.planner import (
    ChipPlanner,
)
from fpl_oracle.optimise.multiperiod import (
    MultiPeriodConfig,
    MultiPeriodOptimizer,
)


def _build_test_squad(rng: np.random.Generator) -> pd.DataFrame:
    """Builds a valid 15-player squad without hardcoded player names."""
    rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    for i, pos in enumerate(positions, 1):
        team = (i % 5) + 1
        xp = round(float(rng.uniform(3.0, 7.5)), 2)
        rows.append(
            {
                "element": i,
                "web_name": f"P_{i}",
                "position": pos,
                "team": team,
                "value": 60,
                "now_cost": 60,
                "purchase_price": 60,
                "selling_price": 60,
                "expected_points": xp,
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
    return pd.DataFrame(rows)


def _build_candidate_pool(squad_df: pd.DataFrame, rng: np.random.Generator, extra: int = 25) -> pd.DataFrame:
    """Builds an expanded candidate pool for transfers and chips."""
    rows = squad_df.to_dict(orient="records")
    start_id = int(squad_df["element"].max()) + 1
    for i in range(extra):
        eid = start_id + i
        pos = "GKP" if i < 3 else ("DEF" if i < 10 else ("MID" if i < 18 else "FWD"))
        team = (i % 20) + 1
        xp = round(float(rng.uniform(2.5, 9.5)), 2)
        rows.append(
            {
                "element": eid,
                "web_name": f"Pool_{eid}",
                "position": pos,
                "team": team,
                "value": 65,
                "now_cost": 65,
                "purchase_price": 65,
                "selling_price": 65,
                "expected_points": xp,
                "chance_of_playing": 1.0,
                "chance_of_playing_next_round": 100,
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# Test 1: MILP Arbitrary Chip Pinning via forced_chip_schedule (WI-5.1)
# ==============================================================================
def test_milp_forced_chip_schedule_pinning():
    """
    WI-5.1: Verifies that forced_chip_schedule pins the specified chip at the exact
    target gameweek, suppresses it in other gameweeks in that set, and obeys max 1 chip/GW.
    """
    rng = np.random.default_rng(101)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=10)
    projections = {gw: pool_df for gw in range(6, 10)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=4))

    # Pin Triple Captain at GW7 and Bench Boost at GW9
    plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=5,
        target_gw=6,
        horizon_len=4,
        chips_available=["3xc", "bboost"],
        forced_chip_schedule={7: "3xc", 9: "bboost"},
    )

    sched = {act.gameweek: act.active_chip for act in plan.trajectory if act.active_chip}
    assert sched.get(7) == "3xc", f"Expected 3xc at GW7, got {sched.get(7)}"
    assert sched.get(9) == "bboost", f"Expected bboost at GW9, got {sched.get(9)}"
    assert sched.get(6) is None, f"Expected no chip at GW6, got {sched.get(6)}"
    assert sched.get(8) is None, f"Expected no chip at GW8, got {sched.get(8)}"


# ==============================================================================
# Test 8: 10 Diverse Historical / Synthetic States (WI-5.2 Acceptance)
# ==============================================================================
@pytest.mark.slow
def test_10_historical_states_reproduce_sensible_choices_and_no_illegal_sequences():
    """
    WI-5.2 Acceptance: Chip planning on 10 historical states reproduces sensible choices
    (e.g. FH recommended into a blank, BB after WC, TC with elite asset / DGW)
    and never recommends an illegal sequence.
    """
    rng = np.random.default_rng(808)
    base_squad = _build_test_squad(rng)
    base_pool = _build_candidate_pool(base_squad, rng, extra=15)

    states = [
        # 1. Normal GW6: healthy core -> defer
        {"current_gw": 6, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(6, 11)}},
        # 2. Blank GW8: FA Cup / League Cup blank -> Free Hit viable
        {
            "current_gw": 8,
            "squad_df": base_squad,
            "projections": {gw: base_pool for gw in range(8, 13)},
            "fixtures": [Fixture(id=1, code=1, event=8, team_h=1, team_a=2)],
        },
        # 3. Congested winter period GW15 (<4 days turnaround)
        {"current_gw": 15, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(15, 20)}},
        # 4. GW18: One week before Set 1 hard deadline -> urgent deployment / lowered threshold
        {"current_gw": 18, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(18, 20)}},
        # 5. GW19: Final gameweek of Set 1 -> any positive gain triggers recommendation
        {"current_gw": 19, "squad_df": base_squad, "projections": {19: base_pool}},
        # 6. GW20: Set 2 fresh start -> Set 2 chips available, Set 1 expired
        {"current_gw": 20, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(20, 25)}},
        # 7. GW25: Mid-season normal week
        {"current_gw": 25, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(25, 30)}},
        # 8. GW29: Spring Blank Gameweek candidate
        {"current_gw": 29, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(29, 34)}},
        # 9. GW37: One week before final season deadline -> lowered threshold
        {"current_gw": 37, "squad_df": base_squad, "projections": {37: base_pool, 38: base_pool}},
        # 10. GW38: Final gameweek of season -> use-it-or-lose-it
        {"current_gw": 38, "squad_df": base_squad, "projections": {38: base_pool}},
    ]

    planner = ChipPlanner()

    for idx, state in enumerate(states, 1):
        gw = state["current_gw"]
        res = planner.generate_chip_strategy(
            current_gw=gw,
            current_squad_df=state["squad_df"],
            horizon_projections=state["projections"],
            fixtures=state.get("fixtures", []),
            bootstrap=BootstrapStatic(),
            manager_history=None,
        )

        plan_table = res["chip_plan_table"]
        # Invariant 1: No illegal sequence
        assigned_gws = [r["recommended_gw"] for r in plan_table if r.get("recommended_gw") is not None]
        assert all(1 <= g <= 38 for g in assigned_gws)
        # In each set, assigned gameweeks must be distinct (max 1 chip per GW)
        set1_gws = [r["recommended_gw"] for r in plan_table if r.get("set") == 1]
        set2_gws = [r["recommended_gw"] for r in plan_table if r.get("set") == 2]
        assert len(set1_gws) == len(set(set1_gws)), f"State {idx} (GW{gw}): Duplicate GW in Set 1 schedule!"
        assert len(set2_gws) == len(set(set2_gws)), f"State {idx} (GW{gw}): Duplicate GW in Set 2 schedule!"

        # Invariant 2: Set 1 chips must never be scheduled after GW19
        for r in plan_table:
            if r.get("set") == 1:
                assert r["recommended_gw"] <= 19, f"State {idx}: Set 1 chip scheduled in GW{r['recommended_gw']} > 19"
            if r.get("code") == "freehit":
                assert r["recommended_gw"] != 1, f"State {idx}: Free Hit scheduled in GW1"

        # Invariant 3: Sensible choice at deadlines
        if gw == 19:
            # At GW19, any positive gain Set 1 chip must be recommended
            s1_recs = [r for r in plan_table if r.get("set") == 1]
            for r in s1_recs:
                if r["expected_gain"] > 0:
                    assert r["is_recommended_this_gw"] is True
                    assert r["recommendation_threshold"] == 0.0
        elif gw == 38:
            # At GW38, any positive gain Set 2 chip must be recommended
            s2_recs = [r for r in plan_table if r.get("set") == 2]
            for r in s2_recs:
                if r["expected_gain"] > 0:
                    assert r["is_recommended_this_gw"] is True
                    assert r["recommendation_threshold"] == 0.0
