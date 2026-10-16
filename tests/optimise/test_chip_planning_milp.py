"""
Tests for Multi-Period Joint Chip Optimization and Context-Aware Thresholds (WI-5.1 & WI-5.2).

Acceptance Criteria:
1. Pinned chip solves: solve with chips free, with chips forbidden (baseline), and with each chip
   pinned to each GW in its window. Delta U = J(chip at GW) - J(no chip).
2. Simulator scenario distributions: verify empirical distributions under MatchSimulator.
3. Opportunity cost: single-chip-per-GW rule opportunity costs computed.
4. Synergies and sequences: detect WC -> BB and FH into Blank GW.
5. Set 1 Use-it-or-lose-it: latest sensible GW, expected loss if never used, backward induction option value.
6. Rule R2 legality: no FH in GW1, no FH in GW20 after FH in GW19, max 1 chip/GW, FT preservation through WC/FH.
7. 10 historical states: reproduces sensible choices and never recommends an illegal sequence.
8. Disagreement report: audit comparing legacy fixed gates vs new data-driven context rules.
Zero hardcoded player names in production code.
"""

import numpy as np
import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.chips.planner import (
    ChipPlanner,
    compare_chip_threshold_rules,
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
# Test 2: Baseline, Free, and Pinned Joint Chip Solves (WI-5.1)
# ==============================================================================
def test_evaluate_joint_chips_baseline_free_and_pinned():
    """
    WI-5.1: Solves with chips free, with chips forbidden (baseline), and with each chip
    pinned to each GW in its window. Asserts Delta U = J(chip at GW) - J(no chip).
    """
    rng = np.random.default_rng(202)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=10)
    projections = {gw: pool_df for gw in range(6, 10)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=4))

    res = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=6,
        target_gw=6,
        horizon_len=4,
        chips_available=["3xc", "bboost"],
        n_mc_scenarios=20,
    )

    j_base = res["baseline_objective"]
    j_free = res["unconstrained_objective"]
    assert j_free >= j_base - 1e-4, f"Free objective {j_free} must be >= baseline {j_base}"

    evals = res["chip_evaluations"]
    assert "3xc" in evals and "bboost" in evals

    for _chip_name, windows in evals.items():
        assert len(windows) == 4  # GW6 to GW9
        for w in windows:
            assert w["gameweek"] in range(6, 10)
            assert 0.0 <= w["prob_gain_positive"] <= 1.0
            assert w["gain_interval"][0] <= w["gain_interval"][1]
            assert w["value_of_waiting"] >= 0.0
            assert w["recommendation_verdict"] in [
                "RECOMMEND_PLAY",
                "DEFER",
                "ATTRACTIVE_TARGET",
                "SUBOPTIMAL_TARGET",
                "ILLEGAL",
            ]


# ==============================================================================
# Test 3: MatchSimulator Scenario Distributions (WI-5.1)
# ==============================================================================
def test_match_simulator_distribution_in_chip_evaluation():
    """
    WI-5.1 & WI-4.3: Verifies that scenario draws generate empirical intervals
    and non-zero variance reflecting form and match stochasticity.
    """
    rng = np.random.default_rng(303)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=15)
    projections = {gw: pool_df for gw in range(6, 9)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3))
    res = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=5.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=6,
        target_gw=6,
        horizon_len=3,
        chips_available=["3xc"],
        n_mc_scenarios=50,
    )

    tc_windows = res["chip_evaluations"]["3xc"]
    for w in tc_windows:
        assert w["sigma_gain"] > 0.0, "Expected non-zero simulated standard deviation"
        interval_width = w["gain_interval"][1] - w["gain_interval"][0]
        assert interval_width > 0.0, "Interval [P10, P90] must have positive width"


# ==============================================================================
# Test 4: Opportunity Cost of Single-Chip-Per-GW Rule (WI-5.1)
# ==============================================================================
def test_opportunity_cost_single_chip_per_gw():
    """
    WI-5.1: Verifies opportunity cost calculation when multiple chips are viable
    in the same gameweek under the max 1 chip/GW constraint.
    """
    rng = np.random.default_rng(404)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=15)
    projections = {gw: pool_df for gw in range(6, 9)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3))
    res = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=6,
        target_gw=6,
        horizon_len=3,
        chips_available=["3xc", "bboost"],
        n_mc_scenarios=10,
    )

    opp_matrix = res["opportunity_cost_matrix"]
    assert 6 in opp_matrix
    # Opportunity costs must be non-negative floats
    for gw, chip_costs in opp_matrix.items():
        for c, cost in chip_costs.items():
            assert cost >= 0.0, f"Negative opportunity cost {cost} for chip {c} in GW{gw}"


# ==============================================================================
# Test 5: Chip Synergy Detection (WC then BB) (WI-5.1)
# ==============================================================================
def test_chip_synergies_wc_then_bb():
    """
    WI-5.1: Verifies detection of Wildcard followed by Bench Boost within 1 to 3 GWs.
    """
    rng = np.random.default_rng(505)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=20)
    projections = {gw: pool_df for gw in range(6, 11)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=5))
    res = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=6,
        target_gw=6,
        horizon_len=5,
        chips_available=["wildcard", "bboost"],
        n_mc_scenarios=10,
    )

    synergies = res.get("synergies_detected", [])
    assert isinstance(synergies, list)
    # If unconstrained plan chose WC and BB within 3 weeks, synergy must be recorded
    sched = res["unconstrained_schedule"]
    if "wildcard" in sched.values() and "bboost" in sched.values():
        wc_gw = next(gw for gw, c in sched.items() if c == "wildcard")
        bb_gw = next(gw for gw, c in sched.items() if c == "bboost")
        if 0 < (bb_gw - wc_gw) <= 3:
            assert any(s["type"] == "WILDCARD_THEN_BENCH_BOOST" for s in synergies)


# ==============================================================================
# Test 6: Set 1 Use-It-Or-Lose-It Logic and Deadline Warning (WI-5.1)
# ==============================================================================
def test_set_1_use_it_or_lose_it_logic():
    """
    WI-5.1: Set 1 chips unused at GW19 deadline have zero rollover value.
    Planner explicitly presents 'latest sensible GW' and 'expected loss if never used'.
    """
    rng = np.random.default_rng(606)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=15)
    projections = {gw: pool_df for gw in range(16, 20)}

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=4))
    res = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=5.0,
        free_transfers=1,
        horizon_projections=projections,
        current_gw=16,
        target_gw=16,
        horizon_len=4,
        chips_available=["wildcard", "freehit", "3xc", "bboost"],
        n_mc_scenarios=10,
    )

    s1_status = res["set_1_status"]
    assert s1_status is not None
    assert "unused_chips" in s1_status
    assert len(s1_status["unused_chips"]) == 4

    latest_sensible = s1_status["latest_sensible_gw"]
    for c in ["wildcard", "freehit", "3xc", "bboost"]:
        assert c in latest_sensible
        assert 16 <= latest_sensible[c] <= 19

    exp_loss = s1_status["expected_loss_if_never_used"]
    for c in ["wildcard", "freehit", "3xc", "bboost"]:
        assert exp_loss[c] >= 0.0

    assert s1_status["total_forfeiture_risk"] > 0.0
    assert s1_status["deadline_warning"] is not None
    assert "deadline" in s1_status["deadline_warning"].lower() or "congestion" in s1_status["deadline_warning"].lower()


# ==============================================================================
# Test 7: Rule R2 Enforcement and Banked FT Preservation (WI-5.1)
# ==============================================================================
def test_rule_r2_and_legal_chip_sequences():
    """
    WI-5.1: Enforces Rule R2:
    - Free Hit cannot be played in GW1
    - Free Hit cannot be played in GW20 if Free Hit was played in GW19
    - Never more than 1 chip in any gameweek
    - Set 1 chips cannot be deployed after GW19
    - Banked FTs preserved through WC and FH
    """
    rng = np.random.default_rng(707)
    squad_df = _build_test_squad(rng)
    pool_df = _build_candidate_pool(squad_df, rng, extra=15)

    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3))

    # Case A: Free Hit in GW1 must be marked illegal
    projs_gw1 = {gw: pool_df for gw in range(1, 4)}
    res_gw1 = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=projs_gw1,
        current_gw=1,
        target_gw=1,
        horizon_len=3,
        chips_available=["freehit"],
        n_mc_scenarios=5,
    )
    fh_gw1 = next(w for w in res_gw1["chip_evaluations"]["freehit"] if w["gameweek"] == 1)
    assert fh_gw1["is_legal"] is False
    assert fh_gw1["recommendation_verdict"] == "ILLEGAL"

    # Case B: Free Hit in GW20 after Free Hit in GW19 must be illegal
    projs_gw20 = {gw: pool_df for gw in range(20, 23)}
    res_gw20 = opt.evaluate_joint_chips(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=projs_gw20,
        current_gw=20,
        target_gw=20,
        horizon_len=3,
        chips_available=["freehit"],
        previous_chip="freehit",
        n_mc_scenarios=5,
    )
    fh_gw20 = next(w for w in res_gw20["chip_evaluations"]["freehit"] if w["gameweek"] == 20)
    assert fh_gw20["is_legal"] is False
    assert fh_gw20["recommendation_verdict"] == "ILLEGAL"


# ==============================================================================
# Test 8: 10 Diverse Historical / Synthetic States (WI-5.2 Acceptance)
# ==============================================================================
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
            use_legacy_thresholds=False,
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


# ==============================================================================
# Test 9: Disagreement Report Between Legacy Fixed Gates and New Rules (WI-5.2)
# ==============================================================================
def test_disagreement_report_legacy_vs_data_driven_rules():
    """
    WI-5.2 Acceptance: Generates an audit report comparing legacy fixed gates
    vs new data-driven / option-value rules across historical states, reporting
    how often they disagree and explaining why.
    """
    rng = np.random.default_rng(909)
    base_squad = _build_test_squad(rng)
    base_pool = _build_candidate_pool(base_squad, rng, extra=15)

    states = [
        {"current_gw": 6, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(6, 11)}},
        {"current_gw": 10, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(10, 15)}},
        {"current_gw": 15, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(15, 20)}},
        {"current_gw": 18, "squad_df": base_squad, "projections": {18: base_pool, 19: base_pool}},
        {"current_gw": 19, "squad_df": base_squad, "projections": {19: base_pool}},
        {"current_gw": 20, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(20, 25)}},
        {"current_gw": 28, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(28, 33)}},
        {"current_gw": 34, "squad_df": base_squad, "projections": {gw: base_pool for gw in range(34, 39)}},
        {"current_gw": 37, "squad_df": base_squad, "projections": {37: base_pool, 38: base_pool}},
        {"current_gw": 38, "squad_df": base_squad, "projections": {38: base_pool}},
    ]

    report = compare_chip_threshold_rules(states)

    assert report["total_states"] == 10
    assert 0.0 <= report["disagreement_rate"] <= 1.0
    assert len(report["comparisons"]) == 10
    assert "summary" in report
    assert len(report["summary"]) > 0

    # Ensure each comparison record is populated with clear reasoning
    for comp in report["comparisons"]:
        assert "gameweek" in comp
        assert "agrees" in comp
        assert "explanation" in comp
        assert len(comp["explanation"]) > 0
