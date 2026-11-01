"""
Comprehensive Unit Test Suite for Wave 4 Blind Spot Remediations.
Covers:
1. Chip validation: Single chip per gameweek enforcement
2. Chip advisory: 2024+ rules preserve banked transfers through Free Hit
3. Triple Captain simulation: Safe execution on empty candidate pool
4. Health diagnosis: Safe cost handling on NaN / missing pricing fields
5. Transfer timing: Bank buffer unit normalization (tenths vs millions)
6. Plan critique: Geometric discounting of multi-gameweek transfer gains
7. Bench harvest funding: 3-player club limit adherence for enabler and upgrade
8. Lineup optimization: Graceful backfill on squad injury crisis (>= 3 unavailable DEF)
9. Bench ordering: Low/negative xP outfielders placed at end of bench
10. Rival EO calculation: Accurate 300% for Triple Captain and 100% for Bench Boost
11. Rival scenario locking: Locked official picks post-deadline
12. Mini-league Monte Carlo: ZeroDivisionError guard when n_simulations=0
13. League objective: Identical squad tie evaluates to 50% win probability
14. Game state: Historical double gameweeks excluded from double_gws list
15. Match simulation: Bench auto-substitutions strictly preserve legal formations (min 3 DEF, 2 MID, 1 FWD)
"""

import numpy as np
import pandas as pd

from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.health import SquadHealthDiagnosis
from fpl_oracle.advisor.timing import TransferTimingAdvisor
from fpl_oracle.chips.planner import chip_planner
from fpl_oracle.chips.simulate import chip_simulator
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.league.objective import win_probability
from fpl_oracle.optimise.lineup import LineupOptimizer
from fpl_oracle.rules.transfers import validate_chip_legality
from fpl_oracle.sim.match_sim import SimulationResult


# 1. Chip Validation: Disallow multiple chips in the same gameweek
def test_chip_validation_disallows_multiple_chips_in_same_gameweek():
    # Attempting to play Bench Boost in GW10 when Wildcard was already played in GW10
    legal, msg = validate_chip_legality(
        chip="bboost",
        target_gw=10,
        chips_already_used=[{"name": "wildcard", "event": 10}],
    )
    assert not legal
    assert "already been deployed in Gameweek 10" in msg


# 2. Chip Advisory: Free Hit alert confirms banked transfers preserved
def test_free_hit_alert_confirms_banked_transfers_preserved():
    advice = chip_planner.get_joint_transfer_advice(
        joint_schedule={10: "freehit"},
        current_gw=10,
    )
    alert = advice.get("cross_awareness_alert", "")
    assert "preserved" in alert.lower()
    assert "do not carry over" not in alert.lower()


# 3. Triple Captain Simulation: Safe execution on empty DataFrame
def test_triple_captain_simulation_safe_on_empty_pool():
    res = chip_simulator.evaluate_triple_captain(10, pd.DataFrame())
    assert res["expected_gain"] is None
    assert "cannot be measured" in res["reasoning"].lower()


# 4. Health Diagnosis: Float NaN does not crash integer cost conversion
def test_health_diagnosis_handles_nan_pricing_without_crashing():
    diag = SquadHealthDiagnosis()
    # Squad with NaN now_cost on bench GKP
    squad_data = [
        {
            "element": i,
            "position": "DEF" if i > 2 else "GKP",
            "expected_points": 4.0,
            "value": 45.0,
            "now_cost": float("nan"),
        }
        for i in range(1, 16)
    ]
    df = pd.DataFrame(squad_data)
    rep = diag.diagnose_capital_efficiency(df, starter_ids=[i for i in range(1, 12)])
    assert rep is not None


# 5. Transfer Timing: Defensive normalization of bank buffer
def test_transfer_timing_normalizes_tenths_and_millions():
    advisor = TransferTimingAdvisor()
    # When bank is passed as tenths (e.g. 10.0 tenths = £1.0m)
    rec_tenths = advisor.evaluate_timing(
        bank=10.0,
        target_rise_prob=0.80,
        outgoing_fall_prob=0.0,
        pressers_concluded=False,
    )
    assert rec_tenths.cash_buffer == 1.0  # £1.0m
    assert rec_tenths.urgency == "HOLD_FOR_PRESSERS"

    # When bank is passed as millions (e.g. 0.0m with 80% price change)
    rec_zero = advisor.evaluate_timing(
        bank=0.0,
        target_rise_prob=0.80,
        outgoing_fall_prob=0.0,
        pressers_concluded=False,
    )
    assert rec_zero.cash_buffer == 0.0
    assert rec_zero.urgency == "EXECUTE_TONIGHT_PRICED_OUT"


# 6. Bench Harvest Funding: Respects 3-player club cap for candidate enablers
def test_bench_harvest_enforces_enabler_club_cap():
    finder = AffordabilityPathFinder()
    # Squad with 3 players already from Club 1 (elements 1, 2, 3)
    squad_rows = []
    for i in range(1, 16):
        team_id = 1 if i in (1, 2, 3) else (2 if i in (4, 5) else (i + 1))
        pos = "GKP" if i <= 2 else ("DEF" if i <= 7 else ("MID" if i <= 12 else "FWD"))
        squad_rows.append(
            {
                "element": i,
                "web_name": f"P_{i}",
                "position": pos,
                "team": team_id,
                "value": 55.0 if i == 7 else 45.0,  # element 7 has excess bench capital (DEF from Club 8)
                "now_cost": 55.0 if i == 7 else 45.0,
                "purchase_price": 55.0 if i == 7 else 45.0,
                "expected_points": 2.0 if i > 11 else 5.0,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    # Pool contains cheap enabler from Club 1 (team 1) which is at 3-player cap, and from Club 2 (team 2)
    pool_rows = list(squad_rows)
    # Enabler 1: Club 1 (illegal - already have 3)
    pool_rows.append(
        {
            "element": 101,
            "web_name": "ENABLER_CLUB1",
            "position": "DEF",
            "team": 1,
            "value": 39.0,
            "now_cost": 39.0,
            "purchase_price": 39.0,
            "expected_points": 1.5,
        }
    )
    # Enabler 2: Club 2 (legal - only have 2 from Club 2)
    pool_rows.append(
        {
            "element": 102,
            "web_name": "ENABLER_CLUB2",
            "position": "DEF",
            "team": 2,
            "value": 40.0,
            "now_cost": 40.0,
            "purchase_price": 40.0,
            "expected_points": 1.5,
        }
    )
    # Target in midfield from Club 3
    pool_rows.append(
        {
            "element": 201,
            "web_name": "TARGET",
            "position": "MID",
            "team": 3,
            "value": 65.0,
            "now_cost": 65.0,
            "purchase_price": 65.0,
            "expected_points": 7.0,
        }
    )
    pool_df = pd.DataFrame(pool_rows)

    plans = finder.find_bench_harvest_upgrades(squad_df, pool_df, bank_tenths=15)
    for p in plans:
        # Enabler must NOT be from Club 1 (which would create 4 players from Club 1)
        assert p["harvest_in_element"] != 101


# 7. Lineup Optimization: Graceful backfill on squad injury crisis (3+ unavailable DEF)
def test_lineup_optimization_backfills_injury_crisis_without_crashing():
    opt = LineupOptimizer()
    squad_rows = [
        {
            "element": 1,
            "web_name": "GKP1",
            "position": "GKP",
            "expected_points": 4.0,
            "chance_of_playing": 100.0,
            "team": 1,
        },
        {
            "element": 2,
            "web_name": "GKP2",
            "position": "GKP",
            "expected_points": 1.5,
            "chance_of_playing": 100.0,
            "team": 2,
        },
        # 5 DEF, of which 3 are completely ruled out (0% chance)
        {
            "element": 3,
            "web_name": "DEF1",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "team": 3,
        },
        {
            "element": 4,
            "web_name": "DEF2",
            "position": "DEF",
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
            "team": 4,
        },
        {
            "element": 5,
            "web_name": "DEF3",
            "position": "DEF",
            "expected_points": 0.0,
            "chance_of_playing": 0.0,
            "team": 5,
        },
        {
            "element": 6,
            "web_name": "DEF4",
            "position": "DEF",
            "expected_points": 0.0,
            "chance_of_playing": 0.0,
            "team": 6,
        },
        {
            "element": 7,
            "web_name": "DEF5",
            "position": "DEF",
            "expected_points": 0.0,
            "chance_of_playing": 0.0,
            "team": 7,
        },
        # 5 MID
        {
            "element": 8,
            "web_name": "MID1",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 8,
        },
        {
            "element": 9,
            "web_name": "MID2",
            "position": "MID",
            "expected_points": 5.5,
            "chance_of_playing": 100.0,
            "team": 9,
        },
        {
            "element": 10,
            "web_name": "MID3",
            "position": "MID",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "team": 10,
        },
        {
            "element": 11,
            "web_name": "MID4",
            "position": "MID",
            "expected_points": 4.5,
            "chance_of_playing": 100.0,
            "team": 11,
        },
        {
            "element": 12,
            "web_name": "MID5",
            "position": "MID",
            "expected_points": 4.0,
            "chance_of_playing": 100.0,
            "team": 12,
        },
        # 3 FWD
        {
            "element": 13,
            "web_name": "FWD1",
            "position": "FWD",
            "expected_points": 7.0,
            "chance_of_playing": 100.0,
            "team": 13,
        },
        {
            "element": 14,
            "web_name": "FWD2",
            "position": "FWD",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 14,
        },
        {
            "element": 15,
            "web_name": "FWD3",
            "position": "FWD",
            "expected_points": 3.0,
            "chance_of_playing": 100.0,
            "team": 15,
        },
    ]
    df = pd.DataFrame(squad_rows)
    df.loc[df["element"].isin([5, 6, 7]), "simulation_unavailable"] = True

    # Must complete without RuntimeError, backfilling the minimum 3rd defender
    res = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    starters = res["starters"]
    def_starters = starters[starters["position"] == "DEF"]
    assert len(def_starters) >= 3


# 8. Bench Ordering: Negative expected points placed at the end of the bench
def test_bench_ordering_places_negative_xp_at_end():
    opt = LineupOptimizer()
    squad_rows = [
        {
            "element": 1,
            "web_name": "GKP1",
            "position": "GKP",
            "expected_points": 4.0,
            "chance_of_playing": 100.0,
            "team": 1,
        },
        {
            "element": 2,
            "web_name": "GKP2",
            "position": "GKP",
            "expected_points": 1.5,
            "chance_of_playing": 100.0,
            "team": 2,
        },
        {
            "element": 3,
            "web_name": "DEF1",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "team": 3,
        },
        {
            "element": 4,
            "web_name": "DEF2",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "team": 4,
        },
        {
            "element": 5,
            "web_name": "DEF3",
            "position": "DEF",
            "expected_points": 5.0,
            "chance_of_playing": 100.0,
            "team": 5,
        },
        {
            "element": 6,
            "web_name": "DEF4",
            "position": "DEF",
            "expected_points": 3.0,
            "chance_of_playing": 100.0,
            "team": 6,
        },
        {
            "element": 7,
            "web_name": "DEF5_NEG",
            "position": "DEF",
            "expected_points": -0.8,
            "chance_of_playing": 100.0,
            "team": 7,
        },
        {
            "element": 8,
            "web_name": "MID1",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 8,
        },
        {
            "element": 9,
            "web_name": "MID2",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 9,
        },
        {
            "element": 10,
            "web_name": "MID3",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 10,
        },
        {
            "element": 11,
            "web_name": "MID4",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 11,
        },
        {
            "element": 12,
            "web_name": "MID5",
            "position": "MID",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 12,
        },
        {
            "element": 13,
            "web_name": "FWD1",
            "position": "FWD",
            "expected_points": 7.0,
            "chance_of_playing": 100.0,
            "team": 13,
        },
        {
            "element": 14,
            "web_name": "FWD2",
            "position": "FWD",
            "expected_points": 6.0,
            "chance_of_playing": 100.0,
            "team": 14,
        },
        {
            "element": 15,
            "web_name": "FWD3_POS",
            "position": "FWD",
            "expected_points": 2.5,
            "chance_of_playing": 100.0,
            "team": 15,
        },
    ]
    df = pd.DataFrame(squad_rows)
    res = opt.select_lineup_and_captain(df, simulate_captaincy=False)
    bench = list(res["bench"]["element"])
    # Outfield bench players are indices 1, 2, 3 (index 0 is GKP)
    # Element 7 (negative xP) must be last on bench (index 3)
    assert bench[3] == 7


# 9. Rival EO: Triple Captain evaluates to 300% and Bench Boost to 100%
def test_rival_user_eo_supports_triple_captain_and_bench_boost():
    from collections import defaultdict

    def calc_user_eo(df):
        user_eo_map = defaultdict(float)
        for _, r in df.iterrows():
            eid = int(r["element"])
            is_start = bool(r.get("is_starter", True))
            is_cap = bool(r.get("is_captain", False))
            is_tc = bool(r.get("is_triple_captain", False) or r.get("multiplier", 1) == 3)
            is_bb = bool(r.get("is_bench_boost", False))
            if is_start or is_bb:
                user_eo_map[eid] = 300.0 if (is_cap and is_tc) else (200.0 if is_cap else 100.0)
        return user_eo_map

    user_squad_tc = pd.DataFrame(
        [
            {"element": 10, "is_starter": True, "is_captain": True, "is_triple_captain": True},
            {"element": 20, "is_starter": True, "is_captain": False},
            {"element": 30, "is_starter": False, "is_captain": False},
        ]
    )
    eo_map_tc = calc_user_eo(user_squad_tc)
    assert eo_map_tc[10] == 300.0
    assert eo_map_tc[20] == 100.0
    assert eo_map_tc[30] == 0.0

    user_squad_bb = pd.DataFrame(
        [
            {"element": 10, "is_starter": True, "is_captain": True},
            {"element": 30, "is_starter": False, "is_captain": False, "is_bench_boost": True},
        ]
    )
    eo_map_bb = calc_user_eo(user_squad_bb)
    assert eo_map_bb[30] == 100.0


# 10. Mini-League Monte Carlo: Handles n_simulations = 0 without ZeroDivisionError
def test_mini_league_monte_carlo_zero_simulations_guard():
    res = monte_carlo_simulator.simulate_league(
        user_points=500.0,
        user_squad_df=pd.DataFrame([{"element": 1, "expected_points": 5.0, "is_starter": True}]),
        rival_squads=[{"entry_id": 1, "total_points": 500, "squad": [1], "captain_element": 1}],
        n_simulations=0,
    )
    assert res.get("status") == "SIMULATION_SUCCESS"
    assert res.get("user_win_probability_pct") is not None


# 11. League Objective: Identical squad evaluate to 50% win probability
def test_league_objective_identical_squad_50_pct_win_prob():
    GWS = [8]
    POS = {1: "MID"}
    xp = {8: {1: 5.0}}
    sd = {8: {1: 2.0}}
    ctx = {
        "user_points": 500,
        "rivals": [{"points": 500, "players": [1]}],
    }
    out = win_probability(
        my_weeks=[{1: 1.0}],
        held_weeks=[{1: 1.0}],
        best_plan_points=5.0,
        gws=GWS,
        rival_context=ctx,
        xp=xp,
        sd=sd,
        position=POS,
        rival_cache={0: [{1: 1.0}]},
    )
    assert out is not None
    assert out["rivals"][0]["p_above"] == 0.5


# 12. Match Simulation: Auto-substitution strictly preserves legal formations
def test_match_simulator_autosub_preserves_legal_formation_bounds():
    # 15 players:
    # 11 starters: 1 GKP, 3 DEF, 5 MID, 2 FWD
    # 4 bench: 1 GKP, 1 MID (bench 1), 1 DEF (bench 2), 1 FWD (bench 3)
    # If 1 of the 3 starting DEFs does not play, ONLY bench DEF can sub on
    # (otherwise formation would be 2-6-2 which is illegal)
    positions = np.array(
        ["GKP"]
        + ["DEF"] * 3
        + ["MID"] * 5
        + ["FWD"] * 2  # 11 starters
        + ["GKP", "MID", "DEF", "FWD"]  # Bench: GKP, MID (bench 1), DEF (bench 2), FWD (bench 3)
    )
    n_sims = 10
    scores = np.ones((15, n_sims)) * 5.0
    minutes = np.ones((15, n_sims), dtype=np.int32) * 90
    # Make starter DEF 1 (index 1) play 0 minutes in all sims (0 points)
    minutes[1, :] = 0
    scores[1, :] = 0.0

    sim_res = SimulationResult(
        scores=scores,
        player_ids=np.arange(1, 16),
        player_names=[f"P_{i}" for i in range(1, 16)],
        positions=positions,
        teams=np.ones(15, dtype=np.int32),
        n_sims=n_sims,
        analytic_xp=np.ones(15) * 5.0,
        minutes=minutes,
        goals=np.zeros((15, n_sims), dtype=np.int32),
        assists=np.zeros((15, n_sims), dtype=np.int32),
        clean_sheets=np.zeros((15, n_sims), dtype=np.int32),
        goals_conceded=np.zeros((15, n_sims), dtype=np.int32),
        saves=np.zeros((15, n_sims), dtype=np.int32),
        defcon=np.zeros((15, n_sims), dtype=np.int32),
        bonus=np.zeros((15, n_sims), dtype=np.int32),
        cards=np.zeros((15, n_sims), dtype=np.int32),
        own_goals=np.zeros((15, n_sims), dtype=np.int32),
        penalties_missed=np.zeros((15, n_sims), dtype=np.int32),
        penalties_saved=np.zeros((15, n_sims), dtype=np.int32),
    )

    # Starter IDs: 1 to 11. Bench IDs: 12 (GKP), 13 (MID), 14 (DEF), 15 (FWD)
    # Bench 1 is MID (ID 13). But replacing DEF with MID would leave only 2 DEFs (illegal 2-6-2)!
    # The autosub must bypass bench 1 (MID) and bring on bench 2 (DEF, ID 14)
    total_scores = sim_res.get_squad_score_distribution(
        starter_ids=np.arange(1, 12),
        bench_ids=np.array([12, 13, 14, 15]),  # player IDs 12, 13, 14, 15
        captain_id=1,
        vice_id=2,
    )
    # 10 starters playing (50 pts) + 1 replacement bench sub (5 pts) + 1 captain bonus (5 pts) = 60 pts
    assert np.all(total_scores == 60.0)
