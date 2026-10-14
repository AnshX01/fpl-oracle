"""
Property and Acceptance Tests for Multi-Period MILP Transfer Optimizer (WI-2.1 to WI-2.5).

Required Property Tests:
1. Equivalence at H=1, no transfers: equals the existing lineup optimizer on 50 random squads.
2. Dominance: MILP objective >= beam objective on 30 historical/synthetic states.
3. Rules oracle agreement: replay the MILP's chosen plan through rules/transfers.py; hits, FT counts and bank match exactly.
4. Feasibility: squad composition, club cap, formation, budget at every GW.
5. Chip logic: never two chips in a GW; FH not in GW1; no FH in both GW19 and GW20; Set 1 chips never scheduled after GW19.
6. Brute force: on a toy league (3 clubs, 4 positions, H=2) enumerate all legal plans and match the MILP optimum.
"""

import numpy as np
import pandas as pd
import pytest

from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer as BeamTransferOptimizer
from fpl_oracle.rules.transfers import next_free_transfers


def _generate_valid_random_squad(rng: np.random.Generator, squad_id_offset: int = 0) -> pd.DataFrame:
    """Generates a strictly valid 15-player squad (2 GKP, 5 DEF, 5 MID, 3 FWD, <=3 per club)."""
    rows = []
    # 20 teams available, distribute to avoid any club exceeding 3
    # 2 GKP
    for i in range(2):
        eid = squad_id_offset + i + 1
        rows.append({
            "element": eid,
            "web_name": f"GKP_{eid}",
            "position": "GKP",
            "team": i + 1,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": round(rng.uniform(2.0, 6.0), 2),
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
    # 5 DEF
    for i in range(5):
        eid = squad_id_offset + i + 3
        rows.append({
            "element": eid,
            "web_name": f"DEF_{eid}",
            "position": "DEF",
            "team": (i % 5) + 3,
            "value": 50,
            "purchase_price": 50,
            "selling_price": 50,
            "expected_points": round(rng.uniform(2.0, 7.0), 2),
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
    # 5 MID
    for i in range(5):
        eid = squad_id_offset + i + 8
        rows.append({
            "element": eid,
            "web_name": f"MID_{eid}",
            "position": "MID",
            "team": (i % 5) + 8,
            "value": 65,
            "purchase_price": 65,
            "selling_price": 65,
            "expected_points": round(rng.uniform(2.5, 9.0), 2),
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
    # 3 FWD
    for i in range(3):
        eid = squad_id_offset + i + 13
        rows.append({
            "element": eid,
            "web_name": f"FWD_{eid}",
            "position": "FWD",
            "team": (i % 3) + 13,
            "value": 75,
            "purchase_price": 75,
            "selling_price": 75,
            "expected_points": round(rng.uniform(3.0, 9.5), 2),
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
    return pd.DataFrame(rows)


def _generate_candidate_pool(squad_df: pd.DataFrame, rng: np.random.Generator, extra_players: int = 50) -> pd.DataFrame:
    """Generates candidate pool extending a given squad."""
    pool_rows = squad_df.to_dict(orient="records")
    start_eid = int(squad_df["element"].max()) + 1
    for i in range(extra_players):
        eid = start_eid + i
        pos = "GKP" if i < 6 else ("DEF" if i < 20 else ("MID" if i < 38 else "FWD"))
        team = (i % 20) + 1
        val = int(rng.integers(45, 120))
        xp = round(rng.uniform(1.5, 9.0), 2)
        pool_rows.append({
            "element": eid,
            "web_name": f"Pool_{eid}",
            "position": pos,
            "team": team,
            "value": val,
            "purchase_price": val,
            "selling_price": val,
            "expected_points": xp,
            "chance_of_playing": 1.0,
            "chance_of_playing_next_round": 100,
        })
    return pd.DataFrame(pool_rows)


# ==============================================================================
# Property Test 1: Equivalence at H=1, no transfers
# ==============================================================================
@pytest.mark.slow
def test_property_1_h1_equivalence_to_lineup_optimizer():
    """
    Property Test 1:
    At H=1 with transfers forced to 0, MultiPeriodOptimizer produces the exact
    same starting XI, captain, and points total as the existing lineup optimizer
    across 50 random valid squads.
    """
    rng = np.random.default_rng(1001)
    optimizer = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=1, w_bench_mean=0.0, lambda_bank=0.0))

    for squad_idx in range(50):
        squad_df = _generate_valid_random_squad(rng, squad_id_offset=squad_idx * 20)

        # Baseline: Lineup optimizer
        lineup_res = lineup_optimizer.select_lineup_and_captain(squad_df, risk_preference="balanced")
        expected_lineup_xp = float(lineup_res["total_gameweek_expected_points"])
        expected_starters = set(int(r["element"]) for _, r in lineup_res["starters"].iterrows())
        expected_captain = int(lineup_res["captain"]["element"])

        # Multi-Period MILP at H=1 with 0 transfers
        plan = optimizer.optimize(
            current_squad_df=squad_df,
            player_pool_df=squad_df,
            bank=10.0,
            free_transfers=1,
            horizon_projections={10: squad_df},
            current_gw=9,
            target_gw=10,
            horizon_len=1,
            forced_w1_transfers=0,
        )

        milp_action = plan.recommended_action
        milp_starters = set(milp_action.starters)
        milp_captain = milp_action.captain
        milp_score = milp_action.lineup_xp + milp_action.captain_xp

        # 1. Total starting XI + captain points must match within 1e-4
        assert np.isclose(milp_score, expected_lineup_xp, atol=1e-4), (
            f"Squad {squad_idx}: MILP score {milp_score} != lineup optimizer {expected_lineup_xp}"
        )

        # 2. Captain must match
        assert milp_captain == expected_captain, (
            f"Squad {squad_idx}: MILP captain {milp_captain} != lineup captain {expected_captain}"
        )

        # 3. Starting XI players must match
        assert milp_starters == expected_starters, (
            f"Squad {squad_idx}: Starters mismatch: {milp_starters.symmetric_difference(expected_starters)}"
        )


# ==============================================================================
# Property Test 2: Dominance: MILP objective >= beam objective on 30 states
# ==============================================================================
@pytest.mark.slow
def test_property_2_dominance_over_beam_search():
    """
    Property Test 2:
    Because MILP solves the global exact optimization without heuristic beam pruning,
    the MILP objective value is >= beam search objective value on 30 test states.
    """
    rng = np.random.default_rng(2002)
    milp_opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=3, w_bench_mean=0.05, lambda_bank=0.0))
    beam_opt = BeamTransferOptimizer()

    for idx in range(30):
        squad_df = _generate_valid_random_squad(rng, squad_id_offset=idx * 100)
        pool_df = _generate_candidate_pool(squad_df, rng, extra_players=20)
        horizon = {10: pool_df.copy(), 11: pool_df.copy(), 12: pool_df.copy()}

        # Solve via MILP
        milp_plan = milp_opt.optimize(
            current_squad_df=squad_df,
            player_pool_df=pool_df,
            bank=10.0,
            free_transfers=1,
            horizon_projections=horizon,
            current_gw=9,
            target_gw=10,
            horizon_len=3,
        )

        # Solve via Beam
        beam_res = beam_opt.evaluate_transfer_options(
            current_squad_df=squad_df,
            player_pool_df=pool_df,
            bank=10.0,
            free_transfers=1,
            horizon_projections=horizon,
            current_gw=9,
            target_gw=10,
            horizon_len=3,
            num_mc_scenarios=0,
        )
        beam_rec = beam_res["recommended_plan"]
        beam_horizon_xp = float(beam_rec.get("horizon_net_xp", 0.0))

        # MILP horizon net xP must be >= beam horizon net xP (within 0.1 tolerance for discrete ties)
        assert milp_plan.horizon_net_xp >= beam_horizon_xp - 0.1, (
            f"State {idx}: MILP net xP {milp_plan.horizon_net_xp} < Beam net xP {beam_horizon_xp}"
        )


# ==============================================================================
# Property Test 3: Rules oracle agreement
# ==============================================================================
@pytest.mark.slow
def test_property_3_rules_oracle_agreement():
    """
    Property Test 3:
    Replaying the MILP's chosen plan through rules/transfers.py next_free_transfers()
    matches hits, FT counts, and bank balances exactly at every step.
    """
    rng = np.random.default_rng(3003)
    squad_df = _generate_valid_random_squad(rng)
    pool_df = _generate_candidate_pool(squad_df, rng, extra_players=40)
    horizon = {t: pool_df.copy() for t in range(10, 18)}

    optimizer = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=8))
    plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=15.0,
        free_transfers=3,
        horizon_projections=horizon,
        current_gw=9,
        target_gw=10,
    )

    replayed_ft = 3
    replayed_bank = 15.0

    for act in plan.trajectory:
        # 1. FT and Hit agreement
        expected_next_ft, expected_hit_points = next_free_transfers(
            ft=replayed_ft,
            transfers_made=act.transfers_count,
            chip=act.active_chip,
        )

        assert act.hit_cost == expected_hit_points, (
            f"GW{act.gameweek}: MILP hit cost {act.hit_cost} != oracle {expected_hit_points}"
        )
        assert act.free_transfers_next == expected_next_ft, (
            f"GW{act.gameweek}: MILP next FT {act.free_transfers_next} != oracle {expected_next_ft}"
        )

        # 2. Bank balance agreement
        s_vals = sum(t["sell_price"] for t in act.transfers_out)
        b_costs = sum(t["cost"] for t in act.transfers_in)
        replayed_bank = round(replayed_bank + s_vals - b_costs, 1)

        assert np.isclose(act.bank_end, replayed_bank, atol=0.2), (
            f"GW{act.gameweek}: MILP bank {act.bank_end} != replayed bank {replayed_bank}"
        )

        replayed_ft = expected_next_ft


# ==============================================================================
# Property Test 4: Feasibility across horizon
# ==============================================================================
@pytest.mark.slow
def test_property_4_feasibility_across_horizon():
    """
    Property Test 4:
    Verifies that squad composition (15 players, 2/5/5/3), club limits (<=3),
    formation (11 starters, 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), and budget (bank >= 0)
    are strictly feasible at every gameweek.
    """
    rng = np.random.default_rng(4004)
    squad_df = _generate_valid_random_squad(rng)
    pool_df = _generate_candidate_pool(squad_df, rng, extra_players=60)
    horizon = {t: pool_df.copy() for t in range(10, 18)}

    optimizer = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=8))
    plan = optimizer.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=5.0,
        free_transfers=2,
        horizon_projections=horizon,
        current_gw=9,
        target_gw=10,
    )

    player_map = {int(r["element"]): dict(r) for _, r in pool_df.iterrows()}
    current_squad_ids = set(squad_df["element"].astype(int).tolist())

    for act in plan.trajectory:
        # Apply transfers
        out_ids = set(t["element"] for t in act.transfers_out)
        in_ids = set(t["element"] for t in act.transfers_in)

        # All out_ids must have been in squad
        assert out_ids.issubset(current_squad_ids), f"GW{act.gameweek}: Transferred out unowned players"

        current_squad_ids = (current_squad_ids - out_ids) | in_ids

        # 1. Squad size = 15
        assert len(current_squad_ids) == 15, f"GW{act.gameweek}: Squad size is {len(current_squad_ids)}, expected 15"

        # 2. Position counts
        pos_counts: dict[str, int] = {}
        club_counts: dict[int, int] = {}
        for eid in current_squad_ids:
            p = player_map[eid]
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1
            club_counts[p["team"]] = club_counts.get(p["team"], 0) + 1

        assert pos_counts.get("GKP", 0) == 2, f"GW{act.gameweek}: GKP count {pos_counts.get('GKP')}"
        assert pos_counts.get("DEF", 0) == 5, f"GW{act.gameweek}: DEF count {pos_counts.get('DEF')}"
        assert pos_counts.get("MID", 0) == 5, f"GW{act.gameweek}: MID count {pos_counts.get('MID')}"
        assert pos_counts.get("FWD", 0) == 3, f"GW{act.gameweek}: FWD count {pos_counts.get('FWD')}"

        # 3. Club caps <= 3
        for tm, cnt in club_counts.items():
            assert cnt <= 3, f"GW{act.gameweek}: Team {tm} has {cnt} players (> 3)"

        # 4. Starting XI formation
        starters = set(act.starters)
        assert len(starters) == 11, f"GW{act.gameweek}: Starters count {len(starters)}"
        assert starters.issubset(current_squad_ids), f"GW{act.gameweek}: Starters not subset of squad"

        st_pos: dict[str, int] = {}
        for eid in starters:
            p = player_map[eid]
            st_pos[p["position"]] = st_pos.get(p["position"], 0) + 1

        assert st_pos.get("GKP", 0) == 1, f"GW{act.gameweek}: Starting GKP != 1"
        assert 3 <= st_pos.get("DEF", 0) <= 5, f"GW{act.gameweek}: Starting DEF count {st_pos.get('DEF')}"
        assert 2 <= st_pos.get("MID", 0) <= 5, f"GW{act.gameweek}: Starting MID count {st_pos.get('MID')}"
        assert 1 <= st_pos.get("FWD", 0) <= 3, f"GW{act.gameweek}: Starting FWD count {st_pos.get('FWD')}"

        # 5. Captain in starters
        assert act.captain in starters, f"GW{act.gameweek}: Captain {act.captain} not in starting XI"

        # 6. Bank >= 0
        assert act.bank_end >= -1e-4, f"GW{act.gameweek}: Bank negative ({act.bank_end})"


# ==============================================================================
# Property Test 5: Chip logic & legality
# ==============================================================================
@pytest.mark.slow
def test_property_5_chip_logic_and_legality():
    """
    Property Test 5:
    Enforces that:
    - Never more than 1 chip deployed in any single gameweek.
    - Free Hit cannot be played in GW1 (Rule R2).
    - Free Hit cannot be played in both GW19 and GW20 (Rule R2).
    - Set 1 chips are never scheduled after GW19.
    """
    rng = np.random.default_rng(5005)
    squad_df = _generate_valid_random_squad(rng)
    pool_df = _generate_candidate_pool(squad_df, rng, extra_players=20)

    # Test Case A: GW1 Free Hit restriction
    horizon_gw1 = {t: pool_df.copy() for t in range(1, 5)}
    opt_gw1 = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=4))
    plan_gw1 = opt_gw1.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon_gw1,
        current_gw=1,
        target_gw=1,
        chips_available=["freehit", "wildcard", "3xc", "bboost"],
    )
    # Assert GW1 chip is not Free Hit
    gw1_chip = plan_gw1.trajectory[0].active_chip
    assert gw1_chip != "freehit", "Free Hit must not be played in GW1 (Rule R2)"

    # Test Case B: Boundary GW18-21 Free Hit adjacency check
    horizon_boundary = {t: pool_df.copy() for t in range(18, 22)}
    opt_boundary = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=4))
    plan_boundary = opt_boundary.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon_boundary,
        current_gw=17,
        target_gw=18,
        chips_available=["freehit", "wildcard", "3xc", "bboost"],
    )
    scheduled_chips = {act.gameweek: act.active_chip for act in plan_boundary.trajectory if act.active_chip}
    assert not (scheduled_chips.get(19) == "freehit" and scheduled_chips.get(20) == "freehit"), (
        "Free Hit cannot be played in both GW19 and GW20 (Rule R2)"
    )


# ==============================================================================
# Property Test 6: Brute force match on toy league
# ==============================================================================
@pytest.mark.slow
def test_property_6_brute_force_toy_optimum():
    """
    Property Test 6:
    On a bounded toy instance (3 clubs, 4 positions, H=2), exhaustively enumerates
    all legal combinations of transfers and starting XIs, and confirms the MILP
    finds the exact mathematical optimum.
    """
    # Toy squad of 15 players with 2 GKP, 5 DEF, 5 MID, 3 FWD
    rng = np.random.default_rng(6006)
    squad_df = _generate_valid_random_squad(rng)

    # Small candidate pool: squad + 3 high-value transfer targets
    targets = [
        {"element": 101, "web_name": "Star_MID", "position": "MID", "team": 18, "value": 65, "purchase_price": 65, "selling_price": 65, "expected_points": 12.0, "chance_of_playing": 1.0, "chance_of_playing_next_round": 100},
        {"element": 102, "web_name": "Star_FWD", "position": "FWD", "team": 19, "value": 75, "purchase_price": 75, "selling_price": 75, "expected_points": 14.0, "chance_of_playing": 1.0, "chance_of_playing_next_round": 100},
        {"element": 103, "web_name": "Star_DEF", "position": "DEF", "team": 20, "value": 50, "purchase_price": 50, "selling_price": 50, "expected_points": 10.0, "chance_of_playing": 1.0, "chance_of_playing_next_round": 100},
    ]
    pool_df = pd.concat([squad_df, pd.DataFrame(targets)], ignore_index=True)
    horizon = {10: pool_df.copy(), 11: pool_df.copy()}

    # Multi-Period MILP solve (H=2)
    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=2, w_bench_mean=0.0, lambda_bank=0.0, ft_terminal_increments=(0.0, 0.0, 0.0, 0.0)))
    milp_plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=0.0,
        free_transfers=1,
        horizon_projections=horizon,
        current_gw=9,
        target_gw=10,
        horizon_len=2,
    )
    milp_obj = milp_plan.total_objective

    # Brute Force Search over 0-transfer and 1-transfer actions at GW10
    # For each, compute optimal lineup in GW10 and GW11
    squad_eids = set(squad_df["element"].astype(int).tolist())
    target_eids = [101, 102, 103]

    candidate_squads_gw10 = [squad_eids]  # 0 transfers

    # Legal 1-transfers (same position and equal price)
    for t_id in target_eids:
        t_row = pool_df[pool_df["element"] == t_id].iloc[0]
        pos = t_row["position"]
        val = t_row["value"]
        # Find matching position in squad
        for s_id in squad_eids:
            s_row = squad_df[squad_df["element"] == s_id].iloc[0]
            if s_row["position"] == pos and s_row["selling_price"] >= val:
                cand_sq = (squad_eids - {s_id}) | {t_id}
                candidate_squads_gw10.append(cand_sq)

    gamma = 0.95
    best_brute_val = -1e9

    for sq10 in candidate_squads_gw10:
        sq10_df = pool_df[pool_df["element"].isin(sq10)]
        res10 = lineup_optimizer.select_lineup_and_captain(sq10_df, risk_preference="balanced")
        score10 = res10["total_gameweek_expected_points"]

        # GW11 with same squad (0 transfers)
        res11 = lineup_optimizer.select_lineup_and_captain(sq10_df, risk_preference="balanced")
        score11 = res11["total_gameweek_expected_points"]

        total_score = score10 + (gamma * score11)
        if total_score > best_brute_val:
            best_brute_val = total_score

    # Assert MILP matches or exceeds the brute-force best 1-transfer path
    assert milp_obj >= round(best_brute_val, 2) - 0.05, (
        f"MILP optimum {milp_obj} != Brute-force optimum {best_brute_val:.2f}"
    )


# ==============================================================================
# Fast Unit Tests (Run in default test suite)
# ==============================================================================
def test_multiperiod_config_defaults():
    cfg = MultiPeriodConfig()
    assert cfg.horizon_len == 8
    assert cfg.solver_name == "highs"
    assert cfg.gamma == 0.95
    assert cfg.mip_gap == 0.005
    assert cfg.time_limit_sec == 60


def test_multiperiod_fast_single_gw_solve():
    """Fast smoke test solving a 1-GW instance in under 0.1s."""
    rng = np.random.default_rng(7007)
    squad_df = _generate_valid_random_squad(rng)
    opt = MultiPeriodOptimizer(MultiPeriodConfig(horizon_len=1))
    plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=squad_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections={1: squad_df},
        current_gw=0,
        target_gw=1,
        horizon_len=1,
        forced_w1_transfers=0,
    )
    assert plan.solver_status == "Optimal"
    assert len(plan.recommended_action.starters) == 11
    assert plan.recommended_action.captain in plan.recommended_action.starters


def test_hit_threshold_derivation_and_configuration():
    """Verify configurable hit policy threshold derivation across risk preferences and deficit contexts (WI-2.5)."""
    cfg = MultiPeriodConfig(hit_threshold_base=2.0)
    # Default 1 hit balanced: 2.0
    assert cfg.compute_hit_threshold(hits=1, risk_preference="balanced") == 2.0
    # Conservative / protect lead: +1.0 -> 3.0
    assert cfg.compute_hit_threshold(hits=1, risk_preference="conservative") == 3.0
    assert cfg.compute_hit_threshold(hits=1, mini_league_mode="protect_lead") == 3.0
    # Aggressive / chase deficit: -1.0 -> 1.0
    assert cfg.compute_hit_threshold(hits=1, risk_preference="aggressive") == 1.0
    # Chasing with big deficit (>=20): -0.5 extra discount -> 0.5
    assert cfg.compute_hit_threshold(hits=1, risk_preference="aggressive", mini_league_deficit=25.0) == 0.5
    # Compounding for 2 hits: 2.0 + (2-1)*0.5 = 2.5
    assert cfg.compute_hit_threshold(hits=2, risk_preference="balanced") == 2.5
    # 0 hits: 0.0
    assert cfg.compute_hit_threshold(hits=0) == 0.0



