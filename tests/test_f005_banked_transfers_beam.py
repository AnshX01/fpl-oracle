"""
Regression test for F-005: Beam Search Incapable of >2 Transfers (Cannot Exploit Banked 3-5 FTs).
Under official 2024-2027 rules, managers can bank up to 5 free transfers.
When 3 FTs are banked and a 3-player transfer yields higher expected gain with 0 hits
than any 1- or 2-player move, the sequential beam search cannot discover it because candidate
generation is strictly limited to 1 and 2 transfers.

Deferred to spec WI-2.1 (Multi-period MILP optimizer).
"""

import pandas as pd
import pytest

from fpl_oracle.optimise.transfers import transfer_optimizer


@pytest.mark.xfail(strict=True, reason="Deferred to spec WI-2.1 (Multi-period MILP optimizer)")
def test_beam_finds_optimal_3_transfer_hit_free_plan():
    # Base pool
    rows = []
    # 2 GKP
    for i in [1, 2]:
        rows.append({"element": i, "web_name": f"GKP_{i}", "position": "GKP", "team": 1, "value": 50, "purchase_price": 50, "selling_price": 50, "expected_points": 4.0, "chance_of_playing_next_round": 100})
    # 5 DEF
    for i in range(3, 8):
        rows.append({"element": i, "web_name": f"DEF_{i}", "position": "DEF", "team": 2, "value": 50, "purchase_price": 50, "selling_price": 50, "expected_points": 4.0, "chance_of_playing_next_round": 100})
    # 5 MID in squad: elements 8, 9, 10 are weak (xP = 1.0), elements 11, 12 are normal (xP = 5.0)
    for i in [8, 9, 10]:
        rows.append({"element": i, "web_name": f"MID_{i}", "position": "MID", "team": 3, "value": 60, "purchase_price": 60, "selling_price": 60, "expected_points": 1.0, "chance_of_playing_next_round": 100})
    for i in [11, 12]:
        rows.append({"element": i, "web_name": f"MID_{i}", "position": "MID", "team": 4, "value": 60, "purchase_price": 60, "selling_price": 60, "expected_points": 5.0, "chance_of_playing_next_round": 100})
    # 3 FWD
    for i in range(13, 16):
        rows.append({"element": i, "web_name": f"FWD_{i}", "position": "FWD", "team": 5, "value": 70, "purchase_price": 70, "selling_price": 70, "expected_points": 5.0, "chance_of_playing_next_round": 100})

    squad = pd.DataFrame(rows).copy()

    # Targets in pool: 3 strong MIDs (elements 20, 21, 22) each with 8.0 xP and cost 60
    targets = [
        {"element": 20, "web_name": "MID_STAR_1", "position": "MID", "team": 6, "value": 60, "purchase_price": 60, "selling_price": 60, "expected_points": 8.0, "chance_of_playing_next_round": 100},
        {"element": 21, "web_name": "MID_STAR_2", "position": "MID", "team": 7, "value": 60, "purchase_price": 60, "selling_price": 60, "expected_points": 8.0, "chance_of_playing_next_round": 100},
        {"element": 22, "web_name": "MID_STAR_3", "position": "MID", "team": 8, "value": 60, "purchase_price": 60, "selling_price": 60, "expected_points": 8.0, "chance_of_playing_next_round": 100},
    ]
    pool = pd.concat([squad, pd.DataFrame(targets)], ignore_index=True)

    horizon = {10: pool.copy(), 11: pool.copy(), 12: pool.copy()}

    # Manager has 3 free transfers banked
    result = transfer_optimizer.evaluate_transfer_options(
        current_squad_df=squad,
        player_pool_df=pool,
        bank=0.0,
        free_transfers=3,
        horizon_projections=horizon,
        current_gw=9,
        target_gw=10,
    )
    rec_plan = result.get("recommended_plan", {})

    # With 3 banked FTs, the 3-transfer move (8->20, 9->21, 10->22) yields +21.0 xP with 0 hit cost.
    # A 2-transfer move only yields at most +14.0 xP.
    # Therefore, the optimal hit-free move is 3 transfers.
    assert rec_plan.get("transfers_count") == 3, f"Expected 3 transfers, got {rec_plan.get('transfers_count')}"
    assert rec_plan.get("hit_cost") == 0, f"Expected 0 hit cost with 3 FTs, got {rec_plan.get('hit_cost')}"
