"""
Unit tests for Mathematical Optimizer (PuLP MILP), Formation rules, and Selling Price math.
"""

import pytest
import pandas as pd
import numpy as np

from fpl_oracle.optimise.squad import squad_optimizer
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer

def make_mock_pool():
    rows = []
    # 4 GKP
    for i in range(1, 5):
        rows.append({"element": i, "web_name": f"GKP_{i}", "position": "GKP", "team": (i % 20) + 1, "value": 50, "expected_points": 4.0 - (i * 0.2)})
    # 10 DEF
    for i in range(5, 15):
        rows.append({"element": i, "web_name": f"DEF_{i}", "position": "DEF", "team": (i % 20) + 1, "value": 50, "expected_points": 5.0 - (i * 0.1)})
    # 10 MID
    for i in range(15, 25):
        rows.append({"element": i, "web_name": f"MID_{i}", "position": "MID", "team": (i % 20) + 1, "value": 65, "expected_points": 6.0 - (i * 0.1)})
    # 6 FWD
    for i in range(25, 31):
        rows.append({"element": i, "web_name": f"FWD_{i}", "position": "FWD", "team": (i % 20) + 1, "value": 75, "expected_points": 7.0 - (i * 0.1)})

    return pd.DataFrame(rows)

def test_squad_optimizer_constraints():
    pool = make_mock_pool()
    res = squad_optimizer.solve_best_squad(pool, budget=1000.0)
    squad = res["squad"]

    assert len(squad) == 15
    assert len(squad[squad["position"] == "GKP"]) == 2
    assert len(squad[squad["position"] == "DEF"]) == 5
    assert len(squad[squad["position"] == "MID"]) == 5
    assert len(squad[squad["position"] == "FWD"]) == 3
    assert squad["value"].sum() <= 1000.0

    # Max 3 per club
    team_counts = squad["team"].value_counts()
    assert (team_counts <= 3).all()

def test_selling_price_math():
    """Verify selling price formula: purchase_price + floor((now_cost - purchase_price) / 2)."""
    # Case 1: Player bought at 10.0m (100), now 10.4m (104). Rise = 4. Selling price = 100 + 2 = 102 (10.2m)
    p1 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=104)
    assert p1 == 102

    # Case 2: Player bought at 10.0m (100), now 10.3m (103). Rise = 3. 3 // 2 = 1. Selling price = 101 (10.1m)
    p2 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=103)
    assert p2 == 101

    # Case 3: Player bought at 10.0m (100), now 9.8m (98). Loss = 2. Sells at current price 98
    p3 = transfer_optimizer.calculate_selling_price(purchase_price=100, now_cost=98)
    assert p3 == 98

def test_lineup_and_captain_formation():
    pool = make_mock_pool()
    squad = squad_optimizer.solve_best_squad(pool, budget=1000.0)["squad"]
    lineup = lineup_optimizer.select_lineup_and_captain(squad)

    starters = lineup["starters"]
    bench = lineup["bench"]

    assert len(starters) == 11
    assert len(bench) == 4
    assert len(starters[starters["position"] == "GKP"]) == 1
    assert 3 <= len(starters[starters["position"] == "DEF"]) <= 5
    assert 2 <= len(starters[starters["position"] == "MID"]) <= 5
    assert 1 <= len(starters[starters["position"] == "FWD"]) <= 3

    # Captain and Vice Captain
    assert lineup["captain"]["element"] != lineup["vice_captain"]["element"]
    assert lineup["captain"]["multiplier"] == 2

def test_compute_squad_selling_prices_and_free_transfers():
    """Verify transfer history parsing for selling prices and banked free transfers calculation."""
    # Test squad with 2 players
    squad_df = pd.DataFrame([
        {"element": 10, "web_name": "Salah", "value": 128, "cost_change_start": 3}, # rose from 12.5 to 12.8
        {"element": 20, "web_name": "Haaland", "value": 152, "cost_change_start": 2} # bought at 15.0 via transfer, now 15.2
    ])

    transfers = [
        {"element_in": 20, "element_in_cost": 150, "element_out": 99, "event": 2}
    ]

    priced_df = transfer_optimizer.compute_squad_selling_prices(squad_df, transfer_history=transfers)
    salah_row = priced_df[priced_df["element"] == 10].iloc[0]
    haaland_row = priced_df[priced_df["element"] == 20].iloc[0]

    # Salah: purchase 125, now 128 -> profit 3 -> selling price = 125 + (3//2) = 126
    assert salah_row["purchase_price"] == 125
    assert salah_row["selling_price"] == 126

    # Haaland: bought in GW2 at 150, now 152 -> profit 2 -> selling price = 150 + 1 = 151
    assert haaland_row["purchase_price"] == 150
    assert haaland_row["selling_price"] == 151

    # Banked free transfers: 2026/27 rules
    # Start: 1 in GW1. Rolled GW1 (transfers_made=0) -> 2 in GW2.
    # Rolled GW2 -> 3 in GW3.
    # Made 1 in GW3 -> remaining 2 -> +1 = 3 in GW4.
    # Rolled GW4 -> 4 in GW5.
    history = [
        {"event": 1, "event_transfers": 0},
        {"event": 2, "event_transfers": 0},
        {"event": 3, "event_transfers": 1},
        {"event": 4, "event_transfers": 0},
        {"event": 5, "event_transfers": 0}
    ]
    ft = transfer_optimizer.compute_available_free_transfers(entry_history=history, current_gw=5)
    # At end of GW5: GW4 was 4, rolled GW5 -> 5 banked
    assert ft == 5

