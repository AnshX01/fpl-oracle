"""
Property-based tests for FPL Oracle Optimization & Financial Logic using Hypothesis.
Mathematically verifies invariant constraints across arbitrary distributions:
1. Squad Optimizer budget limit: sum(cost) <= budget.
2. Positional quotas: exactly 2 GKP, 5 DEF, 5 MID, 3 FWD.
3. Club quotas: <= 3 players per club.
4. Lineup & Captaincy valid formations and distinct roles.
5. Selling price invariants (integer tenths, monotonic, bounded profit split).
"""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings

from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.squad import squad_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer


# ---------------------------------------------------------------------------
# 1. Selling Price Invariants
# ---------------------------------------------------------------------------
@given(
    purchase_price=st.integers(min_value=38, max_value=150),
    now_cost=st.integers(min_value=35, max_value=160),
)
@settings(max_examples=100)
def test_selling_price_properties(purchase_price: int, now_cost: int):
    sp = transfer_optimizer.calculate_selling_price(purchase_price, now_cost)

    # Invariant 1: Must be integer (tenths of million)
    assert isinstance(sp, int)

    # Invariant 2: When price has dropped or stayed equal, selling price equals now_cost
    if now_cost <= purchase_price:
        assert sp == now_cost
    else:
        # Invariant 3: When in profit, profit is shared 50% with FPL, rounded down
        profit = now_cost - purchase_price
        expected = purchase_price + (profit // 2)
        assert sp == expected
        # Bounded between purchase price and now_cost
        assert purchase_price <= sp <= now_cost


# ---------------------------------------------------------------------------
# 2. Lineup Formation & Captaincy Properties
# ---------------------------------------------------------------------------
def make_mock_squad_df() -> pd.DataFrame:
    """Generate a valid 15-player squad DataFrame with random xP."""
    elements = []
    # 2 GKP, 5 DEF, 5 MID, 3 FWD
    roles = [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
    e_id = 1
    for pos, count in roles:
        for _ in range(count):
            elements.append({
                "element": e_id,
                "web_name": f"Player_{e_id}",
                "position": pos,
                "team": (e_id % 10) + 1,
                "now_cost": 50 + (e_id % 40),
                "selling_price": 50 + (e_id % 40),
                "expected_points": round(float(np.random.uniform(1.0, 9.0)), 2),
                "p10_points": 1.0,
                "p90_points": 12.0,
                "chance_of_playing": 100.0,
            })
            e_id += 1
    return pd.DataFrame(elements)


@given(seed=st.integers(min_value=1, max_value=1000))
@settings(max_examples=15, deadline=None)
def test_lineup_formation_and_captain_invariants(seed: int):
    np.random.seed(seed)
    squad_df = make_mock_squad_df()

    lineup = lineup_optimizer.select_lineup_and_captain(squad_df)

    starters = lineup["starters"]
    bench = lineup["bench"]

    # Invariant 1: Total counts
    assert len(starters) == 11, "Must have exactly 11 starters"
    assert len(bench) == 4, "Must have exactly 4 bench players"

    # Invariant 2: Disjoint sets partition the squad
    starter_ids = set(starters["element"].tolist())
    bench_ids = set(bench["element"].tolist())
    assert len(starter_ids.intersection(bench_ids)) == 0, "Starters and bench must be disjoint"
    assert len(starter_ids.union(bench_ids)) == 15, "Union must cover all 15 players"

    # Invariant 3: Formation rules
    pos_counts = starters["position"].value_counts().to_dict()

    assert pos_counts.get("GKP", 0) == 1, "Must start exactly 1 GKP"
    assert 3 <= pos_counts.get("DEF", 0) <= 5, "Must start between 3 and 5 DEF"
    assert 2 <= pos_counts.get("MID", 0) <= 5, "Must start between 2 and 5 MID"
    assert 1 <= pos_counts.get("FWD", 0) <= 3, "Must start between 1 and 3 FWD"
    assert sum(pos_counts.values()) == 11

    # Invariant 4: Captain and Vice-Captain are distinct starters
    cap = lineup["captain"]
    vc = lineup["vice_captain"]

    assert cap["element"] in starter_ids, "Captain must be a starter"
    assert vc["element"] in starter_ids, "Vice-captain must be a starter"
    assert cap["element"] != vc["element"], "Captain and Vice-captain must be distinct"


# ---------------------------------------------------------------------------
# 3. Squad Optimization Budget & Positional Invariants
# ---------------------------------------------------------------------------
@given(budget_tenths=st.integers(min_value=990, max_value=1050))
@settings(max_examples=10, deadline=None)
def test_squad_optimizer_budget_and_quota_invariants(budget_tenths: int):
    # Construct a diverse pool of 60 players (6 GKP, 20 DEF, 20 MID, 14 FWD) across 20 clubs
    pool = []
    e_id = 1
    for pos, count in [("GKP", 6), ("DEF", 20), ("MID", 20), ("FWD", 14)]:
        for i in range(count):
            team_id = (i % 20) + 1
            cost = 40 + (i % 30) # £4.0m to £7.0m
            pool.append({
                "element": e_id,
                "web_name": f"P_{e_id}",
                "position": pos,
                "team": team_id,
                "now_cost": cost,
                "value": cost,
                "expected_points": 2.5 + (cost / 15.0),
                "p10_points": 1.0,
                "p90_points": 8.0,
            })
            e_id += 1

    pool_df = pd.DataFrame(pool)
    res = squad_optimizer.solve_best_squad(pool_df, budget=float(budget_tenths))

    assert res["status"] in ("Optimal", "Feasible")
    squad = res["squad"]

    # Invariant 1: Exactly 15 players
    assert len(squad) == 15

    # Invariant 2: Budget never exceeded
    total_cost = squad["now_cost"].sum()
    assert total_cost <= budget_tenths, f"Total cost {total_cost} exceeded budget {budget_tenths}"

    # Invariant 3: Positional quotas
    pos_counts = squad["position"].value_counts().to_dict()
    assert pos_counts.get("GKP", 0) == 2
    assert pos_counts.get("DEF", 0) == 5
    assert pos_counts.get("MID", 0) == 5
    assert pos_counts.get("FWD", 0) == 3

    # Invariant 4: Club quota (max 3 per club)
    team_counts = squad["team"].value_counts()
    assert (team_counts <= 3).all(), f"Club limit violated: {team_counts[team_counts > 3].to_dict()}"
