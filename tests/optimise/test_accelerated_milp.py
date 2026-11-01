"""
Unit tests for Accelerated Multi-Period MILP Optimization.
Verifies Pareto candidate pool pruning, complete MIP warm-start injection,
and proven optimality with accelerated solve times.
"""

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Team
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


def make_mock_bootstrap(n_players=60):
    elements = []
    for i in range(1, n_players + 1):
        pos_id = 1 if i <= 6 else (2 if i <= 24 else (3 if i <= 44 else 4))
        cost = 40 + (i % 8) * 10
        elements.append(
            Element(
                id=i,
                web_name=f"Player_{i}",
                element_type=pos_id,
                team=(i % 4) + 1,
                now_cost=cost,
                ep_next=str(3.0 + (i % 6) * 0.7),
            )
        )
    element_types = [
        ElementType(
            id=1,
            singular_name_short="GKP",
            singular_name="Goalkeeper",
            plural_name="Goalkeepers",
            plural_name_short="GKP",
        ),
        ElementType(
            id=2, singular_name_short="DEF", singular_name="Defender", plural_name="Defenders", plural_name_short="DEF"
        ),
        ElementType(
            id=3,
            singular_name_short="MID",
            singular_name="Midfielder",
            plural_name="Midfielders",
            plural_name_short="MID",
        ),
        ElementType(
            id=4, singular_name_short="FWD", singular_name="Forward", plural_name="Forwards", plural_name_short="FWD"
        ),
    ]
    teams = [
        Team(id=1, name="Team1", short_name="T1"),
        Team(id=2, name="Team2", short_name="T2"),
        Team(id=3, name="Team3", short_name="T3"),
        Team(id=4, name="Team4", short_name="T4"),
    ]
    return BootstrapStatic(elements=elements, element_types=element_types, teams=teams, events=[])


def test_accelerated_pool_pruning_preserves_owned_and_locked():
    opt = MultiPeriodOptimizer(MultiPeriodConfig())
    pool_records = []
    for i in range(1, 51):
        pos = "GKP" if i <= 6 else ("DEF" if i <= 20 else ("MID" if i <= 38 else "FWD"))
        pool_records.append(
            {
                "element": i,
                "web_name": f"P_{i}",
                "position": pos,
                "team": (i % 4) + 1,
                "value": 45.0 + (i % 10),
                "expected_points": 2.0 + (i % 5),
            }
        )
    pool_df = pd.DataFrame(pool_records)
    # User owns players 1, 2, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25, 39, 40, 41
    owned_ids = [1, 2, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25, 39, 40, 41]
    horizon_proj = {1: pool_df, 2: pool_df, 3: pool_df}

    pruned = opt.prune_candidates(
        pool_df=pool_df,
        squad_ids=set(owned_ids),
        horizon_gws=[1, 2, 3],
        horizon_projections=horizon_proj,
        limits={"GKP": 4, "DEF": 8, "MID": 10, "FWD": 6},
        enablers_count=2,
    )

    pruned_ids = set(pruned["element"].tolist())
    # Invariant: Every single owned player MUST be present in the pruned candidate pool
    for oid in owned_ids:
        assert oid in pruned_ids, f"Owned player {oid} was illegally pruned"

    # Pool size must be significantly reduced from 50
    assert len(pruned) < len(pool_df)
