"""
Unit tests for EliteOwnershipEngine.
Verifies Top-10k Effective Ownership (EO), ghost distortion calculus,
zombie trap identification, and differential leverage index.
"""

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Team
from fpl_oracle.league.elite_ownership import elite_ownership_engine


def make_mock_bootstrap():
    elements = [
        Element(
            id=1,
            web_name="PlayerA",
            element_type=3,
            team=1,
            now_cost=100,
            selected_by_percent="75.0",
            ep_next="6.5",
        ),
        Element(
            id=2,
            web_name="PlayerB",
            element_type=2,
            team=1,
            now_cost=50,
            selected_by_percent="35.0",
            ep_next="1.5",
        ),
        Element(
            id=3,
            web_name="PlayerC",
            element_type=3,
            team=2,
            now_cost=70,
            selected_by_percent="8.0",
            ep_next="5.2",
        ),
    ]
    element_types = [
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
    ]
    teams = [
        Team(id=1, name="Team1", short_name="T1"),
        Team(id=2, name="Team2", short_name="T2"),
    ]
    return BootstrapStatic(elements=elements, element_types=element_types, teams=teams, events=[])


def test_compute_elite_eo_exact():
    boot = make_mock_bootstrap()
    # 4 sample elite managers
    # Player 1 is started by all 4, captained by 3, 3xC by 1
    # Player 2 is started by 0 managers
    # Player 3 is started by 1 manager
    squads = [
        {"squad": [1], "captain": 1, "is_triple_captain": True},
        {"squad": [1], "captain": 1, "is_triple_captain": False},
        {"squad": [1], "captain": 1, "is_triple_captain": False},
        {"squad": [1, 3], "captain": 2, "is_triple_captain": False},
    ]
    proj_df = pd.DataFrame(
        [
            {"element": 1, "expected_points": 6.5},
            {"element": 2, "expected_points": 1.5},
            {"element": 3, "expected_points": 5.2},
        ]
    )

    results = elite_ownership_engine.compute_elite_eo_from_squads(squads, boot, proj_df)

    p1 = results[1]
    assert p1.elite_ownership_pct == 100.0  # 4/4 started
    assert p1.elite_captain_pct == 75.0  # 3/4 captained
    # EO = 100 + 75 + 2*(25) = 225.0
    assert p1.elite_effective_ownership_pct == 225.0
    assert p1.classification == "ELITE_CORE"

    p2 = results[2]
    # Player 2 has 35% overall ownership, but only 25% captain and 0% starters
    # Ghost distortion is high
    assert p2.overall_ownership_pct == 35.0
    assert p2.elite_ownership_pct == 0.0

    p3 = results[3]
    # Player 3 has 25% elite starter, EO = 25%, xP = 5.2 -> ELITE_DIFFERENTIAL
    assert p3.elite_ownership_pct == 25.0
    assert p3.classification == "ELITE_DIFFERENTIAL"
    assert p3.differential_leverage_index > 0.15


def test_model_calibrated_elite_eo():
    boot = make_mock_bootstrap()
    proj_df = pd.DataFrame(
        [
            {"element": 1, "expected_points": 7.0},
            {"element": 2, "expected_points": 1.2},
            {"element": 3, "expected_points": 5.5},
        ]
    )

    results = elite_ownership_engine.model_calibrated_elite_eo(boot, proj_df)
    assert 1 in results
    assert 2 in results
    assert 3 in results

    # High xP player should have higher elite EO than low xP player
    assert results[1].elite_effective_ownership_pct > results[2].elite_effective_ownership_pct
    assert results[2].classification in ("ZOMBIE_TRAP", "CASUAL_POPULAR")
