"""
Tests for Chip Timeline Comparator (WI-8.4).
Validates evaluation of Set 1 chips over every legal GW until GW19.
Asserts that Wildcard comparison contains all three families (now, later, staged FTs)
and fails if the comparison set degenerates to {now, never}.
Ensures zero hardcoded player names in production code.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.advisor.chip_timeline import ChipTimelineComparator


@pytest.fixture
def s1_fixture():
    fixture_path = Path(__file__).parents[1] / "fixtures" / "s1_gw6_state.json"
    with open(fixture_path) as f:
        return json.load(f)


def test_wildcard_evaluates_three_families_and_does_not_degenerate(s1_fixture):
    """
    WI-8.4 Acceptance:
    Wildcard analysis MUST evaluate three distinct families:
    1. Wildcard NOW
    2. Wildcard LATER (at multiple windows)
    3. Staged Free-Transfer Plan (no Wildcard, using banked FTs)
    A regression test MUST fail if the comparison set degenerates to {now, never}.
    """
    squad_df = pd.DataFrame(s1_fixture["squad"])
    pool_rows = []
    # Build minimal pool from squad and projs
    for r in s1_fixture["squad"]:
        pool_rows.append({
            "element": r["element"],
            "position": r["position"],
            "team": r["team"],
            "value": r["now_cost"],
            "expected_points": r["expected_points"],
        })
    pool_df = pd.DataFrame(pool_rows)

    comparator = ChipTimelineComparator()
    res = comparator.evaluate_wildcard_families(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=s1_fixture["bank_tenths"] / 10.0,
        free_transfers=s1_fixture["free_transfers"],
        horizon_projections={6: pool_df, 7: pool_df, 8: pool_df, 9: pool_df, 10: pool_df},
        current_gw=6,
        chips_available=["wildcard"],
        candidate_later_windows=[7, 8, 12, 16],
    )

    # 1. Assert all three families are present in the comparison set
    assert "WILDCARD_NOW" in res.comparison_set_families
    assert "WILDCARD_LATER" in res.comparison_set_families
    assert "STAGED_FREE_TRANSFERS" in res.comparison_set_families
    assert len(res.comparison_set_families) >= 3

    # 2. Regression check: MUST NOT degenerate to {now, never}
    assert res.comparison_set_families != ["now", "never"]
    assert res.comparison_set_families != ["WILDCARD_NOW", "NEVER"]
    assert len(res.later_windows) >= 3, "Expected at least 3 distinct later windows evaluated"
    assert res.staged_ft_plan is not None
    assert "gain_vs_roll" in res.staged_ft_plan

    # 3. Assert winning family and reasoning are populated
    assert res.winning_family in ["WILDCARD_NOW", "WILDCARD_LATER", "STAGED_FREE_TRANSFERS"]
    assert len(res.reasoning) > 0


def test_chip_timeline_covers_every_legal_gw_until_gw19():
    """All legal GWs until GW19 deadline must be evaluated with intervals and option value."""
    comparator = ChipTimelineComparator()
    timeline = comparator.evaluate_chip_timeline(
        chip="3xc",
        current_gw=6,
        chips_available=["3xc"],
        last_legal_gw=19,
    )

    assert len(timeline) == (19 - 6 + 1)  # GW6 to GW19 inclusive = 14 gameweeks
    for entry in timeline:
        assert entry.gameweek >= 6 and entry.gameweek <= 19
        assert entry.expected_gain > 0
        assert 0.0 <= entry.prob_gain_positive <= 1.0
        assert entry.gain_interval[0] < entry.gain_interval[1]
        assert entry.value_of_waiting >= 0.0


def test_no_hardcoded_player_names_in_chip_timeline():
    """Principle: zero hardcoded player names allowed in production code."""
    path = Path(__file__).parents[2] / "src" / "fpl_oracle" / "advisor" / "chip_timeline.py"
    content = path.read_text(encoding="utf-8").lower()
    for forbidden in ["haaland", "saka", "palmer", "mbeumo", "tarkowski", "saliba", "foden"]:
        assert forbidden not in content, f"Found hardcoded player name '{forbidden}' in chip_timeline.py"
