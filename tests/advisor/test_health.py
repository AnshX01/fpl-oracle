"""
Tests for Squad Health Diagnosis (WI-8.1).
Validates KEEP/WATCH/REPLACE logic, functional starters, and structural deficiency score.
Ensures zero hard-coded player names in production code.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_oracle.advisor.health import SquadHealthDiagnosis


@pytest.fixture
def s1_fixture_data():
    fixture_path = Path(__file__).parents[1] / "fixtures" / "s1_gw6_state.json"
    with open(fixture_path) as f:
        return json.load(f)


def test_s1_fixture_classified_as_healthy_core_targeted_fixes(s1_fixture_data):
    """Scenario S1: Owner's real GW6 state must be diagnosed as healthy core with targeted fixes."""
    squad_rows = s1_fixture_data["squad"]
    squad_df = pd.DataFrame(squad_rows)

    diag = SquadHealthDiagnosis()
    report = diag.diagnose_squad(squad_df)

    # 1. Must classify as healthy core, NOT requiring emergency structural overhaul
    assert report.classification == "HEALTHY_CORE_TARGETED_FIXES"

    # 2. Functional starters count must be >= 9 (healthy playable XI exists)
    assert report.functional_starters_count >= 9

    # 3. Targeted fixes only: healthy core is majority of squad
    assert len(report.healthy_core_ids) >= 8
    assert len(report.replace_ids) <= 3

    # 4. Check machine-readable reasons on diagnosed elements
    for _eid, p_stat in report.player_diagnoses.items():
        assert p_stat.status in ["KEEP", "WATCH", "REPLACE"]
        if p_stat.status in ["WATCH", "REPLACE"]:
            assert len(p_stat.reasons) > 0


def test_broken_squad_triggers_structural_overhaul():
    """Severely damaged squad (<9 functional starters, high deficiency) must trigger overhaul."""
    # 15 players where 8 have 0 xP or injuries
    broken_rows = []
    for i in range(1, 16):
        pos = "GKP" if i <= 2 else ("DEF" if i <= 7 else ("MID" if i <= 12 else "FWD"))
        xp = 0.0 if i in [1, 3, 4, 5, 8, 9, 13] else 4.5
        chance = 0.0 if i in [1, 3, 4, 5, 8, 9, 13] else 100.0
        broken_rows.append(
            {
                "element": i,
                "position": pos,
                "expected_points": xp,
                "chance_of_playing": chance,
                "news": "Injured" if chance == 0.0 else "",
                "value": 50.0,
                "purchase_price": 50.0,
            }
        )

    broken_df = pd.DataFrame(broken_rows)
    diag = SquadHealthDiagnosis()
    report = diag.diagnose_squad(broken_df)

    assert report.classification == "STRUCTURAL_OVERHAUL_REQUIRED"
    assert report.functional_starters_count < 9
    assert len(report.replace_ids) >= 7


def test_no_hardcoded_player_names_in_health_module():
    """Principle: zero hardcoded player names allowed in production code."""
    health_py = Path(__file__).parents[2] / "src" / "fpl_oracle" / "advisor" / "health.py"
    content = health_py.read_text(encoding="utf-8").lower()
    for forbidden in ["haaland", "saka", "palmer", "mbeumo", "tarkowski", "saliba", "foden"]:
        assert forbidden not in content, f"Found hard-coded player name '{forbidden}' in health.py"


def test_availability_under_100_triggers_watch_and_reason():
    """Verify that player with 75% availability triggers AVAILABILITY_DOUBT and WATCH status."""
    rows = []
    for i in range(1, 16):
        pos = "GKP" if i <= 2 else ("DEF" if i <= 7 else ("MID" if i <= 12 else "FWD"))
        chance = 75.0 if i == 10 else 100.0
        rows.append(
            {
                "element": i,
                "position": pos,
                "expected_points": 5.0,
                "chance_of_playing": chance,
                "news": "Minor knock" if chance == 75.0 else "",
                "value": 50.0,
                "purchase_price": 50.0,
            }
        )
    df = pd.DataFrame(rows)
    diag = SquadHealthDiagnosis()
    report = diag.diagnose_squad(df)

    p10 = report.player_diagnoses[10]
    assert p10.status == "WATCH"
    assert any("AVAILABILITY_DOUBT" in r for r in p10.reasons)


def test_missing_chance_uses_status_and_xi_is_availability_adjusted():
    rows = [
        {"element": 1, "position": "GKP", "expected_points": 4.0, "chance_of_playing": None, "status": "a"},
        {"element": 2, "position": "DEF", "expected_points": 6.0, "chance_of_playing": None, "status": "i"},
        {"element": 3, "position": "DEF", "expected_points": 5.0, "chance_of_playing": None, "status": "d"},
    ]
    report = SquadHealthDiagnosis().diagnose_squad(pd.DataFrame(rows))
    assert report.player_diagnoses[2].status == "REPLACE"  # injured, no chance given -> 0%
    assert report.player_diagnoses[2].functional_starter is False
    assert report.player_diagnoses[3].chance_of_playing == 50.0
    assert report.player_diagnoses[3].status == "WATCH"
    assert report.player_diagnoses[1].functional_starter is True
