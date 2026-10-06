"""
Unit tests validating that offline recorded fixtures in tests/fixtures/
can be parsed and utilized cleanly without network dependency.
"""

import json
from pathlib import Path

from fpl_oracle.api.models import BootstrapStatic, Fixture, ManagerHistory, SquadPicks

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_offline_bootstrap_static_fixture():
    """Verify offline bootstrap_static.json can be loaded and validated into BootstrapStatic."""
    boot_file = FIXTURES_DIR / "bootstrap_static.json"
    assert boot_file.exists(), "Offline fixture bootstrap_static.json must exist"

    with open(boot_file, encoding="utf-8") as f:
        data = json.load(f)

    boot = BootstrapStatic.model_validate(data)
    assert len(boot.elements) > 500
    assert len(boot.teams) == 20
    assert len(boot.events) == 38


def test_offline_fixtures_list_fixture():
    """Verify offline fixtures.json can be loaded into Fixture list."""
    fix_file = FIXTURES_DIR / "fixtures.json"
    assert fix_file.exists(), "Offline fixture fixtures.json must exist"

    with open(fix_file, encoding="utf-8") as f:
        data = json.load(f)

    fixtures = [Fixture.model_validate(item) for item in data]
    assert len(fixtures) >= 380


def test_offline_manager_history_and_picks_fixtures():
    """Verify offline manager picks and history can be parsed cleanly."""
    picks_file = FIXTURES_DIR / "manager_picks.json"
    hist_file = FIXTURES_DIR / "manager_history.json"
    assert picks_file.exists()
    assert hist_file.exists()

    with open(picks_file, encoding="utf-8") as f:
        picks_data = json.load(f)
    picks = SquadPicks.model_validate(picks_data)
    assert len(picks.picks) == 15

    with open(hist_file, encoding="utf-8") as f:
        hist_data = json.load(f)
    hist = ManagerHistory.model_validate(hist_data)
    assert len(hist.current) == 5
    assert len(hist.chips) == 1
