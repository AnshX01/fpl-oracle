from datetime import UTC, datetime

from fpl_oracle.api.fpl_client import FPLClient
from fpl_oracle.api.models import BootstrapStatic, Fixture, Team
from fpl_oracle.chips.calendar import fixture_calendar


def test_missing_source_timestamp_is_unknown_not_last_sync_or_now():
    client = FPLClient()
    client.last_sync_time = datetime.now(UTC)
    assert client.get_data_as_of("news") is None
    stamp = datetime(2026, 10, 8, tzinfo=UTC)
    client._cache_timestamps["news"] = stamp
    assert client.get_data_as_of("news") == stamp.isoformat()
    assert client.get_data_as_of("league") is None


def test_mixed_blank_double_and_rescheduled_calendar():
    boot = BootstrapStatic(teams=[Team(id=i, name=f"Team{i}", short_name=f"T{i}") for i in range(1, 5)])
    fixtures = [
        Fixture(id=1, code=1, event=7, team_h=1, team_a=2),
        Fixture(id=2, code=2, event=7, team_h=1, team_a=2),
        Fixture(id=3, code=3, event=None, team_h=3, team_a=4),
    ]
    first = fixture_calendar.analyze_calendar(fixtures, boot)
    blank = next(r for r in first["blank_gameweeks"] if r["gameweek"] == 7)
    double = next(r for r in first["double_gameweeks"] if r["gameweek"] == 7)
    assert {t["team_id"] for t in blank["blanking_teams"]} == {3, 4}
    assert {t["team_id"] for t in double["teams_with_doubles"]} == {1, 2}
    fixtures[2].event = 7
    updated = fixture_calendar.analyze_calendar(fixtures, boot)
    assert not any(r["gameweek"] == 7 for r in updated["blank_gameweeks"])
    assert any(r["gameweek"] == 7 for r in updated["double_gameweeks"])
