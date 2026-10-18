"""
Tests for the pre-deadline odds snapshot (WI-3.2): refuses without a real source, persists metadata otherwise.
"""

import json
from pathlib import Path

import pytest

from fpl_oracle.data.odds import FootballDataOddsProvider, OddsProvider, snapshot_odds


class _StubOddsProvider(OddsProvider):
    """Provider returning one explicit, test-owned match (the real provider has no upcoming-fixture source)."""

    def get_match_odds(self, season, gameweek=None):
        odds = [2.10, 3.40, 3.50]
        return [
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "bookmaker": "stub",
                "odds_1x2": dict(zip(("home_win", "draw", "away_win"), odds, strict=True)),
            }
        ]


def test_odds_snapshot_refuses_when_no_real_source(tmp_path: Path):
    """The default provider has no live odds: snapshotting must fail instead of persisting invented odds."""
    assert FootballDataOddsProvider().get_match_odds("2026-27", 8) == []
    with pytest.raises(RuntimeError, match="No odds source"):
        snapshot_odds(target_gw=8, custom_output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_odds_snapshot_persistence(tmp_path: Path):
    """WI-3.2 Step 5: Verify scheduled odds_snapshot stores pre-deadline snapshots with metadata."""
    snap_path = snapshot_odds(target_gw=6, custom_output_dir=tmp_path, provider=_StubOddsProvider())
    assert snap_path.exists()
    assert snap_path.suffix == ".json"

    with open(snap_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["gameweek"] == 6
    assert "snapshot_at_utc" in data
    assert data["matches_count"] > 0
    assert len(data["matches"]) == data["matches_count"]

    first_match = data["matches"][0]
    assert "home_team" in first_match
    assert "away_team" in first_match
    assert "odds_1x2" in first_match
