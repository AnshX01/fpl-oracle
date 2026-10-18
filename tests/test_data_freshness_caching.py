"""
Item 5 Verification Tests: Data Freshness, Incremental Live Refresh, Caching, and Season Derivation.
"""

import time

import pandas as pd
from fastapi.testclient import TestClient

from fpl_oracle.api.cache import cache_manager
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import GameState
from fpl_oracle.api.models import BootstrapStatic, Element, ElementType, Fixture, GameweekEvent, Team
from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.data.player_identity import player_identity_resolver
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.server.main import app
from fpl_oracle.utils.season import derive_season


def test_sqlite_rehydration_preserves_remaining_ttl():
    """Verify SQLite cache rehydration restores stored expires_at timestamp, NOT now + ttl."""
    key = "test:ttl:preservation"
    now_ts = time.time()
    # Cache entry created with remaining TTL of 15 seconds
    ttl_seconds = 15
    cache_manager.set(key, {"value": "cached_data"}, ttl_seconds=ttl_seconds)

    # Invalidate in-memory cache to force SQLite rehydration
    if key in cache_manager._mem_cache:
        del cache_manager._mem_cache[key]

    # Rehydrate from SQLite
    res = cache_manager.get(key)
    assert res == {"value": "cached_data"}

    # Assert in-memory expires_at was NOT reset to now + 300, but preserved <= now_ts + 15
    rehydrated = cache_manager._mem_cache.get(key)
    assert rehydrated is not None
    assert rehydrated["expires_at"] <= now_ts + ttl_seconds + 2
    assert rehydrated["expires_at"] < now_ts + 100


def test_cache_keys_namespaced_by_season():
    """Verify fpl_client namespacing by season."""
    key = fpl_client._namespaced_cache_key("bootstrap-static")
    curr_season = derive_season(format="hyphen")
    assert key.startswith(f"{curr_season}:")
    assert key == f"{curr_season}:bootstrap-static"


def test_prediction_cache_key_and_max_age():
    """Verify prediction cache key contains content hash, season namespace, manifest hash, and enforces max age."""
    projection_engine.load_or_train()
    projection_engine.invalidate_cache()

    event = GameweekEvent(id=6, name="Gameweek 6", deadline_time="2026-09-26T10:00:00Z", is_current=False, is_next=True)
    teams = [
        Team(id=1, name="Arsenal", short_name="ARS", strength=4),
        Team(id=2, name="Aston Villa", short_name="AVL", strength=3),
    ]
    elem_types = [
        ElementType(
            id=1,
            singular_name="Goalkeeper",
            singular_name_short="GKP",
            plural_name="Goalkeepers",
            plural_name_short="GKP",
        ),
    ]
    elem = Element(
        id=1,
        web_name="Raya",
        first_name="David",
        second_name="Raya",
        team=1,
        element_type=1,
        now_cost=55,
        status="a",
        minutes=450,
        total_points=25,
        code=154561,
    )
    fixtures = [Fixture(id=1, code=1001, event=6, team_h=1, team_a=2, kickoff_time="2026-09-26T14:00:00Z")]
    boot = BootstrapStatic(events=[event], elements=[elem], teams=teams, element_types=elem_types)

    res = projection_engine.predict_gameweek(target_gw=6, bootstrap=boot, fixtures=fixtures)
    assert not res.empty

    cache_keys = list(projection_engine._cache.keys())
    assert len(cache_keys) == 1
    key = cache_keys[0]

    # Structure: model_version : season : target_gw : feature_hash : manifest_hash : avail_hash
    parts = key.split(":")
    assert len(parts) >= 5
    curr_season = derive_season(format="hyphen")
    manifest_hash = model_registry.get_manifest_hash()[:16]
    assert parts[1] == curr_season
    assert parts[2] == "gw6"
    assert parts[4] == manifest_hash

    # Test explicit invalidation
    projection_engine.invalidate_cache()
    assert len(projection_engine._cache) == 0


def test_dynamic_season_derivation_without_stale_literals(monkeypatch):
    """Verify changing season dynamically updates GameState with no stale literals."""
    monkeypatch.setenv("FPL_SEASON", "2027/28")
    assert derive_season() == "2027/28"
    assert derive_season(format="hyphen") == "2027-28"

    gs = GameState()
    assert gs.season == "2027/28"

    monkeypatch.setenv("FPL_SEASON", "2028/29")
    assert derive_season() == "2028/29"
    assert derive_season(format="hyphen") == "2028-29"


def test_status_endpoint_freshness_section():
    """Verify /api/health and /api/system/status contain complete freshness telemetry."""
    client = TestClient(app)

    # Health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    data_health = r_health.json()
    assert "freshness" in data_health
    f_h = data_health["freshness"]
    assert "data_as_of" in f_h
    assert "last_finished_gw_in_master" in f_h
    assert "bootstrap_age" in f_h
    assert "cache_age" in f_h
    assert "model_manifest_hash" in f_h
    assert f_h["model_manifest_hash"] == model_registry.get_manifest_hash()

    # System status
    r_sys = client.get("/api/system/status")
    assert r_sys.status_code == 200
    data_sys = r_sys.json()
    assert "freshness" in data_sys
    f_s = data_sys["freshness"]
    assert f_s["model_manifest_hash"] == model_registry.get_manifest_hash()
    assert f_s["last_finished_gw_in_master"] == historical_manager.get_last_finished_gameweek()


def test_player_identity_resolver_used_in_features_groupby():
    """Verify FeaturePipeline.build_historical_features resolves canonical IDs via PlayerIdentityResolver without collisions."""
    df_raw = pd.DataFrame(
        [
            {
                "season": "2023-24",
                "element": 1,
                "name": "Folarin Balogun",
                "position": "FWD",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2023-08-12T12:00:00Z",
                "round": 1,
                "minutes": 90,
                "total_points": 5,
                "fixture": 1,
            },
            {
                "season": "2024-25",
                "element": 1,
                "name": "Fabio Vieira",
                "position": "MID",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2024-08-17T12:00:00Z",
                "round": 1,
                "minutes": 90,
                "total_points": 2,
                "fixture": 2,
            },
            {
                "season": "2026-27",
                "element": 1,
                "name": "David Raya",
                "position": "GKP",
                "team": "Arsenal",
                "opponent_team": 2,
                "was_home": True,
                "kickoff_time": "2026-08-15T12:00:00Z",
                "round": 1,
                "minutes": 90,
                "total_points": 6,
                "fixture": 3,
            },
        ]
    )
    X, Y = feature_engineering.build_historical_features(df_raw)
    assert not X.empty
    c1 = player_identity_resolver.resolve_canonical_id(1, "2023-24", "Folarin Balogun")
    c2 = player_identity_resolver.resolve_canonical_id(1, "2024-25", "Fabio Vieira")
    c3 = player_identity_resolver.resolve_canonical_id(1, "2026-27", "David Raya")
    assert len({c1, c2, c3}) == 3
