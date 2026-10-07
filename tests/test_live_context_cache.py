import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.data.features import FeatureEngineering, player_identity_resolver


def test_horizon_reuses_identity_and_stats_without_changing_values(monkeypatch):
    root = Path(__file__).parent / "fixtures"
    boot = BootstrapStatic.model_validate(json.loads((root / "bootstrap_static.json").read_text()))
    fixtures = [Fixture.model_validate(r) for r in json.loads((root / "fixtures.json").read_text())]
    engine = FeatureEngineering()
    load = Mock(wraps=player_identity_resolver.load)
    monkeypatch.setattr(player_identity_resolver, "load", load)
    first = engine.extract_live_features_for_upcoming(boot, fixtures, 6)
    second = engine.extract_live_features_for_upcoming(boot, fixtures, 6)
    assert load.call_count == 1
    pd.testing.assert_frame_equal(first, second)
    engine.extract_live_features_for_upcoming(boot, fixtures, 7)
    assert load.call_count == 1


def test_external_identity_reload_invalidates_context(monkeypatch):
    root = Path(__file__).parent / "fixtures"
    boot = BootstrapStatic.model_validate(json.loads((root / "bootstrap_static.json").read_text()))
    fixtures = [Fixture.model_validate(r) for r in json.loads((root / "fixtures.json").read_text())]
    engine = FeatureEngineering()
    load = Mock(wraps=player_identity_resolver.load)
    monkeypatch.setattr(player_identity_resolver, "load", load)
    first = engine.extract_live_features_for_upcoming(boot, fixtures, 6)
    player_identity_resolver.load(history_df=pd.DataFrame({"element": [999], "season": ["2020-21"], "name": ["Other"]}))
    second = engine.extract_live_features_for_upcoming(boot, fixtures, 6)
    assert load.call_count == 3
    pd.testing.assert_frame_equal(first, second)
