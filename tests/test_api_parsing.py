"""
Unit tests for API models and defensive parsing.
"""

from fpl_oracle.api.models import BootstrapStatic, Element, Team


def test_team_parsing_with_null_strength():
    """Verify that null strength (as seen in 2026/27 live API) parses cleanly without crashing."""
    raw = {
        "id": 1,
        "name": "Arsenal",
        "short_name": "ARS",
        "strength": None,
        "strength_overall_home": 4,
        "strength_overall_away": 5
    }
    team = Team.model_validate(raw)
    assert team.id == 1
    assert team.name == "Arsenal"
    assert team.strength is None
    assert team.strength_overall_home == 4

def test_element_parsing_with_2026_27_fields():
    """Verify element parsing with price_change_projections and hourly_rate."""
    raw = {
        "id": 10,
        "web_name": "Saka",
        "team": 1,
        "element_type": 3,
        "now_cost": 100,
        "price_change_hourly_rate": 15,
        "price_change_projections": [{"offset": 0, "projected_percent": "1.5"}],
        "defensive_contribution": 8
    }
    elem = Element.model_validate(raw)
    assert elem.id == 10
    assert elem.web_name == "Saka"
    assert elem.defensive_contribution == 8
    assert elem.price_change_hourly_rate == 15

def test_bootstrap_static_defensive_allow_extra():
    """Verify that extra unknown fields in API payload do not crash the app."""
    raw = {
        "events": [],
        "teams": [],
        "elements": [],
        "element_types": [],
        "chips": [],
        "future_unannounced_fpl_feature": {"version": 3}
    }
    boot = BootstrapStatic.model_validate(raw)
    assert boot.elements == []
