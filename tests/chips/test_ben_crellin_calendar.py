"""
Unit tests for Ben Crellin BGW/DGW Calendar Projection & Pre-Emptive Chip Choreography.
"""


from fpl_oracle.chips.calendar import FixtureCalendar


def test_prospective_blank_and_double_gameweeks_projection():
    """Verify that prospective cup-clash BGWs and DGWs are identified based on fixed FA/EFL Cup dates."""
    cal = FixtureCalendar()
    proj = cal.project_prospective_blank_and_double_gameweeks(current_gw=1)

    blanks = proj["prospective_blank_gameweeks"]
    doubles = proj["prospective_double_gameweeks"]

    blank_gws = [b["gameweek"] for b in blanks]
    double_gws = [d["gameweek"] for d in doubles]

    # GW26 (EFL Cup) and GW29 (FA Cup QF) are prospective blanks
    assert 26 in blank_gws
    assert 29 in blank_gws
    # GW34 and GW37 are prospective doubles
    assert 34 in double_gws
    assert 37 in double_gws

    # Check target milestones
    assert proj["bgw_target_gw"] == 29
    assert proj["dgw_target_gw"] == 34


def test_prospective_calendar_filters_past_gameweeks():
    """Verify that prospective milestones in the past are excluded."""
    cal = FixtureCalendar()
    proj = cal.project_prospective_blank_and_double_gameweeks(current_gw=30)

    blank_gws = [b["gameweek"] for b in proj["prospective_blank_gameweeks"]]
    double_gws = [d["gameweek"] for d in proj["prospective_double_gameweeks"]]

    assert 26 not in blank_gws
    assert 29 not in blank_gws
    assert 34 in double_gws
    assert 37 in double_gws


def test_prospective_calendar_does_not_duplicate_confirmed_fixtures():
    """Verify that if GW29 is already in confirmed blank_gameweeks, it is not duplicated."""
    cal = FixtureCalendar()
    confirmed = {
        "blank_gameweeks": [{"gameweek": 29, "type": "BLANK_GAMEWEEK", "blanking_teams": []}],
        "double_gameweeks": [{"gameweek": 34, "type": "DOUBLE_GAMEWEEK", "teams_with_doubles": []}],
    }
    proj = cal.project_prospective_blank_and_double_gameweeks(current_gw=1, confirmed_calendar=confirmed)
    blank_gws = [b["gameweek"] for b in proj["prospective_blank_gameweeks"]]
    double_gws = [d["gameweek"] for d in proj["prospective_double_gameweeks"]]

    assert 29 not in blank_gws  # already confirmed, not duplicated in prospective
    assert 34 not in double_gws  # already confirmed, not duplicated in prospective
    assert 26 in blank_gws
    assert 37 in double_gws
