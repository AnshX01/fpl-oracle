import pytest
from fpl_oracle.api.models import Element
from fpl_oracle.data.fuzzy_match import FuzzyPlayerMatcher


@pytest.fixture
def timber_elements():
    return [
        Element(id=10, web_name="J.Timber", first_name="Jurrien", second_name="Timber", team=1, element_type=2, status="a", now_cost=55),
        Element(id=20, web_name="Q.Timber", first_name="Quinten", second_name="Timber", team=5, element_type=3, status="a", now_cost=60),
    ]


@pytest.fixture
def silva_elements():
    return [
        Element(id=30, web_name="B.Silva", first_name="Bernardo", second_name="Silva", team=11, element_type=3, status="a", now_cost=70),
        Element(id=40, web_name="F.Silva", first_name="Fabio", second_name="Silva", team=20, element_type=4, status="a", now_cost=50),
    ]


def test_shared_surname_requires_club_disambiguation(timber_elements, silva_elements):
    """
    F-010: Querying a shared surname without club or position context must trigger an
    ambiguity penalty (confidence < 0.5 or matched=None). When club is supplied,
    it must resolve to the correct candidate with high confidence.
    """
    matcher = FuzzyPlayerMatcher()
    team_map = {1: "Arsenal", 5: "Feyenoord", 11: "Man City", 20: "Wolves"}

    # 1. Ambiguous query "Timber" without club context
    res_ambiguous = matcher.match_single_name("Timber", timber_elements, team_map=team_map)
    assert (
        res_ambiguous["matched"] is None or res_ambiguous["confidence"] < 0.5
    ), f"Ambiguous surname 'Timber' matched with high confidence: {res_ambiguous['confidence']}"

    # 2. Disambiguated query with club
    res_arsenal = matcher.match_single_name("Timber", timber_elements, team_map=team_map, club="Arsenal")
    assert res_arsenal["matched"] is not None
    assert res_arsenal["matched"]["element"] == 10  # Jurrien Timber
    assert res_arsenal["confidence"] >= 0.8

    # 3. Ambiguous query "Silva" without club context
    res_silva = matcher.match_single_name("Silva", silva_elements, team_map=team_map)
    assert (
        res_silva["matched"] is None or res_silva["confidence"] < 0.5
    ), f"Ambiguous surname 'Silva' matched with high confidence: {res_silva['confidence']}"

    # 4. Disambiguated query "Bernardo Silva" or with club "Man City"
    res_mancity = matcher.match_single_name("Silva", silva_elements, team_map=team_map, club="Man City")
    assert res_mancity["matched"] is not None
    assert res_mancity["matched"]["element"] == 30  # Bernardo Silva
    assert res_mancity["confidence"] >= 0.8
