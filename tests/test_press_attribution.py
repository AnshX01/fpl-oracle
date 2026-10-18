from fpl_oracle.news.extract import text_extractor
from fpl_oracle.news.models import EvidenceCategory


def category(text, name, pid=1):
    return {e.category for e in text_extractor.extract_evidence_from_text(text, name, pid)}


def test_a_past_absence_of_one_player_rules_out_nobody():
    s = "Mbeumo spent the first few games on the right of United's attack when Amad Diallo was ruled out by injury."
    assert EvidenceCategory.RULED_OUT not in category(s, "Mbeumo")
    assert EvidenceCategory.RULED_OUT not in category(s, "Amad")
    s = "With Ryan Gravenberch suspended, Liverpool were light on midfield options."
    assert EvidenceCategory.RULED_OUT not in category(s, "Gravenberch")


def test_current_absences_still_count():
    assert EvidenceCategory.RULED_OUT in category("Amad has been ruled out of Saturday's game.", "Amad")
    assert EvidenceCategory.RULED_OUT in category("Amad was ruled out for Saturday's trip to Spurs.", "Amad")
    assert EvidenceCategory.RULED_OUT in category(
        "Amad will miss the next three games with a hamstring injury.", "Amad"
    )
    assert EvidenceCategory.RULED_OUT in category("Gravenberch is suspended for the visit of Brighton.", "Gravenberch")
