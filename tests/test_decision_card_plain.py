"""Plain-words helpers of the decision card: league standing, move pairing and the planner's confidence check."""

from fpl_oracle.briefing.decision_card import _comparison_sentences, _later_chips, _pair_moves, standing_summary

ROWS = [
    {"entry": 1, "rank": 1, "player_name": "Ann", "entry_name": "A FC", "total": 500, "event_total": 60},
    {"entry": 2, "rank": 2, "player_name": "Me", "entry_name": "My FC", "total": 488, "event_total": 55},
    {"entry": 3, "rank": 3, "player_name": "Cy", "entry_name": "C FC", "total": 470, "event_total": 50},
]


def test_standing_gaps_and_text():
    s = standing_summary({"league_name": "Pals", "standings": ROWS}, 2)
    assert (s["position"], s["gap_to_leader"], s["gap_to_next_above"], s["lead_over_next_below"]) == (2, 12, 12, 18)
    assert "2nd in Pals" in s["text"] and "12 points behind the leader (Ann)" in s["text"]
    assert [r["is_you"] for r in s["table"]] == [False, True, False]
    assert standing_summary({"standings": ROWS}, 1)["is_leader"] is True
    assert standing_summary({"standings": ROWS}, 99) is None


def test_pair_moves_matches_position_first():
    outs = [{"web_name": "D1", "position": "DEF"}, {"web_name": "M1", "position": "MID"}]
    ins = [{"web_name": "M2", "position": "MID"}, {"web_name": "D2", "position": "DEF"}]
    assert [(p["out"]["web_name"], p["in"]["web_name"]) for p in _pair_moves(ins, outs)] == [("D1", "D2"), ("M1", "M2")]


def test_chip_comparison_in_plain_words():
    keep = {"decision_check": {"gain": 4.9, "uncertainty": 1.2, "needed": 2.0, "versus": "keeping the chip"}}
    assert (
        _comparison_sentences(keep, "Wildcard")[0]
        == "Playing it now is about 4.9 points better than keeping your Wildcard for later."
    )
    held = {
        "rejected_alternative": {
            "gain": 1.0,
            "needed": 3.0,
            "uncertainty": 1.5,
            "versus": "keeping the chip",
            "chip": "wildcard",
        }
    }
    text = _comparison_sentences(held, None)[0]
    assert "only about 1.0 points better than waiting" in text
    assert not any(word in text for word in ("xP", "uncertain", "needed", "sure"))


def test_later_chips_are_provisional():
    plan = {
        "trajectory": [
            {"gameweek": 8, "chip": "wildcard"},
            {"gameweek": 12, "chip": "bboost"},
            {"gameweek": 15, "chip": "freehit"},
        ]
    }
    # the 0.4 point Free Hit is noise and is dropped
    later = _later_chips(
        plan, [{"code": "bboost", "expected_gain": 3.456}, {"code": "freehit", "expected_gain": 0.4}], 8
    )
    assert later == [
        {
            "chip": "bboost",
            "chip_display_name": "Bench Boost",
            "gameweek": 12,
            "gain": 3.5,
            "provisional": True,
            "label": "provisional - re-decided each week",
        }
    ]
