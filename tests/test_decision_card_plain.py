"""Plain-words helpers of the decision card: league standing, move pairing and the planner's confidence check."""

import json

import pandas as pd

from fpl_oracle.briefing.decision_card import (
    _bench_note,
    _close_calls,
    _comparison_sentences,
    _later_chips,
    _pair_moves,
    standing_summary,
    track_record_summary,
)

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
    plan = {"trajectory": [{"gameweek": 8, "chip": "wildcard"}, {"gameweek": 12, "chip": "bboost"}]}
    assert _later_chips(plan, 8) == [
        {
            "chip": "bboost",
            "chip_display_name": "Bench Boost",
            "gameweek": 12,
            "provisional": True,
            "label": "provisional - re-decided each week",
        }
    ]


def test_rejected_move_names_the_players_and_gameweeks():
    plan = {
        "trajectory": [{"gameweek": 8}, {"gameweek": 9}],
        "rejected_alternative": {
            "gain": 0.4,
            "needed": 2.0,
            "versus": "rolling",
            "ins": [{"web_name": "Haaland", "position": "FWD"}],
            "outs": [{"web_name": "Wood", "position": "FWD"}],
        },
    }
    text = _comparison_sentences(plan, None)[1]
    assert "Wood for Haaland" in text and "next 2 gameweeks" in text


def test_close_calls_use_the_final_win_chance():
    plan = {
        "trajectory": [{"gameweek": 8}],
        "league": {"p_win": 0.31},
        "league_cover_checks": [
            {"chosen": False, "close": True, "points_vs_plan": -0.2, "owned": True, "player": "Saka", "p_win": 0.28}
        ],
    }
    (line,) = _close_calls(plan)
    assert "28%" in line and "31%" in line and "Selling Saka" in line


def test_bench_note_names_doubtful_bench_players():
    frame = pd.DataFrame({"element": [3, 4], "p_play": [0.25, 0.9]})
    bench = [{"element": 3, "web_name": "Benchy"}, {"element": 4, "web_name": "Fit"}]
    assert _bench_note(frame, bench) == "Benchy (25% to play) is on your bench, so he costs little this week."


def test_track_record_in_plain_words(tmp_path, monkeypatch):
    from fpl_oracle.ml import holdout

    path = tmp_path / "log.json"
    sources = {
        "FPL Oracle": {"rmse": 2.0, "top20_actual_points": 5.5},
        "FPL ep_next": {"rmse": 2.4, "top20_actual_points": 5.0},
    }
    path.write_text(json.dumps({"evaluated_gameweeks": [{"comparison": {"sources": sources}}]}), encoding="utf-8")
    monkeypatch.setattr(holdout, "HOLDOUT_LOG_PATH", path)
    assert track_record_summary() == (
        "How good our forecasts have been (1 gameweek): on average we miss a player's points by 2.0, "
        "FPL's own forecast by 2.4. Our top 20 picks scored 5.5 points on average, FPL's 5.0. "
        "A smaller miss and more points are better."
    )
