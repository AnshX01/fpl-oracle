import pandas as pd

from fpl_oracle.briefing.decision_card import _minutes_note
from fpl_oracle.data.features import _games_since_start


def test_minutes_note_names_only_coin_flip_starters():
    df = pd.DataFrame({"element": [1, 2, 3, 3], "p_starts": [0.95, 0.6, 0.2, 0.5]})
    players = [dict(element=1, web_name="Sure"), dict(element=2, web_name="Rotates"), dict(element=3, web_name="Dgw")]
    note = _minutes_note(df, players)
    assert (
        note
        == "Starting place not certain: Rotates (60% to start), Dgw (50% to start). Check team news before the deadline."
    )
    assert _minutes_note(df, players[:1]) is None


def test_games_since_start_counts_matches_before_each_row():
    starts = pd.Series([0, 1, 0, 0, 1])
    assert _games_since_start(starts).tolist() == [10.0, 10.0, 1.0, 2.0, 3.0]


def test_close_call_names_a_narrow_sale_in_plain_words():
    from fpl_oracle.briefing.decision_card import _close_calls

    plan = {
        "trajectory": [{}] * 12,
        "league": {"p_win": 0.241},
        "league_cover_checks": [
            dict(player="Haaland", owners=7, points_vs_plan=-2.7, p_win=0.22, chosen=False, owned=True, close=True),
            dict(player="Groß", owners=5, points_vs_plan=-9.0, p_win=0.2, chosen=False, owned=True, close=False),
        ],
    }
    assert _close_calls(plan) == [
        "Selling Haaland is only slightly better: 2.7 points over the next 12 gameweeks, inside the forecast's "
        "uncertainty. Keeping him is a fair choice; your chance of finishing first would be about 22% instead of 24%."
    ]
