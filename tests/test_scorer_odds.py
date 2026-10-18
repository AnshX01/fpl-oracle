import numpy as np
import pandas as pd

from fpl_oracle.data.odds import parse_scorer_page
from fpl_oracle.ml.components.attacking import PLAYER_GOALS_PER_TEAM_GOAL, AttackingModel


def test_parse_keeps_valid_prices_from_the_embedded_data():
    html = (
        'x \\"fixtures\\":[{\\"fixture\\":72,\\"home\\":2,\\"scorers\\":[{\\"element\\":411,\\"odds\\":1.73,'
        '\\"implied\\":0.578},{\\"element\\":397,\\"implied\\":1.4},{\\"element\\":\\"x\\",\\"implied\\":0.2}]}] y'
    )
    assert parse_scorer_page(html) == {72: {411: 0.578}}


def test_market_margin_comes_off_long_shots_and_the_total_matches_the_match_odds():
    X = pd.DataFrame({"implied_team_xG": [1.0] * 4, "chance_of_playing": [100.0] * 4})
    goals = np.array([0.4, 0.3, 0.2, 0.1])  # model, unconditional
    p_play = np.ones(4)
    chance = np.array([0.578, 0.30, 0.12, np.nan])  # bookmaker prices with margin; the last player is unpriced
    out = AttackingModel._with_market_scorers(goals, p_play, chance, X, np.ones(4), np.ones(4))
    assert np.isclose(out.sum(), PLAYER_GOALS_PER_TEAM_GOAL * 1.0, rtol=1e-6)
    assert out[3] == 0.1
    raw = -np.log1p(-chance[:3])
    kept = out[:3] / raw  # share of each price kept after the margin comes off
    assert kept[0] > kept[1] > kept[2]
