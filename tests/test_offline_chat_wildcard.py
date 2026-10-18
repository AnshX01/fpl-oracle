from fpl_oracle.llm.provider import OfflineExpertProvider


def test_wildcard_decision_renders_as_a_rebuild_even_with_trimmed_lists():
    rec = {
        "gameweek": 8,
        "transfers": {
            "in": [{"name": "Rogers", "price": 7.9, "xp_gw": 5.9}, "... 5 more"],
            "out": [{"name": "Haaland", "price": 15.6, "xp_gw": 5.2}, "... 5 more"],
        },
        "chip": {"play_now": True, "chip": "Wildcard"},
        "lineup": {"captain": {"web_name": "B.Fernandes"}, "vice_captain": {"web_name": "Palmer"}},
    }
    text = OfflineExpertProvider._render_decision(rec)
    assert "play your Wildcard: bring in Rogers" in text and "... 5 more" in text
    assert "sell Haaland" in text and "and buy" not in text
