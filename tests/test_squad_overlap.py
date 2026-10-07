from types import SimpleNamespace

from fpl_oracle.league.rivals import attach_squad_overlap


def test_overlap_uses_observed_complete_squads_without_fabricating_uninspected_rows():
    standings = [{"entry": 2}, {"entry": 3}]
    owned = [SimpleNamespace(element=i) for i in range(1, 16)]
    rivals = [{"entry_id": 2, "observed_gameweek": 5, "squad": [{"element": i} for i in range(6, 21)]}]
    result = attach_squad_overlap(standings, rivals, owned)
    assert result[0]["shared_players"] == 10
    assert result[0]["squad_overlap_pct"] == 66.7
    assert result[0]["overlap_gameweek"] == 5
    assert "squad_overlap_pct" not in result[1]
    assert "squad_overlap_pct" not in standings[0]
    assert "squad_overlap_pct" not in attach_squad_overlap(standings, rivals, owned[:14])[0]
