from fpl_oracle.server.source_fingerprint import changes


def boot(points, cost):
    return {"elements": [{"id": 1, "now_cost": cost, "event_points": points, "status": "a"}], "events": [], "teams": []}


def test_live_counters_do_not_invalidate_advice_with_season_prefixed_keys():
    assert changes("2026-27:bootstrap-static", boot(2, 50), boot(9, 50)) == []
    assert changes("2026-27:bootstrap-static", boot(2, 50), boot(2, 51)) == ["prices"]


def test_live_league_totals_do_not_invalidate_advice():
    page = lambda total, entries: {"standings": {"results": [{"entry": e, "total": total} for e in entries]}}  # noqa: E731
    assert changes("2026-27:league-classic:435956:page:1", page(400, [1, 2]), page(420, [1, 2])) == []
    assert changes("2026-27:league-classic:435956:page:1", page(400, [1, 2]), page(400, [1, 3])) == [
        "owner_or_rival_data"
    ]
