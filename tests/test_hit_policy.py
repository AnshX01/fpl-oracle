from fpl_oracle.optimise.hit_policy import cooldown_until, expected_transfer_gain, settle_hit


def test_covers_hit_is_success_even_below_prediction():
    assert settle_hit([1], [2], {1: 7, 2: 2}, 4)["status"] == "success"
    assert settle_hit([1], [2], {1: 6, 2: 2}, 4)["status"] == "success"
    assert settle_hit([1], [2], {1: 5, 2: 2}, 4)["status"] == "failed"
    assert settle_hit([1], [2], {1: 5}, 4)["status"] == "unknown"


def test_failed_hit_blocks_six_following_weeks():
    assert cooldown_until([{"gameweek": 6, "status": "failed"}]) == 13
    assert cooldown_until([{"gameweek": 6, "status": "success"}]) == 0


def test_horizon_threshold_counts_raw_in_minus_out():
    maps = {
        6: {1: {"expected_points": 3}, 2: {"expected_points": 2}},
        7: {1: {"expected_points": 8}, 2: {"expected_points": 2}},
    }
    assert expected_transfer_gain([1], [2], maps, [6, 7]) == 7


def test_sequence_cap_threshold_and_cooldown():
    import pandas as pd

    from fpl_oracle.optimise.sequential import search_sequences
    from fpl_oracle.optimise.transfers import TransferOptimizer
    from tests.test_sequential_search import fixture_data

    frame, _, _ = fixture_data()
    extras = []
    for i in range(2):
        row = frame.iloc[13 + i].copy()
        row["element"] = 101 + i
        row["team"] = 100 + i
        row["expected_points"] = 10
        extras.append(row)
    pool = pd.concat([frame, pd.DataFrame(extras)], ignore_index=True)
    pools = {g: pool.copy() for g in (6, 7)}
    maps = {g: {int(r["element"]): dict(r) for _, r in p.iterrows()} for g, p in pools.items()}

    def run(ft, cooldown=0):
        return search_sequences(
            TransferOptimizer(),
            set(frame.element),
            0,
            {int(e): 50 for e in frame.element},
            ft,
            [6, 7],
            pools,
            maps,
            [],
            set(),
            set(),
            set(),
            risk="points",
            hit_cooldown_until=cooldown,
        )

    assert all(r["hits"] <= 1 and r["transfers_count"] <= 2 for s in run(0) for r in s["history"])
    assert run(1)[0]["history"][0]["transfers_count"] == 2
    assert all(r["hits"] == 0 for s in run(0, 13) for r in s["history"])
    assert any(s["history"][0]["transfers_count"] == 0 and s["history"][0]["banked_ft"] == 2 for s in run(1))


def test_bank_ft_beats_small_now_gain_for_future_pair():
    import pandas as pd

    from fpl_oracle.optimise.sequential import search_sequences
    from fpl_oracle.optimise.transfers import TransferOptimizer
    from tests.test_sequential_search import fixture_data

    frame, _, _ = fixture_data()
    extras = []
    for i in range(3):
        row = frame.iloc[13].copy()
        row["element"] = 101 + i
        row["team"] = 100 + i
        row["expected_points"] = 3 if i == 2 else 0
        row["simulation_unavailable"] = i < 2
        extras.append(row)
    first = pd.concat([frame, pd.DataFrame(extras)], ignore_index=True).fillna({"simulation_unavailable": False})
    second = first.copy()
    second.loc[second.element.isin([101, 102]), "expected_points"] = 20
    second.loc[second.element == 103, "expected_points"] = 0
    second["simulation_unavailable"] = False
    pools = {6: first, 7: second}
    maps = {g: {int(r["element"]): dict(r) for _, r in p.iterrows()} for g, p in pools.items()}
    states = search_sequences(
        TransferOptimizer(),
        set(frame.element),
        0,
        {int(e): 50 for e in frame.element},
        1,
        [6, 7],
        pools,
        maps,
        [],
        set(),
        set(),
        set(),
        risk="points",
    )
    best = states[0]["history"]
    assert best[0]["transfers_count"] == 0
    assert best[0]["banked_ft"] == 2
    assert best[1]["transfers_count"] == 2
    assert best[1]["hit_cost"] == 0


def test_price_watch_never_promises_tonight_change():
    from types import SimpleNamespace

    from fpl_oracle.optimise.price_change import PriceChangePredictor

    player = SimpleNamespace(
        id=1,
        web_name="Player",
        team=1,
        now_cost=50,
        selected_by_percent=1,
        transfers_in_event=100000,
        transfers_out_event=0,
        price_change_hourly_rate=0,
        price_change_projections=[],
    )
    row = PriceChangePredictor().analyze_price_changes(SimpleNamespace(elements=[player]))[0]
    assert "unconfirmed" in row["urgency_message"]
    assert "Buy before" not in row["urgency_message"]
    assert row["calibrated_price_forecast"] is False
