import pandas as pd

from fpl_oracle.optimise.sequential import search_sequences
from fpl_oracle.optimise.transfers import TransferOptimizer
from tests.test_sequential_search import fixture_data


def test_future_winner_survives_immediate_shortlist():
    squad, _, _ = fixture_data()
    extras = []
    for i, points in enumerate((5, 4, 3)):
        row = dict(squad.iloc[-1], element=101 + i, team=100 + i, expected_points=points)
        extras.append(row)
    first = pd.concat([squad, pd.DataFrame(extras)], ignore_index=True)
    second = first.copy()
    second.loc[second.element.isin([101, 102]), "expected_points"] = 0
    second.loc[second.element == 103, "expected_points"] = 30
    second.loc[second.element == 103, "value"] = 100
    pools = {6: first, 7: second}
    maps = {g: {int(r["element"]): dict(r) for _, r in frame.iterrows()} for g, frame in pools.items()}
    states = search_sequences(
        TransferOptimizer(),
        set(squad.element),
        0,
        {int(e): 50 for e in squad.element},
        1,
        [6, 7],
        pools,
        maps,
        [],
        set(),
        set(),
        set(),
        risk="points",
        max_hits_per_move=0,
    )
    assert 103 in states[0]["history"][0]["transfers_in"]
    assert all(r["transfers_count"] <= 2 and r["hits"] == 0 for s in states for r in s["history"])


def test_future_pair_survives_immediate_shortlist():
    squad, _, _ = fixture_data()
    extras = []
    for i, points in enumerate((8, 7, 3, 2.5)):
        extras.append(dict(squad.iloc[-1], element=101 + i, team=100 + i, expected_points=points))
    first = pd.concat([squad, pd.DataFrame(extras)], ignore_index=True)
    second = first.copy()
    second.loc[second.element.isin([101, 102]), "expected_points"] = 0
    second.loc[second.element.isin([103, 104]), "expected_points"] = 30
    second.loc[second.element.isin([103, 104]), "value"] = 100
    pools = {6: first, 7: second}
    maps = {g: {int(r["element"]): dict(r) for _, r in frame.iterrows()} for g, frame in pools.items()}
    states = search_sequences(
        TransferOptimizer(),
        set(squad.element),
        0,
        {int(e): 50 for e in squad.element},
        2,
        [6, 7],
        pools,
        maps,
        [],
        set(),
        set(),
        set(),
        risk="points",
        max_hits_per_move=0,
    )
    assert set(states[0]["history"][0]["transfers_in"]) == {103, 104}


def test_forced_chip_date_reserves_resource_until_that_week():
    squad, pools, maps = fixture_data()
    states = search_sequences(
        TransferOptimizer(),
        set(squad.element),
        0,
        {int(e): 50 for e in squad.element},
        2,
        [18, 19],
        pools,
        maps,
        ["3xc"],
        set(squad.element),
        set(),
        set(),
        risk="points",
        forced_chip_schedule={19: "3xc"},
    )
    assert states
    assert all([r["chip"] for r in s["history"]] == [None, "3xc"] for s in states)


def test_date_comparison_has_replanned_transfers_and_paired_baseline():
    from fpl_oracle.chips.date_comparison import compare_dates

    squad, pools, maps = fixture_data()
    kwargs = dict(
        optimizer=TransferOptimizer(),
        elements=set(squad.element),
        bank=0,
        purchase={int(e): 50 for e in squad.element},
        ft=2,
        gameweeks=[18, 19],
        pools=pools,
        maps=maps,
        available=["3xc"],
        locked_in=set(squad.element),
        locked_out=set(),
        excluded=set(),
        risk="points",
    )
    baseline = search_sequences(**{**kwargs, "available": []})[0]
    rows = compare_dates(search_sequences, kwargs, "3xc", baseline, [18, 19])
    assert [row["gameweek"] for row in rows] == [18, 19]
    assert rows[1]["expected_gain"] > rows[0]["expected_gain"]
    for row in rows:
        assert row["expected_gain"] == round(row["plan_score"] - row["baseline_score"], 2)
        assert next(r["gameweek"] for r in row["trajectory"] if r["chip"]) == row["gameweek"]


def test_club_limit_does_not_hide_lower_legal_candidate():
    squad, _, _ = fixture_data()
    extras = [dict(squad.iloc[-1], element=101 + i, team=1 if i < 3 else 100, expected_points=20 - i) for i in range(4)]
    pool = pd.concat([squad, pd.DataFrame(extras)], ignore_index=True)
    mapping = {int(r["element"]): dict(r) for _, r in pool.iterrows()}
    moves = TransferOptimizer()._get_candidate_1_transfers(
        set(squad.element),
        0,
        {int(e): 50 for e in squad.element},
        pool,
        mapping,
        set(),
        set(),
        set(),
        max_per_pos=2,
    )
    assert any(m["transfers_in"] == [104] for m in moves)
    assert all(not set(m["transfers_in"]) & {101, 102, 103} for m in moves)


def test_wc_fh_each_date_replans_ordinary_baseline():
    from fpl_oracle.chips.date_comparison import compare_dates

    squad, pools, maps = fixture_data()
    replacement = dict(squad.iloc[-1], element=100, team=100, expected_points=12)
    for gw in (18, 19):
        pools[gw] = pd.concat([pools[gw], pd.DataFrame([replacement])], ignore_index=True)
        maps[gw][100] = replacement
    for chip in ("wildcard", "freehit"):
        kwargs = dict(
            optimizer=TransferOptimizer(),
            elements=set(squad.element),
            bank=0,
            purchase={int(e): 50 for e in squad.element},
            ft=1,
            gameweeks=[18, 19],
            pools=pools,
            maps=maps,
            available=[chip],
            locked_in=set(),
            locked_out=set(),
            excluded=set(),
            risk="points",
        )
        baseline = search_sequences(**{**kwargs, "available": []})[0]
        assert 100 in baseline["history"][0]["transfers_in"]
        rows = compare_dates(search_sequences, kwargs, chip, baseline, [18, 19])
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "measured"
            deployed = [r for r in row["trajectory"] if r["chip"] == chip]
            assert len(deployed) == 1 and deployed[0]["gameweek"] == row["gameweek"]
            assert row["expected_gain"] == round(row["plan_score"] - row["baseline_score"], 2)


def test_measured_restructure_dates_feed_final_joint_choice():
    squad, pools, _ = fixture_data()
    result = TransferOptimizer().evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad,
        player_pool_df=pools[18],
        bank=0,
        free_transfers=1,
        horizon_projections={18: pools[18], 19: pools[19]},
        current_gw=17,
        target_gw=18,
        available_chips=["freehit"],
        chips_by_set={1: ["freehit"]},
        locked_in_ids=list(squad.element),
        horizon_len=2,
        measure_chip_values=True,
    )
    row = result["measured_chip_roadmap"][0]
    assert [r["gameweek"] for r in row["date_comparison"]] == [18, 19]
    assert row["best_measured_date"] in (18, 19)
    for date in row["date_comparison"]:
        assert date["plan_score"] <= result["best_candidate"]["plan"]["accumulated_discounted_net_xp"] + 0.02
