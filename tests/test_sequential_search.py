import itertools

import pandas as pd
import pytest

from fpl_oracle.optimise.sequential import search_sequences
from fpl_oracle.optimise.transfers import TransferOptimizer, _fast_eval_squad_full


def fixture_data():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    frame = pd.DataFrame(
        [
            dict(
                element=i + 1,
                position=p,
                team=i // 3 + 1,
                value=50,
                purchase_price=50,
                expected_points=2.0,
                web_name=str(i + 1),
            )
            for i, p in enumerate(positions)
        ]
    )
    pools = {g: frame.copy() for g in [18, 19, 20]}
    pools[19].loc[pools[19].element == 15, "expected_points"] = 20
    maps = {g: {int(r["element"]): dict(r) for _, r in f.iterrows()} for g, f in pools.items()}
    return frame, pools, maps


def run(gws, available, **kwargs):
    frame, pools, maps = fixture_data()
    opt = TransferOptimizer()
    # All owned players locked, so this reference tests exhaustive chip schedules
    # over an otherwise fixed legal squad rather than pruned player candidates.
    elems = set(frame.element)
    return (
        search_sequences(
            opt, elems, 0, {e: 50 for e in elems}, 1, gws, pools, maps, available, elems, set(), set(), **kwargs
        ),
        opt,
        maps,
    )


def test_matches_small_exhaustive_chip_schedule():
    states, opt, maps = run([18, 19], ["3xc", "bboost"])
    scores = []
    for a, b in itertools.product([None, "3xc", "bboost"], repeat=2):
        if a and a == b:
            continue
        score = 0
        for j, (gw, chip) in enumerate(zip([18, 19], [a, b], strict=True)):
            xp, _, _, bench, cap = _fast_eval_squad_full(set(range(1, 16)), maps[gw])
            score += opt.discount_factor**j * (xp + (cap if chip == "3xc" else bench if chip == "bboost" else 0))
        scores.append(score)
    assert states[0]["score"] == pytest.approx(max(scores))
    assert [r["chip"] for r in states[0]["history"]] == ["bboost", "3xc"]


def test_chips_expire_and_second_set_requires_explicit_inventory():
    states, _, _ = run([19, 20], ["3xc"])
    assert all(r["chip"] is None for s in states for r in s["history"] if r["gameweek"] == 20)
    states, _, _ = run([19, 20], ["3xc"], chips_by_set={2: ["3xc"]})
    assert any([r["chip"] for r in s["history"]] == ["3xc", "3xc"] for s in states)


def test_one_per_gw_and_banked_ft_cap():
    states, _, _ = run([18, 19], ["3xc", "bboost"])
    for s in states:
        chips = [r["chip"] for r in s["history"] if r["chip"]]
        assert len(chips) == len(set(chips))
        assert all(r["banked_ft"] <= 5 and r["bank"] >= 0 for r in s["history"])


def test_freehit_reverts_and_wildcard_persists(monkeypatch):
    from fpl_oracle.optimise.squad import squad_optimizer

    frame, pools, maps = fixture_data()
    replacement = dict(frame.iloc[-1], element=16, expected_points=25, web_name="replacement")
    for gw in pools:
        pools[gw] = pd.concat([pools[gw], pd.DataFrame([replacement])], ignore_index=True)
        maps[gw][16] = replacement
    opt = TransferOptimizer()
    monkeypatch.setattr(opt, "_get_candidate_1_transfers", lambda *a, **k: [])
    monkeypatch.setattr(opt, "_get_candidate_2_transfers", lambda *a, **k: [])
    solved = pd.concat([frame[frame.element != 15], pd.DataFrame([replacement])])
    monkeypatch.setattr(squad_optimizer, "solve_best_squad", lambda *a, **k: {"squad": solved})
    elems = set(frame.element)
    for chip in ["freehit", "wildcard"]:
        states = search_sequences(
            opt, elems, 0, {e: 50 for e in elems}, 3, [18, 19], pools, maps, [chip], set(), set(), set()
        )
        state = next(s for s in states if s["first_chip"] == chip)
        assert 16 in state["history"][0]["elements"]
        assert (16 in state["history"][1]["elements"]) == (chip == "wildcard")
        assert all(r["hits"] == 0 for r in state["history"])
        assert state["history"][0]["banked_ft"] == 3


def test_gap_utility_is_bounded_and_does_not_change_captain():
    from fpl_oracle.league.objective import select_balanced_sequence

    maps = {18: {1: dict(expected_points=5, p10=1, p90=9), 2: dict(expected_points=5, p10=1, p90=9)}}
    first = dict(score=100, history=[dict(gameweek=18, elements=[1], captain=1)])
    close = dict(score=99.6, history=[dict(gameweek=18, elements=[2], captain=2)])
    weak = dict(score=98, history=[dict(gameweek=18, elements=[2], captain=2)])
    chosen, meta = select_balanced_sequence(
        [first, close, weak], dict(user_points=100, rivals=[dict(points=150, elements=[1])]), maps
    )
    assert chosen is close
    assert meta["actual_xp_loss"] <= 0.5
    assert meta["upcoming_rival_captains_assumed"] is False
    assert chosen["history"][0]["captain"] == 2


def test_fast_sequence_lineup_matches_served_lineup():
    from fpl_oracle.optimise.lineup import lineup_optimizer
    from fpl_oracle.optimise.sequential import evaluate_lineup

    frame, _, _ = fixture_data()
    frame["p90"] = frame.expected_points * 1.4
    frame["p10"] = frame.expected_points * 0.3
    frame.loc[frame.element == 15, "expected_points"] = 8
    frame.loc[frame.element == 12, "p90"] = 25
    mapping = {int(r["element"]): dict(r) for _, r in frame.iterrows()}
    for chip in [None, "3xc", "bboost"]:
        gross, cap, _ = evaluate_lineup(set(frame.element), mapping, chip=chip)
        served = lineup_optimizer.select_lineup_and_captain(
            frame, is_triple_captain=chip == "3xc", is_bench_boost=chip == "bboost"
        )
        assert gross == pytest.approx(served["total_gameweek_expected_points"])
        assert cap == served["captain"]["element"]


def test_league_context_uses_roster_not_captain(monkeypatch):
    from types import SimpleNamespace

    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.data.store import data_store
    from fpl_oracle.league.rivals import rival_analyzer
    from fpl_oracle.league.standings import league_standings_manager
    from fpl_oracle.server.analysis import AnalysisService

    monkeypatch.setattr(data_store, "get_profile", lambda: SimpleNamespace(manager_id=10, target_league_id=20))

    async def standings(*a, **k):
        return dict(standings=[dict(entry=10, rank=2, total=100)], coverage=dict(partial=False))

    async def bootstrap(*a, **k):
        return None, False

    async def rivals(*a, **k):
        return dict(rival_squads=[dict(total_points=150, captain_element=99, squad=[dict(element=1)])])

    monkeypatch.setattr(league_standings_manager, "get_league_standings", standings)
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", bootstrap)
    monkeypatch.setattr(rival_analyzer, "analyze_rivals", rivals)
    import asyncio

    context = asyncio.run(AnalysisService().league_context(6))
    assert context == dict(
        user_points=100, user_rank=2, standings=[dict(rank=2, points=100)], rivals=[dict(points=150, elements=[1])]
    )


def test_actual_previous_freehit_blocks_first_future_deadline():
    states, _, _ = run([20], ["freehit"], previous_chip="freehit")
    assert all(state["first_chip"] is None for state in states)


def test_full_expiry_chip_gain_is_actual_paired_search():
    frame, pools, maps = fixture_data()
    result = TransferOptimizer().evaluate_joint_transfer_and_chip_plan(
        current_squad_df=frame,
        player_pool_df=pools[18],
        bank=0,
        free_transfers=1,
        horizon_projections={18: pools[18], 19: pools[19]},
        current_gw=17,
        target_gw=18,
        available_chips=["3xc", "bboost"],
        chips_by_set={1: ["3xc", "bboost"]},
        locked_in_ids=list(frame.element),
        horizon_len=2,
        measure_chip_values=True,
    )
    assert result["chip_measurement_status"] == "measured_through_expiry"
    assert result["decision_scope"]["horizon_gameweeks"] == [18, 19]
    for row in result["measured_chip_roadmap"]:
        assert row["expected_gain"] == pytest.approx(row["plan_score"] - row["baseline_score"], abs=0.02)
        assert all(step["chip"] != row["code"] for step in row["baseline_trajectory"])
        assert row["forecast_through"] == 19


def test_partial_window_never_claims_chip_dates():
    frame, pools, maps = fixture_data()
    result = TransferOptimizer().evaluate_joint_transfer_and_chip_plan(
        current_squad_df=frame,
        player_pool_df=pools[18],
        bank=0,
        free_transfers=1,
        horizon_projections={18: pools[18]},
        current_gw=17,
        target_gw=18,
        available_chips=["3xc"],
        locked_in_ids=list(frame.element),
        horizon_len=1,
        measure_chip_values=True,
    )
    assert result["chip_measurement_status"] == "incomplete_window"
    assert result["measured_chip_roadmap"] == []


def test_points_captain_matches_exhaustive_xi_and_milp():
    from fpl_oracle.optimise.lineup import lineup_optimizer
    from fpl_oracle.optimise.sequential import evaluate_lineup

    frame, pools, maps = fixture_data()
    frame["p90"] = 3.0
    frame["p10"] = 0.0
    frame.loc[frame.element == 15, "expected_points"] = 8.0
    frame.loc[frame.element == 8, "expected_points"] = 7.0
    frame.loc[frame.element == 8, "p90"] = 40.0
    pmap = {int(row["element"]): row for row in frame.to_dict("records")}
    for chip in (None, "3xc", "bboost"):
        score, cap, formation = evaluate_lineup(set(frame.element), pmap, "points", chip)
        exact = lineup_optimizer.select_lineup_and_captain(
            frame, risk_preference="points", is_triple_captain=chip == "3xc", is_bench_boost=chip == "bboost"
        )
        assert cap == 15
        assert cap == exact["captain"]["element"]
        assert score == pytest.approx(exact["total_gameweek_expected_points"], abs=0.02)


def test_progress_tracks_actual_full_chip_checks_without_changing_plan():
    from fpl_oracle.optimise.progress import search_progress

    frame, pools, _ = fixture_data()
    kwargs = dict(
        current_squad_df=frame,
        player_pool_df=pools[18],
        bank=0,
        free_transfers=1,
        horizon_projections={18: pools[18], 19: pools[19]},
        current_gw=17,
        target_gw=18,
        available_chips=["3xc", "bboost"],
        locked_in_ids=list(frame.element),
        horizon_len=2,
        measure_chip_values=True,
    )
    baseline = TransferOptimizer().evaluate_joint_transfer_and_chip_plan(**kwargs)
    stages = []
    token = search_progress.set(stages.append)
    try:
        observed = TransferOptimizer().evaluate_joint_transfer_and_chip_plan(**kwargs)
    finally:
        search_progress.reset(token)
    import json

    def serialize(plan):
        return json.dumps(plan, sort_keys=True, default=lambda value: value.to_dict("records") if isinstance(value, pd.DataFrame) else str(value))

    assert serialize(baseline["recommended_plan"]) == serialize(observed["recommended_plan"])
    assert stages == [
        "Comparing transfers through Gameweek 19",
        "Checking Triple Captain against saving it",
        "Checking Bench Boost against saving it",
        "Checking the final chip plan",
    ]
