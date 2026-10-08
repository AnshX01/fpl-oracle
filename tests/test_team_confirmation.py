import asyncio

import pytest

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.team_confirmation import TeamConfirmation


def memory_confirmation(monkeypatch):
    store = TeamConfirmation()
    saved = {}
    monkeypatch.setattr(store, "read", lambda mid: saved.get("record"))
    monkeypatch.setattr(
        store, "write", lambda mid, r: saved.update(record=__import__("json").loads(__import__("json").dumps(r)))
    )
    return store


def test_confirm_current_state_and_expire_next_deadline(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    store = memory_confirmation(monkeypatch)
    payload = dict(
        gameweek=6,
        player_ids=[p.element for p in state.squad],
        bank_tenths=0,
        free_transfers=2,
        available_chips=["wildcard"],
        active_chip=None,
        hit_cost=0,
        selling_prices={p.element: p.selling_price for p in state.squad},
        purchase_prices={p.element: p.purchase_price for p in state.squad},
    )
    record = store.confirm(state, boot, payload, {"gameweek": 6, "ins": [], "outs": []})
    assert record["comparison"] == "followed"
    new = store.apply(state.model_copy(deep=True), boot)
    assert new.team_confirmed and new.free_transfers == 2 and new.bank_tenths == 0
    assert new.chips_remaining_set_1 == ["wildcard"]
    later = store.apply(state.model_copy(update={"target_gw": 7}), boot)
    assert later.confirmation_required and not later.team_confirmed


def test_confirm_rejects_duplicate_bad_prices_and_wrong_gw(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    store = memory_confirmation(monkeypatch)
    payload = dict(
        gameweek=6,
        player_ids=[p.element for p in state.squad],
        bank_tenths=0,
        free_transfers=1,
        available_chips=[],
        active_chip=None,
        hit_cost=0,
        selling_prices={p.element: p.selling_price for p in state.squad},
        purchase_prices={p.element: p.purchase_price for p in state.squad},
    )
    for field, value in (
        ("gameweek", 7),
        ("free_transfers", 6),
        ("player_ids", [state.squad[0].element] * 15),
        ("selling_prices", {}),
    ):
        with pytest.raises(ValueError):
            store.confirm(state, boot, dict(payload, **{field: value}))


def test_published_team_gets_advice_without_confirmation(configured_advisor, monkeypatch):
    state, pool, horizon = configured_advisor
    state.confirmation_required = True
    state.team_confirmed = False
    from unittest.mock import AsyncMock

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.analysis import analysis_service

    monkeypatch.setattr(team_confirmation, "read", lambda mid: None)
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    result = asyncio.run(
        analysis_service.joint_plan(
            current_squad_df=pool,
            player_pool_df=pool,
            bank=state.bank_tenths,
            free_transfers=state.free_transfers,
            horizon_projections={6: horizon[6]},
            target_gw=6,
            current_gw=5,
            available_chips=[],
        )
    )
    assert result["recommended_plan"]["lineup"]["captain"]


@pytest.mark.parametrize("problem", ["Active Free Hit needs permanent team", "Official published team differs"])
def test_real_state_conflict_still_blocks_advice(configured_advisor, monkeypatch, problem):
    state, pool, horizon = configured_advisor
    state.confirmation_required = True
    state.team_confirmed = False
    state.error_message = problem
    from fastapi import HTTPException

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.analysis import analysis_service

    monkeypatch.setattr(team_confirmation, "read", lambda mid: None)
    with pytest.raises(HTTPException) as err:
        asyncio.run(
            analysis_service.joint_plan(
                current_squad_df=pool,
                player_pool_df=pool,
                bank=state.bank_tenths,
                free_transfers=state.free_transfers,
                horizon_projections=horizon,
                target_gw=6,
                current_gw=5,
                available_chips=[],
            )
        )
    assert err.value.status_code == 409 and err.value.detail == problem


def test_reconfirm_is_idempotent_and_keeps_comparison(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    store = memory_confirmation(monkeypatch)
    payload = dict(
        gameweek=6,
        player_ids=[p.element for p in state.squad],
        bank_tenths=0,
        free_transfers=2,
        available_chips=[],
        active_chip=None,
        hit_cost=0,
        selling_prices={p.element: p.selling_price for p in state.squad},
        purchase_prices={p.element: p.purchase_price for p in state.squad},
    )
    first = store.confirm(state, boot, payload, {"gameweek": 6, "ins": [], "outs": []})
    second = store.confirm(state, boot, payload, {"gameweek": 6, "ins": [], "outs": []})
    assert first["comparison"] == second["comparison"] == "followed"
    assert second["actual_ins"] == [] and second["actual_outs"] == []


def test_forced_active_chip_cannot_be_saved_for_later():
    from tests.test_sequential_search import run

    states, _, _ = run([18, 19], ["3xc"], forced_current_chip="3xc")
    assert states
    assert all(s["history"][0]["chip"] == "3xc" for s in states)


def test_followed_partial_different_and_applied_team(configured_advisor, monkeypatch):
    from collections import Counter

    from fpl_oracle.optimise.transfers import TransferOptimizer

    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    ids = [p.element for p in state.squad]
    clubs = Counter(p.team for p in state.squad)
    swaps = []
    for owned in state.squad:
        candidate = next(
            (
                e
                for e in boot.elements
                if e.id not in ids
                and e.id not in [s[1].id for s in swaps]
                and e.element_type == {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[owned.position]
                and clubs[e.team] < 3
            ),
            None,
        )
        if candidate:
            swaps.append((owned, candidate))
        if len(swaps) == 2:
            break
    assert len(swaps) == 2
    rec = dict(gameweek=6, baseline=ids, ins=[e.id for _, e in swaps], outs=[p.element for p, _ in swaps])
    for count, expected in ((1, "partly followed"), (2, "followed"), (0, "different")):
        store = memory_confirmation(monkeypatch)
        selected = list(ids)
        prices = {p.element: p.purchase_price for p in state.squad}
        for old, new in swaps[:count]:
            selected.remove(old.element)
            selected.append(new.id)
            prices.pop(old.element)
            prices[new.id] = new.now_cost
        payload = dict(
            gameweek=6,
            player_ids=selected,
            bank_tenths=10,
            free_transfers=0,
            available_chips=[],
            active_chip=None,
            hit_cost=4 if count else 0,
            selling_prices={eid: prices[eid] for eid in selected},
            purchase_prices=prices,
        )
        record = store.confirm(state, boot, payload, rec)
        assert record["comparison"] == expected
        newstate = store.apply(state.model_copy(deep=True), boot)
        assert {p.element for p in newstate.squad} == set(selected)
        assert set(record["actual_ins"]) == {e.id for _, e in swaps[:count]}
        # Candidate generator never buys a player already confirmed as owned.
        pool = newstate.to_squad_dataframe()
        pool["expected_points"] = 5.0
        pmap = {int(r.element): dict(r) for _, r in pool.iterrows()}
        assert (
            TransferOptimizer()._get_candidate_1_transfers(set(selected), 10, prices, pool, pmap, set(), set(), set())
            == []
        )


def test_followed_payload_one_tap_and_idempotent(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    store = memory_confirmation(monkeypatch)
    rec = dict(
        gameweek=6,
        baseline=[p.element for p in state.squad],
        baseline_bank=10,
        baseline_ft=0,
        ins=[],
        outs=[],
        hit_cost=0,
        remaining_bank=10,
        buy_prices={},
        baseline_purchase={p.element: p.purchase_price for p in state.squad},
        baseline_selling={p.element: p.selling_price for p in state.squad},
        captain=state.squad[1].element,
        vice_captain=state.squad[2].element,
        bench=[state.squad[0].element] + [p.element for p in state.squad[-3:]],
        chip=None,
    )
    payload = store.followed_payload(state, boot, rec)
    first = store.confirm(state, boot, payload, rec)
    assert first["captain"] == rec["captain"] and first["bench"] == rec["bench"]
    applied = store.apply(state.model_copy(deep=True), boot)
    assert store.followed_payload(applied, boot, rec)["hit_cost"] == 0
    with pytest.raises(ValueError, match="changed"):
        store.followed_payload(state.model_copy(update={"bank_tenths": 11}), boot, rec)


def test_past_deadline_and_active_freehit_block(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    store = memory_confirmation(monkeypatch)
    payload = dict(
        gameweek=6,
        player_ids=[p.element for p in state.squad],
        bank_tenths=10,
        free_transfers=0,
        available_chips=[],
        active_chip="freehit",
        hit_cost=0,
        selling_prices={p.element: p.selling_price for p in state.squad},
        purchase_prices={p.element: p.purchase_price for p in state.squad},
    )
    with pytest.raises(ValueError, match="Deadline passed"):
        store.confirm(state.model_copy(update={"deadline_utc": "2020-01-01T00:00:00Z"}), boot, payload)
    store.confirm(state, boot, payload)
    applied = store.apply(state, boot)
    assert not applied.team_confirmed and "permanent" in applied.error_message


def test_confirmation_endpoint_invalidates_and_uses_actual_team(configured_advisor, monkeypatch):
    from unittest.mock import AsyncMock

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.optimise.hit_ledger import hit_ledger
    from fpl_oracle.server.advice_job import advice_publisher
    from fpl_oracle.server.analysis import analysis_service
    from fpl_oracle.server.routes.api import ConfirmTeamRequest, confirm_team

    state, _, _ = configured_advisor
    saved = {}
    monkeypatch.setattr(team_confirmation, "read", lambda mid: saved.get("record"))
    monkeypatch.setattr(team_confirmation, "write", lambda mid, r: saved.update(record=r))
    monkeypatch.setattr(team_confirmation, "recommendation", lambda mid: None)
    monkeypatch.setattr(advice_publisher, "invalidate", AsyncMock())
    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    captures = []
    monkeypatch.setattr(hit_ledger, "write", lambda mid, r: captures.append(r))
    req = ConfirmTeamRequest(
        gameweek=6,
        player_ids=[p.element for p in state.squad],
        bank_tenths=10,
        free_transfers=0,
        available_chips=[],
        hit_cost=4,
        selling_prices={p.element: p.selling_price for p in state.squad},
        purchase_prices={p.element: p.purchase_price for p in state.squad},
    )
    result = asyncio.run(confirm_team(req))
    assert result["source"] == "user_confirmed"
    assert captures[0]["source"] == "user_confirmed" and captures[0]["status"] == "pending"
    assert captures[0]["expected_gain"] is None
    advice_publisher.invalidate.assert_awaited_once()
    analysis_service.invalidate.assert_awaited_once()


def test_followed_lock_survives_reload_and_expires(configured_advisor, monkeypatch):
    from fastapi import HTTPException

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.analysis import analysis_service

    state, pool, horizon = configured_advisor
    store = memory_confirmation(monkeypatch)
    store.write(None, dict(gameweek=6, deadline=None, locked=True))
    assert store.locked(state)
    assert not store.locked(state.model_copy(update={"target_gw": 7}))
    monkeypatch.setattr(team_confirmation, "locked", store.locked)
    with pytest.raises(HTTPException, match="Next update"):
        asyncio.run(
            analysis_service.joint_plan(
                current_squad_df=pool,
                player_pool_df=pool,
                bank=10,
                free_transfers=0,
                horizon_projections=horizon,
                target_gw=6,
                current_gw=5,
                available_chips=[],
            )
        )


def test_undo_restores_prior_local_state(configured_advisor, monkeypatch):
    from unittest.mock import AsyncMock

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.optimise.hit_ledger import hit_ledger
    from fpl_oracle.server.advice_job import advice_publisher
    from fpl_oracle.server.analysis import analysis_service
    from fpl_oracle.server.routes.api import undo_followed_team

    prior = dict(gameweek=6, deadline=None, locked=False, hit_cost=0)
    saved = {"record": dict(gameweek=6, deadline=None, locked=True, undo_state=prior)}
    monkeypatch.setattr(team_confirmation, "read", lambda mid: saved["record"])
    monkeypatch.setattr(team_confirmation, "write", lambda mid, r: saved.update(record=r))
    monkeypatch.setattr(hit_ledger, "undo_user_confirmation", lambda *a: None)
    monkeypatch.setattr(advice_publisher, "invalidate", AsyncMock())
    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    assert asyncio.run(undo_followed_team())["status"] == "undone"
    assert saved["record"] == prior


def test_post_deadline_reconciles_only_matching_public_team(configured_advisor, monkeypatch):
    state, _, _ = configured_advisor
    boot, _ = asyncio.run(fpl_client.get_bootstrap_static())
    state.manager_id = 123
    store = memory_confirmation(monkeypatch)
    record = dict(
        gameweek=5,
        deadline=None,
        locked=True,
        player_ids=[p.element for p in state.squad],
        purchase_prices={p.element: p.purchase_price for p in state.squad},
        active_chip=None,
    )
    store.write(123, record)
    new = store.apply(state.model_copy(deep=True), boot)
    assert new.team_confirmed and new.confirmation_comparison == "official reconciled"
    assert store.read(123)["gameweek"] == 6
    store.write(123, dict(record, player_ids=[-1]))
    bad = store.apply(state.model_copy(deep=True), boot)
    assert not bad.team_confirmed and "differs" in bad.error_message


def test_followed_endpoint_saves_exact_published_state(configured_advisor, monkeypatch):
    from unittest.mock import AsyncMock

    from fpl_oracle.domain.team_confirmation import team_confirmation
    from fpl_oracle.server.advice_job import advice_publisher
    from fpl_oracle.server.analysis import analysis_service
    from fpl_oracle.server.routes.api import confirm_followed_team

    state, _, _ = configured_advisor
    saved = {}
    monkeypatch.setattr(team_confirmation, "read", lambda mid: saved.get("record"))
    monkeypatch.setattr(
        team_confirmation,
        "write",
        lambda mid, r: saved.update(record=__import__("json").loads(__import__("json").dumps(r))),
    )
    rec = dict(
        gameweek=6,
        baseline=[p.element for p in state.squad],
        baseline_bank=10,
        baseline_ft=0,
        ins=[],
        outs=[],
        hit_cost=0,
        remaining_bank=10,
        buy_prices={},
        baseline_purchase={p.element: p.purchase_price for p in state.squad},
        baseline_selling={p.element: p.selling_price for p in state.squad},
        captain=state.squad[1].element,
        vice_captain=state.squad[2].element,
        bench=[state.squad[0].element] + [p.element for p in state.squad[-3:]],
        chip=None,
    )
    monkeypatch.setattr(team_confirmation, "recommendation", lambda mid: rec)
    monkeypatch.setattr(advice_publisher, "invalidate", AsyncMock())
    monkeypatch.setattr(analysis_service, "invalidate", AsyncMock())
    result = asyncio.run(confirm_followed_team())
    assert result["locked"] and result["captain"] == rec["captain"]
    assert saved["record"]["locked"]
    assert saved["record"]["followed_recommendation"]["baseline"] == rec["baseline"]
