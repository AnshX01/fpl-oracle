"""Configured, deterministic source fixtures; not live user data or model certification."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.domain.manager_state import EffectiveManagerState, PlayerSquadState


@pytest.fixture
def configured_advisor(monkeypatch):
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.domain.manager_state import manager_state_service
    from fpl_oracle.news.analyse import news_analyzer
    from fpl_oracle.server.analysis import analysis_service

    root = Path(__file__).parent / "fixtures"
    boot = BootstrapStatic.model_validate(json.loads((root / "bootstrap_static.json").read_text()))
    fixtures = [Fixture.model_validate(f) for f in json.loads((root / "fixtures.json").read_text())]
    wanted = {1: 2, 2: 5, 3: 5, 4: 3}
    elements = []
    for pos, count in wanted.items():
        elems = [e for e in boot.elements if e.element_type == pos]
        elems.sort(key=lambda e: (-e.total_points, e.id))
        elements.extend(elems[:count])
    counts = {}
    players = []
    # Pick legal team quota from source records, prefer current source performance ordering.
    elements = []
    for pos, count in wanted.items():
        for e in sorted([e for e in boot.elements if e.element_type == pos], key=lambda e: (-e.total_points, e.id)):
            if counts.get(e.team, 0) >= 3:
                continue
            elements.append(e)
            counts[e.team] = counts.get(e.team, 0) + 1
            if sum(x.element_type == pos for x in elements) == count:
                break
    for i, e in enumerate(elements):
        players.append(
            PlayerSquadState(
                element=e.id,
                web_name=e.web_name,
                team=e.team,
                position={1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e.element_type],
                now_cost=e.now_cost,
                purchase_price=e.now_cost,
                selling_price=e.now_cost,
                is_starter=i < 11,
                is_captain=i == 0,
                is_vice_captain=i == 1,
                bench_order=max(0, i - 10),
                chance_of_playing=100,
            )
        )
    state = EffectiveManagerState(squad=players, current_gw=5, target_gw=6, bank_tenths=10, free_transfers=0)
    pool = state.to_squad_dataframe()
    pool["expected_points"] = [4 + i * 0.1 for i in range(15)]
    pool["p10"] = 1.0
    pool["p90"] = 9.0
    pool["exp_defcon_pts"] = 0.0
    horizon = {g: pool.copy() for g in range(6, 14)}
    monkeypatch.setattr(manager_state_service, "get_current_state", AsyncMock(return_value=state))
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=(fixtures, False)))
    monkeypatch.setattr(fpl_client, "get_current_and_next_gw", AsyncMock(return_value=(5, 6)))
    monkeypatch.setattr(news_analyzer, "get_player_news_signals", AsyncMock(return_value=[]))
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value=horizon))
    analysis_service._plans.clear()
    analysis_service._plan_tasks.clear()
    return state, pool, horizon
