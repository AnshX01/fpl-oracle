import asyncio
from unittest.mock import AsyncMock

from fpl_oracle.server.analysis import analysis_service


def test_expiry_uses_same_inputs_for_short_and_long_window(configured_advisor, monkeypatch):
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.domain.manager_state import manager_state_service
    from fpl_oracle.optimise.transfers import transfer_optimizer

    state, _, horizon = configured_advisor
    boot, fixtures = None, []
    monkeypatch.setattr(manager_state_service, "get_current_state", AsyncMock(return_value=state))
    monkeypatch.setattr(fpl_client, "get_bootstrap_static", AsyncMock(return_value=(boot, False)))
    monkeypatch.setattr(fpl_client, "get_fixtures", AsyncMock(return_value=(fixtures, False)))
    frame = horizon[6]
    projections = {gw: frame for gw in range(6, 20)}
    monkeypatch.setattr(analysis_service, "projections", AsyncMock(return_value=projections))
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return dict(
            decision_scope=dict(horizon_gameweeks=list(range(6, 6 + kwargs["horizon_len"]))),
            recommended_chip=None,
            best_candidate=dict(net_gain_vs_hold=0),
            recommended_plan=dict(trajectory=[]),
            chip_comparison_table=[],
        )

    monkeypatch.setattr(transfer_optimizer, "evaluate_joint_transfer_and_chip_plan", solve)
    result = asyncio.run(analysis_service.expiry_sensitivity())
    assert [c["horizon_len"] for c in calls] == [8, 14]
    assert calls[0]["horizon_projections"] is calls[1]["horizon_projections"]
    assert result["promotion_allowed"] is False
    assert result["first_chip_changes"] is False
