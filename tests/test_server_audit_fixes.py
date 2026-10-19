from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from fpl_oracle.server import analysis
from fpl_oracle.server.analysis import analysis_service


def test_backup_planner_plans_no_chips_and_says_it_ran(monkeypatch):
    from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
    from fpl_oracle.optimise.transfers_beam import TransferOptimizer

    def broken(**_):
        raise RuntimeError("solver crashed")

    seen = {}

    def beam(self, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(multiperiod_optimizer, "evaluate_transfer_options", broken)
    monkeypatch.setattr(TransferOptimizer, "evaluate_joint_transfer_and_chip_plan", beam)
    chips = {"available_chips": ["freehit", "wildcard"], "chips_by_set": {1: ["freehit"], 2: []}}
    result = analysis.execute_authoritative_milp_plan(target_gw=8, **chips)
    assert seen["available_chips"] == [] and seen["chips_by_set"] is None
    assert result["decision_scope"]["fallback_used"] is True

    analysis.execute_authoritative_milp_plan(target_gw=8, forced_current_chip="bboost", **chips)
    assert seen["available_chips"] == ["bboost"]  # a chip already active this week is a fact, not advice


@pytest.mark.anyio
@pytest.mark.parametrize("failed, expect_context", [([], True), ([102], False)])
async def test_league_context_is_off_when_a_rival_failed_to_load(failed, expect_context):
    rows = [{"entry": 1, "rank": 2, "total": 500}, {"entry": 102, "rank": 1, "total": 510}]
    rivals = {"rival_squads": [], "rivals_failed": failed}
    analysis_service._league_context_cache = {}
    with (
        patch(
            "fpl_oracle.data.store.data_store.get_profile",
            return_value=SimpleNamespace(manager_id=1, target_league_id=5),
        ),
        patch(
            "fpl_oracle.league.standings.league_standings_manager.get_league_standings",
            AsyncMock(return_value={"standings": rows, "coverage": {}}),
        ),
        patch("fpl_oracle.api.fpl_client.fpl_client.get_bootstrap_static", AsyncMock(return_value=(object(), False))),
        patch("fpl_oracle.league.rivals.rival_analyzer.analyze_rivals", AsyncMock(return_value=rivals)),
    ):
        context = await analysis_service.league_context(8)
    assert (context is not None) is expect_context
    analysis_service._league_context_cache = {}


async def _card(configured_advisor, *, rivals_failed=None, planner_win=None, backup=False):
    """The real decision card, with the planner result and rival data adjusted as each test needs."""
    from fpl_oracle.briefing.decision_card import decision_card_generator
    from fpl_oracle.data.store import ProfileData
    from fpl_oracle.domain.manager_state import manager_state_service
    from fpl_oracle.league.rivals import rival_analyzer
    from fpl_oracle.league.standings import league_standings_manager

    real_plan = analysis_service.joint_plan

    async def plan(**kw):
        res = await real_plan(**kw)
        if backup:
            res["decision_scope"]["fallback_used"] = True
            res["recommended_chip"] = "freehit"
        if planner_win is not None:
            res["recommended_plan"]["league"] = {"p_win": planner_win}
        return res

    state = configured_advisor[0].model_copy(update={"manager_id": 101, "overall_points": 500})
    standings = {
        "league_name": "Test",
        "coverage": {"partial": False},
        "standings": [
            {"entry": 101, "player_name": "Me", "rank": 2, "total": 500},
            {"entry": 102, "player_name": "Rival", "rank": 1, "total": 510},
        ],
    }
    rivals = {"rival_squads": [], "rivals_failed": rivals_failed or [], "rival_selection_mode": "points window"}
    with (
        patch(
            "fpl_oracle.briefing.decision_card.data_store.get_profile",
            return_value=ProfileData(manager_id=101, target_league_id=555),
        ),
        patch.object(manager_state_service, "get_current_state", AsyncMock(return_value=state)),
        patch.object(league_standings_manager, "get_league_standings", AsyncMock(return_value=standings)),
        patch.object(rival_analyzer, "analyze_rivals", AsyncMock(return_value=rivals)),
        patch.object(analysis_service, "joint_plan", plan),
    ):
        return await decision_card_generator.generate_decision_card()


@pytest.mark.slow
@pytest.mark.anyio
async def test_backup_plan_is_announced_and_gives_no_chip_advice(configured_advisor):
    card = await _card(configured_advisor, backup=True)
    assert card["backup_planner_notice"] == "The main planner failed this run; this is a simpler backup plan."
    assert card["caveats"][0] == card["backup_planner_notice"]
    assert card["chip"]["recommend"] is False and card["chip"]["chip_name"] is None
    assert card["chip"]["reason"] == "Keep your chips this week; the backup planner does not judge chips."
    assert "Free Hit" not in card["headline"]


@pytest.mark.slow
@pytest.mark.anyio
async def test_league_numbers_are_unavailable_when_a_rival_failed_to_load(configured_advisor):
    card = await _card(configured_advisor, rivals_failed=[102])
    assert card["win_prob"]["status"] == "unavailable" and card["win_prob"]["p_first"] is None
    assert "unavailable this run" in card["win_prob"]["simulation_note"]


@pytest.mark.slow
@pytest.mark.anyio
async def test_only_the_planners_chance_of_finishing_first_is_shown(configured_advisor):
    card = await _card(configured_advisor, planner_win=0.15)
    assert card["win_prob"]["p_first"] == 15.0 and card["win_prob"]["mc_se"] is None
    assert "about 15%" in card["rivals"]["summary"]
