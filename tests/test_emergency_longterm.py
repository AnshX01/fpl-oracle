import asyncio
from unittest.mock import AsyncMock

from fpl_oracle.server.analysis import analysis_service
from fpl_oracle.server.routes import api


def test_emergency_equals_normal_model_with_same_constraint(configured_advisor, monkeypatch):
    state, pool, horizon = configured_advisor
    state.chips_remaining_set_1 = []
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    affected = int(pool.query("position == 'GKP'").iloc[0].element)

    async def run():
        result = await api.post_contingency_panic(
            api.PanicRequest(query="", ruled_out_ids=[affected], duration="window")
        )
        constrained = {}
        for gw, frame in horizon.items():
            frame = frame.copy()
            mask = frame.element == affected
            frame.loc[mask, ["expected_points", "p10", "p90"]] = 0
            frame["simulation_unavailable"] = mask
            constrained[gw] = frame
        direct = await analysis_service.joint_plan(
            current_squad_df=state.to_squad_dataframe(),
            player_pool_df=constrained[6],
            bank=state.bank_tenths,
            free_transfers=state.free_transfers,
            horizon_projections=constrained,
            current_gw=5,
            target_gw=6,
            available_chips=[],
            chips_by_set={1: [], 2: state.chips_remaining_set_2},
            chips_already_used=[],
            locked_out_ids=set(),
        )
        assert result["status"] == "recalculated"
        from fpl_oracle.server.safe_json import safe_json_serialize

        assert result["joint_plan"]["recommended_plan"]["trajectory"] == safe_json_serialize(
            direct["recommended_plan"]["trajectory"]
        )
        assert result["joint_plan"]["decision_scope"]["horizon_gameweeks"][-1] == 19
        assert affected not in {int(p) for p in direct["recommended_plan"]["lineup"]["starters"].element}
        for step in direct["recommended_plan"]["trajectory"]:
            assert affected not in step["transfers_in"]
            assert step["captain"] != affected

    asyncio.run(run())


def test_unrecognized_emergency_never_runs_optimizer(configured_advisor, monkeypatch):
    solve = AsyncMock()
    monkeypatch.setattr(analysis_service, "joint_plan", solve)
    result = asyncio.run(api.post_contingency_panic(api.PanicRequest(query="not-a-player")))
    assert result["status"] == "needs_player"
    solve.assert_not_called()


def test_short_absence_returns_to_normal_projections(configured_advisor, monkeypatch):
    state, pool, horizon = configured_advisor
    affected = int(pool.query("position == 'GKP'").iloc[0].element)
    captured = []

    async def solve(**kwargs):
        captured.append(kwargs)
        return {
            "recommended_plan": {
                "lineup": {
                    "formation": "3-4-3",
                    "captain": {"web_name": "Captain"},
                    "vice_captain": {},
                    "total_gameweek_expected_points": 50,
                },
                "transfers_in": [],
                "transfers_out": [],
                "hit_cost": 0,
            }
        }

    monkeypatch.setattr(analysis_service, "joint_plan", solve)
    for duration in ("1", "2"):
        result = asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=[affected], duration=duration)))
        frames = captured[-1]["horizon_projections"]
        for gw in frames:
            row = frames[gw].query("element == @affected").iloc[0]
            assert bool(row.simulation_unavailable) == (gw < 6 + int(duration))
            assert row.expected_points == (
                0 if gw < 6 + int(duration) else float(pool.query("element == @affected").iloc[0].expected_points)
            )
        assert result["excluded_gameweeks"] == list(range(6, 6 + int(duration)))
        assert captured[-1]["locked_out_ids"] == set()


def test_missing_duration_asks_before_search(configured_advisor, monkeypatch):
    state, pool, horizon = configured_advisor
    solve = AsyncMock()
    monkeypatch.setattr(analysis_service, "joint_plan", solve)
    result = asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=[int(pool.iloc[0].element)])))
    assert result["status"] == "needs_duration"
    solve.assert_not_called()


def test_empty_search_returns_review_option_not_invented_advice(configured_advisor, monkeypatch):
    _, pool, _ = configured_advisor
    monkeypatch.setattr(analysis_service, "joint_plan", AsyncMock(side_effect=ValueError("empty beam")))
    result = asyncio.run(
        api.post_contingency_panic(api.PanicRequest(ruled_out_ids=[int(pool.iloc[0].element)], duration="1"))
    )
    assert result["status"] == "needs_decision"
    assert "not a recommendation" in result["recommendation"]


def test_broken_xi_repaired_with_paid_transfer(configured_advisor, monkeypatch):
    import pandas as pd

    state, pool, _ = configured_advisor
    state.chips_remaining_set_1 = []
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    affected = pool.query("position == 'GKP'").element.astype(int).tolist()
    candidate = pool.query("position == 'GKP'").iloc[0].copy()
    candidate["element"] = 10001
    candidate["team"] = 100
    candidate["web_name"] = "Repair keeper"
    candidate["expected_points"] = 5.0
    expanded = pd.concat([pool, candidate.to_frame().T], ignore_index=True)
    monkeypatch.setattr(
        analysis_service, "projections", AsyncMock(return_value={g: expanded.copy() for g in range(6, 20)})
    )
    result = asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=affected, duration="1")))
    plan = result["joint_plan"]["recommended_plan"]
    assert result["status"] == "recalculated"
    assert plan["hit_cost"] == 4
    assert len(plan["lineup"]["starters"]) == 11
    assert 10001 in [p["element"] for p in plan["lineup"]["starters"]]
    assert not set(affected) & {p["element"] for p in plan["lineup"]["starters"]}


def test_repair_can_make_more_than_two_transfers(configured_advisor, monkeypatch):
    import pandas as pd

    state, pool, _ = configured_advisor
    state.chips_remaining_set_1 = []
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    affected = pool.query("position == 'DEF'").element.astype(int).tolist()
    extras = []
    for i in range(3):
        candidate = pool.query("position == 'DEF'").iloc[i].copy()
        candidate["element"] = 10001 + i
        candidate["team"] = 100 + i
        candidate["web_name"] = f"Repair defender {i}"
        candidate["expected_points"] = 5.0
        extras.append(candidate)
    expanded = pd.concat([pool, pd.DataFrame(extras)], ignore_index=True)
    monkeypatch.setattr(
        analysis_service, "projections", AsyncMock(return_value={g: expanded.copy() for g in range(6, 20)})
    )
    result = asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=affected, duration="1")))
    assert result["status"] == "needs_decision"
    assert result["above_cap_option"]["hit_cost"] == 12
    assert len(result["above_cap_option"]["transfers_in"]) == 3
    assert "above your limit" in result["recommendation"]


def test_no_plan_only_when_budget_cannot_buy_legal_keeper(configured_advisor, monkeypatch):
    import pandas as pd
    import pytest
    from fastapi import HTTPException

    state, pool, _ = configured_advisor
    state.chips_remaining_set_1 = []
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    affected = pool.query("position == 'GKP'").element.astype(int).tolist()
    candidate = pool.query("position == 'GKP'").iloc[0].copy()
    candidate["element"] = 10001
    candidate["team"] = 100
    candidate["value"] = 10000
    candidate["expected_points"] = 9.0
    expanded = pd.concat([pool, candidate.to_frame().T], ignore_index=True)
    monkeypatch.setattr(
        analysis_service, "projections", AsyncMock(return_value={g: expanded.copy() for g in range(6, 20)})
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=affected, duration="1")))
    assert error.value.status_code == 409
    assert "no recommendation published" in error.value.detail


def test_chip_escape_survives_missing_ordinary_baseline(configured_advisor, monkeypatch):
    import pandas as pd

    state, pool, _ = configured_advisor
    state.chips_remaining_set_1 = ["wildcard", "freehit"]
    monkeypatch.setattr(analysis_service, "league_context", AsyncMock(return_value=None))
    affected = pool.query("position == 'DEF'").element.astype(int).tolist()
    extras = []
    for i in range(5):
        row = pool.query("position == 'DEF'").iloc[i].copy()
        row["element"] = 10001 + i
        row["team"] = 100 + i
        row["web_name"] = f"Fit defender {i}"
        row["expected_points"] = 5.0
        extras.append(row)
    expanded = pd.concat([pool, pd.DataFrame(extras)], ignore_index=True)
    monkeypatch.setattr(
        analysis_service, "projections", AsyncMock(return_value={g: expanded.copy() for g in range(6, 20)})
    )
    result = asyncio.run(api.post_contingency_panic(api.PanicRequest(ruled_out_ids=affected, duration="1")))
    assert result["status"] == "recalculated"
    assert not result["joint_plan"]["normal_plan_feasible"]
    assert result["joint_plan"]["recommended_plan"]["hit_cost"] == 0
    assert len(result["joint_plan"]["recommended_plan"]["lineup"]["starters"]) == 11
    assert {c["chip"] for c in result["chip_options"]} == {"wildcard", "freehit"}
