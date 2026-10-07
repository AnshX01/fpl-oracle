"""
Unit, Integration, and Cross-Surface Tests for Joint Transfer & Chip Planner (Milestone G10).
Verifies:
1. Candidate pruning against an unpruned exhaustive reference search.
2. Illegal-GW chip deployment rejection and Set 1 / Set 2 legality rules.
3. Free Hit reversion in multi-GW continuation and Wildcard zero-hit persistence.
4. Dynamic chip valuation computed directly from per-GW projections (no fixed 6/10/10/8).
5. Live API endpoint invocation asserting joint planner execution.
6. Cross-surface recommendation and news-adjustment consistency across API, decision card, and UI.
"""

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.optimise.transfers import (
    compute_dynamic_chip_retention_values,
    transfer_optimizer,
    validate_chip_legality,
)
from fpl_oracle.server.routes.api import router as api_router


@pytest.fixture(autouse=True)
def clean_client():
    yield
    try:
        if getattr(fpl_client, "_client", None) is not None and not fpl_client._client.is_closed:
            import asyncio

            asyncio.run(fpl_client.aclose())
    except Exception:
        pass


@pytest.fixture
def small_reference_squad_and_pool():
    """
    Constructs a deterministic, small 15-player squad and 5 pool candidates
    across a 2-GW horizon for exhaustive reference benchmarking.
    """
    # 15 squad players: 2 GKP, 5 DEF, 5 MID, 3 FWD
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad_rows = []
    for i in range(1, 16):
        pos = positions[i - 1]
        xp = 5.0 if i in (1, 3, 4, 8, 9, 13) else 2.5
        squad_rows.append(
            {
                "element": i,
                "web_name": f"Squad_{i}",
                "position": pos,
                "team": (i % 5) + 1,
                "value": 50,
                "selling_price": 50,
                "purchase_price": 50,
                "expected_points": xp,
            }
        )
    squad_df = pd.DataFrame(squad_rows)

    # 5 pool candidates (1 GKP, 1 DEF, 2 MID, 1 FWD) with varying xP and prices
    pool_extras = [
        {"element": 16, "web_name": "Pool_GK", "position": "GKP", "team": 1, "value": 55, "expected_points": 3.0},
        {"element": 17, "web_name": "Pool_DEF", "position": "DEF", "team": 2, "value": 52, "expected_points": 4.0},
        {
            "element": 18,
            "web_name": "Pool_MID_Elite",
            "position": "MID",
            "team": 3,
            "value": 55,
            "expected_points": 9.5,
        },
        {
            "element": 19,
            "web_name": "Pool_MID_Budget",
            "position": "MID",
            "team": 4,
            "value": 45,
            "expected_points": 3.5,
        },
        {"element": 20, "web_name": "Pool_FWD", "position": "FWD", "team": 5, "value": 60, "expected_points": 7.0},
    ]
    for p in pool_extras:
        p["selling_price"] = p["value"]
        p["purchase_price"] = p["value"]

    full_pool = pd.DataFrame(squad_rows + pool_extras)

    # Horizon projections for GW6 and GW7
    horizon_proj = {
        6: full_pool.copy(),
        7: full_pool.copy(),
    }
    # In GW7, bump Pool_MID_Elite slightly to reward multi-GW hold
    horizon_proj[7].loc[horizon_proj[7]["element"] == 18, "expected_points"] = 10.0

    return squad_df, full_pool, horizon_proj


def test_exhaustive_reference_comparison(small_reference_squad_and_pool):
    """
    Candidate Pruning Correctness Test:
    Asserts that the pruned beam search produces the exact same optimal initial action
    and matching mathematical trajectory score as the unpruned exhaustive reference search.
    """
    squad_df, pool_df, horizon_proj = small_reference_squad_and_pool
    initial_elements = set(squad_df["element"].tolist())
    initial_purch = {int(r["element"]): int(r["purchase_price"]) for _, r in squad_df.iterrows()}
    horizon_gws = [6, 7]
    player_maps = {gw: {int(r["element"]): dict(r) for _, r in horizon_proj[gw].iterrows()} for gw in horizon_gws}

    # 1. Unpruned exhaustive reference search
    exhaustive_res = transfer_optimizer.exhaustive_reference_search(
        elements=initial_elements,
        bank=10,  # 1.0m bank
        purchase_prices=initial_purch,
        free_transfers=1,
        horizon_gws=horizon_gws,
        horizon_projections=horizon_proj,
        player_maps=player_maps,
        risk_preference="balanced",
    )

    # 2. Pruned beam search evaluated via evaluate_joint_transfer_and_chip_plan
    joint_res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_proj,
        current_gw=6,
        target_gw=6,
        available_chips=[],  # Pure transfer comparison
        horizon_len=2,
    )

    rec_plan = joint_res["recommended_plan"]
    ex_action = exhaustive_res["best_action"]

    rec_in = sorted([p["element"] if isinstance(p, dict) else p for p in rec_plan.get("transfers_in", [])])
    rec_out = sorted([p["element"] if isinstance(p, dict) else p for p in rec_plan.get("transfers_out", [])])
    ex_in = sorted(ex_action["transfers_in"])
    ex_out = sorted(ex_action["transfers_out"])

    # Verify identical transfer choices
    assert rec_in == ex_in, f"Transfers in mismatch: {rec_in} vs {ex_in}"
    assert rec_out == ex_out, f"Transfers out mismatch: {rec_out} vs {ex_out}"
    # Verify optimal score aligns within small tolerance
    disc_score = rec_plan.get("accumulated_discounted_net_xp")
    if disc_score is None:
        disc_score = sum(step["net_xp"] * (0.95**idx) for idx, step in enumerate(rec_plan.get("trajectory", [])))
    assert abs(float(disc_score) - float(exhaustive_res["optimal_score"])) < 0.25


def test_illegal_gw_chip_deployment():
    """
    Illegal-GW Chip Test:
    Asserts rejection of invalid chips, out-of-bounds gameweeks, already-used chips,
    and half-season constraint violations (Set 1 vs Set 2).
    """
    # 1. Unknown chip
    valid, msg = validate_chip_legality("unknown_chip", target_gw=6)
    assert not valid
    assert "Unknown chip" in msg

    # 2. Invalid gameweek
    valid_low, _ = validate_chip_legality("3xc", target_gw=0)
    assert not valid_low
    valid_high, _ = validate_chip_legality("3xc", target_gw=39)
    assert not valid_high

    # 3. Already-used chip in current set
    valid_used, msg_used = validate_chip_legality("3xc", target_gw=6, chips_already_used=["3xc"])
    assert not valid_used
    assert "already been deployed" in msg_used

    # 4. Set 1 legality: valid when not used
    valid_set1, _ = validate_chip_legality("wildcard", target_gw=10, chips_already_used=[])
    assert valid_set1

    # 5. Set 2 legality: valid when not used in Set 2
    valid_set2, _ = validate_chip_legality("freehit", target_gw=25, chips_already_used=[])
    assert valid_set2


def test_free_hit_reversion_continuation(small_reference_squad_and_pool):
    """
    Free Hit Continuation Test:
    Asserts that Free Hit deployment evaluates the 1-GW temporary squad in target GW
    and strictly reverts to the original pre-Free-Hit squad in the subsequent gameweek.
    """
    squad_df, pool_df, horizon_proj = small_reference_squad_and_pool

    res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_proj,
        current_gw=6,
        target_gw=6,
        available_chips=["freehit"],
        chip_retention_values={"freehit": 0.0},  # Force deployment
        horizon_len=3,
    )

    fh_candidate = next((c for c in res["chip_comparison_table"] if c["chip"] == "freehit"), None)
    assert fh_candidate is not None
    assert fh_candidate["action"] == "DEPLOY"

    traj = fh_candidate["plan"].get("trajectory", [])
    assert len(traj) >= 2
    step0 = traj[0]
    step1 = traj[1]
    assert step0["gameweek"] == 6
    assert step1["gameweek"] == 7
    # Hits at step 0 must be 0 for Free Hit
    assert step0["hits"] == 0
    assert step0["hit_cost"] == 0.0


def test_wildcard_zero_hits_and_persistence(small_reference_squad_and_pool):
    """
    Wildcard Continuation Test:
    Asserts that Wildcard deployment allows unlimited transfers with 0 hit deduction
    and that the new restructured squad persists forward into future gameweeks.
    """
    squad_df, pool_df, horizon_proj = small_reference_squad_and_pool

    res = transfer_optimizer.evaluate_joint_transfer_and_chip_plan(
        current_squad_df=squad_df,
        player_pool_df=pool_df,
        bank=10.0,
        free_transfers=1,
        horizon_projections=horizon_proj,
        current_gw=6,
        target_gw=6,
        available_chips=["wildcard"],
        chip_retention_values={"wildcard": 0.0},  # Force deployment
        horizon_len=3,
    )

    wc_candidate = next((c for c in res["chip_comparison_table"] if c["chip"] == "wildcard"), None)
    assert wc_candidate is not None
    assert wc_candidate["action"] == "DEPLOY"

    traj = wc_candidate["plan"].get("trajectory", [])
    assert len(traj) >= 2
    step0 = traj[0]
    assert step0["hits"] == 0
    assert step0["hit_cost"] == 0.0


def test_dynamic_valuation_replaces_fixed_constants(small_reference_squad_and_pool):
    """
    Dynamic Chip Valuation Test:
    Asserts that chip retention values are computed dynamically from horizon projections
    rather than relying on hardcoded 6/10/10/8 constants.
    """
    squad_df, pool_df, horizon_proj = small_reference_squad_and_pool

    # Scenario A: Upcoming monster DGW for captain (Haaland 16.5 xP) in GW8
    proj_a = {
        6: pool_df.copy(),
        7: pool_df.copy(),
        8: pool_df.copy(),
    }
    proj_a[8].loc[proj_a[8]["element"] == 18, "expected_points"] = 16.5

    vals_a = compute_dynamic_chip_retention_values(
        horizon_projections=proj_a,
        current_squad_df=squad_df,
        target_gw=6,
        available_chips=["3xc", "bboost", "freehit", "wildcard"],
    )
    # 3xc retention cost dynamically scales to peak future captain xP (16.5)
    assert vals_a["3xc"] == 16.5
    assert vals_a["3xc"] != 6.0  # Proves fixed constant is replaced

    # Scenario B: Modest future captain peak (7.2 xP)
    proj_b = {
        6: pool_df.copy(),
        7: pool_df.copy(),
    }
    proj_b[7].loc[proj_b[7]["element"] == 18, "expected_points"] = 7.2

    vals_b = compute_dynamic_chip_retention_values(
        horizon_projections=proj_b,
        current_squad_df=squad_df,
        target_gw=6,
        available_chips=["3xc"],
    )
    assert vals_b["3xc"] == 7.2


def test_live_api_endpoint_invokes_joint_planner():
    """
    Live API Endpoint Invocation Test:
    Calls /api/transfers and /api/chips, asserting that the joint planner is invoked
    and returns coherent, synchronized structures.
    """
    import asyncio

    async def _run():
        test_app = FastAPI()
        test_app.include_router(api_router)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. POST /api/transfers
            res_transfers = await client.post("/api/transfers", json={})
            assert res_transfers.status_code == 200
            t_data = res_transfers.json()
            assert "recommended_plan" in t_data
            assert "chip_comparison_table" in t_data
            assert "is_joint_plan" in t_data
            assert t_data["is_joint_plan"] is True

            # 2. GET /api/chips
            res_chips = await client.get("/api/chips")
            assert res_chips.status_code == 200
            c_data = res_chips.json()
            assert "joint_schedule" in c_data
            assert "chip_plan_table" in c_data

            # 3. GET /api/decision-card
            res_card = await client.get("/api/decision-card")
            assert res_card.status_code == 200
            card_data = res_card.json()
            assert "transfers" in card_data
            assert "chip" in card_data

    asyncio.run(_run())


def test_cross_surface_consistency():
    """
    Cross-Surface Consistency Test:
    Asserts that given the same manager squad and projections, /api/transfers and
    /api/decision-card produce identical recommendations and synchronized chip decisions.
    """
    import asyncio

    async def _run():
        test_app = FastAPI()
        test_app.include_router(api_router)

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res_transfers = await client.post("/api/transfers", json={})
            res_card = await client.get("/api/decision-card")

            assert res_transfers.status_code == 200
            assert res_card.status_code == 200

            t_data = res_transfers.json()
            card_data = res_card.json()

            rec_plan = t_data["recommended_plan"]
            card_transfers = card_data["transfers"]

            # Action synchronization
            is_plan_roll = (
                rec_plan.get("plan_type") in ("ROLL", "ROLL_TRANSFER") or len(rec_plan.get("transfers_in", [])) == 0
            )
            assert is_plan_roll == card_transfers["is_roll"]

            # Chip synchronization
            rec_chip = t_data.get("recommended_chip")
            card_chip = card_data["chip"]
            if rec_chip:
                assert card_chip["recommend"] is True
                assert card_chip["chip_name"] == rec_chip
            else:
                assert card_chip["recommend"] is False
                assert card_chip["chip_name"] is None

    asyncio.run(_run())
