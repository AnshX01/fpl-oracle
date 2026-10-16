import asyncio
import json
from pathlib import Path

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer as BeamSearchTransferOptimizer


async def compare_planners_current_squad():
    state = await manager_state_service.get_current_state()
    boot, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()

    target_gw = state.target_gw
    current_gw = state.current_gw
    bank = state.bank_tenths
    free_transfers = state.free_transfers

    projections = projection_engine.predict_multi_gameweeks(
        start_gw=target_gw, horizon=2, bootstrap=boot, fixtures=fixtures
    )
    target_pool = projections[target_gw]
    user_squad_df = state.to_squad_dataframe()

    for col in target_pool.columns:
        if col not in ("element", "selling_price", "purchase_price", "price_provenance"):
            user_squad_df[col] = user_squad_df["element"].map(target_pool.set_index("element")[col])

    print("=" * 80)
    print(f"C3 PLANNER COMPARISON: CURRENT SQUAD (Manager {state.manager_id}, GW{target_gw})")
    print(f"Bank: {bank / 10:.1f}m | Free Transfers: {free_transfers}")
    print("=" * 80)

    # 1. Beam Search Optimizer
    beam_opt = BeamSearchTransferOptimizer()
    res_beam = beam_opt.evaluate_transfer_options(
        current_squad_df=user_squad_df,
        player_pool_df=target_pool,
        bank=bank,
        free_transfers=free_transfers,
        horizon_projections=projections,
        current_gw=current_gw,
        target_gw=target_gw,
        risk_preference="balanced",
        horizon_len=2,
        num_mc_scenarios=5,
    )
    p_beam = res_beam["recommended_plan"]
    tin_beam = [p["web_name"] for p in p_beam.get("transfers_in", [])]
    tout_beam = [p["web_name"] for p in p_beam.get("transfers_out", [])]
    cap_beam = p_beam.get("lineup", {}).get("captain", {}).get("web_name")
    gain_beam = p_beam.get("horizon_gain_vs_roll")
    type_beam = p_beam.get("plan_type")

    print("\n--- 1. BEAM SEARCH OPTIMIZER OUTPUT ---")
    print(f"Plan Type:      {type_beam}")
    print(f"Transfers In:   {tin_beam}")
    print(f"Transfers Out:  {tout_beam}")
    print(f"Captain:        {cap_beam}")
    print(f"Horizon Gain:   {gain_beam}")
    print(f"Hit Cost:       {p_beam.get('hit_cost')}")

    # 2. Multi-Period MILP Optimizer
    res_milp = multiperiod_optimizer.evaluate_transfer_options(
        current_squad_df=user_squad_df,
        player_pool_df=target_pool,
        bank=bank,
        free_transfers=free_transfers,
        horizon_projections=projections,
        current_gw=current_gw,
        target_gw=target_gw,
        risk_preference="balanced",
        horizon_len=2,
        num_mc_scenarios=5,
    )
    p_milp = res_milp["recommended_plan"]
    tin_milp = [p["web_name"] for p in p_milp.get("transfers_in", [])]
    tout_milp = [p["web_name"] for p in p_milp.get("transfers_out", [])]
    cap_milp = p_milp.get("lineup", {}).get("captain", {}).get("web_name")
    gain_milp = p_milp.get("horizon_gain_vs_roll")
    type_milp = p_milp.get("plan_type")

    print("\n--- 2. MULTI-PERIOD MILP OPTIMIZER OUTPUT ---")
    print(f"Plan Type:      {type_milp}")
    print(f"Transfers In:   {tin_milp}")
    print(f"Transfers Out:  {tout_milp}")
    print(f"Captain:        {cap_milp}")
    print(f"Horizon Gain:   {gain_milp}")
    print(f"Hit Cost:       {p_milp.get('hit_cost')}")

    out_file = Path("reports/c3_planner_comparison.json")
    record = {
        "manager_id": state.manager_id,
        "target_gw": target_gw,
        "bank": bank,
        "free_transfers": free_transfers,
        "beam_search": {
            "plan_type": type_beam,
            "transfers_in": tin_beam,
            "transfers_out": tout_beam,
            "captain": cap_beam,
            "horizon_gain": gain_beam,
            "hit_cost": p_beam.get("hit_cost"),
        },
        "multiperiod_milp": {
            "plan_type": type_milp,
            "transfers_in": tin_milp,
            "transfers_out": tout_milp,
            "captain": cap_milp,
            "horizon_gain": gain_milp,
            "hit_cost": p_milp.get("hit_cost"),
        },
    }
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nSaved C3 comparison to {out_file}")

    if getattr(fpl_client, "_client", None) is not None and not fpl_client._client.is_closed:
        await fpl_client.aclose()


if __name__ == "__main__":
    asyncio.run(compare_planners_current_squad())
