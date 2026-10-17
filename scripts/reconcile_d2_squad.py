import asyncio
import json
from pathlib import Path

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import manager_state_service
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import multiperiod_optimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer as BeamSearchTransferOptimizer


async def run_d2_reconciliation():
    print("=" * 80)
    print("D2. RECONCILE ON THE CURRENT SQUAD (MANAGER 1085141, GW7)")
    print("=" * 80)

    # 1. State resolution
    # A: Raw API state from cached entry_history
    picks_obj, _ = await fpl_client.get_manager_picks(1085141, 6)
    history_obj, _ = await fpl_client.get_manager_history(1085141)
    boot, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()

    # Raw API state values
    entry_hist = getattr(picks_obj, "entry_history", {})
    api_bank_tenths = (
        getattr(entry_hist, "bank", 1)
        if hasattr(entry_hist, "bank")
        else (entry_hist.get("bank", 1) if isinstance(entry_hist, dict) else 1)
    )
    api_ft = 1  # 1 transfer made in GW6, no rollover -> 1 FT

    # Profile override state values
    profile_state = await manager_state_service.get_current_state()
    prof_bank_tenths = profile_state.bank_tenths  # 15 = 1.5m
    prof_ft = profile_state.free_transfers  # 2 FTs

    print("\n--- 1. DEADLINE AND STATE DISCREPANCY RESOLUTION ---")
    print("Target Deadline: Gameweek 7 (GW7)")
    print(
        f"Ground Truth API State:    Bank = {api_bank_tenths / 10.0:.1f}m ({api_bank_tenths} tenths) | Free Transfers = {api_ft}"
    )
    print(
        f"User Profile Override:    Bank = {prof_bank_tenths / 10.0:.1f}m ({prof_bank_tenths} tenths) | Free Transfers = {prof_ft}"
    )
    print("Resolution: The earlier brief used the ground-truth FPL API picks entry_history (bank=0.1m, 1 FT).")
    print(
        "The C3 run invoked manager_state_service.get_current_state(), which applied manual overrides from user_profile table (bank=1.5m, 2 FTs)."
    )

    # Build projections for GW7 horizon
    target_gw = 7
    current_gw = 6
    projections = projection_engine.predict_multi_gameweeks(
        start_gw=target_gw, horizon=2, bootstrap=boot, fixtures=fixtures
    )
    target_pool = projections[target_gw]
    squad_df = profile_state.to_squad_dataframe()
    for col in target_pool.columns:
        if col not in ("element", "selling_price", "purchase_price", "price_provenance"):
            squad_df[col] = squad_df["element"].map(target_pool.set_index("element")[col])

    beam_opt = BeamSearchTransferOptimizer()
    reconciliation_results = {}

    for label, bank_val, ft_val in [
        ("Ground_Truth_API (Bank 0.1m, 1 FT)", api_bank_tenths, api_ft),
        ("Profile_Override (Bank 1.5m, 2 FT)", prof_bank_tenths, prof_ft),
    ]:
        print(f"\n{'=' * 80}\nEVALUATION UNDER {label}\n{'=' * 80}")

        # A. Beam Search (Transfer options <= 2)
        beam_res = beam_opt.evaluate_transfer_options(
            current_squad_df=squad_df,
            player_pool_df=target_pool,
            bank=bank_val,
            free_transfers=ft_val,
            horizon_projections=projections,
            current_gw=current_gw,
            target_gw=target_gw,
            risk_preference="balanced",
            horizon_len=2,
            num_mc_scenarios=5,
        )
        p_beam = beam_res["recommended_plan"]
        tin_beam = sorted([p["web_name"] for p in p_beam.get("transfers_in", [])])
        tout_beam = sorted([p["web_name"] for p in p_beam.get("transfers_out", [])])
        cap_beam = p_beam.get("lineup", {}).get("captain", {}).get("web_name")
        gain_beam = p_beam.get("horizon_gain_vs_roll")
        type_beam = p_beam.get("plan_type")

        # B. MILP Optimizer (Unconstrained, includes chips)
        milp_res_full = multiperiod_optimizer.evaluate_transfer_options(
            current_squad_df=squad_df,
            player_pool_df=target_pool,
            bank=bank_val,
            free_transfers=ft_val,
            horizon_projections=projections,
            current_gw=current_gw,
            target_gw=target_gw,
            risk_preference="balanced",
            horizon_len=2,
            chips_available=["wildcard", "freehit", "bboost", "3xc"],
        )
        p_milp_full = milp_res_full["recommended_plan"]
        tin_milp_full = sorted([p["web_name"] for p in p_milp_full.get("transfers_in", [])])
        tout_milp_full = sorted([p["web_name"] for p in p_milp_full.get("transfers_out", [])])
        cap_milp_full = p_milp_full.get("lineup", {}).get("captain", {}).get("web_name")
        gain_milp_full = p_milp_full.get("horizon_gain_vs_roll")
        type_milp_full = p_milp_full.get("plan_type")

        # C. MILP Optimizer (Chips Forbidden, same action set as Beam transfer search)
        milp_res_nochip = multiperiod_optimizer.evaluate_transfer_options(
            current_squad_df=squad_df,
            player_pool_df=target_pool,
            bank=bank_val,
            free_transfers=ft_val,
            horizon_projections=projections,
            current_gw=current_gw,
            target_gw=target_gw,
            risk_preference="balanced",
            horizon_len=2,
            chips_available=[],
        )
        p_milp_nochip = milp_res_nochip["recommended_plan"]
        tin_milp_nochip = sorted([p["web_name"] for p in p_milp_nochip.get("transfers_in", [])])
        tout_milp_nochip = sorted([p["web_name"] for p in p_milp_nochip.get("transfers_out", [])])
        cap_milp_nochip = p_milp_nochip.get("lineup", {}).get("captain", {}).get("web_name")
        gain_milp_nochip = p_milp_nochip.get("horizon_gain_vs_roll")
        type_milp_nochip = p_milp_nochip.get("plan_type")

        print("1. Beam Search (<= 2 Transfers):")
        print(f"   Plan: {type_beam} | In: {tin_beam} | Out: {tout_beam} | C: {cap_beam} | Gain: {gain_beam}")
        print("2. MILP Optimizer (Unconstrained with Chips):")
        print(
            f"   Plan: {type_milp_full} | In count: {len(tin_milp_full)} | Out count: {len(tout_milp_full)} | C: {cap_milp_full} | Gain: {gain_milp_full}"
        )
        print("3. MILP Optimizer (Chips Forbidden, identical transfer action set):")
        print(
            f"   Plan: {type_milp_nochip} | In: {tin_milp_nochip} | Out: {tout_milp_nochip} | C: {cap_milp_nochip} | Gain: {gain_milp_nochip}"
        )

        reconciliation_results[label] = {
            "bank_tenths": bank_val,
            "free_transfers": ft_val,
            "beam_search": {
                "plan_type": type_beam,
                "transfers_in": tin_beam,
                "transfers_out": tout_beam,
                "captain": cap_beam,
                "horizon_gain": gain_beam,
            },
            "milp_unconstrained": {
                "plan_type": type_milp_full,
                "transfers_in": tin_milp_full,
                "transfers_out": tout_milp_full,
                "captain": cap_milp_full,
                "horizon_gain": gain_milp_full,
            },
            "milp_chips_forbidden": {
                "plan_type": type_milp_nochip,
                "transfers_in": tin_milp_nochip,
                "transfers_out": tout_milp_nochip,
                "captain": cap_milp_nochip,
                "horizon_gain": gain_milp_nochip,
            },
        }

    # Save artifact
    out_path = Path("reports/d2_squad_reconciliation.json")
    out_path.write_text(json.dumps(reconciliation_results, indent=2), encoding="utf-8")
    print(f"\nSaved machine artifact to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_d2_reconciliation())
