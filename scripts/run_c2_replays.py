import asyncio
import json
import sqlite3
from pathlib import Path

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import ManagerStateService
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.transfers import transfer_optimizer


async def load_replays():
    conn = sqlite3.connect("data/fpl_oracle.db")
    cursor = conn.cursor()

    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elements_map = {e.id: e for e in bootstrap.elements}

    # Available picks in cache
    cases = [
        ("1085141", 5, 6),  # Replay 1
        ("1085141", 6, 7),  # Replay 2
        ("2227954", 4, 5),  # Replay 3
        ("2227954", 5, 6),  # Replay 4
        ("2227954", 6, 7),  # Replay 5
        ("4687536", 4, 5),  # Replay 6
        ("4687536", 5, 6),  # Replay 7
        ("4687536", 6, 7),  # Replay 8
        ("8364971", 4, 5),  # Replay 9
        ("8364971", 5, 6),  # Replay 10
    ]

    replays = []
    for mgr, pick_gw, target_gw in cases:
        key = f"entry:{mgr}:event:{pick_gw}:picks"
        row = cursor.execute("SELECT data_json FROM api_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            continue
        data = json.loads(row[0])
        picks = data.get("picks", [])
        if len(picks) < 15:
            continue
        entry_hist = data.get("entry_history", {})
        bank = entry_hist.get("bank", 5)  # in tenths
        replays.append(
            {
                "manager_id": mgr,
                "pick_gw": pick_gw,
                "target_gw": target_gw,
                "bank": bank,
                "picks": picks,
            }
        )

    print(f"Loaded {len(replays)} real squad replays from api_cache.")
    return replays, bootstrap, fixtures, elements_map


async def run_all_replays():
    replays, bootstrap, fixtures, elements_map = await load_replays()
    assert len(replays) >= 10, f"Expected >= 10 replays, got {len(replays)}"

    results = []
    modes = ["conservative", "balanced", "aggressive"]

    for idx, rep in enumerate(replays, 1):
        mgr = rep["manager_id"]
        tgt_gw = rep["target_gw"]
        bank = rep["bank"]
        print("\n=======================================================")
        print(f"REPLAY {idx}/10: Manager {mgr} entering GW{tgt_gw} (Bank={bank / 10:.1f}m)")
        print("=======================================================")

        # Generate projections for target_gw and horizon
        projs = projection_engine.predict_multi_gameweeks(
            start_gw=tgt_gw, horizon=2, bootstrap=bootstrap, fixtures=fixtures
        )
        target_pool = projs[tgt_gw]

        # Reconstruct squad dataframe
        squad_rows = []
        for p in rep["picks"]:
            eid = p["element"]
            elem = elements_map.get(eid)
            if not elem:
                continue
            pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(elem.element_type, "MID")
            now_cost = elem.now_cost
            buy_price = p.get("purchase_price", now_cost)
            sell_price = p.get("selling_price", ManagerStateService.calculate_selling_price(buy_price, now_cost))

            p_row = target_pool[target_pool["element"] == eid]
            xp = float(p_row["expected_points"].iloc[0]) if not p_row.empty else 0.0

            squad_rows.append(
                {
                    "element": eid,
                    "web_name": elem.web_name,
                    "position": pos_str,
                    "team": elem.team,
                    "value": now_cost,
                    "purchase_price": buy_price,
                    "selling_price": sell_price,
                    "expected_points": xp,
                    "chance_of_playing": 1.0,
                    "chance_of_playing_next_round": 100,
                }
            )
        squad_df = pd.DataFrame(squad_rows)

        replay_record = {
            "replay_index": idx,
            "manager_id": mgr,
            "target_gw": tgt_gw,
            "modes": {},
        }

        mode_plans = {}

        for mode in modes:
            # 1. simulate_captaincy = False (Current HEAD)
            res_fast = transfer_optimizer.evaluate_transfer_options(
                current_squad_df=squad_df,
                player_pool_df=target_pool,
                bank=bank,
                free_transfers=1,
                horizon_projections=projs,
                current_gw=tgt_gw - 1,
                target_gw=tgt_gw,
                risk_preference=mode,
                horizon_len=2,
                num_mc_scenarios=5,
            )
            p_fast = res_fast["recommended_plan"]
            tin_fast = sorted([p["web_name"] for p in p_fast.get("transfers_in", [])])
            tout_fast = sorted([p["web_name"] for p in p_fast.get("transfers_out", [])])
            cap_fast = p_fast.get("lineup", {}).get("captain", {}).get("web_name")
            gain_fast = p_fast.get("horizon_gain_vs_roll")
            ptype_fast = p_fast.get("plan_type")

            # 2. simulate_captaincy = True (Before)
            orig_select = lineup_optimizer.select_lineup_and_captain

            def select_with_sim(*args, _orig=orig_select, **kwargs):
                kwargs["simulate_captaincy"] = True
                return _orig(*args, **kwargs)

            lineup_optimizer.select_lineup_and_captain = select_with_sim
            try:
                res_sim = transfer_optimizer.evaluate_transfer_options(
                    current_squad_df=squad_df,
                    player_pool_df=target_pool,
                    bank=bank,
                    free_transfers=1,
                    horizon_projections=projs,
                    current_gw=tgt_gw - 1,
                    target_gw=tgt_gw,
                    risk_preference=mode,
                    horizon_len=2,
                    num_mc_scenarios=5,
                )
            finally:
                lineup_optimizer.select_lineup_and_captain = orig_select

            p_sim = res_sim["recommended_plan"]
            tin_sim = sorted([p["web_name"] for p in p_sim.get("transfers_in", [])])
            tout_sim = sorted([p["web_name"] for p in p_sim.get("transfers_out", [])])
            cap_sim = p_sim.get("lineup", {}).get("captain", {}).get("web_name")
            gain_sim = p_sim.get("horizon_gain_vs_roll")
            ptype_sim = p_sim.get("plan_type")

            match_transfers = bool((tin_fast == tin_sim) and (tout_fast == tout_sim) and (ptype_fast == ptype_sim))
            match_captain = bool(cap_fast == cap_sim)
            match_gain = bool(gain_fast == gain_sim)

            print(
                f"Mode {mode:<12} | Sim=True: {ptype_sim} (In={tin_sim}, Out={tout_sim}, C={cap_sim}, Gain={gain_sim})"
            )
            print(
                f"                 | Sim=False:{ptype_fast} (In={tin_fast}, Out={tout_fast}, C={cap_fast}, Gain={gain_fast})"
            )
            print(f"                 | Match: transfers={match_transfers}, captain={match_captain}, gain={match_gain}")

            replay_record["modes"][mode] = {
                "plan_type_before": ptype_sim,
                "plan_type_after": ptype_fast,
                "transfers_in_before": tin_sim,
                "transfers_in_after": tin_fast,
                "transfers_out_before": tout_sim,
                "transfers_out_after": tout_fast,
                "captain_before": cap_sim,
                "captain_after": cap_fast,
                "gain_before": float(gain_sim) if gain_sim is not None else None,
                "gain_after": float(gain_fast) if gain_fast is not None else None,
                "transfers_match": match_transfers,
                "captain_match": match_captain,
                "gain_match": match_gain,
            }

            mode_plans[mode] = {
                "transfers_in": tin_fast,
                "transfers_out": tout_fast,
                "captain": cap_fast,
                "gain": float(gain_fast) if gain_fast is not None else None,
                "plan_type": ptype_fast,
            }

        # Check if risk modes differ
        diff_modes = bool(
            mode_plans["conservative"] != mode_plans["balanced"] or mode_plans["balanced"] != mode_plans["aggressive"]
        )
        replay_record["risk_modes_differ"] = diff_modes
        print(f"Risk Modes Differ on Replay {idx}: {diff_modes}")
        results.append(replay_record)

    out_json = Path("reports/c2_simulate_captaincy_replays.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved machine-readable replay comparisons to: {out_json}")

    # Summary table
    print("\n" + "=" * 80)
    print("C2 EMPIRICAL SUMMARY TABLE ACROSS 10 REPLAYS & 3 RISK MODES")
    print("=" * 80)
    print(
        f"{'Replay':<8} {'Mgr ID':<10} {'GW':<4} {'Mode':<12} {'Transfers In':<20} {'Transfers Out':<20} {'Captain':<12} {'Gain':<6} {'Sim==NoSim'}"
    )
    print("-" * 105)
    any_diff_sim = False
    any_diff_modes = False

    for r in results:
        idx = r["replay_index"]
        mgr = r["manager_id"]
        gw = r["target_gw"]
        if r["risk_modes_differ"]:
            any_diff_modes = True
        for m, d in r["modes"].items():
            tin = ", ".join(d["transfers_in_after"]) or "None (Roll)"
            tout = ", ".join(d["transfers_out_after"]) or "None (Roll)"
            cap = d["captain_after"]
            gain = d["gain_after"]
            match = d["transfers_match"] and d["captain_match"] and d["gain_match"]
            if not match:
                any_diff_sim = True
            print(f"{idx:<8} {mgr:<10} GW{gw:<2} {m:<12} {tin:<20} {tout:<20} {cap:<12} {gain:<6.2f} {str(match)}")

    print("-" * 105)
    print(f"Any change across all 30 evaluations (10 replays x 3 modes) from simulate_captaincy: {any_diff_sim}")
    print(f"Do risk modes ever differ across the 10 replays: {any_diff_modes}")

    if getattr(fpl_client, "_client", None) is not None and not fpl_client._client.is_closed:
        await fpl_client.aclose()


if __name__ == "__main__":
    asyncio.run(run_all_replays())
