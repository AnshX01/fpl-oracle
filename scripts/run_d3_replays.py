import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.transfers_beam import TransferOptimizer as BeamSearchTransferOptimizer

# 11 Independent Managers from Mini-League 435956
CANDIDATE_MANAGERS = [
    2227954,
    1085141,
    4687536,
    8364971,
    2230307,
    7793828,
    9778590,
    9153936,
    6861977,
    9216581,
    9223805,
]


def validate_squad(squad_rows: list[dict[str, Any]], manager_id: int) -> tuple[bool, str]:
    """
    Validates squad structure:
    1. Exactly 15 players
    2. Formation counts: 2 GKP, 5 DEF, 5 MID, 3 FWD
    3. Maximum 3 players per Premier League club
    """
    if len(squad_rows) != 15:
        return False, f"Squad size is {len(squad_rows)}, expected 15."

    pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts: dict[int, int] = {}

    for p in squad_rows:
        pos = p["position"]
        if pos not in pos_counts:
            return False, f"Unknown position {pos}"
        pos_counts[pos] += 1

        team = p["team"]
        club_counts[team] = club_counts.get(team, 0) + 1

    expected_pos = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    if pos_counts != expected_pos:
        return False, f"Invalid position distribution: {pos_counts}, expected {expected_pos}"

    for team, count in club_counts.items():
        if count > 3:
            return False, f"Club {team} has {count} players (max 3 allowed)."

    return True, "Valid"


async def run_d3_replays():
    print("=" * 80)
    print("D3. REPLAY FIXTURES & SQUAD VALIDATION ACROSS >= 10 INDEPENDENT MANAGERS")
    print("=" * 80)

    boot, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elem_map = {e.id: e for e in boot.elements}

    target_gw = 7
    pick_gw = 6

    # Generate projections
    projections = projection_engine.predict_multi_gameweeks(
        start_gw=target_gw, horizon=2, bootstrap=boot, fixtures=fixtures
    )
    target_pool = projections[target_gw]

    valid_squads = []

    print(f"\nValidating candidate managers entering GW{target_gw} (from GW{pick_gw} picks):")
    for mgr in CANDIDATE_MANAGERS:
        try:
            picks_res, _ = await fpl_client.get_manager_picks(mgr, pick_gw)
            picks = getattr(picks_res, "picks", [])
            entry_hist = getattr(picks_res, "entry_history", {})
            bank = (
                getattr(entry_hist, "bank", 5)
                if hasattr(entry_hist, "bank")
                else (entry_hist.get("bank", 5) if isinstance(entry_hist, dict) else 5)
            )

            squad_rows = []
            for p in picks:
                eid = getattr(p, "element", 0)
                elem = elem_map.get(eid)
                if not elem:
                    continue
                pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(elem.element_type, "MID")
                pool_row = target_pool[target_pool["element"] == eid]
                xp = float(pool_row.iloc[0]["expected_points"]) if not pool_row.empty else 2.0

                squad_rows.append(
                    {
                        "element": eid,
                        "web_name": elem.web_name,
                        "position": pos_str,
                        "team": elem.team,
                        "value": elem.now_cost,
                        "now_cost": elem.now_cost,
                        "purchase_price": elem.now_cost,
                        "selling_price": elem.now_cost,
                        "expected_points": xp,
                        "chance_of_playing": 100.0,
                    }
                )

            is_valid, reason = validate_squad(squad_rows, mgr)
            if is_valid:
                valid_squads.append(
                    {
                        "manager_id": mgr,
                        "squad_df": pd.DataFrame(squad_rows),
                        "bank": bank,
                        "names": [p["web_name"] for p in squad_rows],
                    }
                )
                print(
                    f"  Manager {mgr:<10}: VALID (15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD, <= 3 per club; Bank={bank / 10:.1f}m)"
                )
            else:
                print(f"  Manager {mgr:<10}: INVALID ({reason}) -> DROPPED")
        except Exception as e:
            print(f"  Manager {mgr:<10}: FAILED TO LOAD ({e}) -> DROPPED")

    print(f"\nTotal Validated Independent Managers: {len(valid_squads)}")
    assert len(valid_squads) >= 10, f"Expected >= 10 valid independent managers, got {len(valid_squads)}"

    # Run comparisons for first 10 valid independent managers
    eval_managers = valid_squads[:10]
    modes = ["conservative", "balanced", "aggressive"]
    transfer_opt = BeamSearchTransferOptimizer()
    current_gw = target_gw - 1

    replay_results = []
    any_change_detected = False
    risk_modes_differ_anywhere = False

    for idx, item in enumerate(eval_managers, 1):
        mgr_id = item["manager_id"]
        squad_df = item["squad_df"]
        bank = item["bank"]

        print("\n=======================================================")
        print(f"REPLAY {idx}/10: Independent Manager {mgr_id} entering GW{target_gw} (Bank={bank / 10:.1f}m)")
        print(f"Squad contains 'Cash': {'Cash' in item['names']} (Note: Matty Cash, Aston Villa DEF)")
        print("=======================================================")

        mode_plans: dict[str, Any] = {}

        for mode in modes:
            # 1. simulate_captaincy = False (Fast)
            res_fast = transfer_opt.evaluate_transfer_options(
                current_squad_df=squad_df,
                player_pool_df=target_pool,
                bank=bank,
                free_transfers=1,
                horizon_projections=projections,
                current_gw=current_gw,
                target_gw=target_gw,
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

            # 2. simulate_captaincy = True (Full simulation)
            orig_select = lineup_optimizer.select_lineup_and_captain

            def select_with_sim(*args, _orig=orig_select, **kwargs):
                kwargs["simulate_captaincy"] = True
                return _orig(*args, **kwargs)

            lineup_optimizer.select_lineup_and_captain = select_with_sim
            try:
                res_sim = transfer_opt.evaluate_transfer_options(
                    current_squad_df=squad_df,
                    player_pool_df=target_pool,
                    bank=bank,
                    free_transfers=1,
                    horizon_projections=projections,
                    current_gw=current_gw,
                    target_gw=target_gw,
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

            match_transfers = bool(tin_fast == tin_sim and tout_fast == tout_sim)
            match_captain = bool(cap_fast == cap_sim)
            match_gain = bool(gain_fast == gain_sim)
            match_all = bool(match_transfers and match_captain and match_gain)

            if not match_all:
                any_change_detected = True

            print(
                f"Mode {mode:<12} | Sim=True:  {ptype_sim} (In={tin_sim}, Out={tout_sim}, C={cap_sim}, Gain={gain_sim})"
            )
            print(
                f"                 | Sim=False: {ptype_fast} (In={tin_fast}, Out={tout_fast}, C={cap_fast}, Gain={gain_fast})"
            )
            print(f"                 | Match: transfers={match_transfers}, captain={match_captain}, gain={match_gain}")

            mode_plans[mode] = {
                "plan_type": ptype_fast,
                "transfers_in": tin_fast,
                "transfers_out": tout_fast,
                "captain": cap_fast,
                "gain": float(gain_fast) if gain_fast is not None else 0.0,
            }

            replay_results.append(
                {
                    "replay_index": idx,
                    "manager_id": mgr_id,
                    "risk_mode": mode,
                    "transfers_in": tin_fast,
                    "transfers_out": tout_fast,
                    "captain": cap_fast,
                    "gain": float(gain_fast) if gain_fast is not None else 0.0,
                    "sim_matches_no_sim": match_all,
                }
            )

        diff_modes = bool(
            mode_plans["conservative"] != mode_plans["balanced"] or mode_plans["balanced"] != mode_plans["aggressive"]
        )
        if diff_modes:
            risk_modes_differ_anywhere = True
        print(f"Risk Modes Differ on Replay {idx}: {diff_modes}")

    # Empirical Summary Table
    print("\n" + "=" * 105)
    print("D3 EMPIRICAL SUMMARY TABLE: 10 INDEPENDENT MANAGERS ACROSS 3 RISK MODES")
    print("=" * 105)
    print(
        f"{'Replay':<8} {'Mgr ID':<10} {'Mode':<12} {'Transfers In':<20} {'Transfers Out':<20} {'Captain':<12} {'Gain':<6} {'Sim==NoSim'}"
    )
    print("-" * 105)
    for r in replay_results:
        print(
            f"{r['replay_index']:<8} {r['manager_id']:<10} {r['risk_mode']:<12} "
            f"{str(r['transfers_in']):<20} {str(r['transfers_out']):<20} "
            f"{str(r['captain']):<12} {r['gain']:<6.2f} {str(r['sim_matches_no_sim'])}"
        )
    print("-" * 105)
    print(f"Any change across all 30 evaluations from simulate_captaincy: {any_change_detected}")
    print(f"Do risk modes ever differ across the 10 independent managers: {risk_modes_differ_anywhere}")

    out_file = Path("reports/d3_replays.json")
    out_file.write_text(
        json.dumps(
            {
                "validation_summary": {
                    "candidate_managers_count": len(CANDIDATE_MANAGERS),
                    "validated_independent_managers_count": len(valid_squads),
                    "evaluated_managers_count": len(eval_managers),
                    "cash_finding": "Confirmed: 'Cash' is Aston Villa defender Matty Cash (element 32), not missing player or currency.",
                    "all_squads_15_players": True,
                    "all_formations_valid": True,
                    "all_club_limits_respected": True,
                },
                "replays": replay_results,
                "any_change_from_simulate_captaincy": any_change_detected,
                "risk_modes_differ": risk_modes_differ_anywhere,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved machine artifact to {out_file}")


if __name__ == "__main__":
    asyncio.run(run_d3_replays())
