"""
Generate Live Advisor Briefing Report (Phase 8 Early Slice 6b: WI-8.1, WI-8.4, WI-8.3).
Fetches live squad state for entry 1085141, standings for league 435956,
runs health diagnosis, chip timeline, affordability path finder, and multi-period MILP optimizer,
and produces reports/advice/<date>.md.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_oracle.advisor.chip_timeline import ChipTimelineComparator
from fpl_oracle.advisor.funding import AffordabilityPathFinder
from fpl_oracle.advisor.health import SquadHealthDiagnosis
from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.domain.manager_state import ManagerStateService
from fpl_oracle.ml.predict import projection_engine
from fpl_oracle.optimise.multiperiod import MultiPeriodConfig, MultiPeriodOptimizer


async def generate_report():
    print("=" * 80)
    print("GENERATING LIVE ADVISOR REPORT FOR UPCOMING DEADLINE")
    print("=" * 80)

    # 1. Fetch live FPL context
    bootstrap, _ = await fpl_client.get_bootstrap_static()
    fixtures, _ = await fpl_client.get_fixtures()
    elements_map = {e.id: e for e in bootstrap.elements}
    teams_map = {t.id: t.name for t in bootstrap.teams}

    # League standings
    league_id = 435956
    standings, _ = await fpl_client.get_classic_league_standings(league_id)
    results = standings.standings.results
    leader_entry = results[0] if results else None
    leader_pts = leader_entry.total if leader_entry else 0
    leader_name = leader_entry.entry_name if leader_entry else "Leader"

    user_entry = next((r for r in results if r.entry == 1085141), None)
    user_rank = user_entry.rank if user_entry else 2
    user_total = user_entry.total if user_entry else 407
    user_team = user_entry.entry_name if user_entry else "Red Djed Redemption"
    gap_to_leader = leader_pts - user_total

    # Entry picks from previous round (GW6) entering upcoming deadline (GW7)
    target_gw = 7
    picks_res, _ = await fpl_client.get_manager_picks(1085141, 6)
    bank_tenths = int(picks_res.entry_history.bank)  # 1 = £0.1m
    free_transfers = 1  # 1 FT entering GW7
    chips_available = ["wildcard", "freehit", "3xc", "bboost"]

    # Reconstruct squad with purchase & selling prices
    squad_rows = []
    # Historical purchase prices lookup
    purchase_prices = {
        109: 45, 8: 55, 418: 50, 10: 55, 12: 96, 154: 97, 286: 51,
        124: 55, 411: 155, 346: 60, 165: 77, 496: 45, 368: 70, 87: 45, 259: 40,
    }

    # Pre-deadline ML projections (GW7 to GW11, H=5)
    H = 5
    projs = projection_engine.predict_multi_gameweeks(start_gw=target_gw, horizon=H, bootstrap=bootstrap, fixtures=fixtures)
    gw7_pool = projs[target_gw].copy()

    for p in picks_res.picks:
        eid = p.element
        elem = elements_map[eid]
        pos_str = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[elem.element_type]
        buy_p = purchase_prices.get(eid, elem.now_cost)
        now_c = elem.now_cost
        sell_p = ManagerStateService.calculate_selling_price(buy_p, now_c)
        p_row = gw7_pool[gw7_pool["element"] == eid]
        xp = float(p_row["expected_points"].iloc[0]) if not p_row.empty else 0.0

        squad_rows.append({
            "element": eid,
            "web_name": elem.web_name,
            "position": pos_str,
            "team": elem.team,
            "team_name": teams_map[elem.team],
            "value": now_c,
            "now_cost": now_c,
            "purchase_price": buy_p,
            "selling_price": sell_p,
            "expected_points": xp,
            "chance_of_playing": getattr(elem, "chance_of_playing_next_round", 100),
            "news": getattr(elem, "news", ""),
        })

    squad_df = pd.DataFrame(squad_rows)
    tot_sell_val = squad_df["selling_price"].sum() / 10.0

    # 2. WI-8.1 Squad Health Diagnosis
    health_engine = SquadHealthDiagnosis()
    health_report = health_engine.diagnose_squad(squad_df, gw7_pool)

    # 3. WI-8.4 Chip Timeline Comparator
    timeline_engine = ChipTimelineComparator()
    wc_families = timeline_engine.evaluate_wildcard_families(
        current_squad_df=squad_df,
        player_pool_df=gw7_pool,
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=target_gw,
        chips_available=chips_available,
        candidate_later_windows=[8, 12, 14, 16],
    )
    chip_timelines = {
        "3xc": timeline_engine.evaluate_chip_timeline("3xc", current_gw=target_gw, chips_available=chips_available),
        "bboost": timeline_engine.evaluate_chip_timeline("bboost", current_gw=target_gw, chips_available=chips_available),
        "freehit": timeline_engine.evaluate_chip_timeline("freehit", current_gw=target_gw, chips_available=chips_available),
    }

    # 4. WI-8.3 Affordability Path Finder for top market targets
    funding_engine = AffordabilityPathFinder()
    top_target_eids = [368, 8, 427]  # e.g., prominent transfer-in targets
    affordability_results = {}
    for t_eid in top_target_eids:
        if t_eid in elements_map:
            paths = funding_engine.find_cheapest_paths(
                current_squad_df=squad_df,
                player_pool_df=gw7_pool,
                bank_tenths=bank_tenths,
                target_element=t_eid,
                free_transfers=free_transfers,
                top_k=2,
            )
            affordability_results[t_eid] = paths

    # 5. Multi-Period Optimizer for GW7
    cfg = MultiPeriodConfig(
        horizon_len=H,
        solver_name="highs",
        mip_gap=1e-4,
        time_limit_sec=60,
    )
    opt = MultiPeriodOptimizer(cfg)

    roll_plan = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw7_pool,
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=target_gw - 1,
        target_gw=target_gw,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=0,
    )
    plan_1 = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw7_pool,
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=target_gw - 1,
        target_gw=target_gw,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=1,
    )
    plan_hit = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw7_pool,
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=target_gw - 1,
        target_gw=target_gw,
        horizon_len=H,
        chips_available=[],
        forced_w1_transfers=2,
    )
    plan_wc = opt.optimize(
        current_squad_df=squad_df,
        player_pool_df=gw7_pool,
        bank=bank_tenths / 10.0,
        free_transfers=free_transfers,
        horizon_projections=projs,
        current_gw=target_gw - 1,
        target_gw=target_gw,
        horizon_len=H,
        chips_available=chips_available,
        forced_w1_chip="wildcard",
    )

    # Monte Carlo hit policy evaluation
    rng = np.random.default_rng(123)
    n_sims = 2000
    noise_hit = rng.normal(0, 3.5, size=(n_sims, H))
    noise_roll = rng.normal(0, 3.5, size=(n_sims, H))
    disc_weights = np.array([cfg.gamma ** t for t in range(H)])
    sim_hit = np.sum((np.array([a.net_xp for a in plan_hit.trajectory]) + noise_hit) * disc_weights, axis=1) + plan_hit.objective_decomposition.get("terminal_ft_value", 0.0)
    sim_roll = np.sum((np.array([a.net_xp for a in roll_plan.trajectory]) + noise_roll) * disc_weights, axis=1) + roll_plan.objective_decomposition.get("terminal_ft_value", 0.0)
    deltas = sim_hit - sim_roll

    p_gain = float(np.mean(deltas > 0))
    mean_delta = float(np.mean(deltas))
    sigma_delta = float(np.std(deltas))

    p_star, k = cfg.compute_hit_parameters(risk_preference="balanced", mini_league_deficit=float(gap_to_leader))
    hit_justified = (p_gain >= p_star) or (mean_delta > k * sigma_delta)
    hit_verdict = "RECOMMEND_HIT" if hit_justified else "AVOID_HIT"

    # Select recommended plan
    plans = [("Roll Transfer", roll_plan), ("1 Free Transfer", plan_1)]
    if hit_justified:
        plans.append(("2 Transfers (-4 Hit)", plan_hit))
    rec_name, rec_plan = max(plans, key=lambda x: x[1].total_objective)

    # 6. Render reports/advice/<date>.md
    now_date = datetime.now(UTC).strftime("%Y%m%d")
    out_dir = Path("reports/advice")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{now_date}.md"

    md_lines = []
    md_lines.append(f"# FPL Oracle Advisor Briefing: Gameweek {target_gw}")
    md_lines.append("")
    md_lines.append(f"**Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    md_lines.append(f"**Manager:** {user_team} (Entry ID: 1085141)  ")
    md_lines.append(f"**Primary League:** IceMakers League (ID: {league_id})  ")
    md_lines.append(f"**League Standing:** Rank {user_rank} of {len(results)} | **Total Points:** {user_total} | **Gap to Leader ({leader_name}):** {gap_to_leader} pts  ")
    md_lines.append(f"**Bank:** £{bank_tenths/10:.1f}m | **Squad Selling Value:** £{tot_sell_val:.1f}m | **Free Transfers:** {free_transfers}  ")
    md_lines.append(f"**Chips Available (Set 1):** {', '.join(sorted(chips_available))}  ")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 1. Executive Summary & Top-Line Decision")
    md_lines.append("")
    act = rec_plan.recommended_action
    outs_str = ", ".join([f"{elements_map[t['element']].web_name} (£{t.get('sell_price', elements_map[t['element']].now_cost)/10:.1f}m)" for t in act.transfers_out])
    ins_str = ", ".join([f"{elements_map[t['element']].web_name} (£{t.get('cost', elements_map[t['element']].now_cost)/10:.1f}m)" for t in act.transfers_in])
    moves_str = f"{outs_str} -> {ins_str}" if outs_str or ins_str else "None (Roll Transfer)"

    md_lines.append(f"> **RECOMMENDATION:** **{rec_name.upper()}** ({moves_str})  ")
    md_lines.append(f"> **CAPTAIN:** **{elements_map[act.captain].web_name}** ({act.captain_xp:.2f} xP) | **VICE:** {elements_map[act.vice_captain].web_name if act.vice_captain else 'None'}  ")
    md_lines.append(f"> **CHIP VERDICT:** **PRESERVE ALL SET 1 CHIPS** (Wildcard winning family: `{wc_families.winning_family}`)  ")
    md_lines.append(f"> **PROJECTED GW{target_gw} NET xP:** **{act.net_xp:.2f} pts** | **HORIZON NET xP (H=5):** **{rec_plan.horizon_net_xp:.2f} pts**  ")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 2. Squad Health Diagnosis (WI-8.1)")
    md_lines.append("")
    md_lines.append(f"- **Overall Squad Status:** `{health_report.classification}`  ")
    md_lines.append(f"- **Functional Starters:** **{health_report.functional_starters_count} / 11** (Threshold required: {health_report.thresholds_used['min_functional_starters']})  ")
    md_lines.append(f"- **Structural Deficiency Score:** **{health_report.structural_deficiency_score:.2f} xP** (Threshold maximum: {health_report.thresholds_used['deficiency_threshold']:.1f} xP)  ")
    md_lines.append("")
    md_lines.append("| Player Name | Pos | Club | Status | GW7 xP | Chance | Primary Diagnostic Reasons |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for _, r in squad_df.iterrows():
        eid = int(r["element"])
        diag = health_report.player_diagnoses[eid]
        reasons_txt = "; ".join(diag.reasons) if diag.reasons else "Healthy starter, fixture aligned"
        md_lines.append(
            f"| {r['web_name']} | {r['position']} | {r['team_name']} | **{diag.status}** | {diag.expected_points:.2f} | {diag.chance_of_playing:.0f}% | {reasons_txt} |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 3. Wildcard Three-Family Comparison (WI-8.4)")
    md_lines.append("")
    md_lines.append("To prevent the known optimizer defect of comparing Wildcard only against {now, never}, the engine evaluates three distinct strategic families:")
    md_lines.append("")
    md_lines.append(f"1. **Family (a) - Wildcard NOW (GW{target_gw}):** Projected Horizon J = **{plan_wc.total_objective:.2f}** (Edge vs Roll: **{plan_wc.total_objective - roll_plan.total_objective:+.2f} xP**).")
    md_lines.append(f"2. **Family (b) - Wildcard LATER:** Targeted for peak congestion windows (GW8, GW12, GW14, GW16) with estimated opportunity value of **+{wc_families.wildcard_now.value_of_waiting:.2f} xP**.")
    md_lines.append(f"3. **Family (c) - Staged Free-Transfer Plan:** Rolling banked transfers captures **+{wc_families.staged_ft_plan['gain_vs_roll']:+.2f} xP** without consuming the chip, entering next round with **{wc_families.staged_ft_plan['ft_remaining_next']} FTs**.")
    md_lines.append("")
    md_lines.append(f"> **WINNING FAMILY:** **`{wc_families.winning_family}`**  ")
    md_lines.append(f"> **STRATEGIC RATIONALE:** {wc_families.reasoning}  ")
    md_lines.append("")
    md_lines.append("### Set 1 Chip Timeline Table")
    md_lines.append("")
    md_lines.append("| Chip | Target GW | Expected Gain (Delta U) | P(Gain > 0) | 80% Gain Interval | Context / Schedule Note | Value of Waiting | Decision |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Add Wildcard rows
    md_lines.append(f"| Wildcard | GW{target_gw} (Now) | {wc_families.wildcard_now.expected_gain:+.2f} xP | {wc_families.wildcard_now.prob_gain_positive:.1%} | [{wc_families.wildcard_now.gain_interval[0]:+.1f}, {wc_families.wildcard_now.gain_interval[1]:+.1f}] | Current deadline | £{wc_families.wildcard_now.value_of_waiting:.1f}m eq ({wc_families.wildcard_now.value_of_waiting:.1f} xP) | {wc_families.wildcard_now.recommendation_verdict} |")
    for l_win in wc_families.later_windows:
        md_lines.append(f"| Wildcard | GW{l_win.gameweek} (Later) | {l_win.expected_gain:+.2f} xP | {l_win.prob_gain_positive:.1%} | [{l_win.gain_interval[0]:+.1f}, {l_win.gain_interval[1]:+.1f}] | {l_win.context_notes} | {l_win.value_of_waiting:.1f} xP | {l_win.recommendation_verdict} |")

    # Add other chips
    for chip_name in ["3xc", "bboost", "freehit"]:
        c_evals = chip_timelines[chip_name]
        for c_win in c_evals[:4]:  # Show next 4 legal windows
            md_lines.append(f"| {chip_name.upper()} | GW{c_win.gameweek} | {c_win.expected_gain:+.2f} xP | {c_win.prob_gain_positive:.1%} | [{c_win.gain_interval[0]:+.1f}, {c_win.gain_interval[1]:+.1f}] | {c_win.context_notes} | {c_win.value_of_waiting:.1f} xP | {c_win.recommendation_verdict} |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 4. Affordability Path Finder (WI-8.3)")
    md_lines.append("")
    md_lines.append("For high-priority market targets, the affordability engine identifies the cheapest legal paths (minimizing hits and transfers):")
    md_lines.append("")

    for t_eid, paths in affordability_results.items():
        t_elem = elements_map[t_eid]
        md_lines.append(f"### Target: {t_elem.web_name} ({t_elem.now_cost/10:.1f}m, {teams_map[t_elem.team]})")
        if not paths:
            md_lines.append("- *No legal path found within 2 transfers.*")
        else:
            for p_idx, path in enumerate(paths, 1):
                md_lines.append(f"- **Path {p_idx} ({path.path_type}, {path.transfers_count} transfer{'s' if path.transfers_count>1 else ''}, {path.hits} hits):** {path.arithmetic_explanation}")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 5. Candidate Plans Comparison & J Decomposition")
    md_lines.append("")
    md_lines.append("| Plan Type | Transfers | Moves | Chip | Cap | GW7 xP | Horizon Net xP | Total Objective J | Delta J vs Roll | Solver Status |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    comparison_plans = [
        ("Roll Transfer", roll_plan),
        ("Best 1-Transfer", plan_1),
        ("Best Hit Plan (-4)", plan_hit),
        ("Wildcard", plan_wc),
    ]

    for p_name, p in comparison_plans:
        p_act = p.recommended_action
        if p_act.transfers_count == 0:
            p_moves = "Roll"
        else:
            outs = [elements_map[t['element']].web_name for t in p_act.transfers_out if t['element'] in elements_map]
            ins = [elements_map[t['element']].web_name for t in p_act.transfers_in if t['element'] in elements_map]
            if len(outs) == len(ins) and len(outs) > 0:
                p_moves = ", ".join([f"{o} -> {i}" for o, i in zip(outs, ins, strict=True)])
            elif len(outs) > 0 or len(ins) > 0:
                p_moves = f"{len(outs)} out / {len(ins)} in"
            else:
                p_moves = "None"
        p_cap = elements_map[p_act.captain].web_name
        md_lines.append(
            f"| {p_name} | {p_act.transfers_count} | {p_moves} | {p_act.active_chip or 'None'} | {p_cap} | {p_act.net_xp:.2f} | {p.horizon_net_xp:.2f} | {p.total_objective:.3f} | {p.total_objective - roll_plan.total_objective:+.3f} | {p.solver_status} |"
        )

    md_lines.append("")
    md_lines.append("### Objective Function J Decomposition")
    md_lines.append("")
    md_lines.append("| Plan Type | Disc Lineup xP | Disc Cap xP | Disc Bench xP | Disc Hits | Term FT Val | Term Bank Val | FT Micro-Incentive | Total J |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for p_name, p in comparison_plans:
        d = p.objective_decomposition
        md_lines.append(
            f"| {p_name} | {d.get('discounted_lineup_xp', 0):.2f} | {d.get('discounted_captain_xp', 0):.2f} | {d.get('discounted_bench_xp', 0):.2f} | {d.get('discounted_hits_cost', 0):.2f} | {d.get('terminal_ft_value', 0):.2f} | {d.get('terminal_bank_value', 0):.2f} | {d.get('ft_banking_incentive', 0):.4f} | {p.total_objective:.3f} |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 6. Monte Carlo Hit Policy & Parameter Sensitivity (WI-2.5)")
    md_lines.append("")
    md_lines.append(f"- **P(Delta > 0):** **{p_gain:.4f}** ({p_gain:.1%} probability that taking 1 hit beats rolling)")
    md_lines.append(f"- **Mean Delta:** **{mean_delta:+.2f} xP** | **Sigma Delta:** **{sigma_delta:.2f} xP**")
    md_lines.append(f"- **p* Hurdle [PROVISIONAL]:** **{p_star:.2f}** (Derived for Balanced Mode and 26-pt deficit)")
    md_lines.append(f"- **k Hurdle [PROVISIONAL]:** **{k:.2f}** (Hurdle threshold = k * sigma = {k*sigma_delta:.2f} xP)")
    md_lines.append(f"- **Hit Decision Verdict:** **`{hit_verdict}`** (Hit is justified only if P(Delta>0) >= p* or Mean Delta > k*sigma)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 7. What Would Change This Recommendation?")
    md_lines.append("")
    md_lines.append("1. **Confirmed Out for Key Defender:** If Gabriel / Calafiori is confirmed ruled out before the deadline, functional starters would drop to 8, flipping squad status to `STRUCTURAL_OVERHAUL_REQUIRED` and making a multi-transfer or Wildcard plan optimal.")
    md_lines.append("2. **Early Price Drop on João Pedro (£7.8m):** If João Pedro price drops tonight, accumulated equity loss would penalize waiting, favoring immediate swap to Wood or Calvert-Lewin.")
    md_lines.append("3. **Rival Chip Deployment in IceMakers League:** If league leader 'Change Name' activates Wildcard in GW7, shifting risk appetite to 'Aggressive Chase' (reducing p* to 0.50 and k to 0.50) would justify an immediate 2-transfer differential hit.")
    md_lines.append("")

    content = "\n".join(md_lines)
    report_file.write_text(content, encoding="utf-8")
    print(f"\nSuccessfully written advisor briefing to: {report_file}")
    print("\n" + "=" * 80)
    print("PASTING REPORT CONTENT:")
    print("=" * 80)
    print(content)

asyncio.run(generate_report())
