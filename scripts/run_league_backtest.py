"""
Proxy Simulation: Mini-League Monte Carlo Replay (Synthetic League States).
Evaluates Monte Carlo win probability calibration across real historical gameweek data
from data/historical/master_history.csv without hindsight leaks (Requirement G11 / Rule R1).

Note on League States:
Real mini-league standings and manager squad histories are private data not accessible
in public historical archives without authenticated FPL credentials. Mini-league standings
and rival rosters are therefore modeled as realistic synthetic proxies under strict FPL legality rules.

Generates reports/backtest.json and reports/fix_pass_backtest.md.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_oracle.backtest import assert_no_future_data
from fpl_oracle.league.montecarlo import monte_carlo_simulator
from fpl_oracle.optimise.lineup import lineup_optimizer
from fpl_oracle.optimise.squad import squad_optimizer

logger = logging.getLogger("fpl_oracle.scripts.run_league_backtest")


def assert_legal_squad(
    starters_df: pd.DataFrame,
    full_squad_df: pd.DataFrame | None = None,
    max_budget: float = 1000.0,
) -> None:
    """
    Asserts that the squad strictly satisfies FPL legality:
    - Exactly 11 starters
    - Full squad = 15 players (if provided)
    - Squad value <= max_budget (£100.0m / 1000 tenths)
    - Team limit <= 3 per club
    - Formation: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
    """
    if len(starters_df) != 11:
        raise ValueError(f"Squad must have exactly 11 starters, got {len(starters_df)}")

    if full_squad_df is not None:
        if len(full_squad_df) != 15:
            raise ValueError(f"Full squad must have 15 players, got {len(full_squad_df)}")
        squad_val = float(full_squad_df["value"].sum())
        if squad_val > max_budget + 1e-4:
            raise ValueError(f"Squad value {squad_val:.1f} exceeds budget {max_budget:.1f}")
        team_counts = full_squad_df["team"].value_counts()
        if int(team_counts.max()) > 3:
            raise ValueError(f"Club limit exceeded: {team_counts.max()} players from team {team_counts.idxmax()}")
    else:
        starters_val = float(starters_df["value"].sum())
        if starters_val > max_budget + 1e-4:
            raise ValueError(f"Starters value {starters_val:.1f} exceeds budget {max_budget:.1f}")
        team_counts = starters_df["team"].value_counts()
        if int(team_counts.max()) > 3:
            raise ValueError(f"Club limit exceeded: {team_counts.max()} players from team {team_counts.idxmax()}")

    pos_counts = starters_df["position"].value_counts().to_dict()
    gkp = pos_counts.get("GKP", 0)
    def_cnt = pos_counts.get("DEF", 0)
    mid_cnt = pos_counts.get("MID", 0)
    fwd_cnt = pos_counts.get("FWD", 0)

    if gkp != 1:
        raise ValueError(f"Starters must have exactly 1 GKP, got {gkp}")
    if not (3 <= def_cnt <= 5):
        raise ValueError(f"Starters must have 3-5 DEF, got {def_cnt}")
    if not (2 <= mid_cnt <= 5):
        raise ValueError(f"Starters must have 2-5 MID, got {mid_cnt}")
    if not (1 <= fwd_cnt <= 3):
        raise ValueError(f"Starters must have 1-3 FWD, got {fwd_cnt}")


def run_proxy_simulation(quick: bool = False) -> dict[str, Any]:
    print("Starting Proxy Simulation Replay (Synthetic League States, Strict Legality & Zero-Hindsight)...")
    t0 = time.time()

    hist_file = Path("data/historical/master_history.csv")
    if not hist_file.exists():
        raise FileNotFoundError(f"Historical master data not found at {hist_file}")

    hist_df = pd.read_csv(hist_file)

    # 26 realistic historical scenarios across 3 complete seasons
    # Covering diverse checkpoint rounds t and horizons (1-GW, 3-GW, 5-GW)
    scenarios = [
        # 2023-24 Season
        {"season": "2023-24", "start_gw": 10, "horizon": 5, "user_lead": 12.0, "name": "2023-24 GW10-15 Narrow Lead"},
        {"season": "2023-24", "start_gw": 15, "horizon": 5, "user_lead": -15.0, "name": "2023-24 GW15-20 Winter Chase"},
        {"season": "2023-24", "start_gw": 20, "horizon": 5, "user_lead": 5.0, "name": "2023-24 GW20-25 Tight Battle"},
        {
            "season": "2023-24",
            "start_gw": 25,
            "horizon": 5,
            "user_lead": -8.0,
            "name": "2023-24 GW25-30 Spring Deficit",
        },
        {"season": "2023-24", "start_gw": 30, "horizon": 5, "user_lead": 22.0, "name": "2023-24 GW30-35 Solid Margin"},
        {"season": "2023-24", "start_gw": 33, "horizon": 5, "user_lead": -18.0, "name": "2023-24 GW33-38 Final Chase"},
        {"season": "2023-24", "start_gw": 35, "horizon": 3, "user_lead": 6.0, "name": "2023-24 GW35-38 Late Squeeze"},
        {
            "season": "2023-24",
            "start_gw": 37,
            "horizon": 1,
            "user_lead": -3.0,
            "name": "2023-24 GW37-38 Penultimate Sprint",
        },
        # 2024-25 Season
        {
            "season": "2024-25",
            "start_gw": 10,
            "horizon": 5,
            "user_lead": -10.0,
            "name": "2024-25 GW10-15 Early Deficit",
        },
        {"season": "2024-25", "start_gw": 15, "horizon": 5, "user_lead": 18.0, "name": "2024-25 GW15-20 Winter Margin"},
        {
            "season": "2024-25",
            "start_gw": 20,
            "horizon": 5,
            "user_lead": -5.0,
            "name": "2024-25 GW20-25 Midseason Clash",
        },
        {"season": "2024-25", "start_gw": 25, "horizon": 5, "user_lead": 14.0, "name": "2024-25 GW25-30 Spring Lead"},
        {"season": "2024-25", "start_gw": 28, "horizon": 5, "user_lead": 0.0, "name": "2024-25 GW28-33 Dead Heat"},
        {"season": "2024-25", "start_gw": 30, "horizon": 5, "user_lead": -20.0, "name": "2024-25 GW30-35 Deep Chase"},
        {"season": "2024-25", "start_gw": 33, "horizon": 5, "user_lead": 10.0, "name": "2024-25 GW33-38 Run-In Finale"},
        {"season": "2024-25", "start_gw": 35, "horizon": 3, "user_lead": -7.0, "name": "2024-25 GW35-38 Late Chase"},
        {
            "season": "2024-25",
            "start_gw": 37,
            "horizon": 1,
            "user_lead": 4.0,
            "name": "2024-25 GW37-38 Penultimate Round",
        },
        # 2025-26 Season
        {"season": "2025-26", "start_gw": 10, "horizon": 5, "user_lead": 8.0, "name": "2025-26 GW10-15 Autumn Run"},
        {"season": "2025-26", "start_gw": 15, "horizon": 5, "user_lead": -12.0, "name": "2025-26 GW15-20 Winter Surge"},
        {
            "season": "2025-26",
            "start_gw": 20,
            "horizon": 5,
            "user_lead": 15.0,
            "name": "2025-26 GW20-25 Midseason Clash",
        },
        {"season": "2025-26", "start_gw": 25, "horizon": 5, "user_lead": -6.0, "name": "2025-26 GW25-30 Spring Shift"},
        {"season": "2025-26", "start_gw": 28, "horizon": 5, "user_lead": 7.0, "name": "2025-26 GW28-33 Spring Stretch"},
        {
            "season": "2025-26",
            "start_gw": 30,
            "horizon": 5,
            "user_lead": -16.0,
            "name": "2025-26 GW30-35 Run-In Pressure",
        },
        {"season": "2025-26", "start_gw": 33, "horizon": 5, "user_lead": 12.0, "name": "2025-26 GW33-38 Final Stretch"},
        {"season": "2025-26", "start_gw": 35, "horizon": 3, "user_lead": 2.0, "name": "2025-26 GW35-38 Late Sprint"},
        {
            "season": "2025-26",
            "start_gw": 37,
            "horizon": 1,
            "user_lead": -2.0,
            "name": "2025-26 GW37-38 Penultimate Round",
        },
    ]

    if quick:
        scenarios = scenarios[:10]

    all_predictions: list[float] = []
    all_actuals: list[float] = []
    all_naive_uniform: list[float] = []
    all_naive_lead: list[float] = []
    scenario_results: list[dict[str, Any]] = []

    # Actual points tracked per strategy
    oracle_gw_pts_list: list[float] = []
    prev_gw_pts_list: list[float] = []
    season_avg_pts_list: list[float] = []
    template_pts_list: list[float] = []

    for sc_idx, sc in enumerate(scenarios):
        season = sc["season"]
        t = sc["start_gw"]
        h = sc["horizon"]
        user_lead = float(sc["user_lead"])

        season_df = hist_df[hist_df["season"] == season]

        # 1. Past historical data available strictly up to checkpoint round t
        past_df = season_df[season_df["round"] <= t].copy()
        assert_no_future_data(past_df, t)

        # 2. Future actual data over horizon (t+1 .. t+h)
        future_df = season_df[(season_df["round"] > t) & (season_df["round"] <= t + h)].copy()

        # Build pre-deadline player pool
        player_stats = (
            past_df.groupby("element")
            .agg(
                {
                    "total_points": ["count", "mean", "sum"],
                    "team": "last",
                    "position": "last",
                    "name": "last",
                    "value": "last",
                }
            )
            .reset_index()
        )
        player_stats.columns = ["element", "n_games", "mean_pts", "sum_pts", "team", "position", "name", "value"]
        player_stats = player_stats[player_stats["n_games"] >= 3].copy()
        player_stats["web_name"] = player_stats["name"]
        player_stats["value"] = player_stats["value"].fillna(50.0).astype(float)
        player_stats["expected_points"] = player_stats["mean_pts"].clip(0.5, 10.0)

        # Compute rolling recent form (last 3 matches)
        recent = past_df[past_df["round"] >= max(1, t - 2)].groupby("element")["total_points"].sum().reset_index()
        recent.columns = ["element", "recent_pts"]
        player_stats = player_stats.merge(recent, on="element", how="left")
        player_stats["recent_pts"] = player_stats["recent_pts"].fillna(0.0)

        # Compute previous gameweek points (round t)
        prev_gw = past_df[past_df["round"] == t].groupby("element")["total_points"].sum().reset_index()
        prev_gw.columns = ["element", "prev_gw_pts"]
        player_stats = player_stats.merge(prev_gw, on="element", how="left")
        player_stats["prev_gw_pts"] = player_stats["prev_gw_pts"].fillna(0.0)
        if player_stats["prev_gw_pts"].sum() == 0.0:
            player_stats["prev_gw_pts"] = player_stats["recent_pts"]

        # Points per million
        player_stats["ppm"] = player_stats["sum_pts"] / (player_stats["value"] / 10.0).clip(lower=4.0)

        # 6 Distinct Manager Strategies
        strategy_configs = [
            ("Oracle (Model xP)", "expected_points", 1000.0),
            ("Recent Form", "recent_pts", 1000.0),
            ("Season Average", "mean_pts", 1000.0),
            ("Template", "sum_pts", 1000.0),
            ("Value Hunter", "ppm", 985.0),
            ("Prev GW Hauler", "prev_gw_pts", 1000.0),
        ]

        # Establish realistic mini-league points standings at round t
        base_t = 60.0 * t
        spreads = [user_lead, 0.0, -10.0, -20.0, -30.0, -45.0]
        if user_lead < 0:
            spreads = [0.0, abs(user_lead), abs(user_lead) - 10.0, abs(user_lead) - 20.0, -15.0, -30.0]

        managers = []
        for m_idx, (name, metric, budget) in enumerate(strategy_configs):
            squad_res = squad_optimizer.solve_best_squad(player_stats, budget=budget, metric_col=metric)
            lineup_res = lineup_optimizer.select_lineup_and_captain(squad_res["squad"])
            starters_df = lineup_res["starters"]
            starters = starters_df["element"].tolist()
            cap = int(lineup_res["captain"]["element"])

            # Verify 100% strict FPL squad legality
            assert_legal_squad(starters_df, squad_res["squad"], max_budget=budget)

            pts_at_t = base_t + spreads[m_idx]
            managers.append(
                {
                    "entry_id": 100 + m_idx,
                    "name": name,
                    "points_at_t": pts_at_t,
                    "starters": starters,
                    "captain": cap,
                    "starters_df": starters_df,
                    "squad_df": squad_res["squad"],
                }
            )

        # Evaluate actual future points scored by each squad across the horizon
        final_totals = []
        horizon_actual_pts_by_mgr = []
        for m in managers:
            act_pts = 0.0
            for r_fut in range(t + 1, t + h + 1):
                r_df = future_df[future_df["round"] == r_fut]
                for eid in m["starters"]:
                    p_pts = r_df[r_df["element"] == eid]["total_points"].sum()
                    mult = 2.0 if eid == m["captain"] else 1.0
                    act_pts += float(p_pts) * mult
            final_totals.append(m["points_at_t"] + act_pts)
            horizon_actual_pts_by_mgr.append(act_pts)

        actual_winner_idx = int(np.argmax(final_totals))
        user_won = bool(actual_winner_idx == 0)
        y_user_actual = 1.0 if user_won else 0.0

        # Track strategy points per GW
        oracle_gw_pts_list.append(horizon_actual_pts_by_mgr[0] / h)
        prev_gw_pts_list.append(horizon_actual_pts_by_mgr[5] / h)
        season_avg_pts_list.append(horizon_actual_pts_by_mgr[2] / h)
        template_pts_list.append(horizon_actual_pts_by_mgr[3] / h)

        # Run Monte Carlo simulator for User (Manager 0)
        user_mgr = managers[0]
        user_squad_df = user_mgr["starters_df"][["element"]].copy()
        user_squad_df["is_starter"] = True
        user_squad_df["is_captain"] = user_squad_df["element"] == user_mgr["captain"]

        rival_entries = []
        for r_m in managers[1:]:
            rival_entries.append(
                {
                    "entry_id": r_m["entry_id"],
                    "player_name": r_m["name"],
                    "total_points": r_m["points_at_t"],
                    "captain_element": r_m["captain"],
                    "squad": [{"element": eid, "is_starter": True} for eid in r_m["starters"]],
                }
            )

        sim_res = monte_carlo_simulator.simulate_league(
            user_points=user_mgr["points_at_t"],
            user_squad_df=user_squad_df,
            rival_squads=rival_entries,
            projections_df=player_stats[["element", "expected_points", "team", "position"]],
            horizon_gws=h,
            seed=42 + sc_idx,
        )

        p_win_user = sim_res["user_win_probability_pct"] / 100.0

        # Naive Reference Probabilities
        p_uniform = 1.0 / len(managers)
        tau = max(10.0, 25.0 * np.sqrt(h))
        pts_arr = np.array([m["points_at_t"] for m in managers])
        exp_pts = np.exp((pts_arr - np.max(pts_arr)) / tau)
        p_lead = float(exp_pts[0] / np.sum(exp_pts))

        all_predictions.append(p_win_user)
        all_actuals.append(y_user_actual)
        all_naive_uniform.append(p_uniform)
        all_naive_lead.append(p_lead)

        brier_model = (p_win_user - y_user_actual) ** 2
        brier_unif = (p_uniform - y_user_actual) ** 2
        brier_lead = (p_lead - y_user_actual) ** 2

        scenario_results.append(
            {
                "scenario": sc["name"],
                "season": season,
                "start_gw": t,
                "horizon_gws": h,
                "user_lead_pts": user_lead,
                "model_win_prob_pct": round(p_win_user * 100.0, 1),
                "naive_lead_win_prob_pct": round(p_lead * 100.0, 1),
                "actual_won": user_won,
                "winner": managers[actual_winner_idx]["name"],
                "brier_model": round(brier_model, 4),
                "brier_uniform": round(brier_unif, 4),
                "brier_lead": round(brier_lead, 4),
                "squad_legality_verified": True,
            }
        )

    overall_brier_model = float(np.mean([(p - y) ** 2 for p, y in zip(all_predictions, all_actuals, strict=True)]))
    overall_brier_unif = float(np.mean([(p - y) ** 2 for p, y in zip(all_naive_uniform, all_actuals, strict=True)]))
    overall_brier_lead = float(np.mean([(p - y) ** 2 for p, y in zip(all_naive_lead, all_actuals, strict=True)]))

    # Reliability calibration curve
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    calib_rows = []
    for b_low, b_high in bins:
        mask = [b_low <= p < b_high or (b_high == 1.0 and p == 1.0) for p in all_predictions]
        n_bin = sum(mask)
        if n_bin > 0:
            mean_pred = float(np.mean([p for p, m in zip(all_predictions, mask, strict=True) if m]))
            mean_act = float(np.mean([y for y, m in zip(all_actuals, mask, strict=True) if m]))
        else:
            mean_pred = (b_low + b_high) / 2.0
            mean_act = 0.0
        calib_rows.append(
            {
                "bin": f"{int(b_low * 100)}%-{int(b_high * 100)}%",
                "count": n_bin,
                "pred": round(mean_pred * 100, 1),
                "actual": round(mean_act * 100, 1),
            }
        )

    # Strategy points comparison
    mean_oracle_pts = float(np.mean(oracle_gw_pts_list))
    mean_prev_gw_pts = float(np.mean(prev_gw_pts_list))
    mean_season_avg_pts = float(np.mean(season_avg_pts_list))
    mean_template_pts = float(np.mean(template_pts_list))

    skill_demonstrated = bool(overall_brier_model < overall_brier_unif and overall_brier_model < overall_brier_lead)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "simulation_type": "proxy_simulation",
        "is_synthetic_league": True,
        "synthetic_league_reason": (
            "Real mini-league standings and squad histories are private manager data requiring authenticated API access "
            "not present in historical offline dumps. Mini-league states and rival squads are modeled as synthetic proxy "
            "scenarios under strict FPL legality rules."
        ),
        "sample_size": len(scenarios),
        "model_brier": round(overall_brier_model, 4),
        "naive_uniform_brier": round(overall_brier_unif, 4),
        "naive_points_lead_brier": round(overall_brier_lead, 4),
        "brier_delta_vs_uniform": round(overall_brier_model - overall_brier_unif, 4),
        "brier_delta_vs_lead": round(overall_brier_model - overall_brier_lead, 4),
        "skill_demonstrated": skill_demonstrated,
        "squad_points_summary": {
            "model_mean_pts_per_gw": round(mean_oracle_pts, 2),
            "prev_gw_mean_pts_per_gw": round(mean_prev_gw_pts, 2),
            "season_avg_mean_pts_per_gw": round(mean_season_avg_pts, 2),
            "template_mean_pts_per_gw": round(mean_template_pts, 2),
            "uplift_vs_prev_gw": round(mean_oracle_pts - mean_prev_gw_pts, 2),
            "uplift_vs_season_avg": round(mean_oracle_pts - mean_season_avg_pts, 2),
            "uplift_vs_template": round(mean_oracle_pts - mean_template_pts, 2),
        },
        "reliability_table": calib_rows,
        "scenarios": scenario_results,
    }

    # Write reports/backtest.json
    out_json = Path("reports/backtest.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Generate Markdown report
    md_lines = [
        "# Proxy Simulation: Mini-League Monte Carlo Replay (Synthetic League States)",
        "",
        f"**Generated:** {payload['generated_at']}  ",
        "**Methodology:** Proxy simulation across historical gameweeks with strict `assert_no_future_data` barriers.  ",
        f"**Sample Size:** {len(scenarios)} overlapping scenarios across 3 seasons (2023-24, 2024-25, 2025-26).  ",
        f"**Model Brier Score:** {overall_brier_model:.4f}  ",
        f"**Naive Reference Brier (Uniform):** {overall_brier_unif:.4f}  ",
        f"**Naive Reference Brier (Points Lead):** {overall_brier_lead:.4f}  ",
        f"**Skill Demonstrated:** {'Yes' if skill_demonstrated else 'No'}  ",
        "",
        "> [!NOTE]",
        "> Real mini-league standings and squad pick histories are private manager data requiring authenticated API access.",
        "> League states and rival squad compositions are modeled as synthetic proxies subject to 100% strict FPL legality rules",
        "> (15-man squad within £100.0m budget, 11 starters with 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, max 3 players per club).",
        "",
        "---",
        "",
        "## 1. Squad Points Comparison vs Naive References",
        "",
        "| Strategy | Mean Actual Pts / GW | Uplift vs Strategy |",
        "| :--- | :---: | :---: |",
        f"| **FPL Oracle Model (xP)** | **{mean_oracle_pts:.2f}** | - |",
        f"| **Previous GW Hauler** | {mean_prev_gw_pts:.2f} | +{mean_oracle_pts - mean_prev_gw_pts:.2f} |",
        f"| **Season Average** | {mean_season_avg_pts:.2f} | +{mean_oracle_pts - mean_season_avg_pts:.2f} |",
        f"| **Template (Total Points)** | {mean_template_pts:.2f} | +{mean_oracle_pts - mean_template_pts:.2f} |",
        "",
        "---",
        "",
        "## 2. Real Historical Scenarios & Outcomes",
        "",
        "| Scenario | Season | Horizon | Initial Lead | Forecast Win Prob (%) | Naive Lead Prob (%) | Winner | Brier | Legality |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for sr in scenario_results:
        md_lines.append(
            f"| {sr['scenario']} | {sr['season']} | {sr['horizon_gws']} GW | {sr['user_lead_pts']:+.0f} pts | "
            f"{sr['model_win_prob_pct']}% | {sr['naive_lead_win_prob_pct']}% | {sr['winner']} | {sr['brier_model']:.4f} | PASS |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Reliability Calibration Curve",
            "",
            "| Forecast Bin | Sample Count | Mean Forecast Win Prob (%) | Realized Win Frequency (%) |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )

    for cr in calib_rows:
        md_lines.append(f"| {cr['bin']} | {cr['count']} | {cr['pred']}% | {cr['actual']}% |")

    md_lines.append("")
    Path("reports/fix_pass_backtest.md").write_text("\n".join(md_lines), encoding="utf-8")

    elapsed = time.time() - t0
    print(
        f"Proxy simulation completed in {elapsed:.2f}s. "
        f"Sample Size: {len(scenarios)}, Model Brier: {overall_brier_model:.4f}, "
        f"Naive Lead Brier: {overall_brier_lead:.4f}"
    )

    # Legacy compatibility return format
    payload["brier_score"] = overall_brier_model
    return payload


# Alias for backwards compatibility
run_backtest = run_proxy_simulation


if __name__ == "__main__":
    run_proxy_simulation()
