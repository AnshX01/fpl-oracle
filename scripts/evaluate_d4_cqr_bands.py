"""
D4: CQR Production Bands Empirical Evaluation & Recalibration
Refit and recalibrate on rolling finalised data with multi-season calibration
and cross-position shrinkage, reporting per-position coverage with 95% CIs
against the SPEC gate [77.0%, 83.0%].
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_oracle.data.features import feature_engineering
from fpl_oracle.data.historical import historical_manager
from fpl_oracle.ml.quantiles import ConformalQuantileRegressor, calc_pinball_loss, calc_wilson_ci


def compute_gw_block_bootstrap_ci(
    df_eval: pd.DataFrame, n_boot: int = 2000, random_state: int = 42
) -> tuple[float, float]:
    """Computes gameweek-block bootstrap 95% CI for coverage percentage."""
    rng = np.random.default_rng(random_state)
    gws = df_eval["round"].unique()
    if len(gws) == 0:
        return (0.0, 0.0)

    boot_covs = []
    for _ in range(n_boot):
        sample_gws = rng.choice(gws, size=len(gws), replace=True)
        blocks = [df_eval[df_eval["round"] == gw] for gw in sample_gws]
        sample_df = pd.concat(blocks, ignore_index=True)
        in_band = (sample_df["actual_points"] >= sample_df["p10"]) & (sample_df["actual_points"] <= sample_df["p90"])
        boot_covs.append(float(np.mean(in_band) * 100.0))

    return (
        float(np.round(np.percentile(boot_covs, 2.5), 2)),
        float(np.round(np.percentile(boot_covs, 97.5), 2)),
    )


def run_d4_cqr_evaluation():
    print("=" * 80)
    print("D4: CQR PRODUCTION BANDS EMPIRICAL EVALUATION & RECALIBRATION")
    print("=" * 80)

    # 1. Load historical data
    df = historical_manager.ensure_dataset_ready()
    X, Y, meta = feature_engineering.build_historical_features(df, return_meta=True)
    seasons = meta["season"]

    print("\n--- 1. ROOT CAUSE ANALYSIS OF C5 GOALKEEPER UNDER-COVERAGE (71.7%) ---")
    print("Prior CQR model was calibrated strictly on 2025-26 GW11-18 (8 gameweeks).")
    print("In that small 8-week slice, there were only N=161 Goalkeepers.")
    print("This small sample produced a loose offset (0.1905) that under-covered on the 2026-27 holdout (71.67%).")
    print("Remediation: (1) Multi-season calibration data (N=405-887 GKPs) and (2) Cross-position empirical shrinkage.")

    # 2. Forward Holdout Protocol Evaluation (Zero Leakage)
    # Train on 2023-24 + 2024-25, Calibrate on 2025-26 full season (multi-season) with shrinkage
    # Evaluate strictly on 2026-27 GW1-6 active players (minutes > 0)
    print("\n--- 2. FORWARD HOLDOUT EVALUATION (2026-27 GW1-6 ACTIVE PLAYERS) ---")
    train_mask = seasons.isin(["2023-24", "2024-25"])
    cal_mask = (seasons == "2025-26") & (Y["target_minutes"] > 0)
    eval_mask = (seasons == "2026-27") & (Y["target_minutes"] > 0)

    cqr_eval = ConformalQuantileRegressor(nominal_coverage=0.80, n_estimators=50, random_state=42)
    cqr_eval.fit(X[train_mask], Y.loc[train_mask, "target_points"], meta.loc[train_mask, "position"])
    cal_offsets = cqr_eval.calibrate_conformal(
        X[cal_mask],
        Y.loc[cal_mask, "target_points"],
        meta.loc[cal_mask, "position"],
        shrink_across_positions=True,
    )
    print(f"Calibration Sample Counts: {cqr_eval.cal_sample_counts}")
    print(f"Conformal Offsets (Multi-season Cal): {cal_offsets}")

    X_eval = X[eval_mask].copy()
    y_eval = Y.loc[eval_mask, "target_points"].values
    pos_eval = meta.loc[eval_mask, "position"].values
    gw_eval = meta.loc[eval_mask, "round"].values

    p10_eval, p90_eval, _ = cqr_eval.predict_quantiles(X_eval, pos_eval, enforce_monotonicity=True)

    df_eval_records = pd.DataFrame(
        {
            "round": gw_eval,
            "position": pos_eval,
            "actual_points": y_eval,
            "p10": p10_eval,
            "p90": p90_eval,
        }
    )

    positions = ["GKP", "DEF", "MID", "FWD", "ALL"]
    results = {}

    print("\nEmpirical Coverage vs SPEC Gate [77.0%, 83.0%]:")
    print(
        f"{'Position':<8} | {'N':<6} | {'Coverage':<8} | {'Wilson 95% CI':<16} | {'Block Boot 95% CI':<18} | {'Width':<6} | {'PB10':<6} | {'PB90':<6} | {'SPEC Gate [77, 83]'}"
    )
    print("-" * 105)

    for pos in positions:
        sub = df_eval_records if pos == "ALL" else df_eval_records[df_eval_records["position"] == pos]
        n = len(sub)
        y = sub["actual_points"].values
        p10 = sub["p10"].values
        p90 = sub["p90"].values

        in_band = (y >= p10) & (y <= p90)
        cov = float(np.round(np.mean(in_band) * 100.0, 2))
        k = int(np.sum(in_band))

        w_ci = calc_wilson_ci(k, n)
        boot_ci = compute_gw_block_bootstrap_ci(sub)
        width = float(np.round(np.mean(p90 - p10), 2))
        pb10 = float(np.round(calc_pinball_loss(y, p10, 0.10), 4))
        pb90 = float(np.round(calc_pinball_loss(y, p90, 0.90), 4))
        within_gate = 77.0 <= cov <= 83.0

        status_str = "PASS" if within_gate else "FAIL"
        print(
            f"{pos:<8} | {n:<6} | {cov:.2f}%  | [{w_ci[0]:.2f}, {w_ci[1]:.2f}]%   | [{boot_ci[0]:.2f}, {boot_ci[1]:.2f}]%     | {width:<6.2f} | {pb10:<6.4f} | {pb90:<6.4f} | {status_str}"
        )

        results[pos] = {
            "sample_count": n,
            "coverage_pct": cov,
            "wilson_ci_95": w_ci,
            "block_bootstrap_ci_95": list(boot_ci),
            "mean_interval_width": width,
            "pinball_loss_p10": pb10,
            "pinball_loss_p90": pb90,
            "within_spec_gate_77_83": within_gate,
        }

        # Assert every position satisfies the SPEC gate
        assert within_gate, f"Position {pos} coverage {cov}% failed SPEC gate [77.0%, 83.0%]!"

    # 3. Production Rolling Recalibration (All Finalised Seasons to Date)
    print("\n--- 3. PRODUCTION CQR RECALIBRATION (ALL FINALISED DATA TO DATE) ---")
    prod_train_mask = seasons.isin(["2023-24", "2024-25"])
    prod_cal_mask = (seasons.isin(["2025-26", "2026-27"])) & (Y["target_minutes"] > 0)

    prod_cqr = ConformalQuantileRegressor(nominal_coverage=0.80, n_estimators=50, random_state=42)
    prod_cqr.fit(X[prod_train_mask], Y.loc[prod_train_mask, "target_points"], meta.loc[prod_train_mask, "position"])
    prod_offsets = prod_cqr.calibrate_conformal(
        X[prod_cal_mask],
        Y.loc[prod_cal_mask, "target_points"],
        meta.loc[prod_cal_mask, "position"],
        shrink_across_positions=True,
    )

    print(f"Production Calibrated Sample Counts: {prod_cqr.cal_sample_counts}")
    print(f"Production Conformal Offsets: {prod_offsets}")

    output_payload = {
        "evaluation_protocol": "Forward holdout on active players (minutes > 0) in 2026-27 GW1-6",
        "spec_gate": [77.0, 83.0],
        "holdout_results": results,
        "production_recalibration": {
            "seasons_trained": ["2023-24", "2024-25"],
            "seasons_calibrated": ["2025-26", "2026-27 GW1-6"],
            "cal_sample_counts": prod_cqr.cal_sample_counts,
            "conformal_offsets": prod_offsets,
        },
    }

    out_file = Path("reports/d4_cqr_bands.json")
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nSaved machine artifact to {out_file}")
    print("\nD4 CQR PRODUCTION BANDS EVALUATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_d4_cqr_evaluation()
