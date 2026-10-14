"""
CLI entry point for WI-0.1 Walk-Forward Backtest Runner.

Usage:
    python scripts/run_backtest.py --seasons 2024-25 2025-26 --window last-20-gw --retrain-every 4
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpl_oracle.eval.backtest import WalkForwardBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_backtest")


def main():
    parser = argparse.ArgumentParser(description="Run authentic walk-forward backtest across historical gameweeks.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2024-25", "2025-26"],
        help="Seasons to evaluate (e.g. --seasons 2024-25 2025-26)",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="last-20-gw",
        help="Evaluation window: 'last-20-gw', 'last-N-gw', or 'all'",
    )
    parser.add_argument(
        "--retrain-every",
        type=int,
        default=4,
        help="Retrain component ML models every N gameweeks (default: 4 for speed, 1 for strict reference)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom directory to store evaluation reports (default: reports/eval/)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    tester = WalkForwardBacktester(output_base_dir=out_dir)

    print("=" * 80)
    print("FPL ORACLE: WALK-FORWARD BACKTEST RUNNER (WI-0.1 / WI-0.2 / WI-0.3)")
    print("=" * 80)
    print(f"Seasons:       {args.seasons}")
    print(f"Window:        {args.window}")
    print(f"Retrain Every: {args.retrain_every} GW(s)")
    print(f"Random Seed:   {args.seed}")
    print("=" * 80)

    report = tester.run(
        seasons=args.seasons,
        window=args.window,
        retrain_every=args.retrain_every,
        seed=args.seed,
    )

    va = report["metrics"]["view_a_all_available_players"]
    vb = report["metrics"]["view_b_played_players_only"]
    vb_base = vb.get("baselines", {})
    boot = vb.get("gameweek_bootstrap", {})

    print("\n" + "=" * 80)
    print("WALK-FORWARD BACKTEST SUMMARY RESULTS")
    print("=" * 80)
    print(f"Evaluated Observations: {report['total_observations']:,}")
    print(f"View (a) All Players:   Model MAE: {va.get('model_mae')} | RMSE: {va.get('model_rmse')} | Spearman (Tied): {va.get('model_spearman_tied')}")
    print(f"View (b) Active Only:   Model MAE: {vb.get('model_mae')} | Baseline 2 MAE: {vb_base.get('b2_roll5_mae')} | Gain: {vb.get('gain_vs_baseline_2_mae'):+0.3f}")
    print(f"                        Model RMSE: {vb.get('model_rmse')} | Baseline 2 RMSE: {vb_base.get('b2_roll5_rmse')} | Gain: {vb.get('gain_vs_baseline_2_rmse'):+0.3f}")
    print(f"                        Spearman (Tied):   Model: {vb.get('model_spearman_tied')} vs Base 2: {vb_base.get('b2_roll5_spearman_tied')} (Price: {vb_base.get('b6_price_spearman_tied')}, Prev-GW: {vb_base.get('b5_prev_gw_spearman_tied')})")
    print(f"                        Spearman (NoTies): Model: {vb.get('model_spearman_noties')} vs Base 2: {vb_base.get('b2_roll5_spearman_noties')}")
    print(f"                        Empirical 80% Coverage: {vb.get('coverage_80_pct')}% (Nominal: 80.0%)")
    print("                        Gameweek Bootstrap (B=10,000):")
    print(f"                          MAE Gain 95% CI:  [{boot.get('ci_95_mae', [0, 0])[0]}, {boot.get('ci_95_mae', [0, 0])[1]}] (p = {boot.get('bootstrap_p_value_mae')})")
    print(f"                          RMSE Gain 95% CI: [{boot.get('ci_95_rmse', [0, 0])[0]}, {boot.get('ci_95_rmse', [0, 0])[1]}] (p = {boot.get('bootstrap_p_value_rmse')})")
    print("=" * 80)
    print("Reports successfully generated in reports/eval/")


if __name__ == "__main__":
    main()
