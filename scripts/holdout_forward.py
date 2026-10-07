"""
Forward Holdout Logging & Scoring Script (Requirement G6).

Guarantees honest forward evaluation of the FPL Oracle ML model on unseen future gameweeks:
1. Pre-deadline freeze: Before each gameweek deadline, serializes served predictions
   to data/holdout/gw<N>_predictions.json with manifest hash, model version, and UTC timestamp.
2. Post-gameweek scoring: Once the gameweek concludes and official points finalize,
   scores frozen predictions against actual points and logs metrics to reports/holdout_log.json.
3. Honesty guarantee: GW1-5 is acknowledged as contaminated by prior inspection.
   Forward scoring commences strictly with GW6 onwards.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fpl_oracle.ml.holdout import (
    HOLDOUT_LOG_PATH,
    compute_manifest_sha256,
    ensure_holdout_log_initialized,
    freeze_predictions,
    run_holdout_cycle,
    score_frozen_predictions,
)

__all__ = [
    "HOLDOUT_LOG_PATH",
    "compute_manifest_sha256",
    "ensure_holdout_log_initialized",
    "freeze_predictions",
    "run_holdout_cycle",
    "score_frozen_predictions",
]


def main():
    parser = argparse.ArgumentParser(description="FPL Oracle Forward Holdout Freeze and Score Tool")
    parser.add_argument("--freeze", action="store_true", help="Freeze served predictions for next gameweek")
    parser.add_argument("--score", action="store_true", help="Score frozen predictions for finished gameweek")
    parser.add_argument("--gw", type=int, default=None, help="Target gameweek number")
    parser.add_argument("--dry-run", action="store_true", help="Execute without mutating disk")
    parser.add_argument("--status", action="store_true", help="Print current forward holdout log status")
    args = parser.parse_args()

    if args.freeze:
        asyncio.run(freeze_predictions(target_gw=args.gw, dry_run=args.dry_run))
    elif args.score:
        if args.gw is None:
            print("Error: --gw <number> is required for --score.")
            sys.exit(1)
        from fpl_oracle.ml.holdout import fetch_finalized_actuals

        actuals = asyncio.run(fetch_finalized_actuals(args.gw))
        result = score_frozen_predictions(gw=args.gw, actual_points_map=actuals, dry_run=args.dry_run)
        print(result if result is not None else "Awaiting official finalization or valid frozen predictions.")
    else:
        asyncio.run(run_holdout_cycle(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
