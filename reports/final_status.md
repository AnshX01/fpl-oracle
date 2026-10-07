# FPL Oracle — Final Correction Pass Status (Milestones G1 – G14)

**Generated**: 2026-10-07T02:45:07.253343+00:00
**Active Production Model**: `v2026.10.06.1713`
**Status**: All milestones G1 through G14 verified under non-negotiable evidence gates (Rules R1–R9).

---

## 1. Verified Model & Pipeline Deliverables

1. **Release Integrity (G1)**: Atomic candidate staging and `os.replace` swap; locked manifest SHA256 integrity covering all 6 models and `calibration.json`. Evaluation runs never mutate production weights.
2. **CI Pipeline & Quality (G2)**: Zero mypy typing errors across 67 source files; clean ruff lint and formatting; fail-closed `check_secrets.py` without shell=True.
3. **UI Honesty & Design System (G3)**: Purged all fabricated metrics and template defaults; integrated Atlas and the-council design system; visual screenshot capture on populated data.
4. **Player Identity & Prediction Freshness (G4)**: Resolved 841 cross-season player ID collisions in historical data via canonical player resolver; content-hashed prediction cache key; incremental API refresh.
5. **Honest Evaluation & Gate (G5)**: Unified evaluation recipe across training, evaluation, and serving; out-of-fold validation stacking; zero-degradation promotion gate.
6. **Forward Holdout Logging (G6)**: Implemented pre-deadline freeze mechanism and post-deadline scoring against actual FPL points.
7. **True Retraining Feature Ablation (G7)**: From-scratch component retraining per feature group with grouped-by-GW bootstrap 95% confidence intervals; all groups verified present.
8. **Rivals Logic & Standings (G8)**: Exact rival selection within configured point window (default 20); removed forced 10-chaser expansion; full pagination with safety limits.
9. **Monte Carlo Clean-Sheet Realism (G9)**: Position-specific component decomposition with correlated Bernoulli clean sheet draws; simulated mean matches model xP within Monte Carlo error across all values (including xP 0.1).
10. **Joint Chip and Transfer Trajectory Planning (G10)**: Real multi-GW stateful search (transfers + chips) called from pipeline, API, and decision card; single consistent plan across all surfaces.
11. **Proxy Simulation Replay (G11)**: Evaluated 26 historical proxy replay scenarios across 3 seasons with pre-deadline data isolation and 100% legal squads. Model Brier: **0.2502** vs naive points lead **0.3740**.
12. **Production News Extraction & Benchmark Gate (G12)**: 70 held-out cases evaluated through real deterministic fallback extractor with clause attribution and negation handling; **70/70 passed (100.0% precision)**, 0 false ruled out; production gate OPEN.
13. **Real Multi-Season Feature Parity (G13)**: 100% numerical parity across all 62 canonical features on real multi-season snapshot with colliding IDs, transfers, BGW, DGW, and refreshed snapshots.
14. **Ledger & Final Gate Tooling (G14)**: Verified evidence files, command outputs, exit codes, manifest SHA256 locks, and doc consistency.

---

## 2. Authoritative Metrics Table

| Metric | Measured Value | Baseline / Reference | Artifact Source |
|:---|:---:|:---:|:---|
| **Production Model Holdout MAE (2026-27)** | **1.895 pts** | 1.952 pts (+0.057 gain) | `reports/model_eval.json` |
| **Spearman Rank Correlation ($\\rho$)** | **0.424** | 0.365 (Season Average) | `reports/model_eval.json` |
| **Credible Interval Coverage ($[P_{10}, P_{90}]$)** | **71.28%** | 80.0% nominal | `reports/model_eval.json` |
| **Retraining Ablation (Fixture Difficulty And Context)** | **+1.17%** | [0.0064, 0.0155] 95% CI | `reports/ablation.json` |
| **Proxy Simulation Replay Brier** | **0.2502** | 0.3740 (Naive Points Lead) | `reports/backtest.json` |
| **News Extractor Benchmark** | **100.0%** | 70/70 passed, 0 false ruled out | `reports/news_benchmark.json` |
| **Train/Serve Parity Discrepancy** | **0.000000** | 8 scenarios across 62 features | `reports/evidence/G13-parity.txt` |
