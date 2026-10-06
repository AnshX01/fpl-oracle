# FPL Oracle — Final Fix Pass Complete Status

**Audit Date**: 2026-10-06
**Active Production Model**: `v2026.10.06.1420`
**Status**: All milestones F1 through F14 verified under non-negotiable evidence gates.

---

## 1. Verified Model & Pipeline Deliverables

1. **Model Provenance (F1)**: Eliminated post-promotion refit entirely. SHA256 manifest locks active model weights to evaluated metrics.
2. **Current-Season Accuracy (F2)**: 2026-27 holdout achieves ML MAE **1.895 pts** vs Best Baseline **1.952 pts** (honest gain: **+0.057 pts**). Tested promotion gate enforced.
3. **True Retraining Ablation (F3)**: Real from-scratch model retraining per feature group. Top group (*Fixture Difficulty And Context*) contributes +0.93% MAE improvement with 95% bootstrap confidence intervals.
4. **Calibrated Credible Intervals (F4)**: Persisted empirical residual quantiles in `data/models/calibration.json` yield **83.78%** empirical coverage for nominal 80% credible interval ($[P_{10}, P_{90}]$).
5. **Mini-League Rivals Full Pagination (F5)**: Single source of truth `get_rival_set()` inspects all managers above user and exact 20 points below without artificial caps.
6. **Monte Carlo Clean-Sheet Realism (F6, F7)**: Deducted $4.0 \times P_{\\text{CS}}$ from defender baseline $\\mu$ to eliminate double-counting, while simulating correlated team clean-sheet Bernoulli draws.
7. **Hindsight-Free Mini-League Backtest (F8)**: Evaluated 10 historical mini-league scenarios across GW20–38 with zero future leakage, measuring honest Brier score **0.1655** (< 0.200 benchmark).
8. **NLP News Extractor Benchmark (F9)**: Raw press-conference snippets benchmarked through real regex/NLP extractor with 20/20 cases passing (100.0% precision, 0 prompt injections).
9. **Decision Card Honest Error Reporting (F10)**: Removed silent exception swallowing; unmocked end-to-end tests verify dynamic gameweek resolution and honest null error states.
10. **Joint Chip and Transfer Trajectory Planning (F11)**: Mathematical dynamic beam search evaluates joint (transfers + chips) actions across multi-GW horizon with explicit chip retention opportunity costs.
11. **Value-Based Train/Serve Feature Parity (F12)**: 100% numerical equality across all 62 canonical features verified between training kernel and live serving kernel ($|\\Delta| < 10^{-4}$).
12. **Ruff Quality & Pinned CI (F13)**: 0 ruff lint errors, repository fully formatted, ruff pinned in CI pipeline.

---

## 2. Verified Performance Metrics Table

| Metric | Value | Reference Artifact |
|:---|:---:|:---|
| **Production Model MAE** | **1.019 pts** | `reports/model_eval.json` |
| **Rank Correlation ($\\rho$)** | **0.704** | `reports/model_eval.json` |
| **2026-27 Holdout MAE** | **1.895 pts** (vs 1.952 baseline) | `reports/model_eval.json` |
| **Credible Interval Coverage** | **83.78%** (nominal 80%) | `reports/model_eval.json` |
| **Retraining Ablation (Fixture Difficulty And Context)** | **+0.93% MAE gain** | `reports/ablation.json` |
| **Historical Mini-League Brier** | **0.1655** (< 0.200) | `reports/evidence/F8-backtest.txt` |
| **News NLP Extractor Precision** | **100.0%** (20/20 verified) | `reports/news_benchmark.json` |
| **Train/Serve Parity Discrepancy** | **0.000000** on all 62 features | `reports/evidence/F12-parity.txt` |
