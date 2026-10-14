# FPL Oracle Improvement Log

Record of changes, metrics before/after, and acceptance criteria per work item.

---

## Baseline Verification
- **Date:** 2026-10-10
- **Test Suite:** 426 tests passed (100% pass rate) in 326.90s.
- **Historical Dataset:** `master_history.csv` with 90,303 player-GW records across 2023-24, 2024-25, 2025-26, and 2026-27.

---

## WI-1.1: Centralised Rules Module & All-Seasons Scoring Contract Tests
- **Date:** 2026-10-10 / 2026-10-14
- **Commits:** `3e4070a`, `82afc35` (`WI-1.1: Scoring contract test covering all historical seasons with rules_version`)
- **Summary:**
  - Created `src/fpl_oracle/rules/scoring.py` as single source of truth for all game rules and scoring constants.
  - Fixed defect D1 (corrected GKP goal points from 10 to official 6).
  - Extended `calculate_match_points` to support `rules_version` parameter (`2023-standard` without DefCon for 2023-24 & 2024-25; `2025-defcon` with DefCon active for 2025-26 & 2026-27).
- **Historical Contract Results across 90,303 rows in `master_history.csv`:**
  - `2023-24`: 29,725 rows -> 0 mismatches.
  - `2024-25`: 27,676 rows -> 27,354 player rows (0 mismatches), exactly 322 rows (251 with points > 0) belonging to 21 Premier League manager elements from the mid-season FPL Challenge trial (Arteta, Guardiola, Emery, Slot, etc.).
  - `2025-26`: 29,757 rows -> 0 mismatches.
  - `2026-27`: 3,216 rows -> 0 mismatches.
  - **Player mismatch rate: 0.000% across all 90,052 player rows.**
- **Tests Added:** `tests/rules/test_scoring_contract.py` (5 tests passing).

---

## WI-1.2: Ensemble Completeness & Exact Arithmetic Decomposition
- **Date:** 2026-10-11
- **Implementation:** `src/fpl_oracle/ml/ensemble.py`
- **Tests:** `tests/rules/test_ensemble_completeness.py` (8 acceptance tests passing).
- **Summary:**
  - Enforced exact arithmetic decomposition:
    `raw_expected_points == exp_appearance + exp_goals_pts + exp_assists_pts + exp_cs_pts + exp_defcon_pts + exp_bonus_pts + exp_saves_pts + exp_pen_saves_pts - exp_gc_deduction - exp_card_deduction - exp_og_deduction - exp_pen_miss_deduction` holds with 0 error across all rows.
  - Conceded goals deduction applies strictly to GKP and DEF only (0.0 for MID/FWD).
  - Clean sheet points require $\ge 60$ minutes (`p_min60`).
  - DefCon actions never apply to GKP.
  - Added expectations for own goals (-2 pts per expected own goal) and penalties missed (-2 pts per expected miss).

---

## WI-1.3 & WI-1.4: Pure Transfer State Machine & Selling Price Oracle
- **Date:** 2026-10-11
- **Summary:**
  - Implemented `src/fpl_oracle/rules/transfers.py` with pure functions:
    - `sell_price(purchase_tenths, current_tenths) -> int`: Exact FPL 50% profit retention rounded down (tenths of a million), passes through full price drop.
    - `sell_price_millions(purchase, current) -> float`: Float wrapper for convenience.
    - `next_free_transfers(ft, transfers_made, chip=None, *, config=None) -> (ft_next, hit_points)`: Pure state transition across normal GWs and chip deployments (WC/FH preserves FTs per R4, configurable `preserve` vs `preserve_plus_one`).
    - `validate_chip_legality(chip, target_gw, chips_already_used, previous_chip)`: Validates 2026/27 chip rules (Set 1 GW1-19, Set 2 GW20-38, no back-to-back Free Hit across GW19/20, GW1 Free Hit restriction per Rule R2).
  - Delegated `ManagerStateService.calculate_selling_price`, `ManagerStateService.calculate_banked_free_transfers`, and `TransferOptimizer` routines directly to `fpl_oracle.rules.transfers`.
- **Tests Added:** `tests/rules/test_transfers.py` (24 table-driven tests).

---

## WI-1.5: Recompute Points for All Finalised Current-Season Gameweeks
- **Status:** NOT DONE. Scheduled for execution in Phase 1 following completion of Phase 0 backtesting infrastructure.

---

## Adversarial Audit Hotfix Phase (H0–H2) Summary
- **Date:** 2026-10-13 / 2026-10-14
- **Commits:** `2b1162b` through `8b5c809`
- **Findings Remediated:**
  - `F-001`: Muted `httpx`/`httpcore`, regex secret redaction filter for logs & exceptions, sanitized local logs, confirmed 0 real keys in git history.
  - `F-002`: Exact fractional tie-handling `1.0 / tied_count` in Monte Carlo simulation and candidate robustness ranking.
  - `F-003`: Publisher allowlist, syndication clustering (shared wire agency / canonical URL), configurable single-source cap.
  - `F-004`: Dynamic Poisson summation bound terminating at residual `< 1e-12`, eliminating truncation bias up to rate 60.
  - `F-006` & `F-008`: Automatic model bundle quarantine & rollback to last verified checkpoint; cross-process `filelock.FileLock`.
  - `F-007`: Automatic multi-chip joint schedule assigning chips to distinct gameweeks; early Set 1 warning from GW14.
  - `F-009`: Rolling-origin temporal CV across 2026/27 splits; proposal logged (1.0x DefCon, 1.0x-1.5x Bonus). Production weights unchanged.
  - `F-010`: Disambiguated shared surnames using club and position context; unresolved ambiguities drop match to `None`.
  - `F-011`: Configurable risk lineup parameters with flag default `false`; verified real squad sensitivity.
  - `F-013`: Removed brittle wall-clock elapsed time assertions.
  - `F-014`: Resolved 20 ruff lint errors across all modules.
  - `F-015`: Verified mypy passes with 0 errors across all 89 source files without type-silencing.
  - `F-016`: Documented all 62 columns in architecture doc with parity CI test.
  - `Scenarios 2 & 3`: Verified full trajectory optimizer recommends `WILDCARD` in injury crisis and `FREE_HIT` in legal blank gameweek (GW10).
  - `F-005` & `F-012`: Protected by strict `xfail` regression tests tracking Spec `WI-2.1` and `WI-3.1`.
- **Suite Quality Gates:**
  - Total Tests: 497 items (495 passed, 2 strict xfailed).
  - Line Coverage: 75.8% (8,868 / 11,698 stmts) [Target >= 75.4%].
  - Branch Coverage: 73.0% (3,708 branches) [Target >= 73.0%].
  - `mypy src`: Checked 89 source files, 0 errors.
  - `ruff check .`: 0 errors.

---

## Phase 0: Evaluation Harness & Baselines (Completed & Validated)
- **Date:** 2026-10-14
- **Modules Created/Updated:**
  - `src/fpl_oracle/eval/backtest.py`: `WalkForwardBacktester` implementing strict temporal walk-forward evaluation, retraining barriers (`--retrain-every k`), dual evaluation views (View a: all available players; View b: played players with minutes > 0), return-bucket stratification, 6 baselines, tied & untied (ordinal) Spearman correlations, and gameweek block bootstrap.
  - `src/fpl_oracle/data/features.py`: Added strictly shifted `prev_points` (lag 1) and expanding `season_points_mean` to `meta_cols` and resolved index alignment so `X_curr`, `Y_curr`, and `meta_curr` share identical sorted row order with 0 index scrambling.
  - `src/fpl_oracle/data/odds.py`: Added `OddsProvider`, `FootballDataOddsProvider`, Shin and power de-vigging algorithms, and `snapshot_odds` saving raw pre-deadline match odds into `data/snapshots/odds/`.
  - `src/fpl_oracle/server/jobs.py`: Added `odds_snapshot_job` scheduled hourly and callable on-demand via `run_job_on_demand("odds_snapshot")`.
  - `scripts/run_backtest.py`: CLI entry point with gameweek block bootstrap reporting.
  - `tests/test_eval_backtest.py`: 6 unit and acceptance tests covering backtest structure, official EP snapshots, tied/untied Spearman, gameweek block bootstrap, and odds devigging/snapshotting.
- **Acceptance Run:**
  - Command: `python scripts/run_backtest.py --seasons 2024-25 2025-26 --window last-20-gw --retrain-every 4`
  - Duration: 113.61s (< 2 minutes, well within 30 min budget).
  - Evaluated Observations: 16,352 player-gameweek observations (6,004 active appearances).
  - Reports Generated: `reports/eval/20261014_054543/summary.json` and `summary.md`.
- **Results vs Baseline 2 (Rolling 5 Appearances):**
  - **View (a) All Available Players (16,352 rows):** Model MAE 0.973 | RMSE 1.912 | Spearman (Tied): 0.705 | Spearman (NoTies): 0.617 | 80% Coverage: 80.2%.
  - **View (b) Active Players Only (6,004 rows):**
    - Model MAE: **2.106** vs Baseline 2 **2.203** (Gain: **+0.097**).
    - Model RMSE: **2.956** vs Baseline 2 **3.133** (Gain: **+0.177**).
    - Spearman (With Ties): Model **0.311** vs Baseline 2 **0.266** (Gain: **+0.045**).
    - Spearman (Without Ties, Ordinal): Model **0.329** vs Baseline 2 **0.260** (Gain: **+0.069**).
    - Empirical 80% Coverage: **73.8%** (Nominal: 80.0%).
    - **Gameweek Block Bootstrap (B=10,000 resamples):**
      - MAE Gain 95% CI: **[+0.0511, +0.1469]** ($p = 0.00000$).
      - RMSE Gain 95% CI: **[+0.1548, +0.2000]** ($p = 0.00000$).
  - **Benchmark Feature Correlations on Active Players (View b):**
    - Model xP: **0.311** (tied) / **0.329** (ordinal)
    - Baseline 2 (Rolling 5): **0.266** (tied) / **0.260** (ordinal)
    - Baseline 1 (Season-to-Date Mean): **0.235** (tied) / **0.233** (ordinal)
    - Baseline 5 (Previous GW Points): **0.234** (tied) / **0.253** (ordinal)
    - Player Price / Value: **0.118** (tied) / **0.123** (ordinal)
  - **Positional Breakdown (Active Players):**
    - GKP (405 rows): Model MAE 2.167 (Gain +0.208) | RMSE 2.906 vs 3.102 (Gain +0.196) | Spearman 0.079 vs 0.049. Model beats Baseline 2 on all metrics.
    - DEF (2,080 rows): Model MAE 2.257 (Gain +0.139) | RMSE 3.083 vs 3.288 (Gain +0.205) | Spearman 0.223 vs 0.174. Model beats Baseline 2 on all metrics.
    - MID (2,727 rows): Model MAE 1.961 (Gain +0.059) | RMSE 2.823 vs 2.977 (Gain +0.154) | Spearman 0.384 vs 0.344. Model beats Baseline 2 on all metrics.
    - FWD (792 rows): Model MAE 2.175 (Gain +0.064) | RMSE 3.085 vs 3.252 (Gain +0.167) | Spearman 0.410 vs 0.361. Model beats Baseline 2 on all metrics.


