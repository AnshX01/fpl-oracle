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

## Phase 0: Evaluation Harness & Baselines (In Progress)
- **WI-0.1:** Walk-forward backtest runner (`src/fpl_oracle/eval/backtest.py`, `scripts/run_backtest.py`).
- **WI-0.2:** Baselines & `ep_next` snapshotting (`holdout_forward` extension to `data/snapshots/`).
