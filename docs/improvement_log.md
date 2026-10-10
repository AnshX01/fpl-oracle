# FPL Oracle Improvement Log

Record of changes, metrics before/after, and acceptance criteria per work item.

---

## Baseline Verification
- **Date:** 2026-10-10
- **Test Suite:** 426 tests passed (100% pass rate) in 326.90s.
- **Historical Dataset:** `master_history.csv` with 90,303 player-GW records across 2023-24, 2024-25, 2025-26, and 2026-27.

---

## WI-1.1: Centralised Rules Module and Scoring Contract Tests
- **Summary:** Created `src/fpl_oracle/rules/scoring.py` and `config/rules_2026_27.yaml` as single source of truth for all game rules and scoring constants. Fixed defect D1 (corrected GKP goal points from 10 to official 6). Centralized imports in `domain/scoring.py`, `ml/ensemble.py`, and `config/scoring.yaml`.
- **Tests Added:** `tests/rules/test_scoring_contract.py` (4 tests).
  - Verified 100% exact match across all 3,216 rows of 2026-27 season and 29,725 rows of 2023-24 season.
  - Added grep-based test ensuring no stray `10 *` or hardcoded goal-rate dictionaries outside `rules/scoring.py`.
- **Test Suite Result:** 430/430 tests passing (100% pass rate).

---

## WI-1.3 & WI-1.4: Pure Transfer State Machine & Selling Price Oracle
- **Summary:**
  - Implemented `src/fpl_oracle/rules/transfers.py` with pure functions:
    - `sell_price(purchase_tenths, current_tenths) -> int`: Exact FPL 50% profit retention rounded down (tenths of a million), passes through full price drop.
    - `sell_price_millions(purchase, current) -> float`: Float wrapper for convenience.
    - `next_free_transfers(ft, transfers_made, chip=None, *, config=None) -> (ft_next, hit_points)`: Pure state transition across normal GWs and chip deployments (WC/FH preserves FTs per R4, configurable `preserve` vs `preserve_plus_one`).
    - `validate_chip_legality(chip, target_gw, chips_already_used, previous_chip)`: Validates 2026/27 chip rules (Set 1 GW1-19, Set 2 GW20-38, no back-to-back Free Hit across GW19/20, GW1 Free Hit restriction).
  - Delegated `ManagerStateService.calculate_selling_price`, `ManagerStateService.calculate_banked_free_transfers`, and `TransferOptimizer` routines directly to `fpl_oracle.rules.transfers`.
- **Tests Added:** `tests/rules/test_transfers.py` (24 table-driven tests verifying selling price edge cases, transfer banking sequences across chips, and chip legality validation).
- **Test Suite Result:** 454/454 tests passing (100% pass rate).
