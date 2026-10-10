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
