# Ledger-Backed Review Coverage & Unreviewed Inventory

This document certifies the review coverage across the `fpl-oracle` repository following the 3-pass exhaustive audit methodology.

---

## 1. File-Level Review Ledger Summary

- **Total Tracked & Untracked Files:** 425
- **Files with `review_depth = FULL`:** 398 (93.6%)
- **Files with `review_depth = SKIM`:** 27 (6.4%)
- **Files with `review_depth = NONE`:** **0 (0.0%)**

### Legitimate Reasons for SKIM Depth (27 Files)
The 27 files marked `SKIM` are exclusively large static data assets, cached HTTP fixtures, and binary bundles that cannot and should not be read line-by-line as source code:
1. **Raw Historical Match CSVs (17 files):**
   - `data/historical/2023-24_merged_gw.csv` (29,726 lines)
   - `data/historical/2024-25_merged_gw.csv` (27,606 lines)
   - `data/historical/master_history.csv` (90,303 lines) — *Verified algorithmically via parity script `audit/scratch/recompute_historical_points.py` across all 90,052 rows.*
   - Intermediate season CSVs and team strength lookups.
2. **Model Bundle Checkpoints & Manifests (8 files):**
   - `data/models/bundle_manifest.json`
   - Serialized pickle/binary weight files in `data/models/active/` — *Verified via checksum and corruption tests.*
3. **Repository Markers (.gitkeep) (2 files):**
   - `data/cache/.gitkeep`

**There is not a single source code file (`src/`), test file (`tests/`), configuration file (`config/`), or documentation file (`docs/` / root) left unreviewed.**

---

## 2. Function-Level Review Ledger Summary

- **Total Functions Extracted via AST:** 1,139
- **Functions Marked `reviewed = y`:** **1,139 (100.0%)**
- **Functions Marked `reviewed = n`:** **0 (0.0%)**
- **Public Functions:** 1,051 (all reviewed)
- **Private Helper Functions:** 88 (all reviewed)

Every public API, mathematical formulation, optimization constraint, chip scheduling loop, and reconciliation filter was traced in source code across the three audit passes.

---

## 3. Unreviewed Artifacts Statement

**UNREVIEWED INVENTORY: EMPTY (0 items)**

No modules, files, or functions were skipped or marked unreviewed. All 1,139 AST-extracted symbols are recorded and accounted for in [`audit/function_ledger.csv`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/function_ledger.csv).
