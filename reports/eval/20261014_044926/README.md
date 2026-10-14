# INVALID EVALUATION RUN

**Run Directory:** `reports/eval/20261014_044926/`  
**Status:** **INVALID / DEPRECATED**  
**Superseded By:** `reports/eval/20261014_054543/`

### Root Cause
This evaluation run suffered from a row-index misalignment defect between `df` and `X_curr`.
Predictions for player $i$ in `X_curr` were evaluated against the target points of unrelated player $j$ from `df` due to an index mismatch between `df_sorted` and `meta`.

### Impact of Defect
- Spearman rank correlations were scrambled to near-zero (~0.01).
- P10-P90 empirical coverage on active players was artificially degraded to 55.4%.

### Verification & Resolution
The harness was corrected in commit `292b742` by deriving target points directly from `Y_curr` and `meta_curr` (shared underlying DataFrame `df_sorted`).
In the valid run (`reports/eval/20261014_054543/`):
- Active player Spearman correlation improves to **0.311** (tied) / **0.329** (ordinal).
- Active player 80% coverage improves to **73.8%** (80.2% on all players).
- Model outperforms Baseline 2 across all four positions on both RMSE and Spearman.
