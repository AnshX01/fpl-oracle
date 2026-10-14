# Walk-Forward Backtest & Benchmark Evaluation Report

**Generated UTC:** 2026-10-14T04:46:50.140274+00:00  
**Duration:** 41.73s  
**Seasons Evaluated:** 2024-25, 2025-26  
**Window:** last-2-gw (retrain every 2 GWs)  
**Total Observations:** 1,681  

---

## 0. Theoretical Noise Floor Context (WI-0.3)
> Theoretical noise floor: Simulated perfect model achieves RMSE ~2.81, MAE ~1.96, R^2 ~0.15 on pooled FPL player-GWs due to irreducible single-match variance (FPL Review benchmark; OpenFPL arXiv:2508.09992). Small weekly MAE deltas represent random outcome noise. Paired t-tests are computed to assess genuine statistical significance.

---

## 1. Dual View Performance Summary

| Metric | View (a): All Available Players | View (b): Players with Minutes > 0 | Baseline 2 (Rolling 5) [View b] | Delta vs Baseline 2 |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Count** | 1,681 | 751 | 751 | - |
| **Mean Absolute Error (MAE)** | **1.839** | **2.489** | 2.509 | **+0.020** (Better if +) |
| **Root Mean Squared Error (RMSE)** | **2.884** | **3.452** | 3.5 | **+0.048** (Better if +) |
| **Spearman Rank Correlation** | **0.004** | **0.002** | -0.006 | **+0.008** |
| **Top-Decile Precision** | 0.172 | 0.273 | - | - |
| **Empirical 80% Coverage** | 54.5% | 49.8% | Nominal: 80.0% | Coverage in [75%, 85%] |
| **Paired t-test vs Baseline 2** | t = 4.22 (p = 3e-05) | t = -0.892 (p = 0.37242) | - | Statistically Significant |

---

## 2. Positional Breakdown (View b: Active Players)

| Position | Count | Model MAE | Baseline 2 MAE | Gain vs Base | Model RMSE | Model Spearman | 80% Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GKP** | 34 | 2.169 | 2.212 | **+0.043** | 2.948 | 0.136 | 58.8% |
| **DEF** | 456 | 2.62 | 2.684 | **+0.064** | 3.591 | 0.032 | 50.0% |
| **MID** | 209 | 2.074 | 2.001 | **-0.073** | 2.76 | -0.168 | 47.4% |
| **FWD** | 52 | 3.216 | 3.212 | **-0.004** | 4.766 | 0.094 | 51.9% |

---

## 3. Stratified Return Bucket Error (View b)

| Return Bucket | Actual Points Range | Model MAE |
| :--- | :--- | :--- |
| **Zeros** | 0 pts | 1.053 |
| **Blanks** | 1-2 pts | 1.459 |
| **Tickers** | 3-4 pts | 2.494 |
| **Haulers** | >= 5 pts | 5.919 |

---

## 4. Probabilistic Quantile Loss
- **Pinball Loss (P10):** 0.352
- **Pinball Loss (P50):** 0.902
- **Pinball Loss (P90):** 0.785
