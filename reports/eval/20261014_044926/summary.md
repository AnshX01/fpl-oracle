# Walk-Forward Backtest & Benchmark Evaluation Report

**Generated UTC:** 2026-10-14T04:47:43.219376+00:00  
**Duration:** 103.34s  
**Seasons Evaluated:** 2024-25, 2025-26  
**Window:** last-20-gw (retrain every 4 GWs)  
**Total Observations:** 16,352  

---

## 0. Theoretical Noise Floor Context (WI-0.3)
> Theoretical noise floor: Simulated perfect model achieves RMSE ~2.81, MAE ~1.96, R^2 ~0.15 on pooled FPL player-GWs due to irreducible single-match variance (FPL Review benchmark; OpenFPL arXiv:2508.09992). Small weekly MAE deltas represent random outcome noise. Paired t-tests are computed to assess genuine statistical significance.

---

## 1. Dual View Performance Summary

| Metric | View (a): All Available Players | View (b): Players with Minutes > 0 | Baseline 2 (Rolling 5) [View b] | Delta vs Baseline 2 |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Count** | 16,352 | 6,207 | 6,207 | - |
| **Mean Absolute Error (MAE)** | **1.685** | **2.453** | 2.486 | **+0.033** (Better if +) |
| **Root Mean Squared Error (RMSE)** | **2.713** | **3.498** | 3.549 | **+0.050** (Better if +) |
| **Spearman Rank Correlation** | **0.005** | **0.017** | 0.016 | **+0.001** |
| **Top-Decile Precision** | 0.112 | 0.144 | - | - |
| **Empirical 80% Coverage** | 59.1% | 55.4% | Nominal: 80.0% | Coverage in [75%, 85%] |
| **Paired t-test vs Baseline 2** | t = 5.784 (p = 0.0) | t = -4.176 (p = 3e-05) | - | Statistically Significant |

---

## 2. Positional Breakdown (View b: Active Players)

| Position | Count | Model MAE | Baseline 2 MAE | Gain vs Base | Model RMSE | Model Spearman | 80% Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GKP** | 310 | 2.603 | 2.67 | **+0.068** | 3.603 | 0.041 | 48.4% |
| **DEF** | 2,442 | 2.616 | 2.654 | **+0.037** | 3.65 | -0.002 | 54.4% |
| **MID** | 2,846 | 2.297 | 2.323 | **+0.026** | 3.338 | 0.034 | 56.4% |
| **FWD** | 609 | 2.452 | 2.488 | **+0.036** | 3.553 | 0.013 | 58.3% |

---

## 3. Stratified Return Bucket Error (View b)

| Return Bucket | Actual Points Range | Model MAE |
| :--- | :--- | :--- |
| **Zeros** | 0 pts | 1.025 |
| **Blanks** | 1-2 pts | 1.408 |
| **Tickers** | 3-4 pts | 2.408 |
| **Haulers** | >= 5 pts | 6.31 |

---

## 4. Probabilistic Quantile Loss
- **Pinball Loss (P10):** 0.31
- **Pinball Loss (P50):** 0.824
- **Pinball Loss (P90):** 0.698
