# Walk-Forward Backtest & Benchmark Evaluation Report

**Generated UTC:** 2026-10-15T11:30:41.780206+00:00
**Duration:** 137.95s
**Seasons Evaluated:** 2025-26
**Window:** last-20-gw (retrain every 4 GWs)
**Total Observations:** 16,352

---

## 0. Theoretical Noise Floor & Statistical Framework (WI-0.3)
> Theoretical noise floor: Simulated perfect model achieves RMSE ~2.81, MAE ~1.96, R^2 ~0.15 on pooled FPL player-GWs due to irreducible single-match variance (FPL Review benchmark; OpenFPL arXiv:2508.09992). Small weekly MAE deltas represent random outcome noise. Gameweek block bootstrapping is computed to assess genuine statistical significance without naive i.i.d. row assumptions.

---

## 1. Dual View Performance Summary

| Metric | View (a): All Available Players | View (b): Active Players (Mins > 0) | Baseline 2 (Rolling 5) [View b] | Delta vs Baseline 2 |
| :--- | :--- | :--- | :--- | :--- |
| **Observation Count** | 16,352 | 6,004 | 6,004 | - |
| **Mean Absolute Error (MAE)** | **0.905** | **2.01** | 2.203 | **+0.193** (Better if +) |
| **Root Mean Squared Error (RMSE)** | **1.883** | **2.97** | 3.133 | **+0.163** (Better if +) |
| **Spearman (With Ties)** | **0.707** | **0.33** | 0.266 | **+0.064** |
| **Spearman (Without Ties, Ordinal)** | **0.618** | **0.318** | 0.26 | **+0.058** |
| **Top-Decile Precision** | 0.382 | 0.21 | - | - |
| **Empirical 80% Coverage** | 85.2% | 78.7% | Nominal: 80.0% | Coverage in [75%, 85%] |

---

## 2. Benchmark Correlations on Active Players (View b)
| Feature / Predictor | Spearman (With Ties) | Spearman (Without Ties) | Target Correlation Status |
| :--- | :--- | :--- | :--- |
| **Model xP** | **0.33** | **0.318** | **Primary Evaluation Target** |
| **Baseline 2: Rolling 5 Appearances** | 0.266 | 0.26 | Benchmark |
| **Baseline 1: Season-to-Date Mean** | 0.235 | 0.233 | Benchmark (~0.2+) |
| **Baseline 5: Previous GW Points (Lag 1)** | 0.234 | 0.253 | Benchmark (~0.2+) |
| **Player Price / Value** | 0.118 | 0.123 | Benchmark (~0.2+) |

---

## 3. Gameweek Block Bootstrap (Replacing Row-Level t-test)
- **Number of Evaluated Gameweeks:** 20
- **MAE Mean Gain vs Baseline 2:** 0.1948
- **MAE 95% Bootstrap CI:** [0.1673, 0.222]
- **MAE Bootstrap p-value:** < 1e-4
- **RMSE Mean Gain vs Baseline 2:** 0.1634
- **RMSE 95% Bootstrap CI:** [0.1388, 0.1883]
- **RMSE Bootstrap p-value:** < 1e-4

---

## 4. Positional Breakdown (View b: Active Players)

| Position | Count | Model MAE | Base 2 MAE | Gain vs Base | Model RMSE | Base 2 RMSE | Model Spearman | Base 2 Spearman | Coverage (80%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GKP** | 405 | 2.178 | 2.374 | **+0.196** | 2.913 | 3.102 | 0.074 | 0.049 | 72.3% |
| **DEF** | 2,080 | 2.177 | 2.396 | **+0.218** | 3.113 | 3.288 | 0.247 | 0.174 | 77.6% |
| **MID** | 2,727 | 1.846 | 2.02 | **+0.174** | 2.824 | 2.977 | 0.398 | 0.344 | 80.0% |
| **FWD** | 792 | 2.05 | 2.24 | **+0.190** | 3.1 | 3.252 | 0.421 | 0.361 | 80.6% |

---

## 5. Stratified Return Bucket Error (View b)

| Return Bucket | Actual Points Range | Model MAE |
| :--- | :--- | :--- |
| **Zeros** | 0 pts | 1.738 |
| **Blanks** | 1-2 pts | 1.06 |
| **Tickers** | 3-4 pts | 1.3 |
| **Haulers** | >= 5 pts | 4.918 |

---

## 6. Probabilistic Quantile Loss
- **Pinball Loss (P10):** 0.135
- **Pinball Loss (P50):** 0.448
- **Pinball Loss (P90):** 0.355
