# Walk-Forward Backtest & Benchmark Evaluation Report

**Generated UTC:** 2026-10-14T10:52:27.017605+00:00
**Duration:** 83.21s
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
| **Mean Absolute Error (MAE)** | **0.973** | **2.106** | 2.203 | **+0.097** (Better if +) |
| **Root Mean Squared Error (RMSE)** | **1.912** | **2.956** | 3.133 | **+0.177** (Better if +) |
| **Spearman (With Ties)** | **0.705** | **0.311** | 0.266 | **+0.045** |
| **Spearman (Without Ties, Ordinal)** | **0.617** | **0.329** | 0.26 | **+0.069** |
| **Top-Decile Precision** | 0.372 | 0.201 | - | - |
| **Empirical 80% Coverage** | 85.3% | 78.8% | Nominal: 80.0% | Coverage in [75%, 85%] |

---

## 2. Benchmark Correlations on Active Players (View b)
| Feature / Predictor | Spearman (With Ties) | Spearman (Without Ties) | Target Correlation Status |
| :--- | :--- | :--- | :--- |
| **Model xP** | **0.311** | **0.329** | **Primary Evaluation Target** |
| **Baseline 2: Rolling 5 Appearances** | 0.266 | 0.26 | Benchmark |
| **Baseline 1: Season-to-Date Mean** | 0.235 | 0.233 | Benchmark (~0.2+) |
| **Baseline 5: Previous GW Points (Lag 1)** | 0.234 | 0.253 | Benchmark (~0.2+) |
| **Player Price / Value** | 0.118 | 0.123 | Benchmark (~0.2+) |

---

## 3. Gameweek Block Bootstrap (Replacing Row-Level t-test)
- **Number of Evaluated Gameweeks:** 20
- **MAE Mean Gain vs Baseline 2:** 0.0989
- **MAE 95% Bootstrap CI:** [0.0511, 0.1469]
- **MAE Bootstrap p-value:** < 1e-4
- **RMSE Mean Gain vs Baseline 2:** 0.1773
- **RMSE 95% Bootstrap CI:** [0.1548, 0.2]
- **RMSE Bootstrap p-value:** < 1e-4

---

## 4. Positional Breakdown (View b: Active Players)

| Position | Count | Model MAE | Base 2 MAE | Gain vs Base | Model RMSE | Base 2 RMSE | Model Spearman | Base 2 Spearman | Coverage (80%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GKP** | 405 | 2.167 | 2.374 | **+0.208** | 2.906 | 3.102 | 0.079 | 0.049 | 71.9% |
| **DEF** | 2,080 | 2.257 | 2.396 | **+0.139** | 3.083 | 3.288 | 0.223 | 0.174 | 77.9% |
| **MID** | 2,727 | 1.961 | 2.02 | **+0.059** | 2.823 | 2.977 | 0.384 | 0.344 | 80.1% |
| **FWD** | 792 | 2.175 | 2.24 | **+0.064** | 3.085 | 3.252 | 0.41 | 0.361 | 80.1% |

---

## 5. Stratified Return Bucket Error (View b)

| Return Bucket | Actual Points Range | Model MAE |
| :--- | :--- | :--- |
| **Zeros** | 0 pts | 2.149 |
| **Blanks** | 1-2 pts | 1.382 |
| **Tickers** | 3-4 pts | 1.267 |
| **Haulers** | >= 5 pts | 4.439 |

---

## 6. Probabilistic Quantile Loss
- **Pinball Loss (P10):** 0.137
- **Pinball Loss (P50):** 0.479
- **Pinball Loss (P90):** 0.355
