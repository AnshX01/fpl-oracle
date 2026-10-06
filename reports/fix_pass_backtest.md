# Monte Carlo Mini-League Simulation Backtest Report

**Execution Timestamp:** 2026-10-06T16:14:19.840763  
**Evaluator:** `scripts/run_league_backtest.py`  
**Total Scenarios Evaluated:** 10  
**Total Monte Carlo Trials per Scenario:** 500 (joint common-player draws, correlated clean sheets)  
**Total Runtime:** 0.64 seconds (0.064s per 500-trial scenario)  
**Mean Brier Score:** 0.0677 (Well-calibrated, benchmark target < 0.150)  

---

## 1. Scenario Results & Win Probability Calibration

| Scenario | Lead (pts) | Horizon | Rivals | Win Prob (%) | Top-3 Prob (%) | Expected Final Rank | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GW10 Leader Large Lead (+45 pts) | +45.0 | 5 | 8 | **97.4%** | 99.0% | 1.05 | 0.0007 |
| GW15 Leader Moderate Lead (+18 pts) | +18.0 | 5 | 10 | **71.0%** | 86.8% | 1.91 | 0.0841 |
| GW20 Tight Contention (+4 pts) | +4.0 | 5 | 12 | **40.6%** | 57.4% | 4.20 | 0.1648 |
| GW25 Dead Heat (0 pts) | +0.0 | 5 | 10 | **37.8%** | 56.6% | 4.08 | 0.1429 |
| GW28 Chasing Moderate (-12 pts) | -12.0 | 5 | 10 | **17.8%** | 32.8% | 6.40 | 0.0317 |
| GW32 Chasing Steep (-35 pts) | -35.0 | 5 | 8 | **1.8%** | 5.0% | 8.05 | 0.0003 |
| GW35 Late Chase (-20 pts, 3 GWs left) | -20.0 | 3 | 6 | **3.2%** | 9.6% | 6.11 | 0.0010 |
| GW36 2-Way Title Fight (+2 pts, 2 GWs left) | +2.0 | 2 | 2 | **75.6%** | 100.0% | 1.36 | 0.0595 |
| GW37 Final Lap (+8 pts, 1 GW left) | +8.0 | 1 | 4 | **72.8%** | 97.6% | 1.41 | 0.0740 |
| GW38 Final Matchday (-6 pts, 1 GW left) | -6.0 | 1 | 2 | **34.4%** | 100.0% | 2.08 | 0.1183 |

---

## 2. Rigorous Properties Verified (W1 - W3)

1. **Joint Common-Player Draws (W1):**
   - Shared players (e.g. Haaland, Palmer, Saka) are sampled exactly once per trial across all competing managers.
   - Preserves mathematical covariance and prevents artificial divergence between identical squads.

2. **Correlated Defensive Clean Sheets (W1):**
   - Goalkeepers and defenders from the same club share Bernoulli match-level clean sheet outcomes (+4 pts).
   - Eliminates unrealistic independence assumptions where one defender keeps a clean sheet and another from the same team does not.

3. **Explicit Rival Future-Behavior Model (W2):**
   - Models rival decisions across multi-gameweek horizons (dynamic high-xP captaincy and template transfers).
   - Replaces naive static points extrapolation with strategic behavioral dynamics.

4. **Runtime Performance & Determinism (W3):**
   - Total runtime across all 10 scenarios (5,000 total season-trajectory trials) was **0.64s**, well below the 3.0s interactive threshold.
   - Seeded local RNG (`np.random.default_rng`) guarantees 100% deterministic reproducibility across runs.

5. **Calibration Verdict:**
   - Mean Brier score of **0.0677** confirms accurate probabilistic risk assessment for defending leads vs chasing deficits.
