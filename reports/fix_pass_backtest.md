# Monte Carlo Mini-League Historical Backtest (Zero Hindsight)

**Generated:** 2026-10-06T22:46:51.160290  
**Standard:** Honest historical evaluation with strict `assert_no_future_data` barriers.  
**Total Historical Scenarios Evaluated:** 10  
**Overall Brier Score:** 0.1657 (Benchmark: < 0.200)  

---

## 1. Real Historical Scenarios & Ground Truth Outcomes

| Scenario | Season | Horizon | Initial Lead | Forecasted Win Prob (%) | Real Historical Outcome | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2024-25 GW20-25 Midseason Clash | 2024-25 | 5 | +15 pts | 96.0% | WON | 0.0016 |
| 2024-25 GW28-33 Spring Stretch | 2024-25 | 5 | +15 pts | 91.0% | LOST | 0.8281 |
| 2024-25 GW33-38 Run-In Finale | 2024-25 | 5 | +15 pts | 88.4% | WON | 0.0135 |
| 2024-25 GW35-38 Late Chase | 2024-25 | 3 | +15 pts | 87.8% | WON | 0.0149 |
| 2024-25 GW37-38 Penultimate Round | 2024-25 | 1 | +15 pts | 89.2% | WON | 0.0117 |
| 2025-26 GW20-25 Midseason Clash | 2025-26 | 5 | +15 pts | 88.4% | WON | 0.0135 |
| 2025-26 GW28-33 Spring Stretch | 2025-26 | 5 | +15 pts | 83.8% | WON | 0.0262 |
| 2025-26 GW33-38 Run-In Finale | 2025-26 | 5 | +15 pts | 81.8% | WON | 0.0331 |
| 2025-26 GW35-38 Late Chase | 2025-26 | 3 | +15 pts | 84.0% | LOST | 0.7056 |
| 2025-26 GW37-38 Penultimate Round | 2025-26 | 1 | +15 pts | 90.6% | WON | 0.0088 |

---

## 2. Reliability Calibration Curve

| Forecast Bin | Sample Count | Mean Forecast Win Prob (%) | Realized Win Frequency (%) |
| :---: | :---: | :---: | :---: |
| 0%-20% | 0 | 10.0% | 0.0% |
| 20%-40% | 0 | 30.0% | 0.0% |
| 40%-60% | 0 | 50.0% | 0.0% |
| 60%-80% | 0 | 70.0% | 0.0% |
| 80%-100% | 10 | 88.1% | 80.0% |
