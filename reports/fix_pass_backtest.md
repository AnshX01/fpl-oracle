# Monte Carlo Mini-League Historical Backtest (Zero Hindsight)

**Generated:** 2026-10-06T20:14:25.151975  
**Standard:** Honest historical evaluation with strict `assert_no_future_data` barriers.  
**Total Historical Scenarios Evaluated:** 10  
**Overall Brier Score:** 0.1655 (Benchmark: < 0.200)  

---

## 1. Real Historical Scenarios & Ground Truth Outcomes

| Scenario | Season | Horizon | Initial Lead | Forecasted Win Prob (%) | Real Historical Outcome | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 2024-25 GW20-25 Midseason Clash | 2024-25 | 5 | +15 pts | 96.8% | WON | 0.0010 |
| 2024-25 GW28-33 Spring Stretch | 2024-25 | 5 | +15 pts | 91.0% | LOST | 0.8281 |
| 2024-25 GW33-38 Run-In Finale | 2024-25 | 5 | +15 pts | 88.0% | WON | 0.0144 |
| 2024-25 GW35-38 Late Chase | 2024-25 | 3 | +15 pts | 85.2% | WON | 0.0219 |
| 2024-25 GW37-38 Penultimate Round | 2024-25 | 1 | +15 pts | 89.2% | WON | 0.0117 |
| 2025-26 GW20-25 Midseason Clash | 2025-26 | 5 | +15 pts | 88.2% | WON | 0.0139 |
| 2025-26 GW28-33 Spring Stretch | 2025-26 | 5 | +15 pts | 84.2% | WON | 0.0250 |
| 2025-26 GW33-38 Run-In Finale | 2025-26 | 5 | +15 pts | 83.8% | WON | 0.0262 |
| 2025-26 GW35-38 Late Chase | 2025-26 | 3 | +15 pts | 83.8% | LOST | 0.7022 |
| 2025-26 GW37-38 Penultimate Round | 2025-26 | 1 | +15 pts | 89.8% | WON | 0.0104 |

---

## 2. Reliability Calibration Curve

| Forecast Bin | Sample Count | Mean Forecast Win Prob (%) | Realized Win Frequency (%) |
| :---: | :---: | :---: | :---: |
| 0%-20% | 0 | 10.0% | 0.0% |
| 20%-40% | 0 | 30.0% | 0.0% |
| 40%-60% | 0 | 50.0% | 0.0% |
| 60%-80% | 0 | 70.0% | 0.0% |
| 80%-100% | 10 | 88.0% | 80.0% |
