# Proxy Simulation: Mini-League Monte Carlo Replay (Synthetic League States)

**Generated:** 2026-10-19T17:27:43.572209  
**Methodology:** Proxy simulation across historical gameweeks with strict `assert_no_future_data` barriers.  
**Sample Size:** 26 overlapping scenarios across 3 seasons (2023-24, 2024-25, 2025-26).  
**Model Brier Score:** 0.2154  
**Naive Reference Brier (Uniform):** 0.3868  
**Naive Reference Brier (Points Lead):** 0.3310  
**Skill Demonstrated:** Yes  

> [!NOTE]
> Real mini-league standings and squad pick histories are private manager data requiring authenticated API access.
> League states and rival squad compositions are modeled as synthetic proxies subject to 100% strict FPL legality rules
> (15-man squad within £100.0m budget, 11 starters with 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, max 3 players per club).

---

## 1. Squad Points Comparison vs Naive References

| Strategy | Mean Actual Pts / GW | Uplift vs Strategy |
| :--- | :---: | :---: |
| **Clipped season-mean proxy** | **52.09** | - |
| **Previous GW Hauler** | 44.52 | +7.57 |
| **Season Average** | 51.84 | +0.26 |
| **Template (Total Points)** | 50.44 | +1.66 |

---

## 2. Real Historical Scenarios & Outcomes

| Scenario | Season | Horizon | Initial Lead | Forecast Win Prob (%) | Naive Lead Prob (%) | Winner | Brier | Legality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2023-24 GW10-15 Narrow Lead | 2023-24 | 5 GW | +12 pts | 88.3% | 25.8% | Clipped season-mean proxy | 0.0137 | PASS |
| 2023-24 GW15-20 Winter Chase | 2023-24 | 5 GW | -15 pts | 59.2% | 17.7% | Clipped season-mean proxy | 0.1665 | PASS |
| 2023-24 GW20-25 Tight Battle | 2023-24 | 5 GW | +5 pts | 82.0% | 23.5% | Season Average | 0.6724 | PASS |
| 2023-24 GW25-30 Spring Deficit | 2023-24 | 5 GW | -8 pts | 70.7% | 19.0% | Clipped season-mean proxy | 0.0858 | PASS |
| 2023-24 GW30-35 Solid Margin | 2023-24 | 5 GW | +22 pts | 98.8% | 29.4% | Clipped season-mean proxy | 0.0001 | PASS |
| 2023-24 GW33-38 Final Chase | 2023-24 | 5 GW | -18 pts | 41.6% | 17.1% | Season Average | 0.1731 | PASS |
| 2023-24 GW35-38 Late Squeeze | 2023-24 | 3 GW | +6 pts | 84.1% | 25.9% | Recent Form | 0.7073 | PASS |
| 2023-24 GW37-38 Penultimate Sprint | 2023-24 | 1 GW | -3 pts | 61.3% | 23.6% | Clipped season-mean proxy | 0.1498 | PASS |
| 2024-25 GW10-15 Early Deficit | 2024-25 | 5 GW | -10 pts | 61.0% | 18.6% | Clipped season-mean proxy | 0.1521 | PASS |
| 2024-25 GW15-20 Winter Margin | 2024-25 | 5 GW | +18 pts | 95.5% | 27.9% | Clipped season-mean proxy | 0.0020 | PASS |
| 2024-25 GW20-25 Midseason Clash | 2024-25 | 5 GW | -5 pts | 69.3% | 19.5% | Recent Form | 0.4802 | PASS |
| 2024-25 GW25-30 Spring Lead | 2024-25 | 5 GW | +14 pts | 66.9% | 26.5% | Clipped season-mean proxy | 0.1096 | PASS |
| 2024-25 GW28-33 Dead Heat | 2024-25 | 5 GW | +0 pts | 75.6% | 21.9% | Recent Form | 0.5715 | PASS |
| 2024-25 GW30-35 Deep Chase | 2024-25 | 5 GW | -20 pts | 0.0% | 16.7% | Season Average | 0.0000 | PASS |
| 2024-25 GW33-38 Run-In Finale | 2024-25 | 5 GW | +10 pts | 93.4% | 25.1% | Clipped season-mean proxy | 0.0044 | PASS |
| 2024-25 GW35-38 Late Chase | 2024-25 | 3 GW | -7 pts | 56.1% | 19.8% | Clipped season-mean proxy | 0.1927 | PASS |
| 2024-25 GW37-38 Penultimate Round | 2024-25 | 1 GW | +4 pts | 74.8% | 31.2% | Clipped season-mean proxy | 0.0635 | PASS |
| 2025-26 GW10-15 Autumn Run | 2025-26 | 5 GW | +8 pts | 73.7% | 24.4% | Recent Form | 0.5432 | PASS |
| 2025-26 GW15-20 Winter Surge | 2025-26 | 5 GW | -12 pts | 30.5% | 18.2% | Season Average | 0.0930 | PASS |
| 2025-26 GW20-25 Midseason Clash | 2025-26 | 5 GW | +15 pts | 84.8% | 26.8% | Clipped season-mean proxy | 0.0231 | PASS |
| 2025-26 GW25-30 Spring Shift | 2025-26 | 5 GW | -6 pts | 56.6% | 19.3% | Recent Form | 0.3204 | PASS |
| 2025-26 GW28-33 Spring Stretch | 2025-26 | 5 GW | +7 pts | 79.8% | 24.1% | Clipped season-mean proxy | 0.0408 | PASS |
| 2025-26 GW30-35 Run-In Pressure | 2025-26 | 5 GW | -16 pts | 32.0% | 17.5% | Recent Form | 0.1024 | PASS |
| 2025-26 GW33-38 Final Stretch | 2025-26 | 5 GW | +12 pts | 73.4% | 25.8% | Season Average | 0.5388 | PASS |
| 2025-26 GW35-38 Late Sprint | 2025-26 | 3 GW | +2 pts | 55.9% | 24.2% | Clipped season-mean proxy | 0.1945 | PASS |
| 2025-26 GW37-38 Penultimate Round | 2025-26 | 1 GW | -2 pts | 44.7% | 24.1% | Season Average | 0.1998 | PASS |

---

## 3. Reliability Calibration Curve

| Forecast Bin | Sample Count | Mean Forecast Win Prob (%) | Realized Win Frequency (%) |
| :---: | :---: | :---: | :---: |
| 0%-20% | 1 | 0.0% | 0.0% |
| 20%-40% | 2 | 31.2% | 0.0% |
| 40%-60% | 6 | 52.3% | 50.0% |
| 60%-80% | 10 | 70.6% | 60.0% |
| 80%-100% | 7 | 89.6% | 71.4% |
