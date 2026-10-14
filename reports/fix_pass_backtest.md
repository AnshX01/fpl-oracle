# Proxy Simulation: Mini-League Monte Carlo Replay (Synthetic League States)

**Generated:** 2026-10-14T16:31:40.555067  
**Methodology:** Proxy simulation across historical gameweeks with strict `assert_no_future_data` barriers.  
**Sample Size:** 26 overlapping scenarios across 3 seasons (2023-24, 2024-25, 2025-26).  
**Model Brier Score:** 0.2530  
**Naive Reference Brier (Uniform):** 0.4380  
**Naive Reference Brier (Points Lead):** 0.3740  
**Skill Demonstrated:** Yes  

> [!NOTE]
> Real mini-league standings and squad pick histories are private manager data requiring authenticated API access.
> League states and rival squad compositions are modeled as synthetic proxies subject to 100% strict FPL legality rules
> (15-man squad within £100.0m budget, 11 starters with 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, max 3 players per club).

---

## 1. Squad Points Comparison vs Naive References

| Strategy | Mean Actual Pts / GW | Uplift vs Strategy |
| :--- | :---: | :---: |
| **Clipped season-mean proxy** | **52.89** | - |
| **Previous GW Hauler** | 43.76 | +9.13 |
| **Season Average** | 51.63 | +1.26 |
| **Template (Total Points)** | 50.44 | +2.45 |

---

## 2. Real Historical Scenarios & Outcomes

| Scenario | Season | Horizon | Initial Lead | Forecast Win Prob (%) | Naive Lead Prob (%) | Winner | Brier | Legality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2023-24 GW10-15 Narrow Lead | 2023-24 | 5 GW | +12 pts | 95.8% | 25.8% | Clipped season-mean proxy | 0.0018 | PASS |
| 2023-24 GW15-20 Winter Chase | 2023-24 | 5 GW | -15 pts | 54.6% | 17.7% | Clipped season-mean proxy | 0.2061 | PASS |
| 2023-24 GW20-25 Tight Battle | 2023-24 | 5 GW | +5 pts | 96.0% | 23.5% | Season Average | 0.9216 | PASS |
| 2023-24 GW25-30 Spring Deficit | 2023-24 | 5 GW | -8 pts | 77.4% | 19.0% | Clipped season-mean proxy | 0.0511 | PASS |
| 2023-24 GW30-35 Solid Margin | 2023-24 | 5 GW | +22 pts | 100.0% | 29.4% | Clipped season-mean proxy | 0.0000 | PASS |
| 2023-24 GW33-38 Final Chase | 2023-24 | 5 GW | -18 pts | 31.4% | 17.1% | Season Average | 0.0986 | PASS |
| 2023-24 GW35-38 Late Squeeze | 2023-24 | 3 GW | +6 pts | 96.2% | 25.9% | Clipped season-mean proxy | 0.0014 | PASS |
| 2023-24 GW37-38 Penultimate Sprint | 2023-24 | 1 GW | -3 pts | 76.6% | 23.6% | Clipped season-mean proxy | 0.0548 | PASS |
| 2024-25 GW10-15 Early Deficit | 2024-25 | 5 GW | -10 pts | 81.2% | 18.6% | Clipped season-mean proxy | 0.0353 | PASS |
| 2024-25 GW15-20 Winter Margin | 2024-25 | 5 GW | +18 pts | 99.8% | 27.9% | Clipped season-mean proxy | 0.0000 | PASS |
| 2024-25 GW20-25 Midseason Clash | 2024-25 | 5 GW | -5 pts | 85.0% | 19.5% | Recent Form | 0.7225 | PASS |
| 2024-25 GW25-30 Spring Lead | 2024-25 | 5 GW | +14 pts | 100.0% | 26.5% | Clipped season-mean proxy | 0.0000 | PASS |
| 2024-25 GW28-33 Dead Heat | 2024-25 | 5 GW | +0 pts | 99.4% | 21.9% | Recent Form | 0.9880 | PASS |
| 2024-25 GW30-35 Deep Chase | 2024-25 | 5 GW | -20 pts | 0.0% | 16.7% | Season Average | 0.0000 | PASS |
| 2024-25 GW33-38 Run-In Finale | 2024-25 | 5 GW | +10 pts | 100.0% | 25.1% | Clipped season-mean proxy | 0.0000 | PASS |
| 2024-25 GW35-38 Late Chase | 2024-25 | 3 GW | -7 pts | 92.4% | 19.8% | Clipped season-mean proxy | 0.0058 | PASS |
| 2024-25 GW37-38 Penultimate Round | 2024-25 | 1 GW | +4 pts | 93.2% | 31.2% | Clipped season-mean proxy | 0.0046 | PASS |
| 2025-26 GW10-15 Autumn Run | 2025-26 | 5 GW | +8 pts | 98.4% | 24.4% | Recent Form | 0.9683 | PASS |
| 2025-26 GW15-20 Winter Surge | 2025-26 | 5 GW | -12 pts | 43.2% | 18.2% | Clipped season-mean proxy | 0.3226 | PASS |
| 2025-26 GW20-25 Midseason Clash | 2025-26 | 5 GW | +15 pts | 99.4% | 26.8% | Clipped season-mean proxy | 0.0000 | PASS |
| 2025-26 GW25-30 Spring Shift | 2025-26 | 5 GW | -6 pts | 85.2% | 19.3% | Recent Form | 0.7259 | PASS |
| 2025-26 GW28-33 Spring Stretch | 2025-26 | 5 GW | +7 pts | 95.2% | 24.1% | Clipped season-mean proxy | 0.0023 | PASS |
| 2025-26 GW30-35 Run-In Pressure | 2025-26 | 5 GW | -16 pts | 50.2% | 17.5% | Recent Form | 0.2520 | PASS |
| 2025-26 GW33-38 Final Stretch | 2025-26 | 5 GW | +12 pts | 93.2% | 25.8% | Season Average | 0.8686 | PASS |
| 2025-26 GW35-38 Late Sprint | 2025-26 | 3 GW | +2 pts | 90.2% | 24.2% | Clipped season-mean proxy | 0.0096 | PASS |
| 2025-26 GW37-38 Penultimate Round | 2025-26 | 1 GW | -2 pts | 58.0% | 24.1% | Season Average | 0.3364 | PASS |

---

## 3. Reliability Calibration Curve

| Forecast Bin | Sample Count | Mean Forecast Win Prob (%) | Realized Win Frequency (%) |
| :---: | :---: | :---: | :---: |
| 0%-20% | 1 | 0.0% | 0.0% |
| 20%-40% | 1 | 31.4% | 0.0% |
| 40%-60% | 4 | 51.5% | 50.0% |
| 60%-80% | 2 | 77.0% | 100.0% |
| 80%-100% | 18 | 94.5% | 66.7% |
