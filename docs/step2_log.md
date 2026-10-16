# Step 2 Carry-Ins Log: Machine-Grounded Reconciliation

**Date:** 2026-10-16  
**Branch:** `remediation/step1b`  
**Commit HEAD:** `503882673ed9e781f20497626f9e7f71f9f7e672`  
**Execution Environment:** Python 3.14.2 (Windows)

---

## C1. Machine-Generated Artifact Grounding Notice

Pursuant to directive **C1**, every numeric value, sample count, gain metric, and table row in this document is copied directly from machine-generated artifacts produced and hashed by `scripts/record_evidence.py`. No numbers have been manually entered or approximated.

### Artifact Registry:
| Directive | Evidence Record File | Machine Artifact (JSON) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **C2** | `reports/evidence/C2-replays.txt` | `reports/c2_simulate_captaincy_replays.json` | `b5b8e8a550655105d39662f51466a45fc2409ab0b9230a40cadf9781fbd6532b` |
| **C3** | `reports/evidence/C3-planners.txt` | `reports/c3_planner_comparison.json` | `debe052e42549a7791b4b0a321a28ab14a8610ed12bcf6ae6c963515b19959c3` |
| **C4** | `reports/evidence/C4-corpus.txt` | `reports/c4_training_corpus.json` | `2424216ea269c6b030832db05da7f25035d428562b4d9fe5c805e882283d0884` |
| **C5** | `reports/evidence/C5-active-holdout.txt` | `reports/c5_active_holdout_metrics.json` | `8178e7e1b43f1405f4cb4ff39c8c7de6415380ac64fc2cade147cb6c296cdd98` |

---

## C2. Captaincy Simulation Replay Comparison (`simulate_captaincy=True` vs `False`)

Evaluated across 10 real historical squad replays extracted from `api_cache` across past gameweeks (GW5, GW6, GW7) under all three risk modes (`conservative`, `balanced`, `aggressive`), totaling 30 comparative optimization runs.

### Empirical Replay Comparison Table
*Source: `reports/c2_simulate_captaincy_replays.json`*

| Replay | Manager ID | Entering GW | Bank | Risk Mode | Transfers In | Transfers Out | Captain | Horizon Gain | Sim == NoSim |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | 1085141 | GW6 | 1.8m | conservative | `['Cherki']` | `['Szoboszlai']` | Haaland | 3.73 | **True** |
| 1 | 1085141 | GW6 | 1.8m | balanced | `['Cherki']` | `['Szoboszlai']` | Haaland | 3.73 | **True** |
| 1 | 1085141 | GW6 | 1.8m | aggressive | `['Cherki']` | `['Szoboszlai']` | Haaland | 3.73 | **True** |
| 2 | 1085141 | GW7 | 0.1m | conservative | `['Hall']` | `['White']` | Haaland | 3.94 | **True** |
| 2 | 1085141 | GW7 | 0.1m | balanced | `['Hall']` | `['White']` | Haaland | 3.94 | **True** |
| 2 | 1085141 | GW7 | 0.1m | aggressive | `['Hall']` | `['White']` | Haaland | 3.94 | **True** |
| 3 | 2227954 | GW5 | 0.7m | conservative | `['De Cuyper']` | `['Cash']` | Haaland | 1.33 | **True** |
| 3 | 2227954 | GW5 | 0.7m | balanced | `['De Cuyper']` | `['Cash']` | Haaland | 1.33 | **True** |
| 3 | 2227954 | GW5 | 0.7m | aggressive | `['De Cuyper']` | `['Cash']` | Haaland | 1.33 | **True** |
| 4 | 2227954 | GW6 | 3.2m | conservative | `['Raya']` | `['Dubravka']` | Haaland | 3.56 | **True** |
| 4 | 2227954 | GW6 | 3.2m | balanced | `['Raya']` | `['Dubravka']` | Haaland | 3.56 | **True** |
| 4 | 2227954 | GW6 | 3.2m | aggressive | `['Raya']` | `['Dubravka']` | Haaland | 3.56 | **True** |
| 5 | 2227954 | GW7 | 2.6m | conservative | `['Raya']` | `['Dubravka']` | Haaland | 1.88 | **True** |
| 5 | 2227954 | GW7 | 2.6m | balanced | `['Raya']` | `['Dubravka']` | Haaland | 1.88 | **True** |
| 5 | 2227954 | GW7 | 2.6m | aggressive | `['Raya']` | `['Dubravka']` | Haaland | 1.88 | **True** |
| 6 | 4687536 | GW5 | 2.6m | conservative | `['Gibbs-White']` | `['Szoboszlai']` | Haaland | 4.69 | **True** |
| 6 | 4687536 | GW5 | 2.6m | balanced | `['Gibbs-White']` | `['Szoboszlai']` | Haaland | 4.69 | **True** |
| 6 | 4687536 | GW5 | 2.6m | aggressive | `['Gibbs-White']` | `['Szoboszlai']` | Haaland | 4.69 | **True** |
| 7 | 4687536 | GW6 | 0.2m | conservative | `['Groß']` | `['Szoboszlai']` | Haaland | 4.03 | **True** |
| 7 | 4687536 | GW6 | 0.2m | balanced | `['Groß']` | `['Szoboszlai']` | Haaland | 4.03 | **True** |
| 7 | 4687536 | GW6 | 0.2m | aggressive | `['Groß']` | `['Szoboszlai']` | Haaland | 4.03 | **True** |
| 8 | 4687536 | GW7 | 0.1m | conservative | `['Calvert-Lewin']` | `['Gonzalo']` | Haaland | 2.69 | **True** |
| 8 | 4687536 | GW7 | 0.1m | balanced | `['Calvert-Lewin']` | `['Gonzalo']` | Haaland | 2.69 | **True** |
| 8 | 4687536 | GW7 | 0.1m | aggressive | `['Calvert-Lewin']` | `['Gonzalo']` | Haaland | 2.69 | **True** |
| 9 | 8364971 | GW5 | 0.4m | conservative | `['Barry']` | `['Isak']` | B.Fernandes | 1.92 | **True** |
| 9 | 8364971 | GW5 | 0.4m | balanced | `['Barry']` | `['Isak']` | B.Fernandes | 1.92 | **True** |
| 9 | 8364971 | GW5 | 0.4m | aggressive | `['Barry']` | `['Isak']` | B.Fernandes | 1.92 | **True** |
| 10 | 8364971 | GW6 | 2.1m | conservative | `['Calafiori']` | `['Robertson']` | B.Fernandes | 1.60 | **True** |
| 10 | 8364971 | GW6 | 2.1m | balanced | `['Calafiori']` | `['Robertson']` | B.Fernandes | 1.60 | **True** |
| 10 | 8364971 | GW6 | 2.1m | aggressive | `['Calafiori']` | `['Robertson']` | B.Fernandes | 1.60 | **True** |

### Findings & Invariants:
1. **Any change from `simulate_captaincy`:** `False`.  
   Across all 30 evaluations (10 replays × 3 modes), passing `simulate_captaincy=False` produced identical recommended transfers, identical captain choices, and identical horizon gain values.
   - *Technical Mechanism:* In `src/fpl_oracle/optimise/transfers_beam.py`, the starting XI and captain are deterministically solved via linear programming on expected points prior to simulation. `simulate_captaincy=True` only calculates empirical Monte Carlo win probabilities (`prob_outscore`) for presentation metadata, but does not alter the objective function.
2. **Do risk modes ever differ across the 10 replays:** `False`.  
   - *Technical Mechanism:* In `src/fpl_oracle/optimise/lineup.py`, the `risk_aware_lineup` feature flag is disabled by default (`is_flag_enabled("risk_aware_lineup", default=False)`). Consequently, variance penalties ($\lambda \cdot \sigma^2$) are bypassed unless explicitly enabled via feature toggle, causing conservative, balanced, and aggressive modes to solve the standard expectation maximization.

---

## C3. Planner Architecture Parity & Current Squad Execution

### Architectural Call Map
Across the codebase, all consumer-facing surfaces call the **Beam Search Joint Planner** (`fpl_oracle.optimise.transfers_beam` via `analysis_service.joint_plan`):
- **Live REST API** (`src/fpl_oracle/server/routes/api.py`):
  - `/api/transfers` $\to$ `analysis_service.joint_plan`
  - `/api/optimize` $\to$ `analysis_service.joint_plan`
  - `/api/contingency/matrix` $\to$ `analysis_service.joint_plan`
  - `/api/contingency/panic` $\to$ `analysis_service.joint_plan`
  - `/api/league/scenarios` $\to$ `analysis_service.joint_plan`
- **Pipeline Background Daemon** (`src/fpl_oracle/server/pipeline.py`):
  - `_execute_optimisation` $\to$ `analysis_service.joint_plan`
- **Weekly Briefing Generator** (`src/fpl_oracle/briefing/weekly_briefing.py`):
  - `generate_briefing` $\to$ `analysis_service.joint_plan`
- **Decision Card Generator** (`src/fpl_oracle/briefing/decision_card.py`):
  - `generate_decision_card` $\to$ `analysis_service.joint_plan`
- **CLI Commands** (`src/fpl_oracle/cli.py`):
  - `fpl optimize` and `fpl briefing` $\to$ `analysis_service.joint_plan`

The **Multi-Period MILP Optimizer** (`src/fpl_oracle/optimise/multi_period_optimizer.py`) is an alternative whole-horizon mathematical solver used for multi-week scenario explorations and benchmark audits.

### Current Squad Execution Comparison
Executed on Manager `1085141` entering GW7 with Bank = 1.5m and Free Transfers = 2:
*Source: `reports/c3_planner_comparison.json`*

```
================================================================================
C3 PLANNER COMPARISON: CURRENT SQUAD (Manager 1085141, GW7)
Bank: 1.5m | Free Transfers: 2
================================================================================

--- 1. BEAM SEARCH OPTIMIZER OUTPUT ---
Plan Type:      2_TRANSFERS
Transfers In:   ['Cherki', 'Hall']
Transfers Out:  ['Szoboszlai', 'White']
Captain:        Haaland
Horizon Gain:   5.9
Hit Cost:       0.0

--- 2. MULTI-PERIOD MILP OPTIMIZER OUTPUT ---
Plan Type:      WILDCARD
Transfers In:   ['Rogers', 'Kelleher', 'De Cuyper', 'McBurnie', 'Davis', 'Bogle', 'Semenyo', 'Cherki', 'Hall', 'Tzolakis']
Transfers Out:  ['White', 'Ajer', 'Verbruggen', 'Palmer', 'João Pedro', 'Diop', 'Belloumi', 'Szoboszlai', 'Maguire', 'Kinsky']
Captain:        None
Horizon Gain:   12.16
Hit Cost:       0
```

### Authoritative Architecture Designation:
The **Beam Search Joint Planner** (`transfer_optimizer.evaluate_joint_transfer_and_chip_plan`) is designated as the **single authoritative planner** across all live surfaces (Live API, Briefing, Decision Card, CLI). All production endpoints and views consistently invoke this engine via `analysis_service.joint_plan`.

---

## C4. Training Corpus Specification & DefCon Resolution

### Corpus Specification
*Source: `reports/c4_training_corpus.json`*

1. **Component Models Corpus:**
   - **Seasons Included:** `['2023-24', '2024-25', '2025-26']`
   - **Max Round:** `38`
   - **Total Training Rows:** `87,087`
   - **Breakdown by Season:**
     - `2023-24`: Max Round = 38, Rows = 29,725
     - `2024-25`: Max Round = 38, Rows = 27,605
     - `2025-26`: Max Round = 38, Rows = 29,757

2. **CQR Bands Corpus:**
   - **Training Set:**
     - **Seasons Included:** `['2023-24', '2024-25', '2025-26']`
     - **Max Round in 2025-26:** `10`
     - **Total CQR Training Rows:** `64,651`
     - Breakdown: `2023-24` (29,725 rows), `2024-25` (27,605 rows), `2025-26` GW1-10 (7,321 rows)
   - **Disjoint Calibration Fold:**
     - **Season:** `2025-26`
     - **Round Range:** GW11 to GW18
     - **Total Calibration Rows (`minutes > 0`):** `2,421`
     - **Positional Counts:** `{'MID': 1133, 'DEF': 829, 'FWD': 298, 'GKP': 161}`

### Resolution of DefCon `is_2026_27` Inconsistency:
Inspection of the training dataset features confirmed:
- Value counts for `is_2026_27` across all 87,087 training rows: `{0.0: 87087}`.
- **Verdict:** The DefCon model does **not** include 2026-27 rows. The era indicator feature `is_2026_27` is **identically constant** (`0.0` for 100% of rows) during model training, meaning 2026-27 data was never leaked or incorporated into the DefCon component model.

---

## C5. 2026-27 GW1-6 Active Players Forward Holdout Evaluation

Evaluated strictly on active players (`actual_minutes > 0`) over the 2026-27 forward holdout window (GW1 through GW6).
- **Total 2026-27 Holdout Observations:** `3,883`
- **Active Player Observations (`minutes > 0`):** `1,853`
- **Positional Breakdown:** `{'MID': 878, 'DEF': 640, 'FWD': 215, 'GKP': 120}`

### Comprehensive Active Holdout Metrics Table
*Source: `reports/c5_active_holdout_metrics.json` and `reports/evidence/C5-active-holdout.txt`*

| Pos | N | ML MAE | Recent MAE | Season MAE | Fixture MAE | ML RMSE | Recent RMSE | Season RMSE | Fixture RMSE | ML Spearman | Recent Spearman | Season Spearman | Fixture Spearman | 80% Coverage | Interval Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GKP** | 120 | **2.249** | 2.567 | 2.565 | 2.546 | **3.080** | 3.439 | 3.404 | 3.407 | **0.138** | 0.080 | 0.061 | 0.184 | 71.67% | 6.19 |
| **DEF** | 640 | **2.280** | 2.585 | 2.545 | 2.656 | **3.375** | 3.669 | 3.637 | 3.701 | **0.248** | 0.195 | 0.185 | 0.232 | 78.91% | 5.85 |
| **MID** | 878 | **1.863** | 2.182 | 2.142 | 2.278 | **2.946** | 3.270 | 3.215 | 3.368 | **0.410** | 0.354 | 0.357 | 0.375 | 78.36% | 4.87 |
| **FWD** | 215 | **2.008** | 2.289 | 2.331 | 2.335 | **3.065** | 3.393 | 3.462 | 3.449 | **0.396** | 0.341 | 0.295 | 0.363 | 77.67% | 5.19 |
| **ALL** | 1853 | **2.049** | 2.358 | 2.330 | 2.433 | **3.122** | 3.438 | 3.407 | 3.498 | **0.339** | 0.285 | 0.276 | 0.315 | **78.04%** | 5.33 |

### Additional Probabilistic Calibration Loss (Active Players):
- **Pinball Loss (P10):** `0.3051`
- **Pinball Loss (P90):** `0.7227`
- **Overall Empirical 80% Coverage:** `78.04%` (within nominal tolerance $[75.0\%, 85.0\%]$)

---
