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
| **D2** | `reports/evidence/D2-reconcile.txt` | `reports/d2_squad_reconciliation.json` | `fc5b02a3e05197dc9cffd8103a714e335e9e9bab27ec9fb943bfba2ff43d660a` |
| **D3** | `reports/evidence/D3-replays.txt` | `reports/d3_replays.json` | `aa04c61217f4166241203f07ca3c99c2abf15327c435a0b982672a72d62389f0` |
| **D4** | `reports/evidence/D4-cqr.txt` | `reports/d4_cqr_bands.json` | `75cae8862f8cfdfe1c418692f0cadaedae231ab392240b91c159b89b03019fed` |
| **D6** | `reports/evidence/D6-holdout.txt` | `reports/d6_holdout_stats.json` | `abd2143487f466e6b8aea0e133ce0dcb36b54e2b13c3c98871572156923cc0da` |

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

## D1. Planner Authority: Multi-Period MILP & Unified Surface Routing

Pursuant to directive **D1** (Spec Phase 2/5):
1. **Authoritative Designation:** The **Multi-Period Mixed-Integer Linear Program (MILP)** (`fpl_oracle.optimise.multiperiod.MultiPeriodOptimizer`) is established as the single authoritative planning engine across the system.
2. **Fallback & Cross-Check:** The bounded **Beam Search Optimizer** (`fpl_oracle.optimise.transfers_beam.TransferOptimizer`) is retained strictly as a fast fallback engine and validation cross-check for <= 2 transfers.
3. **Unified Surface Routing:** All live consumer surfaces (`/api/transfers`, `/api/optimize`, `/api/contingency/*`, `/api/league/scenarios`, pipeline background runner, weekly briefing generator, decision card generator, and CLI) route through `analysis_service.plan()`, which directly executes the authoritative MILP plan with the full legal action set (all chips, hits, FT banking).
4. **Resolution of "Captain: None":** In `src/fpl_oracle/optimise/multiperiod.py`, `_to_plan_dict` now reconstructs the legal 15-player squad following transfers and invokes `lineup_optimizer.select_lineup_and_captain` to select the starting XI, captain, and vice-captain. The output contains `plan["lineup"]["captain"]["web_name"]`, ensuring complete captain parity across all surfaces.

---

## D2. Current Squad Reconciliation (Manager 1085141, GW7)

Reconciled entering Gameweek 7 under identical player universe, action set, and optimizer shrinkage.  
*Source: `reports/d2_squad_reconciliation.json` and `reports/evidence/D2-reconcile.txt`*

### State Discrepancy Resolution:
- **Ground Truth FPL API State:** `entry_history` for GW6 picks reports **Bank = 0.1m** (1 tenth), **Free Transfers = 1**.
- **User Profile Override State:** The SQLite table `user_profile` in `data/fpl_oracle.db` had manual overrides enabled (`bank_override_enabled=1`, `ft_override_enabled=1`), reporting **Bank = 1.5m** (15 tenths), **Free Transfers = 2**.
- **Resolution:** The earlier briefing invoked the live profile service which picked up the override, whereas raw API audit scripts read `entry:1085141:event:6:picks`. Both states were evaluated side-by-side.

### Comparative Execution Table:
| Setting | Planner Engine | Constraints / Chips | Transfers In | Transfers Out | Captain | Horizon Gain | Discrepancy Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Ground Truth** (0.1m, 1 FT) | Beam Search | Normal (<= 2 Xfers) | `['Hall']` | `['White']` | Haaland | +3.94 | Exact transfer match |
| **Ground Truth** (0.1m, 1 FT) | Multi-Period MILP | Chips Forbidden | `['Hall']` | `['White']` | Haaland | +2.59 | Exact transfer match; gain discounted ($\gamma=0.9$) |
| **Ground Truth** (0.1m, 1 FT) | Multi-Period MILP | Full Legal Action Set | 11 Players In | 11 Players Out | Haaland | +11.78 | Optimal Wildcard discovered |
| **Profile Override** (1.5m, 2 FT) | Beam Search | Normal (<= 2 Xfers) | `['Cherki', 'Hall']` | `['Szoboszlai', 'White']` | Haaland | +7.06 | Exact transfer match |
| **Profile Override** (1.5m, 2 FT) | Multi-Period MILP | Chips Forbidden | `['Cherki', 'Hall']` | `['Szoboszlai', 'White']` | Haaland | +4.79 | Exact transfer match; gain discounted ($\gamma=0.9$) |
| **Profile Override** (1.5m, 2 FT) | Multi-Period MILP | Full Legal Action Set | 8 Players In | 8 Players Out | Haaland | +10.83 | Optimal Wildcard discovered |

**Key Findings:** Under identical action sets (chips forbidden), Beam Search and MILP choose **identical transfers** (Hall for White on 0.1m/1FT; Cherki & Hall for Szoboszlai & White on 1.5m/2FT) and identical captaincy (Haaland). Gain differences stem purely from MILP's multi-period discount factor and shrinkage. Unconstrained MILP discovers superior horizon value via early Wildcard (+11.78 / +10.83 xP).

---

## D3. Replay Fixtures & Squad Validation

Evaluated across 10 independent managers from mini-league `435956` in all 3 risk modes (30 total optimization runs).  
*Source: `reports/d3_replays.json` and `reports/evidence/D3-replays.txt`*

### Clarification of "Out: Cash":
- Replay 3 showed transfer `"Out: Cash"`. Investigation verified that player element 32 is **Matty Cash** (Aston Villa defender, value £4.6m).
- Manager `2227954` legitimately owned Matty Cash in their 15-player squad and sold him for De Cuyper. No missing cash or currency corruption occurred.

### Squad Validation:
- All 11 independent managers audited from mini-league `435956` (`2227954`, `1085141`, `4687536`, `8364971`, `2230307`, `7793828`, `9778590`, `9153936`, `6861977`, `9216581`, `9223805`) strictly validated: exactly 15 players (2 GKP, 5 DEF, 5 MID, 3 FWD; $\le 3$ players per Premier League club). Zero invalid squads were found.

### Replay Invariant Verification (30 comparative runs):
1. **Any change detected from `simulate_captaincy=False`:** `False` (`0` transfer changes, `0` captain changes across all 30 runs).
2. **Risk modes difference:** `False` (identical plans across conservative, balanced, aggressive because `risk_aware_lineup` is default-off).

---

## D4. CQR Production Bands Empirical Recalibration & SPEC Gate Audit

*Source: `reports/d4_cqr_bands.json` and `reports/evidence/D4-cqr.txt`*

### Root Cause Analysis of Prior Goalkeeper Under-Coverage (71.7%):
- The earlier snapshot in C5 was calibrated strictly on 2025-26 GW11-18 (8 gameweeks), providing only N=161 played Goalkeepers.
- The small sample size generated high variance and an underestimated offset (0.1905), leading to 71.67% empirical coverage on the 2026-27 forward holdout.
- **Remediation:** (1) Multi-season calibration data (N=767 played GKPs in 2025-26, N=887 played GKPs when rolling finalised data is incorporated) and (2) Cross-position empirical Bayes shrinkage for low-sample strata.

### Forward Holdout Empirical Coverage vs SPEC Gate `[77.0%, 83.0%]`:
Evaluated strictly on active players (`actual_minutes > 0`) over 2026-27 GW1-6 (N=1,853):

| Position | Sample Size (N) | Coverage % | Wilson 95% CI | Gameweek-Block Boot 95% CI | Mean Width | PB10 Loss | PB90 Loss | SPEC Gate `[77, 83]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GKP** | 120 | **78.33%** | [70.15, 84.76]% | [71.67, 84.17]% | 6.22 | 0.3305 | 0.7331 | **PASS** |
| **DEF** | 640 | **79.69%** | [76.40, 82.62]% | [75.58, 83.86]% | 5.63 | 0.3338 | 0.8027 | **PASS** |
| **MID** | 878 | **78.93%** | [76.11, 81.50]% | [71.85, 83.75]% | 5.24 | 0.2815 | 0.7043 | **PASS** |
| **FWD** | 215 | **77.21%** | [71.15, 82.31]% | [69.12, 85.19]% | 5.51 | 0.2817 | 0.7149 | **PASS** |
| **ALL** | 1853 | **78.95%** | [77.04, 80.75]% | [73.55, 82.19]% | 5.47 | 0.3027 | 0.7414 | **PASS** |

*All positions strictly satisfy the SPEC gate `[77.0%, 83.0%]`. Do NOT use relaxed `[75, 85]`.*

### Production Rolling Recalibration (All Finalised Data to Date):
- **Training Set:** Seasons `2023-24` and `2024-25` (57,330 rows).
- **Rolling Calibration Window:** Season `2025-26` + finalised `2026-27 GW1-6` (`minutes > 0`).
- **Calibration Sample Counts:** `{'GKP': 887, 'DEF': 4590, 'MID': 6225, 'FWD': 1649}` (total N=17,351).
- **Calibrated Production Conformal Offsets:** `{'GKP': 0.3519, 'DEF': 0.4658, 'MID': 0.0, 'FWD': 0.0}`.

---

## D5. Dead Feature Protocol & Weekly Retraining Specification

1. **Dead Feature Context:** In pre-2026 historical training (seasons 2023-24 through 2025-26), the feature `is_2026_27` was constant `0.0` for 100% of rows (87,087 observations).
2. **Weekly Retraining Protocol (Step Plan Specification):**
   - Retraining is triggered weekly immediately following gameweek $g-1$ finalization (`event.finished == True` and `event.data_checked == True`).
   - The training and calibration dataset pool incorporates all historical seasons plus 2026-27 rows through gameweek $g-1$ only.
   - Predictions for gameweek $g$ are strictly frozen before the official gameweek deadline (forward holdout protocol).
   - Because rows with `season == '2026-27'` through $g-1$ enter the training corpus, `is_2026_27` becomes non-constant (containing both 0 and 1 values), allowing the model to learn 2026-27 era dynamics (such as revised DefCon points rules and 5-FT banking) without data leakage.
   - Retrained candidate models must pass the automated bootstrap promotion gate against active served weights before production promotion.
   - *Status:* Specification added to step plan. Standing scheduler implementation deferred.

---

## D6. 2026-27 GW1-6 Forward Holdout Per-Gameweek Performance & Bootstrap CIs

Evaluated strictly on active players (`actual_minutes > 0`) over 2026-27 GW1 through GW6 (N=1,853 observations).  
*Source: `reports/d6_holdout_stats.json` and `reports/evidence/D6-holdout.txt`*

### Per-Gameweek Performance Breakdown:
| GW | Active N | ML MAE | Best Baseline MAE | MAE Gain vs Best | ML RMSE | ML Spearman | Best Baseline Spearman | Spearman Gain vs Best |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GW1** | 310 | 2.323 | 2.541 (Season) | **+0.218** | 3.499 | 0.238 | 0.192 (Season) | **+0.046** |
| **GW2** | 312 | 1.940 | 2.331 (Recent) | **+0.391** | 3.004 | 0.369 | 0.300 (Fixture) | **+0.069** |
| **GW3** | 307 | 1.956 | 2.290 (Season) | **+0.334** | 2.923 | 0.283 | 0.287 (Fixture) | -0.004 |
| **GW4** | 307 | 2.101 | 2.301 (Season) | **+0.200** | 3.162 | 0.379 | 0.387 (Fixture) | -0.008 |
| **GW5** | 302 | 2.077 | 2.313 (Season) | **+0.236** | 3.192 | 0.404 | 0.386 (Fixture) | **+0.018** |
| **GW6** | 315 | 1.901 | 2.194 (Season) | **+0.293** | 2.919 | 0.362 | 0.363 (Fixture) | -0.001 |
| **TOT** | **1853** | **2.049** | **2.330 (Season)** | **+0.281** | **3.122** | **0.339** | **0.315 (Fixture)** | **+0.024** |

### Gameweek-Block Bootstrap 95% Confidence Intervals (5,000 Resamples):
- **Pooled MAE Gain vs Best Baseline:** `+0.281` pts | **95% Block Bootstrap CI:** `[+0.2273, +0.3401]`
- **Pooled Spearman Gain vs Best Baseline:** `+0.024` | **95% Block Bootstrap CI:** `[-0.0005, +0.0484]`

### Sample Size Caveat:
The 2026-27 forward holdout comprises 6 early-season gameweeks (GW1-6) encompassing N=1,853 active player observations (mean ~308 active players/GW). While observation counts are statistically substantial for overall point estimates, temporal clustering at the gameweek level yields an effective sample size governed by B=6 gameweek blocks. Gameweek-level systemic shocks (e.g., gameweek fixture variance, tactical anomalies, referee cards) induce intra-gameweek correlation. Consequently, gameweek-block bootstrap confidence intervals are reported to account for block-level dependence.

---

## D7. Risk Modes Owner Decision

Pursuant to directive **D7**:
- The feature flag `risk_aware_lineup` in `config/flags.yaml` defaults to `false`.
- Consequently, the risk modes (`conservative`, `balanced`, `aggressive`) are **inert** (all 3 modes solve standard linear expectation maximization, confirmed by the 30 replay runs in C2 and D3).
- **Owner Decision:** `risk_aware_lineup` remains **default-off** (`false`). It has **NOT** been turned on.
- **Repository Record:** Formally recorded in `docs/needs_human.md` Section 2.
- **Superseding Note:** Item 4's simulator-based mini-league mode (which simulates opponent lineups and match distributions to dynamically optimize chasing vs defending strategies) is designed to supersede this heuristic quadratic variance penalty (`lambda * sigma^2`). Any overhaul is deferred to the mini-league simulator integration.

