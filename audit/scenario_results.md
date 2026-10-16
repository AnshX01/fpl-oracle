# Product Behaviour Scenarios (Section 3.12 Audit)

This document records the exact test inputs, expected behavior, actual runtime behavior, and adversarial verdicts for the 8 canonical product scenarios executed against `fpl-oracle` on Python 3.14.2 (Windows 11).

Verification script: [`audit/scratch/run_scenarios.py`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/scratch/run_scenarios.py)  
Execution result artifact: [`audit/scratch/scenario_results.json`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/scratch/scenario_results.json)

---

## Scenario Summary Matrix

| # | Scenario Name | Area | Expected Behaviour | Actual Behaviour | Verdict |
|---|---|---|---|---|---|
| 1 | Healthy core, bank in hand, one affordable target | Transfer Optimizer | 1 FT upgrade, 0 hits, no lateral churn | Recommends 1 FT (`MID_19` → `MID_STAR`), 0 hits, +10.45 xP | **PASS** |
| 2 | Broken squad (5 flagged starters) | Chip / Squad Solver | Diagnose crisis, deploy Wildcard, solve injured XI | Recommends `ROLL_TRANSFER` (no action); WC retains 0 xP players | **FAIL** |
| 3 | Large blank gameweek (6 blanking players) | Chip / FH Planner | Deploy Free Hit for temporary 1-GW relief | Recommends permanent `WILDCARD` over temporary `FREE_HIT` | **PARTIAL** |
| 4 | Chips near expiry (GW18/GW19 with Set 1 unused) | Chip Planner / Cutoff | Hard GW19 expiry warning; force chip usage | Warning null; 0 chips recommended at GW19 (forfeiture) | **FAIL** |
| 5 | Marginal hit proposal (+0.6 xP vs -4.0 hit) | Hit Policy / Optimizer | Reject negative net return (-3.4 xP); Roll FT | Recommends `ROLL_TRANSFER`, 0 transfers, 0 hit cost | **PASS** |
| 6 | Small mini-league lead (GW36, +12 pts ahead) | League Strategy / MC | Enter `DEFENDING_LEAD`; mirror rival Haaland | Enters `DEFENDING_LEAD`; selects Haaland within 0.5 xP margin | **PASS** |
| 7 | Imminent price rise (rising tonight, budget tight) | Price Urgency / Timing | Recommends buying tonight before price rise | Detects `RISE_IMMINENT` but optimizer ignores execution timing | **PARTIAL** |
| 8 | Unresolved injury doubt (75% flag, late presser) | Risk Preferences | Conservative drops player; Aggressive starts | Starters XI 100% identical across all modes; captaincy only | **PARTIAL** |

---

## Detailed Scenario Reports

### Scenario 1: Healthy Core, Bank in Hand, One Affordable Target
- **Module:** [`src/fpl_oracle/optimise/transfers.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/transfers.py#L877-L910)
- **Inputs:**
  - Squad: 15 viable, healthy starters/bench players.
  - Bank: £1.5m in bank (15 tenths).
  - Free transfers: 1 banked FT.
  - Target: `MID_STAR` (£8.0m, 9.0 xP) available to replace low-yield starter `MID_19` (£6.5m, 3.65 xP).
- **Expected Behaviour:** Recommends exactly 1 Free Transfer (`MID_19` → `MID_STAR`) with 0 hits and no unnecessary moves.
- **Actual Runtime Behaviour:**
  - `plan_type`: `1_TRANSFER`
  - `transfers_count`: 1
  - `hit_cost`: 0.0
  - `transfers_in`: `["MID_STAR"]`
  - `transfers_out`: `["MID_19"]`
  - `expected_gain`: +10.45 net xP across horizon
- **Verdict:** **PASS** (Optimal single upgrade without churn).

---

### Scenario 2: Broken Squad (5 Flagged Starters)
- **Module:** [`src/fpl_oracle/optimise/transfers.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/transfers.py#L1604-L1650), [`src/fpl_oracle/chips/simulate.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/chips/simulate.py#L101-L137)
- **Inputs:**
  - 5 key starters red-flagged with 0.0 xP (`DEF_5`, `DEF_6`, `MID_15`, `MID_16`, `FWD_25`).
  - Bench assets unable to provide sufficient replacement cover.
  - Wildcard available in chip inventory.
- **Expected Behaviour:** Recommends triggering the Wildcard chip to perform a comprehensive squad restructuring, completely replacing injured players with active, healthy starters.
- **Actual Runtime Behaviour:**
  - In `evaluate_joint_transfer_and_chip_plan`: Recommends `ROLL_TRANSFER` (`recommended_chip: null`).
  - In `chip_simulator.evaluate_wildcard`: The target squad returned still contains all 5 broken players (`broken_players_retained_in_wc`: 5).
- **Proof & Mechanism:**
  - `evaluate_joint_transfer_and_chip_plan` applies high static retention values to chips in early gameweeks (`retention_values["wildcard"] = 10.0` to `20.0`), heavily discouraging playing the chip even when 5 starters have 0.0 xP.
  - In `simulate.py:122`, Wildcard gain is estimated via `single_gw_gain * 2.5` without properly accounting for multi-gameweek compound zero-point bleeds.
- **Verdict:** **FAIL** (Defective crisis management; rolls transfer instead of curing broken squad).

---

### Scenario 3: Large Blank Gameweek (6 Blanking Players)
- **Module:** [`src/fpl_oracle/chips/planner.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/chips/planner.py#L117-L122), [`src/fpl_oracle/optimise/transfers.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/transfers.py#L1604-L1650)
- **Inputs:**
  - 6 starting players have no fixture in GW10 (0.0 xP).
  - Free Hit, Wildcard, Triple Captain, Bench Boost available.
- **Expected Behaviour:** Deploys Free Hit chip to navigate the extreme temporary 1-gameweek fixture void while preserving the long-term squad structure.
- **Actual Runtime Behaviour:**
  - Recommends permanent `WILDCARD` over temporary `FREE_HIT` (`recommended_plan_type: "WILDCARD"`).
  - `evaluate_free_hit` computes `expected_gain = 0.0` when the player pool contains blanking players and fallback solver exception handlers kick in.
- **Verdict:** **PARTIAL** (Recommends a chip to address the crisis, but misclassifies a temporary blank as a permanent squad rebuild).

---

### Scenario 4: Chips Near Expiry (GW18 / GW19)
- **Module:** [`src/fpl_oracle/chips/planner.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/chips/planner.py#L294-L360)
- **Inputs:**
  - Current Gameweek: GW18 and GW19.
  - Set 1 chips: All 4 chips (WC, FH, 3xC, BB) unplayed.
  - Official rules: Hard cutoff after GW19 deadline; all unused Set 1 chips expire.
- **Expected Behaviour:** Generates urgent congestion/deadline alerts; forces chip activation in GW18 and GW19 to prevent permanent forfeiture.
- **Actual Runtime Behaviour:**
  - `strat_gw18["set_1_warning"]`: `None`
  - `strat_gw19["set_1_warning"]`: `None`
  - At GW19: `gw19_chip_plan_table_recommended`: `[]` (Empty list; recommends NO chip).
- **Proof & Mechanism:**
  - In `planner.py:356-358`:
    ```python
    threshold = CHIP_RECOMMENDATION_THRESHOLDS.get(c, 8.0)
    is_recommended = (gain >= threshold) and (rec_gw == current_gw)
    ```
    If expected gain is below the static threshold (e.g. TC gain of 3.8 < 4.0 threshold, or FH gain of 8.5 < 10.0 threshold), the planner marks `is_recommended = False`. At GW19 (the final week of Set 1), the chip is allowed to expire unused for 0 points rather than capturing the sub-threshold points!
- **Verdict:** **FAIL** (Unused chips forfeited due to rigid gating thresholds at deadline).

---

### Scenario 5: Marginal Hit Proposal (+0.6 xP vs -4.0 Hit Cost)
- **Module:** [`src/fpl_oracle/optimise/transfers.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/transfers.py#L877-L950)
- **Inputs:**
  - Player A in squad: `MID_19` (4.8 xP).
  - Target Player B: `MID_MARG` (5.4 xP).
  - Delta: +0.6 xP.
  - Free transfers: 0 FT (Transfer incurs a -4.0 point penalty).
- **Expected Behaviour:** Rejects transfer hit (net expectation is +0.6 - 4.0 = -3.4 points); rolls transfer.
- **Actual Runtime Behaviour:**
  - `plan_type`: `ROLL_TRANSFER`
  - `transfers_count`: 0
  - `hit_cost`: 0.0
  - `expected_gain`: 0.0
- **Verdict:** **PASS** (Properly rejects value-destroying hit).

---

### Scenario 6: Small Mini-League Lead (GW36, +12 pts Ahead)
- **Module:** [`src/fpl_oracle/league/strategy.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/league/strategy.py#L20-L115)
- **Inputs:**
  - GW36 (2 gameweeks remaining).
  - User has 2,112 points; Rival 2nd place has 2,100 points (+12 point lead).
  - Rival owns Erling Haaland (100% rival EO); User does not own Haaland.
- **Expected Behaviour:**
  - Posture set to `DEFENDING_LEAD` (lead >= 10 pts).
  - Recommends template coverage / mirroring rival assets to neutralize variance.
- **Actual Runtime Behaviour:**
  - `strategy_mode`: `DEFENDING_LEAD`
  - Tactical advice: "Template Protection: Mirror key players owned by your immediate chasers."
  - When candidate plans are within `margin_xp = 0.5` xP: Tie-breaker triggers and selects Haaland (`tie_breaker_narrow_applied: True`).
  - When differential candidate beats Haaland by > 0.5 xP: Model ignores rival EO and selects differential (`tie_breaker_wide_applied: False`).
- **Verdict:** **PASS** (Qualitative strategy and bounded tie-breaker behave according to specification).

---

### Scenario 7: Imminent Price Rise (Rising Tonight, Exact Budget)
- **Module:** [`src/fpl_oracle/optimise/price_change.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/price_change.py), [`src/fpl_oracle/optimise/transfers.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/transfers.py)
- **Inputs:**
  - Target player has massive net transfers and hourly rate; projected to rise tonight from £6.5m to £6.6m.
  - User has exactly £0.0m bank and £6.5m selling capacity. If transfer is delayed until deadline, user is priced out.
- **Expected Behaviour:** The transfer planner warns of imminent price lock-out and advises executing the transfer tonight before midnight price changes.
- **Actual Runtime Behaviour:**
  - `price_change_predictor` correctly flags `direction: "RISE_IMMINENT"`.
  - `evaluate_transfer_options` in `transfers.py` has **zero timing parameters** and does not model intraday price rises.
  - Price warnings are merely appended as static text in a disconnected "Price watch" section of `weekly.py` without correlating with the recommended transfer.
- **Verdict:** **PARTIAL** (Price predictor functions in isolation, but transfer optimizer has zero execution-timing intelligence).

---

### Scenario 8: Unresolved Injury Doubt (75% Flag, Late Press Conference)
- **Module:** [`src/fpl_oracle/optimise/lineup.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/lineup.py#L52-L75), [`src/fpl_oracle/optimise/sequential.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/optimise/sequential.py#L58-L70)
- **Inputs:**
  - Star starter has a 75% flag with wide distribution uncertainty (`p10 = 2.0`, `p90 = 10.0`).
  - Tested across `risk_preference`: `"conservative"`, `"balanced"`, `"aggressive"`.
- **Expected Behaviour:**
  - Conservative mode hedges against a 0-pointer/bench cameo by benching the doubtful player or prioritizing secure minutes.
  - Aggressive mode starts the player to maximize upside.
- **Actual Runtime Behaviour:**
  - `starters_identical_across_all_modes: True` (Starters XI is 100% identical in all 3 modes).
  - Captaincy changed: Conservative picked `GKP_1` (safe floor), Balanced and Aggressive picked `FWD_25`.
- **Proof & Mechanism:**
  - In `lineup.py:67`:
    ```python
    prob += pulp.lpSum([
        df.loc[i, "expected_points"] * starter[i] + cap_scores.loc[i] * (cap_multiplier - 1.0) * captain[i]
        for i in df.index
    ])
    ```
    `risk_preference` only modulates `cap_scores.loc[i]`. The starter coefficients are strictly `df.loc[i, "expected_points"]`, completely ignoring variance, `p10`, `p90`, and downside risk.
- **Verdict:** **PARTIAL** (Risk preference is purely a captaincy dial; it has zero impact on starting lineup selection).
