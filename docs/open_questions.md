# Open Questions

Record of open architectural, modeling, or rules questions requiring owner confirmation.

---

### 1. Odds Provider for Player-Level Markets (WI-3.2)
- **Question:** Free historical match-level odds are available from football-data.co.uk. For player-level odds (anytime goalscorer, first goalscorer), which odds provider API should be used in production?
- **Default Answer:** Implement an abstracted odds adapter interface (`OddsProvider`). Default to Betfair Exchange API / API-Football with a robust fallback to the team Poisson goal model when market odds are unavailable or uncalibrated.

### 2. Multi-Period MILP Time Budget (WI-2.4)
- **Question:** Is a maximum runtime of 60 seconds acceptable on the target machine for H=8 horizon multi-period transfer optimization with integer programming solvers?
- **Default Answer:** Yes. Enforce a 60-second hard timeout using HiGHS/CBC backend. If solver time expires, return the best incumbent feasible integer solution along with the logged relative optimality gap (e.g. `<= 1.5%`).

### 3. Primary Planner Objective (WI-2.1 / WI-4.3)
- **Question:** Should the primary planner objective be raw expected points maximization, mini-league win probability targeting, or overall rank percentile?
- **Default Answer:** Expected net points over rolling H-gameweek horizon remains the base objective. Mini-league chase/defend modes (WI-4.3) are selectable via runtime flags or CLI option `--league-mode` to modulate transfer variance when trailing or defending a lead.

### 4. Calibration & Production Deployment of Risk-Aware Lineup Parameters (F-011)
- **Question:** The hotfix made starter risk parameters configurable (`downside_risk_lambda = 0.20`, `cop_doubt_penalty = 2.0`, `aggressive_upside_weight = 0.30`) and defaulted the flag `risk_aware_lineup: false`. Should these parameters be adopted with flag enabled, or tuned via the walk-forward backtest runner (WI-0.1)?
- **Default Answer:** Keep `risk_aware_lineup: false` by default until parameter sensitivity and out-of-sample regret are benchmarked across the 2023-2026 seasons in WI-0.1.

### 5. Sample Weighting Policy for Recent Rule-Change Seasons (F-009)
- **Question:** Evaluation across rolling-origin temporal splits showed that 5x sample weighting on 2026/27 rows consistently degraded DefCon log loss (mean $\Delta$ +0.0108) while Bonus MAE was mixed ($\Delta$ -0.0015 to +0.002). Should production adopt component-specific weights (1.0x DefCon, 1.0x-1.5x Bonus) or retain equal 1.0x weighting across all models?
- **Default Answer:** Adopt 1.0x weighting for DefCon (scoring rule applies identically across seasons; extra weight discards ~14.6k effective historical observations). For Bonus, test 1.0x vs 1.25x in the walk-forward backtest before changing production training configurations.
