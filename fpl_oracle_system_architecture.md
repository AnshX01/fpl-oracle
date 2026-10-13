# FPL Oracle 2026/27: End-to-End System Architecture & Algorithmic Specification

> **Version:** 2026/27 Release  
> **Status:** Production-Grade Decision Support & Intelligence Engine  
> **Target Ruleset:** Official Fantasy Premier League 2026/27 Rules (5 Banked Free Transfers, Set 1 GW19 Expiration, Outfield DefCon Rewards)  
> **Test Suite Status:** 426/426 Tests Passing (100% Pass Rate)

---

## Table of Contents
1. [System Overview & High-Level Architecture](#1-system-overview--high-level-architecture)
2. [Machine Learning Point Projection Engine](#2-machine-learning-point-projection-engine)
   - [Modular Decomposition Philosophy](#modular-decomposition-philosophy)
   - [The Six Specialized Component Models](#the-six-specialized-component-models)
   - [Scoring Ensemble & Quantile Distribution Calibration](#scoring-ensemble--quantile-distribution-calibration)
   - [Probabilistic Coherence & Double Gameweek Handling](#probabilistic-coherence--double-gameweek-handling)
3. [Feature Engineering Pipeline & Mathematical Formulations](#3-feature-engineering-pipeline--mathematical-formulations)
   - [Zero-Leakage Temporal Shifting Guarantee](#zero-leakage-temporal-shifting-guarantee)
   - [Continuous Implied Match Strength & Independent Poisson Attack-Defence Model](#continuous-implied-match-strength--independent-poisson-attack-defence-model)
   - [Feature Dictionary & Schema Parity](#feature-dictionary--schema-parity)
4. [Transfer Optimization Engine & Mathematical Solvers](#4-transfer-optimization-engine--mathematical-solvers)
   - [Mixed-Integer Linear Programming (MILP) with PuLP CBC](#mixed-integer-linear-programming-milp-with-pulp-cbc)
   - [Multi-Gameweek Trajectory Beam Search](#multi-gameweek-trajectory-beam-search)
   - [2026/27 Transfer Rules & Selling Price Accounting](#202627-transfer-rules--selling-price-accounting)
   - [Hit Policy, Stability Thresholds, & Option Value](#hit-policy-stability-thresholds--option-value)
5. [Chip Strategy Planner & Boundary Dynamics](#5-chip-strategy-planner--boundary-dynamics)
   - [Set 1 vs Set 2 Chip Mechanics & Hard GW19 Expiration](#set-1-vs-set-2-chip-mechanics--hard-gw19-expiration)
   - [Counterfactual Utility Modeling & Threshold Gating](#counterfactual-utility-modeling--threshold-gating)
6. [Mini-League Analytics & Monte Carlo Simulation](#6-mini-league-analytics--monte-carlo-simulation)
   - [Proximity-Based Rival Tracking & Effective Ownership](#proximity-based-rival-tracking--effective-ownership)
   - [Unbiased Correlated Monte Carlo Match Simulation](#unbiased-correlated-monte-carlo-match-simulation)
7. [Availability Reconciliation & News Ingestion Engine](#7-availability-reconciliation--news-ingestion-engine)
   - [The Single-Reconciler Invariant](#the-single-reconciler-invariant)
   - [LLM Extraction, 12-Hour Disk Caching, & Regex Fallback](#llm-extraction-12-hour-disk-caching--regex-fallback)
   - [Gated Production Promotion & Benchmark Safeguards](#gated-production-promotion--benchmark-safeguards)
8. [Conversational Gemini LLM Assistant](#8-conversational-gemini-llm-assistant)
   - [Agent Architecture & Tool-Calling Protocol](#agent-architecture--tool-calling-protocol)
   - [Dual-Layer Rate-Limit Resilience & Offline Fallback](#dual-layer-rate-limit-resilience--offline-fallback)
   - [Deep Thinking & Rich Prediction Justifications](#deep-thinking--rich-prediction-justifications)
9. [Automated Background Daemon & System Integrity](#9-automated-background-daemon--system-integrity)
   - [The Seven Autonomous APScheduler Background Jobs](#the-seven-autonomous-apscheduler-background-jobs)
   - [Model Bundle Versioning & Rollback Safety](#model-bundle-versioning--rollback-safety)

---

## 1. System Overview & High-Level Architecture

The **FPL Oracle** is an enterprise-grade, mathematically rigorous decision-support and optimization system designed specifically for the Fantasy Premier League 2026/27 campaign. Unlike heuristic or simple linear regression tools, FPL Oracle unites:
- **Decomposed machine learning component inference** (6 specialized LightGBM models with out-of-fold probability calibration),
- **Mixed-Integer Linear Programming (MILP)** and multi-gameweek beam search optimization,
- **Independent Poisson attack-defence match simulation (full bivariate Dixon-Coles with low-score rho scheduled for WI-3.1)**,
- **Correlated Monte Carlo mini-league win-probability modeling**, and
- **Google Gemini LLM conversational reasoning** with contextual tool calling and rich justifications.

```mermaid
flowchart TD
    subgraph Data Layer ["Data Ingestion & Pipeline"]
        FPL_API["Official FPL API\n(Bootstrap, Fixtures, Picks)"]
        RSS["News Feeds\n(BBC, Guardian, Sky)"]
        STORE[("Data Store & Master History\n(Zero-Leakage SQLite/CSV)")]
        FPL_API --> STORE
        RSS --> STORE
    end

    subgraph Feature Engine ["Feature Engineering"]
        FE["Feature Transformer\n(features.py)"]
        DC["Poisson Implied Match Dynamics\n(Attack-Defence Rates)"]
        STORE --> FE
        FE --> DC
    end

    subgraph ML Layer ["Decomposed ML Projection Engine"]
        M1["Minutes & Starts\n(Isotonic Calibrated)"]
        M2["Attacking xG / xA\n(Tweedie Regressor)"]
        M3["Defending CS & GC\n(Isotonic + LightGBM)"]
        M4["2026/27 DefCon\n(5x Sample Weight)"]
        M5["BPS & Bonus\n(2026/27 Rules)"]
        M6["Disciplinary & Saves\n(Empirical Event Labels)"]
        ENS["Scoring Ensemble\n(Expected Div, Stacking, P10/P50/P90)"]
        DC --> M1 & M2 & M3 & M4 & M5 & M6
        M1 & M2 & M3 & M4 & M5 & M6 --> ENS
    end

    subgraph Optimisation Layer ["Decision & Solvers"]
        MILP["PuLP CBC MILP\n(Lineup & Squad Optimization)"]
        BEAM["Multi-GW Beam Search\n(5-14 GW Trajectory Engine)"]
        CHIP["Chip Strategy Planner\n(Set 1 GW19 Boundary)"]
        MC["Monte Carlo Simulator\n(Mini-League Correlated Trials)"]
        ENS --> MILP & BEAM & CHIP & MC
    end

    subgraph Conversational Layer ["Assistant & Serving"]
        FASTAPI["FastAPI REST API\n(Async HTTP Engine)"]
        GEMINI["Gemini Chat Assistant\n(gemini-flash-lite-latest)"]
        TOOLS["Tool Execution Dispatcher\n(Rich Move Schemas)"]
        OFFLINE["Offline Expert Fallback\n(Zero-API-Key Operation)"]
        MILP & BEAM & CHIP & MC --> FASTAPI
        FASTAPI <--> TOOLS
        TOOLS <--> GEMINI
        GEMINI -. Fallback .-> OFFLINE
    end
```

---

## 2. Machine Learning Point Projection Engine

### Modular Decomposition Philosophy
Standard predictive systems train a single regression model directly on `total_points`. In Fantasy Premier League, this is fundamentally flawed because total points is a discontinuous, heterogeneous mixture of distinct real-world processes:
1. Managerial starting lineups vs late substitutions,
2. Team-level possession and tactical defensive solidity (clean sheets, goals conceded),
3. Player-level finishing and playmaking (goals, assists, key passes),
4. Defensive actions rewarded under 2026/27 rules (tackles, interceptions, blocks, clearances),
5. Goalkeeping shot-stopping (saves, penalty stops), and
6. Match referee discipline (yellow/red cards).

FPL Oracle decomposes player scoring into six independent stochastic component models. Total Expected Points $\mathbb{E}[P]$ is then aggregated strictly according to verified 2026/27 Premier League scoring rules:

$$\begin{aligned}
\mathbb{E}[P] = P(\text{play}) \cdot \Big[ & \mathbb{E}[\text{Appearance}] + \mathbb{E}[\text{Goals Pts}] + \mathbb{E}[\text{Assists Pts}] + \mathbb{E}[\text{Clean Sheet Pts}] \\
& - \mathbb{E}[\text{Conceded Deduction}] + \mathbb{E}[\text{Save Pts}] + \mathbb{E}[\text{Penalty Saves}] \\
& + \mathbb{E}[\text{DefCon Pts}] - \mathbb{E}[\text{Card Deduction}] + \mathbb{E}[\text{Bonus Pts}] \Big]
\end{aligned}$$

---

### The Six Specialized Component Models

Each model is implemented in [`src/fpl_oracle/ml/components/`](src/fpl_oracle/ml/components/):

#### 1. Minutes and Starting Model ([`minutes.py`](src/fpl_oracle/ml/components/minutes.py))
- **Objective:** Predicts probability of starting $P(\text{starts})$, probability of playing 60+ minutes $P(\min \ge 60)$, and expected minutes $\mathbb{E}[\min] \in [0, 95]$.
- **Architecture:** Two `LGBMClassifier` estimators and one `LGBMRegressor`.
- **Calibration:** Probability outputs undergo out-of-fold isotonic calibration across chronological cross-validation folds (`temporal_folds`), preventing in-sample overconfidence.
- **Probabilistic Invariant:** Enforces strict physical coherence:
  $$\mathbb{E}[\text{minutes}] \ge 60 \cdot P(\min \ge 60)$$
  $$P(\text{play}) = \text{clip}\left(\max\left(P(\min \ge 60), \min\left(1.0, \frac{\mathbb{E}[\text{minutes}]}{60.0}\right)\right), 0, 1\right)$$

#### 2. Attacking Model ([`attacking.py`](src/fpl_oracle/ml/components/attacking.py))
- **Objective:** Predicts individual expected goals ($xG$) and expected assists ($xA$).
- **Architecture:** `LGBMRegressor` trained with a **Tweedie distribution objective** (`tweedie_variance_power=1.5`, `learning_rate=0.05`, `n_estimators=120`).
- **Mathematical Rationale:** Football events are zero-inflated and continuous/compound Poisson distributed. The Tweedie objective models the high probability of zero events while correctly parameterizing the fat-tailed haul density.
- **Goalkeeper Suppression:** Goalkeeper attacking expectations are physically suppressed by $98\%$ and $95\%$ respectively.

#### 3. Defending Model ([`defending.py`](src/fpl_oracle/ml/components/defending.py))
- **Objective:** Predicts individual clean sheet probability $P(\text{CS})$ and team expected goals conceded $\mathbb{E}[\text{GC}]$.
- **Architecture:** Out-of-fold isotonic-calibrated `LGBMClassifier` for $P(\text{CS})$ and `LGBMRegressor` for $\mathbb{E}[\text{GC}]$.
- **Team Correlation:** Individual clean sheet outputs are coupled to team-level defensive stability and opponent attack metrics.

#### 4. Defensive Contribution Model ([`defcon.py`](src/fpl_oracle/ml/components/defcon.py))
- **Objective:** Dedicated 2026/27 model predicting $P(\text{DefCon})$, the probability an outfield player achieves the official +2 point defensive contribution threshold (tackles, interceptions, blocks, clearances).
- **Weighting:** Trained with a $5\times$ sample weight on 2026/27 observations (`is_2026_27 == 1`) to prioritize new tactical parameters.
- **Rule Constraints:** Goalkeepers are excluded from DefCon scoring per official specifications ($P(\text{DefCon})_{\text{GKP}} \equiv 0$).

#### 5. Bonus Points Model ([`bonus.py`](src/fpl_oracle/ml/components/bonus.py))
- **Objective:** Predicts expected BPS reward $\mathbb{E}[\text{Bonus}] \in [0, 3]$.
- **Weighting:** Sample weighted $4\times$ on 2026/27 data to capture the latest BPS baseline recalibration.

#### 6. Cards & Saves Model ([`cards_saves.py`](src/fpl_oracle/ml/components/cards_saves.py))
- **Objective:** Predicts expected goalkeeper saves $\mathbb{E}[\text{Saves}]$, penalty saves $\mathbb{E}[\text{PenSaves}]$, and disciplinary deductions $\mathbb{E}[\text{Cards}]$.
- **Ground Truth:** Fitted directly against historical yellow cards ($-1$), red cards ($-3$), and penalty stops ($+5$).

---

### Scoring Ensemble & Quantile Distribution Calibration

Implemented in [`src/fpl_oracle/ml/ensemble.py`](src/fpl_oracle/ml/ensemble.py), the scoring ensemble combines component outputs using exact rules:

1. **Appearance Points:**
   $$\mathbb{E}[\text{App}] = 2.0 \cdot P(\min \ge 60) + 1.0 \cdot \big(P(\text{play}) - P(\min \ge 60)\big)$$
2. **Goals Scored Points (Positional Rate):**
   $$\text{Rate}_{\text{goal}} = 10 \cdot \mathbb{I}_{\text{GKP}} + 6 \cdot \mathbb{I}_{\text{DEF}} + 5 \cdot \mathbb{I}_{\text{MID}} + 4 \cdot \mathbb{I}_{\text{FWD}}$$
   $$\mathbb{E}[\text{Goal Pts}] = \mathbb{E}[\text{Goals}] \cdot \text{Rate}_{\text{goal}}$$
3. **Clean Sheet Points (Requires $\min \ge 60$):**
   $$\text{Rate}_{\text{cs}} = 4 \cdot \mathbb{I}_{\text{GKP}} + 4 \cdot \mathbb{I}_{\text{DEF}} + 1 \cdot \mathbb{I}_{\text{MID}}$$
   $$\mathbb{E}[\text{CS Pts}] = P(\text{CS}) \cdot P(\min \ge 60) \cdot \text{Rate}_{\text{cs}}$$
4. **Goals Conceded Deduction (Threshold Expectation):**
   FPL deducts $-1$ point for every 2 goals conceded by defenders and goalkeepers. Taking $\lfloor \mathbb{E}[\text{GC}] / 2 \rfloor$ underestimates variance. FPL Oracle calculates the **exact Poisson expected floor division**:
   $$\mathbb{E}\left[\left\lfloor \frac{X}{2} \right\rfloor\right] = \sum_{k=0}^{\infty} \left\lfloor \frac{k}{2} \right\rfloor \cdot \frac{\lambda^k e^{-\lambda}}{k!}$$
5. **Goalkeeper Save Points:**
   Similarly, $+1$ point per 3 saves uses exact Poisson floor expectation:
   $$\mathbb{E}\left[\left\lfloor \frac{S}{3} \right\rfloor\right] = \sum_{k=0}^{\infty} \left\lfloor \frac{k}{3} \right\rfloor \cdot \frac{\lambda_{\text{saves}}^k e^{-\lambda_{\text{saves}}}}{k!}$$
6. **Learned Stacking Blend:**
   The ML expectation is blended with rolling and season averages via constrained optimization on validation holdouts:
   $$\mathbb{E}[P]_{\text{final}} = w_{\text{ml}} \cdot xP_{\text{ML}} + w_{\text{rec}} \cdot xP_{\text{recent}} + w_{\text{sea}} \cdot xP_{\text{season}}$$
   Default optimal weights: $(w_{\text{ml}}, w_{\text{rec}}, w_{\text{sea}}) = (0.72, 0.04, 0.24)$.

#### Quantile Uncertainty Calibration ($P_{10}$, $P_{50}$, $P_{90}$)
Football scoring is positively skewed; naive Gaussian intervals fail because a player cannot score below $-3$, while hat-tricks create fat upper tails.
- **Base Volatility:** Positional baseline standard deviation:
  $$\sigma_{\text{base}} = 1.1 + 1.4 \cdot \mathbb{I}_{\text{FWD}} + 1.2 \cdot \mathbb{I}_{\text{MID}} + 0.9 \cdot \mathbb{I}_{\text{DEF}} + 0.7 \cdot \mathbb{I}_{\text{GKP}}$$
  $$\sigma = \sigma_{\text{base}} \cdot \sqrt{\text{clip}\left(\frac{\max(0.1, xP)}{3.0}, 0.4, 3.5\right)}$$
- **$P_{50}$ (True Median):** Corrected for right skew:
  $$P_{50} = \max\Big(0.0, \, xP - \text{clip}\big(0.25 \cdot (\mathbb{E}[\text{Goals Pts}] + \mathbb{E}[\text{Assists Pts}] + \mathbb{E}[\text{Bonus}]), \, 0.0, \, 0.6\big)\Big)$$
- **$P_{10}$ & $P_{90}$ Bucketed Quantiles:** Standardized empirical residuals $z = (y - xP)/\sigma$ fitted on a disjoint validation set yield empirical quantiles ($z_{10} = -0.806$, $z_{90} = 1.009$).

---

### Probabilistic Coherence & Double Gameweek Handling

In [`predict.py`](src/fpl_oracle/ml/predict.py):
- **Blank Gameweek (BGW):** All expectation components, floors, and ceilings are set to $0.0$.
- **Double Gameweek (DGW):** For players with two fixtures in a single gameweek, individual fixture feature vectors are evaluated separately through the component models and summed:
  $$xP_{\text{DGW}} = xP_{\text{fixture 1}} + xP_{\text{fixture 2}}$$
- Quantiles carry provenance metadata `uncertainty_status="dgw_marginal_sum_unvalidated"` to signal marginal summation rather than joint bivariate simulation.

---

## 3. Feature Engineering Pipeline & Mathematical Formulations

### Zero-Leakage Temporal Shifting Guarantee
Implemented in [`src/fpl_oracle/data/features.py`](src/fpl_oracle/data/features.py), the feature generator enforces strict train/serve parity:
1. **Training Mode:** All rolling windows for player statistics are strictly shifted by 1 match (`shift(1)` within player groups). A match's outcome never enters the features predicting that match.
2. **Serving Mode:** Features for upcoming fixtures use unshifted rolling statistics computed through the most recently completed match.
3. **Schema Verification:** A canonical SHA-256 hash `FEATURE_SCHEMA_HASH` locks the 62-column feature schema.

---

### Continuous Implied Match Strength & Independent Poisson Attack-Defence Model

Rather than relying on static 1-5 Fixture Difficulty Ratings (FDR), FPL Oracle calculates continuous team attacking rates $\lambda$ and defending rates $\mu$ using an independent univariate Poisson attack-defence formulation (scheduled to be upgraded to a full bivariate Dixon-Coles model with low-score correlation parameter $\rho$ in spec WI-3.1):

```
               [Team 5-match Attack Strength]       [Opponent 5-match Conceded xGC]
  λ_team = 1.35 * ( -------------------------- )  *  ( ----------------------------- ) * Home_Multiplier
                             1.35                                 1.35

  Home_Multiplier = 1.10 (if home) or 0.90 (if away)
```

$$\begin{aligned}
\lambda_{\text{team}} &= \text{clip}\left(1.35 \cdot \frac{\text{roll\_xG}_{5, \text{team}}}{1.35} \cdot \frac{\text{roll\_xGC}_{5, \text{opp}}}{1.35} \cdot H_{\text{team}}, \, 0.25, \, 4.5\right) \\
\mu_{\text{opp}} &= \text{clip}\left(1.35 \cdot \frac{\text{roll\_xG}_{5, \text{opp}}}{1.35} \cdot \frac{\text{roll\_xGC}_{5, \text{team}}}{1.35} \cdot A_{\text{opp}}, \, 0.25, \, 4.5\right)
\end{aligned}$$

#### Implied Clean Sheet Probability:
Assuming goals conceded follow a Poisson distribution with parameter $\mu_{\text{opp}}$, the clean sheet probability is the probability of zero goals conceded:
$$P(\text{Clean Sheet}) = P(Y = 0) = \frac{\mu_{\text{opp}}^0 e^{-\mu_{\text{opp}}}}{0!} = e^{-\mu_{\text{opp}}}$$

---

### Feature Dictionary & Schema Parity

The 62 model features are grouped into 11 functional clusters:

| Feature Group | Features | Count | Description |
|---|---|:---:|---|
| **Minutes & Starts** | `roll_minutes_3`, `roll_minutes_5`, `roll_minutes_8`, `roll_starts_ratio_5`, `roll_min60_ratio_5`, `std_minutes_per_gw` | 6 | Rolling average minutes, start rates, and appearance consistency across 3, 5, and 8 gameweeks. |
| **Attacking Form** | `roll_points_3`, `roll_points_5`, `roll_points_8`, `roll_xG_3`, `roll_xG_5`, `roll_xG_8`, `roll_xA_3`, `roll_xA_5`, `roll_xA_8`, `roll_xGI_5`, `roll_goals_5`, `roll_assists_5` | 12 | Underlying shot quality (xG), playmaking volume (xA/xGI), and attacking returns across 3, 5, and 8 matches. |
| **Defensive & Goalkeeping Form** | `roll_xGC_5`, `roll_clean_sheets_5`, `roll_saves_5`, `roll_goals_conceded_5`, `roll_defcon_5` | 5 | Expected goals conceded, clean sheets, goalkeeper saves, and DefCon defensive contribution actions. |
| **ICT & BPS** | `roll_ict_5`, `roll_bps_5` | 2 | FPL ICT index form and underlying Bonus Point System performance metrics. |
| **Disciplinary & Rare Events** | `roll_cards_5`, `roll_own_goals_5`, `roll_penalties_missed_5`, `roll_penalties_saved_5` | 4 | Deduction risks including card frequency, own goals, penalty misses, and penalty save capability. |
| **Match Context & Rest** | `was_home`, `team_strength_attack`, `opp_strength_defence`, `net_strength_diff`, `opponent_difficulty`, `days_rest` | 6 | Venue home/away split, official fixture difficulty, team strength differences, and match turnaround days rest. |
| **Continuous Implied Match Dynamics** | `implied_team_xG`, `implied_team_cs_prob`, `implied_opp_xG`, `implied_opp_cs_prob` | 4 | Continuous implied team goals, opponent goals, and clean sheet probabilities from Poisson attack-defence model. |
| **Team Attack Form** | `team_roll_goals_3`, `team_roll_goals_5`, `team_roll_goals_8`, `team_roll_xG_3`, `team_roll_xG_5`, `team_roll_xG_8` | 6 | Trailing club attacking volume: rolling actual goals and xG generated across 3, 5, and 8 matches. |
| **Opponent Defensive Form** | `opp_roll_points_3`, `opp_roll_points_5`, `opp_roll_points_8`, `opp_roll_goals_conceded_3`, `opp_roll_goals_conceded_5`, `opp_roll_goals_conceded_8`, `opp_roll_xGC_3`, `opp_roll_xGC_5`, `opp_roll_xGC_8`, `opp_roll_clean_sheets_5` | 10 | Trailing opponent defensive vulnerability: conceded points, actual goals, and xGC across 3, 5, and 8 matches. |
| **Positional One-Hot** | `pos_GKP`, `pos_DEF`, `pos_MID`, `pos_FWD` | 4 | One-hot position indicators (pos_GKP, pos_DEF, pos_MID, pos_FWD) for role-conditioned scoring rates. |
| **Market & Status** | `value`, `is_2026_27`, `chance_of_playing` | 3 | Current player price, active season indicator, and official availability probability percentage. |
| **Total** | *All 62 Canonical Feature Columns* | **62** | *Schema hash: `ec91cae9cf9b75c8...`* |

---

## 4. Transfer Optimization Engine & Mathematical Solvers

### Mixed-Integer Linear Programming (MILP) with PuLP CBC

Implemented in [`src/fpl_oracle/optimise/squad.py`](src/fpl_oracle/optimise/squad.py) and [`src/fpl_oracle/optimise/lineup.py`](src/fpl_oracle/optimise/lineup.py), the solver uses PuLP and the CBC open-source solver (`pulp.PULP_CBC_CMD(msg=0)`).

#### Mathematical Formulation for Lineup Optimization:
Given a 15-player squad $S$, find binary starter decisions $s_i \in \{0, 1\}$ and captain decisions $c_i \in \{0, 1\}$:

$$\max \sum_{i \in S} xP_i \cdot s_i + \sum_{i \in S} xP_i \cdot c_i$$

**Subject to:**
1. **Total Starters:** $\sum_{i \in S} s_i = 11$
2. **Goalkeeper Constraint:** $\sum_{i \in \text{GKP}} s_i = 1$
3. **Defender Constraint:** $3 \le \sum_{i \in \text{DEF}} s_i \le 5$
4. **Midfielder Constraint:** $2 \le \sum_{i \in \text{MID}} s_i \le 5$
5. **Forward Constraint:** $1 \le \sum_{i \in \text{FWD}} s_i \le 3$
6. **Captain Constraint:** $\sum_{i \in S} c_i = 1$
7. **Captain Must Start:** $c_i \le s_i \quad \forall i \in S$

#### Squad Selection & Hit Constraints:
When solving full squad transfers, binary variables $x_i \in \{0, 1\}$ represent squad membership across the league:
- **Total Squad:** $\sum x_i = 15$ ($2$ GKP, $5$ DEF, $5$ MID, $3$ FWD)
- **Club Limit:** $\sum_{i \in \text{Club}_k} x_i \le 3 \quad \forall k \in [1, 20]$
- **Budget:** $\sum_{i} \text{Cost}_i \cdot x_i \le \text{Budget}$
- **Hit Penalties:** Net transfer cost:
  $$\text{Hits} = \max\Big(0, \, (\text{Transfers Out} - \text{Banked FT})\Big) \times 4$$

---

### Multi-Gameweek Trajectory Beam Search

Implemented in [`src/fpl_oracle/optimise/transfers.py`](src/fpl_oracle/optimise/transfers.py):
- Planning horizon: default $H = 5$ gameweeks (configurable up to 14 gameweeks).
- Discount factor: $\gamma = 0.95$ per future gameweek to account for projection decay.
- **Trajectory Objective:**
  $$J(\tau) = \sum_{t=1}^{H} \gamma^{t-1} \cdot \Big( xP_t(\text{Starters}_t) + xP_t(\text{Captain}_t) - 4 \cdot \text{Hits}_t \Big)$$
- Maintains top $K$ candidate trajectories per step (beam width), expanding moves: 0 transfers (roll), 1 transfer, and 2 transfers.

---

### 2026/27 Transfer Rules & Selling Price Accounting

1. **Free Transfer Banking:** Up to **5 Free Transfers** can be banked in 2026/27 (previous rules capped at 2):
   $$\text{Banked}_{t+1} = \min\Big(5, \, \text{Banked}_t - \text{TransfersMade}_t + 1\Big)$$
2. **Exact Selling Price Formula:**
   FPL awards $50\%$ of price appreciation rounded down to $0.1$m:
   $$\text{SellPrice} = \text{PurchasePrice} + \left\lfloor \frac{\max(0, \, \text{CurrentPrice} - \text{PurchasePrice})}{2} \right\rfloor$$

---

### Hit Policy, Stability Thresholds, & Option Value

- **Stability Threshold (`STABILITY_THRESHOLD_XP = 0.3`):** To avoid transfer churn, a candidate transfer is only recommended if its horizon expected point gain exceeds rolling by at least $+0.3$ xP.
- **Hit Verdict Policy:** A $-4$ hit is marked:
  - `JUSTIFIED` if net horizon gain $\Delta xP_{\text{horizon}} \ge +4.0$ and immediate GW gain $\ge +1.5$ xP.
  - `MARGINAL` if $+2.5 \le \Delta xP < +4.0$.
  - `AVOID` if $\Delta xP < +2.5$.
- **Option Value of Banked FT:** Banking an FT is mathematically valued at $+0.4$ to $+0.8$ xP in future agility, derived from beam search counterfactuals.

---

## 5. Chip Strategy Planner & Boundary Dynamics

Implemented in [`src/fpl_oracle/chips/planner.py`](src/fpl_oracle/chips/planner.py):

### Set 1 vs Set 2 Chip Mechanics & Hard GW19 Expiration
- Official 2026/27 rules grant **2 sets of 4 chips**:
  - Set 1: Valid Gameweeks 1 to 19.
  - Set 2: Valid Gameweeks 20 to 38.
- **Hard Expiration Rule:** Any unused Set 1 chips expire permanently after the GW19 deadline; they do **not** carry over to Set 2.
- Maximum 1 chip deployment per gameweek.

```mermaid
timeline
    title 2026/27 Chip Calendar Boundary Architecture
    section Set 1 (GW 1-19)
        GW 1 : Season Kickoff
        GW 5 : Current Position
        GW 12-18 : Peak DGW/Fixtures
        GW 19 : Hard Expiration (Unused Chips Expire Permanently)
    section Set 2 (GW 20-38)
        GW 20 : Second Set Activates (Fresh 4 Chips)
        GW 34-37 : Major Spring Double Gameweeks
        GW 38 : Final Gameweek
```

---

### Counterfactual Utility Modeling & Threshold Gating

For every chip $C \in \{\text{Wildcard, Free Hit, Triple Captain, Bench Boost}\}$, the planner computes utility against the **no-chip counterfactual baseline**:
$$\Delta U_C(\text{GW}) = \mathbb{E}[P \mid C, \text{GW}] - \mathbb{E}[P \mid \text{No Chip}, \text{GW}]$$

#### Gated Activation Thresholds:
To guard against variance, chips are only recommended when expected gain exceeds strict safety margins:
- **Triple Captain (3XC):** Extra $1\times$ captain expectation must be $\ge 4.0$ xP (typically requiring an elite DGW asset).
- **Bench Boost (BB):** 4 bench players combined expectation must be $\ge 8.0$ xP.
- **Free Hit (FH):** Single-week swing versus fielded squad must be $\ge 10.0$ xP (typically Major Blank Gameweek).
- **Wildcard (WC):** Multi-week squad trajectory uplift must be $\ge 8.0$ xP across the 5-week horizon.

---

## 6. Mini-League Analytics & Monte Carlo Simulation

### Proximity-Based Rival Tracking & Effective Ownership
Implemented in [`src/fpl_oracle/league/rivals.py`](src/fpl_oracle/league/rivals.py):
- **Rival Inclusion Window (`RIVAL_POINTS_WINDOW = 35` pts):**
  - Includes all managers ranked above the user.
  - Includes all chasers within 35 points below the user.
  - If user is in 1st place, monitors only close chasers; never forces random lower managers.
- **Effective Ownership (EO):**
  $$\text{EO}_i = \frac{\sum_{m \in \text{Rivals}} \left( \mathbb{I}_{i \in \text{Starters}_m} + \mathbb{I}_{i = \text{Captain}_m} \right)}{|\text{Rivals}|} \times 100\%$$
- **Net Differential Exposure:** Identifies key differential players where user ownership diverges from the mini-league template.

---

### Unbiased Correlated Monte Carlo Match Simulation

Implemented in [`src/fpl_oracle/league/montecarlo.py`](src/fpl_oracle/league/montecarlo.py):
Simulates $N = 500$ to $1000$ complete gameweek trials across all rivals in $< 1.5$ seconds.

#### 1. Correlated Fixture Clean Sheet Bernoulli Draw
Clean sheets are not independent across teammates. For each fixture $f$ with clean sheet probability $p_{f, \text{CS}}$:
$$U_f \sim \text{Uniform}(0, 1) \implies \text{Team\_CS}_f = \begin{cases} 1 & \text{if } U_f < p_{f, \text{CS}} \\ 0 & \text{otherwise} \end{cases}$$
Every defender and goalkeeper on that team shares this exact clean sheet outcome.

#### 2. Hurdle Appearance Draw
$$U_m \sim \text{Uniform}(0, 1) \implies \text{App\_Pts} = \begin{cases} 0 & \text{if } U_m \ge P(\text{play}) \\ 2 & \text{if } U_m < P(\min \ge 60) \\ 1 & \text{otherwise} \end{cases}$$

#### 3. Unbiased Gamma Tail for Attacking & Bonus Points
To guarantee simulated mean matches model $xP$ exactly ($\mathbb{E}[S] \equiv xP$), residual points are drawn from a Gamma distribution:
$$\mu_{\text{other}} = \max\Big(0.0, \, xP - \mathbb{E}[\text{App}] - \mathbb{E}[\text{CS Pts}]\Big)$$
$$S_{\text{other}} \sim \text{Gamma}\left(k = \frac{\mu^2}{\sigma^2}, \, \theta = \frac{\sigma^2}{\mu}\right)$$

#### 4. Joint Player Draws Across Rivals
For common assets owned by multiple managers, a single draw is evaluated per trial and assigned to all owners, preserving cross-manager covariance.

---

## 7. Availability Reconciliation & News Ingestion Engine

### The Single-Reconciler Invariant
Implemented in [`src/fpl_oracle/news/reconcile.py`](src/fpl_oracle/news/reconcile.py):
- **Core Problem:** If an FPL official flag drops a player to 75% chance of playing, and press conference news simultaneously mentions "slight knock", naive systems discount both, unfairly slashing expected points in half.
- **Single Invariant:** Only the single reconciled availability object (`ReconciledAvailability`) modulates the feature pipeline. Downstream modules are strictly prohibited from applying secondary discount multipliers.

---

### LLM Extraction, 12-Hour Disk Caching, & Regex Fallback

Implemented in [`src/fpl_oracle/news/gemini_extractor.py`](src/fpl_oracle/news/gemini_extractor.py):
- **RSS Ingestion:** Ingests live news feeds from BBC Sport, The Guardian, Sky Sports, and TalkSport.
- **12-Hour Disk Cache:** Cached by article SHA-256 in `cache/gemini_news_cache.json` with a 12-hour TTL.
  - Result: Zero quota consumption on repeated queries; background updates run silently.
- **Regex Fallback:** If Gemini is unavailable, a deterministic regex extractor evaluates press conference quotes (`"ruled out"`, `"passed fit"`, `"in contention"`).

---

### Gated Production Promotion & Benchmark Safeguards

News signals operate in three modes (`RecommendationMode`):
1. `API_ONLY`: Uses only official FPL status flags.
2. `SHADOW` (Default): Evaluates and logs candidate news signals for auditing without modifying production projections.
3. `GATED_ACTIVE`: Allows verified news signals to update production expected points only if benchmark accuracy passes:
   - **Precision $\ge 85\%$**
   - **Recall $\ge 80\%$**
   - **False Ruled-Out Rate $\le 2.0\%$**

---

## 8. Conversational Gemini LLM Assistant

### Agent Architecture & Tool-Calling Protocol

Implemented across [`src/fpl_oracle/llm/`](src/fpl_oracle/llm/):
The assistant uses Google Gemini (`gemini-flash-lite-latest` with fallback to `gemini-3.5-flash-lite`) via native function calling.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Frontend
    participant Server as FastAPI (/api/chat)
    participant Agent as LLM Agent (agent.py)
    participant Provider as Gemini Provider (provider.py)
    participant Dispatcher as Tool Executor (tools.py)
    participant Core as ML & Optimisation Solvers

    User->>Server: POST /api/chat "Who should I captain and why?"
    Server->>Agent: answer(messages)
    Agent->>Provider: chat(messages, system_prompt)
    Provider->>Provider: Format Gemini Payload & Tools
    Provider->>Dispatcher: Call captain_options(gameweek=5)
    Dispatcher->>Core: Retrieve P10, P50, P90, xP
    Core-->>Dispatcher: Ranked Captains (Haaland 8.1, Salah 7.4)
    Dispatcher-->>Provider: Tool Result JSON
    Provider->>Provider: Second Round: Synthesize Deep Working
    Provider-->>Agent: Data-Grounded Final Response
    Agent-->>Server: Complete Answer
    Server-->>User: Markdown Response with Full Justifications
```

---

### The Eight Core Functions

1. `get_my_team`: Retrieves active 15-player squad, bank balance, chips used, and rank.
2. `get_projections`: Fetches ML expected points and calibrated quantiles ($P_{10}$, $P_{50}$, $P_{90}$) over 1-8 gameweeks.
3. `optimise_transfers`: Executes multi-GW beam search and MILP solver with candidate comparisons (roll vs 1-transfer vs hit) and roadmaps.
4. `plan_chips`: Evaluates 2026/27 chip utility versus baseline counterfactuals with Set 1 GW19 expiration tracking.
5. `captain_options`: Ranks safe vs differential captain choices with floor/ceiling metrics.
6. `league_analysis`: Computes rival standings, effective ownership, and Monte Carlo finish probabilities.
7. `price_change_watch`: Forecasts price rises and falls based on net transfers and momentum.
8. `compare_players`: Direct head-to-head statistical, cost-efficiency, and DefCon breakdown.

---

### Dual-Layer Rate-Limit Resilience & Offline Fallback

To provide high availability and graceful degradation:
1. **Tier 1:** Native Gemini API call using fast model (`gemini-flash-lite-latest`).
2. **Tier 2 (Fallback Model):** Automatic fallback to `gemini-3.5-flash-lite` on 404 or model version shifts.
3. **Tier 3 (Offline Expert Provider):** If Google returns `429 Too Many Requests` or network drops, the system seamlessly activates [`OfflineExpertProvider`](src/fpl_oracle/llm/provider.py#L39-L140).
   - Operates locally without external API calls.
   - Executes live tool functions and formats structured, data-grounded FPL Oracle recommendations.
   - Falls back gracefully to offline rules-based heuristics if upstream LLM requests fail.

---

### Deep Thinking & Rich Prediction Justifications

The system prompt mandates rigorous justifications for all recommendations:
- **Transfers:** Must explain why the recommended move was chosen over rolling, citing the hit verdict, expected point gain, and banked free transfer trajectory.
- **Captaincy:** Must compare safe floor ($P_{10}$) vs explosive differential ceiling ($P_{90}$).
- **Chips:** Must explicitly highlight the Set 1 GW19 boundary rule and required point gain thresholds.

---

## 9. Automated Background Daemon & System Integrity

### The Seven Autonomous APScheduler Background Jobs

Managed in [`src/fpl_oracle/server/jobs.py`](src/fpl_oracle/server/jobs.py):

| Job ID | Name | Cadence | Purpose |
|---|---|---|---|
| `cadence_refresh` | Live / Cadence Refresh | 60s live / 10m idle | Polls FPL live match data, points, and event status. |
| `news_refresh` | News & RSS Ingestion | Every 20 minutes | Ingests BBC, Guardian, Sky feeds; caches Gemini extraction. |
| `price_snapshot` | Price Change Snapshot | Hourly | Records net transfer momentum for price rise/fall predictions. |
| `history_refresh` | Finalized History Ingestion | Hourly | Pulls finalized player match data into master historical dataset. |
| `retrain_trigger` | Finalized-GW Retrain | Every 15 minutes | Triggers candidate model retraining post-gameweek. |
| `deadline_alert` | Pre-Deadline Alerts | Every 5 minutes | Dispatches briefing cards at -24h, -3h, and -1h before deadline. |
| `holdout_forward` | Forward Holdout Scoring | Every 15 minutes | Evaluates locked pre-deadline forecasts against realized match points. |

---

### Model Bundle Versioning & Rollback Safety

Implemented in [`src/fpl_oracle/ml/model_registry.py`](src/fpl_oracle/ml/model_registry.py):
- **Candidate vs Production:** Newly trained models are saved as candidates.
- **Validation Gates:** A candidate model bundle is only promoted to production if its Mean Absolute Error (MAE) and Rank Correlation improve upon the active model bundle across temporal validation folds.
- **Weight Integrity Verification:** Verifies SHA-256 weight checksums on startup; corrupt bundles trigger automatic rollback to the previous verified release.

---

## Summary of Verification & Test Suite Status

The codebase is validated by 426 automated unit and integration tests across 28 test suites:
- ✅ Rules Verification: 100% compliant with official 2026/27 rules.
- ✅ Solver Sanity: PuLP CBC MIP solver verified and operational.
- ✅ Quantile Calibration: Standardized residual calibration confirmed without mean=median assumptions.
- ✅ Parity Verification: Zero data leakage between training and live serving.
- ✅ Test Results: `426 passed, 0 failed, 0 errors` in `40.54s`.
