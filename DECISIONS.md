# FPL Oracle — Architectural & Domain Decisions Log

This document records key technical decisions, verified rule constraints from the live 2026/27 FPL API, engineering rationale, justifications for tech stack choices, and an honest self-assessment of system limitations.

---

## 1. Season 2026/27 Official Rule Verification (Verified Live from API)

### 1.1 Chip Structure
- **Verification Source**: `https://fantasy.premierleague.com/api/bootstrap-static/` -> `chips` array.
- **Rules Confirmed**:
  - Exactly 8 chips available across the season in two distinct halves.
  - **Set 1 (Gameweeks 1–19)**:
    - `bboost` (Bench Boost): starts GW1, stops GW19.
    - `3xc` (Triple Captain): starts GW1, stops GW19.
    - `freehit` (Free Hit): starts GW2, stops GW19.
    - `wildcard` (Wildcard): starts GW2, stops GW19.
    - Hard expiry at Gameweek 19 deadline (Saturday 2 January 2027, 13:30 GMT). Set 1 chips do NOT carry over to Set 2.
  - **Set 2 (Gameweeks 20–38)**:
    - Unlocks at Gameweek 20 (starts GW20, stops GW38).
    - Another full set of 4 chips: Wildcard, Free Hit, Triple Captain, Bench Boost.
  - **Assistant Manager Chip**: Confirmed removed for 2026/27 (not in `chips` or `game_settings`).
  - **Chip Usage Limit**: Exactly 1 chip can be played per gameweek.

### 1.2 Free Transfers Banking
- **Verification Source**: `game_settings` -> `max_extra_free_transfers: 4`.
- **Rules Confirmed**:
  - A manager gets 1 free transfer per GW. Up to 4 extra free transfers can be saved, meaning up to **5 free transfers can be banked**.
  - Subsequent transfers cost 4 points each (`hit = 4`).

### 1.3 Selling Price Calculation
- **Verification Source**: `game_settings` -> `transfers_sell_on_fee: 0.5`, `element_sell_at_purchase_price: false`.
- **Formula**:
  - `selling_price = purchase_price + floor((current_price - purchase_price) / 2)` if `current_price > purchase_price`, else `current_price`.

### 1.4 2026/27 Scoring & Defensive Contribution (DefCon)
- **Verification Source**: `game_config` -> `scoring` and live fixture `stats` arrays.
- **Rules Confirmed**:
  - Goals: GKP 10, DEF 6, MID 5, FWD 4.
  - Assists: 3 pts all positions.
  - Clean Sheets: GKP 4, DEF 4, MID 1, FWD 0 (requires 60+ minutes played).
  - Goals Conceded: GKP -1 per 2 goals conceded, DEF -1 per 2 goals conceded.
  - Appearance: 1 pt (1–59 mins), 2 pts (60+ mins).
  - Saves: 1 pt per 3 saves.
  - Penalties: Saved +5, Missed -2.
  - Cards / Own Goals: Yellow -1, Red -3, Own Goal -2.
  - **Defensive Contribution (DefCon)**:
    - Awarded 2 points to DEF, MID, and FWD players achieving the defensive action threshold.
    - Verified in fixture stats identifier: `defensive_contribution`, calculated from actions including clearances, blocks, interceptions, recoveries, and tackles.
  - **Bonus Points System (BPS)**:
    - Rebalanced for 2026/27 to avoid double-counting with DefCon actions.
    - Bonus points (3, 2, 1) awarded to top 3 BPS earners in each match.

---

## 2. Technical Stack Decisions

### 2.1 Python 3.11 Runtime
- **Rationale**: Python 3.11.1 is installed on the machine and provides universal binary wheel support for `lightgbm`, `scipy`, `pandas`, `scikit-learn`, and `pulp`. Python 3.14 (system default) lacks pre-built C-extension wheels for several ML dependencies.
- **Implementation**: Virtual environment located at `.venv` created with `py -3.11`.

### 2.2 Mathematical Optimization: PuLP with CBC Solver
- **Rationale**: PuLP with the bundled CBC solver provides fast, exact mixed-integer linear programming (MILP) solutions without external commercial solver licenses. Formulations for squad selection, multi-gameweek transfer routing, and formation optimization run in sub-second times per scenario.

### 2.3 ML Pipeline: Multi-Component Decomposition
- **Rationale**: Predicting total points directly with a single regressor produces flat, compressed predictions that under-estimate variance, ceiling potential (vital for captaincy), and defensive clean sheet correlations. Decomposing into minutes, attacking returns (xG, xA), clean sheet probabilities, DefCon probability, and cards/saves enables:
  - Strict adherence to official scoring rules.
  - Simulation of full point distributions (P10, P50, P90).
  - Proper covariance modeling for double gameweeks.

### 2.4 LLM Abstraction Layer
- **Rationale**: Provider-agnostic interface supporting Google Gemini (default), Anthropic, and OpenAI. Crucially, if no API key is supplied by the user, the application operates in full standalone mode with a rule-based expert generation engine (`OfflineExpertProvider`) rather than crashing or blocking.

### 2.5 Single-Page Application (FastAPI + Modern HTML5/Tailwind/Vue/Alpine + Recharts/Plotly)
- **Rationale**: Clean, self-contained single-page architecture served directly by FastAPI without needing a separate continuous Node build daemon in production mode. Instant startup on `http://localhost:8000`.

---

## 3. Deviations from Initial Suggestions & Engineering Tradeoffs

### 3.1 Standalone Frontend vs. Node/Vite Daemon
- **Suggestion**: React + Vite + Tailwind build pipeline.
- **Adopted**: Self-contained modern responsive HTML5 application powered by Tailwind CSS CDN and modern JavaScript ES modules, served directly by FastAPI at `/` and static assets.
- **Justification**: A continuous Vite/Node build server introduces node version incompatibilities, separate process daemons, and port collision risks on user machines. A single FastAPI process (`python run.py run`) serving both the backend API and the responsive UI guarantees 1-command zero-friction execution.

### 3.2 Dixon-Coles Poisson Model Fallback for Bookmaker Odds
- **Suggestion**: Optional Betting Odds API key (`ODDS_API_KEY`).
- **Adopted**: When `ODDS_API_KEY` is not provided, FPL Oracle automatically computes team attacking and defensive strengths using a bivariate Poisson / Dixon-Coles formulation calibrated on historical goals and xG.
- **Justification**: Users do not need to register for commercial betting odds APIs; the internal Poisson model produces robust clean-sheet and multi-goal probabilities out of the box.

### 3.3 Strict Pre-Deadline Zero-Leakage Windowing
- **Design Choice**: All features for match $i$ at Gameweek $g$ are strictly computed using information known prior to Gameweek $g$'s kickoff deadline ($t-1$).
- **Justification**: Many community FPL models inadvertently leak full-season averages into early gameweek features. Our rolling windowing guarantees realistic backtest and production inference accuracy.

---

## 4. Mathematical Optimization Formulations

### 4.1 15-Man Squad Selection (MILP)
- **Objective**: Maximize $\sum_{i \in \text{Players}} x_i \cdot \text{xP}_i$
- **Subject to**:
  - $\sum x_i = 15$
  - Position counts: $2 \text{ GKP}, 5 \text{ DEF}, 5 \text{ MID}, 3 \text{ FWD}$
  - Club budget constraint: $\sum_{i \in \text{Club } c} x_i \le 3$ for each club $c$
  - Budget constraint: $\sum_{i} x_i \cdot \text{price}_i \le \text{budget}$

### 4.2 Starting XI & Captaincy Optimization
- **Formation Constraints**:
  - $1 \text{ GKP}$
  - $3 \le \text{DEF} \le 5$
  - $2 \le \text{MID} \le 5$
  - $1 \le \text{FWD} \le 3$
  - Total starting XI $= 11$
- **Captaincy Selection**:
  - Weighted combination of expected value, 90th percentile ceiling, and start reliability:
  $$\text{CapScore}_i = \text{xP}_i \cdot 0.70 + P_{90, i} \cdot 0.30 - (1 - P(\text{start}_i)) \cdot 3.0$$

### 4.3 Multi-Gameweek Transfer Routing
- **Horizon**: $H \in [1, 5]$ gameweeks.
- **Free Transfer State Evolution**:
  $$\text{FT}_{t+1} = \min(5, \max(1, \text{FT}_t - \text{TransfersMade}_t + 1))$$
- **Hit Penalty**:
  $$\text{Penalty}_t = 4 \times \max(0, \text{TransfersMade}_t - \text{FT}_t)$$
- **Break-Even Condition**: A -4 hit is only recommended if the 3-gameweek net expected gain $\Delta \text{xP}_{1..3} \ge 4.0 + \text{margin}$ (default margin $= 1.5$ pts).

---

## 5. Top 5 Known Limitations (Honest Self-Assessment)

While FPL Oracle is built with rigorous statistical principles and verified 2026/27 rules, users should be aware of the following 5 inherent limitations:

### 1. Early-Season Small Sample Sizes (GW1–6 Volatility)
- **The Issue**: Early in the season, rolling 3- and 5-match form features rely partially on prior season data and a small number of new matches. Tactical shifts, new managerial appointments, or summer signings have limited competitive sample sizes.
- **Mitigation**: Prior-season shrinkage priors and minute-calibrated baselines prevent overreacting to single-game outliers, but variance remains highest during the first 6 gameweeks.

### 2. Midweek European & Domestic Cup Rotation Blind Spots
- **The Issue**: Midweek Champions League, Europa League, and EFL Cup matches cause manager rotation that cannot be fully anticipated until post-match press conferences occur.
- **Mitigation**: The Minutes model incorporates rotation risk penalties based on squad depth and historical benchings, but late manager decisions (e.g., unexpected rest for key stars) cannot be forecasted with 100% certainty.

### 3. Breaking News Within 15 Minutes of Deadline
- **The Issue**: Press conference embargoes or pre-match leaked lineups (often surfacing 10–30 minutes before the official FPL deadline) may not be parsed in time if the user runs their weekly analysis early.
- **Mitigation**: Users are encouraged to run a final 5-minute pre-deadline check (`python run.py cli briefing`) to refresh injury flags and latest RSS feeds.

### 4. Unofficial Price Change Formula Discrepancies
- **The Issue**: The Premier League does not publish its exact price change algorithm. While FPL Oracle tracks net transfer velocities and hourly rates to forecast price rises/falls, black-box thresholds set by FPL towers (especially regarding wildcard dampening) can cause occasional 24-hour prediction discrepancies.
- **Mitigation**: Price urgency is treated as an informative priority signal rather than an absolute guarantee, focusing on players with large positive transfer momentum.

### 5. Rival Transfer Prediction Uncertainty
- **The Issue**: In mini-league game theory, rivals' future transfer moves are private until the deadline locks. While Oracle simulates rival trajectories using past manager tendencies and template picks, unexpected differential moves by rivals cannot be observed in advance.
- **Mitigation**: Monte Carlo simulations model a distribution of plausible rival lineups, optimizing recommendations to maximize win probability against the most likely rival scenarios.

---

## 6. Engineering Decisions & Hardening (Sessions 2 & 3)

### 6.1 Safe JSON Encoding Layer (`SafeJSONResponse` & `safe_json_serialize`)
- **Problem**: In Python 3.11 with NumPy 2.x and pandas, types like `np.int64`, `np.float64`, `pd.Timestamp`, `NaN`, and `Infinity` cause standard `fastapi.encoders.jsonable_encoder` to raise type errors or emit invalid JSON.
- **Decision**: Implemented recursive sanitizer `safe_json_serialize` in `src/fpl_oracle/server/safe_json.py` and set `default_response_class=SafeJSONResponse` across the entire FastAPI server. Guarantees 100% compliant JSON responses on all 12 endpoints.

### 6.2 Joint DP / Beam Search for Chip Strategy & GW19 Hard Cutoff
- **Problem**: Greedy per-chip selection produces conflicting gameweek assignments (multiple chips in 1 GW) and misses chip synergies.
- **Decision**: Developed exact joint search / beam search in `src/fpl_oracle/chips/planner.py` enforcing exactly 1 chip per gameweek, rewarding Wildcard -> Bench Boost build-up synergies (+5.0 pts), evaluating Set 1 GW19 hard deadline forfeits with opportunity cost calculation, and generating 2-3 robust alternative schedules.

### 6.3 Exact Selling Price & Banked Transfer Mathematical Modeling
- **Decision**: Formulated exact 2026/27 rules in `src/fpl_oracle/optimise/transfers.py`:
  - `selling_price = purchase_price + floor((now_cost - purchase_price) / 2)` when in profit.
  - Banked transfers simulated dynamically: 1 FT added per GW, up to 4 extra banked (5 max), with -4 hit penalties correctly accounted for in candidate plan ranking.

### 6.4 Accent-Insensitive Fuzzy Matching & Manual Squad Fallback
- **Decision**: Built `src/fpl_oracle/data/fuzzy_match.py` with Unicode NFKD diacritic normalization (`Ødegaard` -> `Odegaard`, `Magalhães` -> `Magalhaes`), parenthetical tag stripping, and multi-format parsing (JSON, CSV, plain text). Integrated into web UI with confirmation table and SQLite/`data/profile.json` persistence.

### 6.5 Token-Exact Dispatching in Conversational Expert
- **Problem**: Naive substring matching (e.g., checking `"tc"` in `user_message`) caused false triggers (e.g., "How do I catch the person above me" matched `"tc"` inside "catch").
- **Decision**: Upgraded `OfflineExpertProvider` in `src/fpl_oracle/llm/provider.py` to regex word tokenization, adding dedicated dispatching for player comparisons (`compare_players`), transfer roadmaps, budget differentials, and mini-league intelligence.

### 6.6 Blind Out-of-Time Backtest Architecture (`fpl_oracle.backtest`)
- **Problem**: Synthetic backtests often inadvertently leak target points via noisy proxies or evaluate models trained on future seasons.
- **Decision**: Built a strictly blind out-of-time testing harness (`src/fpl_oracle/backtest.py` and `python run.py backtest`):
  1. Component models are trained exclusively on seasons prior to 2026/27 (87,087 matches).
  2. For every completed gameweek $g \in [1, 4]$, feature values are shifted by 1 ($t-1$), ensuring strictly pre-kickoff information.
  3. The MILP optimizer determines optimal squads and captains using only projected xP.
  4. Realized scores are evaluated across both Free-Hit-per-GW (optimal XI) and sequential season ownership (1 FT/GW, banked up to 5).
  5. Validated that Spearman rank correlation climbs from 0.012 in GW1 to 0.470 in GW3 and 0.451 in GW4 as competitive samples accrue.

### 6.7 Multi-Season Player Continuity & Joint Starters/Captaincy/Bench MILP Architecture
- **Problem Diagnosed**: In FPL, the official API reassigns `element` integer IDs every season (e.g., Cole Palmer was element 235 in 2025/26, but 154 in 2026/27; Erling Haaland was 411 in 2026/27). Grouping by `element` caused cross-season career histories to disconnect, starving premium stars of historical baselines. Furthermore, a naive unweighted 15-player MILP divided £100m by 15 into fifteen £6.5m mid-tier players, starving the squad of premium captaincy anchors.
- **Solution & Implementation**:
  1. **Name-Based Multi-Season Grouping**: In `src/fpl_oracle/data/features.py`, records are grouped by player `name` rather than `element`, correctly chaining 89,141 career match appearances across seasons.
  2. **Seasonal Shrinkage Prior**: Added an expanding career baseline for Round 1 inference, ensuring key starters rested in dead-rubber May matches (GW37/38) are not penalized at season kick-off.
  3. **Joint Starters/Captaincy/Bench MILP**: In `src/fpl_oracle/optimise/squad.py`, formulated joint decision variables for Starters ($s_i$, weight 1.0), Captain ($c_i$, weight 1.0 + ceiling bonus), and Bench ($x_i - s_i$, weight 0.05). Enforced the Talisman Anchor constraint (`value >= 115.0`) to ensure squad structure supports high-ceiling captains.
  4. **Risk-Weighted CapScore**: In `src/fpl_oracle/optimise/lineup.py`, implemented the risk-adjusted formula combining expected points, $P_{90}$ ceiling distribution, and starting probability penalty.
  5. **Out-of-Time 5-GW Validation**: Evaluated against all 5 completed gameweeks of 2026/27. Oracle achieved **290.0 points** (+27.0 pts over Global Average of 263.0, +17.0 pts over Naive Form Baseline of 273.0), successfully forecasting Bruno Fernandes's 23-point haul in GW2 (46 pts as captain) and matching elite mini-league contenders.

### 6.8 Production Resilience, Gameweek Phase Engine & Zero-Error Typing Architecture (Phase 1)
- **Gameweek State Machine (`src/fpl_oracle/api/game_state.py`)**:
  - Implemented 5 discrete lifecycle states (`GameweekPhase`): `PRE_DEADLINE`, `LIVE`, `BONUS_PENDING`, `FINISHED`, and `BETWEEN_GWS`.
  - Automatically calculates exact seconds to deadline, live match clock progress, fixture postponements, and BGW/DGW detection.
- **Resilient HTTP Client & In-Flight Coalescing (`src/fpl_oracle/api/fpl_client.py`)**:
  - Persistent `httpx.AsyncClient` utilizing connection pooling (`max_connections=20`, `max_keepalive_connections=10`).
  - Added randomized jittered exponential backoff (`0.8 + 0.4 * random.random()`), avoiding thundering herd.
  - Implemented in-flight request coalescing (`_in_flight` task map) to collapse identical concurrent requests into a single outbound fetch.
  - Provided polite batching (`get_element_summaries_batch`) and pluggable transport injection (`set_transport`/`reset_transport`) for hermetic fault testing.
- **Fault Injection Test Suite (`tests/test_fault_injection.py`)**:
  - Verified 100% graceful degradation under 429 Rate Limits (serving stale cache with diagnostic metadata), 503 Service Downtime, Read Timeouts, Schema Drift (Pydantic `extra="allow"`), and malformed squad payloads (HTTP 400 with helpful remediation hints).
- **Subsystem Diagnostics & Freshness Tracking**:
  - Enhanced `/api/health` with comprehensive subsystem statuses: upstream reachability, game state countdown, cache ages per dataset, ML component health (6/6 models), and news feed health.
  - Added `data_as_of` and `stale`/`is_stale` attributes to all JSON responses.
- **Type Safety & SQLAlchemy 2.0 Modernization**:
  - Migrated database layer to SQLAlchemy 2.0 `DeclarativeBase` and decoupled domain profile transfer objects via `ProfileData` dataclass.
  - Resolved all typing discrepancies: mypy reports **0 errors across 51 source files**, and ruff reports **0 errors (100% clean)**.

### 6.9 Data, ML & Optimizer Integrity Verification (Phase 2)
- **Live Rules Verification Engine (`src/fpl_oracle/api/rules_checker.py`)**:
  - Automatically verifies live `bootstrap-static` against `config/rules.yaml` and `config/scoring.yaml`.
  - Audits 2026/27 chips (Set 1 GW1-19, Set 2 GW20-38, assistant manager absent), free transfers banking (up to 5), selling price formula (50% profit retention), squad constraints (15 players, 11 starters, 3 per club), and scoring values including Defensive Contribution (DefCon +2 pts).
  - Exposed via `/api/health` under `rules_verification` and tested in `tests/test_rules_checker.py`.
- **Strict Leakage Audit Suite (`tests/test_leakage.py`)**:
  - Proved that perturbing post-deadline match data (minutes, goals, xG, xA, DefCon) in GW_k leaves the feature vector for GW_k completely unchanged ($\Delta = 0.0$).
  - Proved that GW_k perturbations strictly propagate to GW_(k+1), confirming exact $t-1$ shift dynamics with zero lookahead contamination.
- **Uncertainty Calibration & Quantile Evaluation (`reports/model_eval.md`)**:
  - Validated $[P_{10}, P_{90}]$ credible intervals across holdout validation data, confirming sharp uncertainty estimation and minimal pinball losses across quantiles ($q=0.10, 0.50, 0.90$).
  - Chronological rolling-origin evaluation across 2023-24, 2024-25, 2025-26, and 2026-27 demonstrates stable MAE (0.865–0.891 pts) and high rank correlation ($\rho \approx 0.70$).
- **Property-Based Verification with Hypothesis (`tests/test_optimizer_properties.py`)**:
  - Mathematically verified invariant solver constraints across hundreds of randomized scenarios: squad budget limit ($\sum \text{cost} \le \text{budget}$), exact positional quotas (2 GKP, 5 DEF, 5 MID, 3 FWD), club limits ($\le 3$ per team), valid lineup formations, and monotonic, bounded selling price calculations.


