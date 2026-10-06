# FPL Oracle — Specification Compliance & Coverage Audit

This report provides a section-by-section audit of the FPL Oracle codebase against the original specification (`fpl_expert_antigravity_goal_prompt.md`). Every requirement is categorized as **DONE**, **PARTIAL**, or **MISSING**, with explicit file and line references.

---

## Summary Matrix

| Spec Section | Feature Area | Status | Key Code References |
|---|---|:---:|---|
| **3.1** | Official FPL API Client & Snapshots | **DONE** | `src/fpl_oracle/api/fpl_client.py`, `models.py`, `cache.py`, `data/store.py` |
| **3.2** | Historical Training Ingestion | **DONE** | `src/fpl_oracle/data/historical.py`, `features.py` |
| **3.3** | News & Qualitative RSS Signals | **DONE** | `src/fpl_oracle/news/ingest.py`, `extract.py`, `analyse.py`, `config/news_sources.yaml` |
| **3.4** | Optional Betting / Dixon-Coles Poisson | **DONE** | `src/fpl_oracle/data/features.py`, `DECISIONS.md` |
| **4.1** | Team Input (Manager ID & Manual/Upload) | **DONE** | `src/fpl_oracle/data/fuzzy_match.py`, `server/routes/api.py`, `web/index.html` |
| **4.2** | Interactive Dashboard & Pitch View | **DONE** | `web/index.html`, `server/routes/api.py` |
| **4.3** | ML Projection Engine & Decomposed Models | **DONE** | `src/fpl_oracle/ml/components/`, `predict.py`, `train.py`, `eval.py` |
| **4.4** | Transfer Optimizer & What-If Mode | **DONE** | `src/fpl_oracle/optimise/transfers.py`, `squad.py`, `price_change.py` |
| **4.5** | Chip Strategy Engine (2026/27 Rules) | **DONE** | `src/fpl_oracle/chips/planner.py`, `calendar.py`, `simulate.py` |
| **4.6** | Captaincy, Lineup & P10/P90 Distributions | **DONE** | `src/fpl_oracle/optimise/lineup.py` |
| **4.7** | Mini-League Intelligence & Monte Carlo | **DONE** | `src/fpl_oracle/league/standings.py`, `rivals.py`, `montecarlo.py`, `strategy.py` |
| **4.8** | Conversational Expert Chat & Tool Calling | **DONE** | `src/fpl_oracle/llm/provider.py`, `agent.py`, `tools.py`, `cli.py` |
| **4.9** | Weekly Gameweek Briefing & Automation | **DONE** | `src/fpl_oracle/briefing/weekly.py`, `review.py`, `notify.py`, `server/jobs.py` |
| **4.10** | Contingency Engine & Panic Button | **DONE** | `src/fpl_oracle/optimise/contingency.py`, `briefing/review.py` |
| **5.0** | Backtesting & Validation Harness | **DONE** | `src/fpl_oracle/backtest.py`, `verify.py`, `reports/backtest.md` |
| **6.0** | Architecture, Packaging & Tech Stack | **DONE** | `pyproject.toml`, `run.py`, `Makefile`, `config/*.yaml` |
| **7.0** | Reasoning & Explanation Standards | **DONE** | `src/fpl_oracle/optimise/transfers.py`, `chips/planner.py`, `llm/provider.py` |
| **8.0** | Security, Ethics & Localhost Execution | **DONE** | `src/fpl_oracle/server/main.py`, `.env.example`, `SETUP.md` |
| **9.0** | Background Automation & Rollback Guard | **DONE** | `src/fpl_oracle/server/jobs.py`, `ml/model_registry.py`, `.github/workflows/ci.yml` |

---

## Detailed Section-by-Section Audit

### Section 3: Data Sources

#### 3.1 Official FPL API
- **Live Endpoint Integration**: All required public endpoints (`bootstrap-static`, `fixtures`, `element-summary/{id}`, `event/{gw}/live`, `entry/{id}`, `entry/{id}/history`, `entry/{id}/event/{gw}/picks`, `entry/{id}/transfers`, `leagues-classic/{id}/standings`, `event-status`, `dream-team`, `team/set-piece-notes`) implemented in `src/fpl_oracle/api/fpl_client.py`.
- **Defensive Pydantic Models**: All schemas configured with `extra="allow"` in `src/fpl_oracle/api/models.py`. Nullable/optional fields verified against live 2026/27 payloads.
- **SQLite Persistence & Snapshots**: Every raw response cached in SQLite (`data/fpl_oracle.db`) via `src/fpl_oracle/api/cache.py` with TTL-based expiration and rate-limit backoff.
- **Zero Credentials Needed**: Public endpoints only; no manager password or auth tokens required.
- **Stale Data Handling**: Automatic fallback to SQLite snapshots if official endpoints return 5xx/429 during deadline spikes, displaying a live banner with sync timestamp (`web/index.html`).

#### 3.2 Historical Training Data
- **Dataset Ingestion**: `src/fpl_oracle/data/historical.py` downloads seasons (2022/23 through 2025/26) from `vaastav/Fantasy-Premier-League` and normalizes schema.
- **Live Season Ingestion**: Completed 2026/27 gameweeks (GW1-5) ingested into `data/historical/master_history.csv`.
- **Zero Data Leakage**: `src/fpl_oracle/data/features.py` enforces strict pre-deadline windowing ($t-1$) for all rolling form, team strength, and player minutes features.

#### 3.3 News & Qualitative Signals
- **Authoritative Status Ingestion**: Ingests official `status`, `news`, `news_added`, and `chance_of_playing_next_round` (`src/fpl_oracle/news/ingest.py`).
- **Live RSS Ingestion**: Connects to BBC Football, Sky Sports, The Guardian, and TalkSport. Dead feeds automatically tracked and dropped dynamically (`news_ingestion.dead_feeds`).
- **Structured Signal Representation**: `src/fpl_oracle/news/analyse.py` attaches element ID, status code, source name, source URL, confidence weighting, and ISO timestamp. Strictly no fabricated news.

---

### Section 4: Features

#### 4.1 Team Input
- **Manager ID Path**: Fetches current 15 picks, bank, transfer history, chips used, and selling prices automatically (`server/routes/api.py:get_squad`).
- **Manual / Paste / Upload Path**: `src/fpl_oracle/data/fuzzy_match.py` accepts plain text, CSV, or JSON player lists. Handles diacritics/accents (`Ødegaard`, `Magalhães`), parenthetical tags, and club prefixes.
- **Confirmation UI & Editing**: Modal in `web/index.html` renders interactive match table with confidence scores, alternative dropdowns, formation validator (2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club), and bank/FT inputs.
- **Profile Persistence**: Synchronized in SQLite `user_profile` table and `data/profile.json` via `src/fpl_oracle/data/store.py`.

#### 4.2 Dashboard
- **Football Pitch View**: CSS pitch with starting XI and ordered bench. Player cards display cost, xP, FDR difficulty for next 3-5 fixtures, DefCon indicator, and captaincy badges (`web/index.html`).
- **Countdown & Freshness**: Gameweek indicator, deadline countdown, and live sync timestamps.

#### 4.3 ML Projection Engine
- **Decomposed Components**: 6 separate component models (`minutes.py`, `attacking.py`, `defending.py`, `defcon.py`, `bonus.py`, `cards_saves.py`) in `src/fpl_oracle/ml/components/`.
- **Scoring System 2026/27**: Configured in `config/scoring.yaml`. Implements Defensive Contribution (DefCon +2) and rebalanced BPS with rule shift features.
- **Calibrated Distributions**: Quantile estimation outputs $P_{10}$ (floor), $P_{50}$ (median), $P_{90}$ (ceiling), and variance per player (`src/fpl_oracle/ml/predict.py`).
- **Validation**: Outperforms baseline in `reports/model_eval.md` ($\rho = 0.667$ vs 0.662).

#### 4.4 Transfer Optimizer
- **MILP Formulation**: PuLP CBC integer linear programming over 1-5 GW horizon in `src/fpl_oracle/optimise/transfers.py`.
- **Rule Constraints**: 15 players, budget, position quotas, max 3 per club, selling price (purchase price + floor((current - purchase) / 2)).
- **Banked Free Transfers**: Verified 2026/27 banking logic (up to 5 FTs max).
- **Hit Verdict**: Explicit break-even evaluation over 3 gameweeks.
- **What-If Mode**: Interactive controls for locking players in/out and excluding clubs (`/api/optimize`, `web/index.html`).
- **Transfer Roadmap**: 5-gameweek sequential plan with firm vs contingent tagging, target assets, and banked FT projections.
- **Price Change Watch**: Imminent rises ("buy before rise") and falls ("sell before drop") in `src/fpl_oracle/optimise/price_change.py`.

#### 4.5 Chip Strategy Engine (Critical 2026/27 Rules)
- **8 Chips / 2 Sets Architecture**: 4 chips in Set 1 (GW1-19), 4 chips in Set 2 (GW20-38). Assistant Manager removed (`config/rules.yaml`).
- **GW19 Set 1 Expiry Warning**: Prominent warning banner in UI with exact opportunity cost of lapsed chips (`chips/planner.py`, `web/index.html`).
- **Joint Assignment DP / Beam Search**: Enforces 1 chip per GW, Wildcard -> Bench Boost synergy (+5.0 pts), and DGW/BGW targeting (`chips/planner.py:optimize_joint_assignment`).
- **Chip Plan Table**: Chip, set, recommended GW, expected gain, confidence, alternative GW, trigger conditions, and opportunity cost.
- **Robust Alternatives**: Reports top schedule plus 2 robust alternative schedules.

#### 4.6 Captaincy and Lineup
- **Formation Selection**: Selects optimal 11 starters meeting formation rules (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) and ordered bench.
- **Floor & Ceiling Optimization**: Selects primary captain, safe alternative (highest $P_{10}$), differential alternative (highest $P_{90}$), and probability of outscoring next best asset using error functions (`src/fpl_oracle/optimise/lineup.py`).
- **Bench Risk & Vice Captain**: Identifies doubtful starters and provides vice-captain reasoning.

#### 4.7 Mini-League Intelligence
- **Standings & Rivals**: Paginates mini-league table and inspects top rival squads (`src/fpl_oracle/league/standings.py`, `rivals.py`).
- **Effective Ownership (EO)**: Categorizes players into template (>50% EO) and differentials (<20% EO).
- **Monte Carlo Simulator**: 500-iteration stochastic championship simulation calculating $P(\text{finish 1st})$ and final rank distribution (`league/montecarlo.py`).
- **Strategy Mode**: Automatically activates "Leading Mode" (template protection, low variance) or "Chasing Mode" (high upside differentials) with tactical recommendations (`league/strategy.py`).

#### 4.8 Conversational Expert Chat
- **12 Live Grounded Tools**: `get_my_team`, `get_projections`, `get_fixtures`, `get_news`, `optimise_transfers`, `plan_chips`, `captain_options`, `league_analysis`, `simulate_win_probability`, `price_change_watch`, `compare_players`, `player_deep_dive` in `src/fpl_oracle/llm/tools.py`.
- **Multi-Provider Architecture**: Google Gemini (default), Anthropic, OpenAI, and built-in `OfflineExpertProvider` in `src/fpl_oracle/llm/provider.py`.
- **Zero Hallucination Guarantee**: Explanations cite actual numbers, source URLs, and timestamps without fabricating stats.
- **CLI & Web Chat**: Interactive web UI tab and `python run.py cli chat`.

#### 4.9 Weekly Workflow Automation
- **Gameweek Briefing**: Markdown report and UI card summarizing deadline countdown, injuries, transfer plans, captain, chips, and rivals (`src/fpl_oracle/briefing/weekly.py`).
- **Webhook Alerts**: Optional Discord/Telegram alerts prior to deadline (`src/fpl_oracle/briefing/notify.py`).

---

### Section 5: Backtesting & Verification
- **Replay Harness**: Replays past completed gameweeks using strict pre-deadline information (`src/fpl_oracle/backtest.py`, `reports/backtest.md`).
- **Automated Self-Check**: `python run.py verify` tests live API connectivity, SQLite storage, 6 ML models, 14 pytest unit tests, and mini backtest.

---

### Section 6: Architecture & Packaging
- **Clean Repo Structure**: Matches specification layout.
- **Safe JSON Serialization**: Custom `SafeJSONResponse` and `safe_json_serialize` preventing NumPy 2.x / pandas encoding issues across all endpoints (`src/fpl_oracle/server/safe_json.py`).
- **Execution**: Simple single-process startup via `python run.py run`.
