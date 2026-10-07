> See [DELIVERY_STATUS.md](DELIVERY_STATUS.md) for Windows startup, verified repairs, actual model results and remaining limits. This source delivery is not a full-quality certification.

# FPL Oracle - 2026/27 decision support checkpoint

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
Testing status: see current CI results and the scoped repair checkpoints below; no full-suite pass is claimed.


> **FPL Oracle** is an autonomous, machine-learning-driven Fantasy Premier League decision engine and conversational AI expert for comparing feasible actions for a configured squad. Running 100% locally on your machine, it couples a decomposed ML projection engine with a stateful multi-gameweek beam-search transfer optimizer, correlated clean-sheet Monte Carlo simulation, and an Atlas/Council decision dashboard.

---

## Implemented paths (see CHECKPOINT_STATUS.md for remaining validation)

1. **Decomposed ML Projection Engine**:
   - Predicts individual scoring components: Minutes $P(\text{start})$, $P(\ge 60)$, Attacking ($xG, xA$), Clean Sheets, Defensive Contribution (DefCon +2), Cards, Saves, and Rebalanced BPS.
   - Generates point distributions (coverage validation is still required) ($P_{10}, P_{50}, P_{90}, \text{variance}$) rather than flat point estimates.
2. **2026/27 Official Rule Verification**:
   - **8 Chips Total (2 Distinct Sets)**: 4 chips in Set 1 (GW1–19, hard cutoff at GW19 deadline), 4 chips in Set 2 (GW20–38).
   - **Assistant Manager Chip**: Confirmed removed from 2026/27 game settings.
   - **5 Banked Free Transfers**: Maximum 4 extra transfers can be banked (up to 5 total).
   - **Defensive Contribution (DefCon)**: Outfield players (DEF, MID, FWD) meeting the defensive action threshold earn +2 bonus points.
   - **50% Selling Price Rule**: Correctly retains 50% of price gains rounded down.
3. **Stateful Transfer Optimizer & What-If Mode**:
   - Implements sequential 5-GW beam search optimizer.
   - Accurately tracks squad changes, bank balances, exact 50% profit selling prices, free transfer rollover (1–5 banked FTs), and club quotas.
   - Computes future-hit avoidance, free transfer option values, and robustness re-ranking (win rate $\ge 0.70$) across projection noise.
   - Full What-If simulation mode: lock in must-keep players, exclude force-sell players, or test price change sensitivities.
4. **Flexible Team Input (Multiple Paths)**:
   - Automated ingestion via FPL Manager ID.
   - Manual entry via plain-text, CSV, or JSON paste / file upload with accent-insensitive fuzzy matching and alternative candidate confirmation UI.
5. **Dynamic Chip Strategy Planner**:
   - Live Blank and Double Gameweek detector across all 38 gameweeks.
   - Evaluates multi-GW incremental benefit against a no-chip baseline, enforcing Set 1 GW19 hard expiry and uncertainty gating.
6. **Mini-League Game Theory & Correlated Monte Carlo**:
   - Paginates up to 10 pages (500 teams) with proximity-based rival selection (all teams ahead + within 30 points below).
   - Monte Carlo stochastic simulator sampling shared players once per trial and sharing correlated clean-sheet Bernoulli draws (+4 pts) across club defenders.
   - Dynamic risk advisor: "Defending Lead" vs. "Chasing Pack" bounded tie-breakers.
7. **Per-Gameweek Unified Decision Card**:
   - Unified battle plan served via `/api/decision-card` and exportable markdown `/api/decision-card/export`.
   - Combines Starting XI, captain/vice-captain, transfer trajectory, chip recommendations, rival proximity context, win probabilities, and tactical caveats.
8. **Multi-Turn Conversational Expert**:
   - Tool-calling agent capable of running projections, chip plans, transfer optimizations, and rival scouting on demand.
   - Provider-agnostic: Supports Google Gemini (default), OpenAI, Anthropic, or an offline rule-based expert engine.

---

## ⚙️ Complete Setup: Baseline, Repaired Model & Integrated LLM Extraction

FPL Oracle runs **locally with a no-key public-data path** on your machine. You can run the entire pipeline out of the box using public official FPL endpoints without any API keys, or optionally enable an optional Google Gemini account tier for AI press conference extraction and chat.

### 1. Environment Configuration (`.env`)
All configuration, manager IDs, and API keys reside exclusively in the local, gitignored `.env` file. There are **zero credentials or ID inputs in the web interface** for privacy and security.

Create your `.env` file by copying the template:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```ini
# ==============================================================================
# FPL User & League Settings
# ==============================================================================
# Find your Team ID on the FPL website under the 'Points' or 'Gameweek history' URL:
# https://fantasy.premierleague.com/entry/<YOUR_ID>/event/1
FPL_MANAGER_ID=1234567

# Find your Mini-League ID under 'Leagues & Cups' -> Standings URL:
# https://fantasy.premierleague.com/leagues/<YOUR_LEAGUE_ID>/standings/c
FPL_TARGET_LEAGUE_ID=7654321

# ==============================================================================
# Google Gemini Free-Tier Integration (Optional; verify account quota and pricing)
# ==============================================================================
# Check API key availability, quota and billing terms: https://aistudio.google.com/
GEMINI_API_KEY=your_free_key_here

# Safety confirmation safeguard: MUST be set to true to enable Gemini Free Tier.
# If omitted or false, the system safely disables Gemini calls and falls back.
GEMINI_FREE_TIER_CONFIRMED=true

# Fast, non-billable free-tier model (defaults to gemini-2.5-flash-lite)
GEMINI_MODEL=gemini-2.5-flash-lite

# ==============================================================================
# Availability & News Recommendation Mode
# ==============================================================================
# Operating modes:
#   - 'shadow' (Default & Recommended): News quotes, publication age, and candidate
#     reconciliations are logged and displayed in the UI, but production projections
#     remain strictly anchored to official FPL API availability.
#   - 'gated_active': Candidate news updates (e.g. manager quotes) actively adjust
#     expected availability in production once verified.
#   - 'api_only': Completely ignores RSS/news feeds and relies 100% on official FPL API.
NEWS_RECOMMENDATION_MODE=shadow

# News ingestion polling interval in minutes (default: 60)
NEWS_FETCH_INTERVAL_MINUTES=60
```

> [!NOTE]
> **Free-Tier Protection Guaranteed:** The built-in `GeminiBudgetManager` persistently enforces a maximum limit of **150 requests/day** and **500,000 tokens/day** in `data/cache/gemini_budget.json`. If this threshold is reached, or if you run without an API key, the system automatically falls back to official FPL API data without raising any errors.

---

### 2. Quickstart Execution Commands

```bash
# 1. Install dependencies (Python 3.11 recommended)
pip install -e .

# 2. Run the automated test suite (duration depends on network/solver tests)
pytest tests/ -v

# 3. Train or evaluate the 6 LightGBM component models
python -m fpl_oracle.ml.train

# 4. Start the local FastAPI server and dashboard
uvicorn fpl_oracle.server.main:app --host 127.0.0.1 --port 8000
```
Open your browser at **http://127.0.0.1:8000** to access the dashboard.

---

```mermaid
flowchart TD
    subgraph Data Layer
        API[Official FPL Live API\nbootstrap-static, fixtures, picks] --> Cache[SQLite Cache & Snapshots\ndata/fpl_oracle.db]
        Hist[Historical Match Dataset\nvaastav community data + 2026/27] --> Feat[Zero-Leakage Feature Pipeline\nsrc/fpl_oracle/data/features.py]
        News[News & RSS Ingestion\nOfficial status + BBC/Sky feeds] --> Feat
    end

    subgraph Machine Learning Engine
        Feat --> Train[Model Trainer\nLightGBM 6-Component Ensemble]
        Train --> Models[Calibrated Pickles\ndata/models/*.pkl]
        Models --> Predict[ML Inference Engine\nsrc/fpl_oracle/ml/predict.py]
    end

    subgraph Mathematical Optimization
        Predict --> MILP[PuLP CBC Solver\nMulti-GW Transfer Optimizer]
        Predict --> Lineup[Lineup & Captain Optimizer\nFloor / Ceiling weighting]
        Predict --> Chips[Chip Scheduler\nGW19 Set 1 hard cutoff & Set 2]
    end

    subgraph Intelligence & Strategy
        API --> League[Mini-League Tracker\nStandings & Rival Squads]
        Predict --> League
        League --> MC[Monte Carlo Simulation\nP(1st) & Risk Mode Advisor]
    end

    subgraph User Interfaces
        MILP --> Web[FastAPI Web App\nhttp://localhost:8000]
        Lineup --> Web
        Chips --> Web
        MC --> Web
        MILP --> CLI[Typer CLI\npython run.py cli ...]
        Lineup --> CLI
        Chips --> CLI
        LLM[LLM Agent + Tool Calling\nGemini / Claude / GPT / Offline] --> Web
        LLM --> CLI
    end
```

---

## ⚡ The 5-Minute Weekly Workflow

FPL Oracle is engineered to turn your weekly deadline routine into a 5-minute high-confidence process:

| Step | Time | What to Do | What FPL Oracle Delivers |
|---|---|---|---|
| **1. Ingest & Brief** | Minute 1 | Run `python run.py cli briefing` or click "Briefing" in Web UI | Scans latest press conferences, confirmed injuries, DefCon candidates, and deadline countdown. |
| **2. Review Lineup** | Minute 2 | Open Pitch View on `http://localhost:8000` | Displays optimal starting XI, auto-sub order, DefCon badges, and captain / vice-captain recommendations. |
| **3. Transfer Decisions** | Minute 3 | View Transfer Planner tab | Recommends optimal moves, rolls/banks transfers (up to 5), and flags whether a -4 hit breaks even over 3 GWs. |
| **4. Check Rival Threats** | Minute 4 | Open Mini-League tab | Displays rival Effective Ownership, your $P(\text{1st})$ championship probability, and high-upside differentials. |
| **5. Chat & Confirm** | Minute 5 | Ask any specific questions in Chat | e.g., *"Should I take a -4 to bring in Saka or roll my transfer?"* — the AI validates with live numbers. |

---

## 💻 CLI Commands Reference

All features are accessible directly from the command line:

```bash
# 1. Inspect your current squad, projected points, and lineup
python run.py cli squad --manager-id <ID>

# 2. Compute the optimal transfer roadmap for the next 3 gameweeks
python run.py cli optimize --manager-id <ID> --horizon 3

# 3. Formulate the season-long chip strategy (enforcing GW19 Set 1 cutoff)
python run.py cli chips --manager-id <ID>

# 4. Analyze mini-league standings, rival differentials, and championship odds
python run.py cli analyze --league-id <LEAGUE_ID> --manager-id <ID>

# 5. Generate and print the full Gameweek Briefing report
python run.py cli briefing --manager-id <ID>

# 6. Launch interactive conversational assistant
python run.py cli chat
```

---

## 🚀 Web Dashboard Tour

When running `python run.py run`, navigate to `http://localhost:8000` for the single-page dark-mode dashboard:

- **Pitch View**: Dynamic football pitch rendering your starting XI and bench. Badges display expected points, fixture difficulty ratings (FDR), DefCon return likelihood, and yellow/red injury warnings.
- **Transfer Roadmap**: Tabular view of recommended transfers over the next 1–5 gameweeks, showing selling price gains, banked FT count, and hit break-even analysis.
- **Chip Horizon Calendar**: Visual timeline across all 38 gameweeks highlighting Blank and Double Gameweeks, the Gameweek 19 Set 1 expiry, and optimal windows for Bench Boost, Triple Captain, Free Hit, and Wildcard.
- **Mini-League Battleground**: Interactive leaderboards showing Effective Ownership of key threats, manager rank trajectories, and Monte Carlo probability gauges.
- **AI Analyst Chat**: Conversational interface with direct access to all backend tools and data snapshots.

---

## 📊 Model Validation & Performance

FPL Oracle's checked-in metrics are historical reports, not a fresh certification of the current code.

- The recorded nominal 80% interval coverage is 71.28%, below the unchanged 75-85% acceptance gate.
- The historical proxy replay is not a replay of the complete served advisor. Its Brier result does not certify the production mini-league simulation or the recommended actions.
- The deterministic news benchmark does not establish Gemini accuracy. Each extractor needs its own measured passing evidence before production overrides.
- Forward scoring requires a genuine pre-deadline prediction snapshot and finalized official actuals. Synthetic tests never count as forward performance.
- Current repairs are on a working branch. Passing scoped tests do not mean every feature or the full suite is complete.

---

## 🔄 Retraining & Feature Engineering

To retrain the ML ensemble with newly completed gameweek data:
```bash
python run.py train
```
The training pipeline executes:
1. **Match Ingestion**: Reads `data/historical/master_history.csv` (89,141 historical player-match records) and live 2026/27 match data.
2. **Zero-Leakage Lag Engineering**: Builds rolling 3, 5, and 8-match forms, per-90 metrics, and opponent defensive strength strictly as of the kickoff deadline ($t-1$).
3. **Model Refitting**: Fits 6 specialized LightGBM models (`minutes.pkl`, `attacking.pkl`, `defending.pkl`, `defcon.pkl`, `bonus.pkl`, `cards_saves.pkl`).
4. **Isotonic Calibration**: Calibrates probability curves against empirical outcomes.
5. **Report Generation**: Updates `reports/model_eval.md` automatically.

---

## 🛠 Self-Check & Verification

Before every gameweek, verify that all systems are healthy:
```bash
python run.py verify
```
The command reports the checks it actually ran. Treat failures and unavailable data as blockers for their affected advice; do not interpret a stored example as a current pass.

---

## 🔍 Honest Limitations & Caveats

In accordance with the project's evidence-gated engineering standards:
1. **Unconstrained Model Learning**: Fixture difficulty, opponent defensive form, and team strength are provided strictly as continuous ML features. No hard constraints, arbitrary floors, or artificial point caps are placed on elite players.
2. **Headless Environment Limitation**: When running in headless terminal environments without display servers or browser automation frameworks (Playwright/Selenium), real browser screenshot rasterization cannot be visually captured; all FastAPI REST endpoints, static assets, and Vue components are verified programmatically.
3. **Free-Tier Boundaries**: Google Gemini integration operates under strict local daily request (150/day) and token budget limits (500k tokens/day). If unconfigured or exhausted, the system seamlessly falls back to deterministic official FPL API availability.

---

## 📜 Documentation & Decisions

- See [SETUP.md](SETUP.md) for full setup instructions, API key provisioning, and troubleshooting.
- See [DECISIONS.md](DECISIONS.md) for detailed rationale on official rule constraints, mathematical formulations, and top limitations.
- See [reports/repair_audit.md](reports/repair_audit.md) and [reports/gap_closure.md](reports/gap_closure.md) for complete verification audit trails.
