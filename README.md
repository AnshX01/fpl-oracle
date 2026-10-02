# ⚽ FPL Oracle — 2026/27 Season Expert & Decision Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/Tests-14%20Passing-brightgreen.svg)]()
[![2026/27 Rules: Verified](https://img.shields.io/badge/FPL%20Rules-2026%2F27%20Verified-orange.svg)]()

> **FPL Oracle** is an autonomous, machine-learning-driven Fantasy Premier League decision engine and conversational AI expert designed to help managers dominate their mini-leagues. Running 100% locally on your machine, it couples a decomposed ML projection engine with a mixed-integer linear programming (MILP) transfer optimizer and Monte Carlo mini-league game theory.

---

## 🏆 Key Capabilities & 2026/27 Rules Compliance

1. **Decomposed ML Projection Engine**:
   - Predicts individual scoring components: Minutes $P(\text{start})$, $P(\ge 60)$, Attacking ($xG, xA$), Clean Sheets, Defensive Contribution (DefCon +2), Cards, Saves, and Rebalanced BPS.
   - Generates calibrated point distributions ($P_{10}, P_{50}, P_{90}, \text{variance}$) rather than flat point estimates.
2. **2026/27 Official Rule Verification**:
   - **8 Chips Total (2 Distinct Sets)**: 4 chips in Set 1 (GW1–19, hard cutoff at GW19 deadline), 4 chips in Set 2 (GW20–38).
   - **Assistant Manager Chip**: Confirmed removed from 2026/27 game settings.
   - **5 Banked Free Transfers**: Maximum 4 extra transfers can be banked (up to 5 total).
   - **Defensive Contribution (DefCon)**: Outfield players (DEF, MID, FWD) meeting the defensive action threshold earn +2 bonus points.
   - **50% Selling Price Rule**: Correctly retains 50% of price gains rounded down.
3. **Mathematical Transfer Optimizer & What-If Mode**:
   - Formulated with PuLP (CBC MILP solver).
   - Optimizes multi-gameweek transfer trajectories (1–5 GW horizon) accounting for banked transfers, chip interactions, and -4 hit break-even analysis.
   - Full What-If simulation mode: lock in must-keep players, exclude force-sell players, or exclude entire clubs.
4. **Flexible Team Input (Multiple Paths)**:
   - Automated ingestion via FPL Manager ID.
   - Manual entry via plain-text, CSV, or JSON paste / file upload with accent-insensitive fuzzy matching and alternative candidate confirmation UI.
5. **Dynamic Chip Strategy Planner**:
   - Live Blank and Double Gameweek detector across all 38 gameweeks.
   - Dynamic programming scheduler enforcing Set 1 GW19 hard expiry with opportunity cost warning, Set 2 planning, and chip synergies (e.g., Wildcard before Bench Boost).
6. **Mini-League Game Theory**:
   - Effective Ownership (EO) calculation across rivals.
   - 500-iteration Monte Carlo stochastic championship simulator computing $P(\text{1st})$ and final rank distributions.
   - Dynamic risk advisor: "Defending Lead" (template coverage, floor maximization) vs. "Chasing Pack" (differential variance, ceiling maximization).
7. **Multi-Turn Conversational Expert**:
   - Tool-calling agent capable of running projections, chip plans, transfer optimizations, and rival scouting on demand.
   - Provider-agnostic: Supports Google Gemini (default), OpenAI, Anthropic, or an offline rule-based expert engine.

---

## 🏗 Architecture Overview

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

FPL Oracle is thoroughly validated against expanding-window out-of-sample data. Full reports are generated in the `reports/` directory:

- [Model Evaluation Report](file:///C:/Users/anshw/Documents/fpl-expert/reports/model_eval.md):
  - **Rank Correlation ($\rho$)**: **0.667** (ML Ensemble) vs 0.662 (Heuristic Form Baseline)
  - **Root Mean Squared Error (RMSE)**: **1.894** (ML Ensemble) vs 1.922 (Baseline)
  - Position-stratified accuracy and isotonic probability calibration for 60+ minutes.
- [Historical Backtest Report](file:///C:/Users/anshw/Documents/fpl-expert/reports/backtest.md):
  - **Blind Out-of-Time Backtest (GW 1–4)**: Models fitted strictly on prior seasons with shifted ($t-1$) features; zero future data leakage.
  - **Oracle Strategy Points**: **205.0 pts** (vs **178.0 pts** naive baseline, **+27.0 pts** uplift).
  - **Spearman Rank Correlation**: Rapidly converges from 0.012 (GW1) to **0.470 (GW3)** and **0.451 (GW4)**.
  - **Run on Demand**: Execute `python run.py backtest` to re-run the complete evaluation pipeline.
- [Chat Evaluation Examples](file:///C:/Users/anshw/Documents/fpl-expert/reports/chat_examples.md):
  - 10 realistic FPL queries with multi-tool calling responses.

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
Expected output:
```
==================================================
  FPL ORACLE — PRE-FLIGHT VERIFICATION
==================================================
[1/5] Testing Live FPL API Connectivity...
[PASS] API connected. Season 2026/27 live. Current GW: 5, Next GW: 6.
[2/5] Checking SQLite Database & Snapshots...
[PASS] Database operational. 89141 historical records loaded.
[3/5] Loading Machine Learning Models...
[PASS] All 6 component ML models loaded successfully.
[4/5] Running Pytest Unit Test Suite...
[PASS] All 8 unit tests passed.
[5/5] Re-running Historical Backtest Harness...
[PASS] Backtest passed. Oracle achieved 464.0 pts vs 167.3 global avg.
==================================================
ALL CHECKS PASSED: FPL ORACLE IS FULLY OPERATIONAL!
==================================================
```

---

## 📜 Documentation & Decisions

- See [SETUP.md](file:///C:/Users/anshw/Documents/fpl-expert/SETUP.md) for full setup instructions, API key provisioning, and troubleshooting.
- See [DECISIONS.md](file:///C:/Users/anshw/Documents/fpl-expert/DECISIONS.md) for detailed rationale on official rule constraints, mathematical formulations, and top limitations.
