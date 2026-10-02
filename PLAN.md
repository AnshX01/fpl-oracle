# FPL Oracle — Master Implementation Plan

## Overview
FPL Oracle is a local, ML-driven Fantasy Premier League decision support system and conversational AI expert designed to maximize expected points and mini-league win probability for the 2026/27 season (commencing October 2026, GW6 upcoming).

This document outlines the detailed architecture, milestone sequence, validation plan, and component specifications.

---

## 1. Milestones & Work Breakdown

### Milestone 1: Environment, Configuration & Live Rule Verification
- [x] Python 3.11 virtual environment setup.
- [ ] Base directory layout and package configuration (`pyproject.toml`, `.env.example`, `Makefile`, `run.py`).
- [ ] Rule & scoring verification against live API (`bootstrap-static`, `fixtures`):
  - 2026/27 Chips: 8 total (2 sets of 4: Wildcard, Free Hit, Triple Captain, Bench Boost; GW1-19 and GW20-38).
  - Assistant Manager removed for 2026/27.
  - 5 bankable free transfers max (`max_extra_free_transfers: 4`).
  - Scoring: Goals, Assists, Clean Sheets, Appearance, DefCon (+2 for DEF/MID/FWD), BPS rebalanced.
  - Config files: `config/rules.yaml`, `config/scoring.yaml`, `config/news_sources.yaml`, `config/settings.yaml`.

### Milestone 2: API Client, Ingestion & Defensive Data Store
- [ ] Async FPL Client (`src/fpl_oracle/api/fpl_client.py`) with `httpx`:
  - Rate limiting, concurrency semaphore, backoff/retry.
  - Endpoints: `bootstrap-static`, `fixtures`, `element-summary/{id}`, `event/{gw}/live`, `entry/{id}`, `entry/{id}/history`, `entry/{id}/event/{gw}/picks`, `entry/{id}/transfers`, `leagues-classic/{id}/standings`, `event-status`, `dream-team/{gw}`, `team/set-piece-notes`.
- [ ] Caching & snapshot persistence (`src/fpl_oracle/api/cache.py`, `src/fpl_oracle/data/store.py`):
  - SQLite database via SQLAlchemy (`data/fpl_oracle.db`).
  - Raw JSON snapshot storage for offline replay and model retraining.
  - Defensive Pydantic schemas (`extra="allow"`).

### Milestone 3: Historical Dataset Ingestion & Feature Engineering
- [ ] Historical downloader (`src/fpl_oracle/data/historical.py`):
  - Download past seasons (2022-23, 2023-24, 2024-25, 2025-26) from `vaastav/Fantasy-Premier-League`.
  - Ingest 2026-27 live completed gameweeks (GW1-5) into normalized match-level schema.
- [ ] Strict Pre-Deadline Feature Engineering (`src/fpl_oracle/data/features.py`):
  - Zero data leakage: features computed strictly using information prior to kickoff/deadline.
  - Rolling form (last 3, 5, 8 GWs minutes-weighted).
  - Per-90 xG, xA, xGI, xGC, ICT index, BPS tendencies, DefCon action rates.
  - Team strength model (Elo / Dixon-Coles attack/defense ratings with home/away split).
  - Opponent difficulty & fixture congestion.
  - Manager rotation tendencies & player minutes distribution.
  - Double and blank gameweek detection.
  - Cold-start priors for new transfers/promoted players based on price, position, team strength.

### Milestone 4: ML Projection Engine & Component Decomposition
- [ ] Component models (`src/fpl_oracle/ml/components/`):
  - `start_prob.py`: Calibrated classifier (starts, 60+ min, <60 min, 0 min).
  - `attacking.py`: Expected goals (xG), expected assists (xA) per 90.
  - `defending.py`: Clean sheet probability (team-level, player-level given minutes), goals conceded distribution.
  - `defcon.py`: Expected defensive contribution actions and probability of achieving DefCon +2 pts.
  - `bonus.py`: 2026/27 rebalanced BPS model trained with rule-shift feature.
  - `cards_saves.py`: Yellow/red cards, goalkeeper saves per 90.
- [ ] Aggregation & Uncertainty (`src/fpl_oracle/ml/predict.py`):
  - Combine components using 2026/27 scoring rules.
  - Quantile / bootstrap estimation: P10, P50 (median), P90, variance.
- [ ] Training & Evaluation Harness (`src/fpl_oracle/ml/train.py`, `eval.py`):
  - Time-aware expanding window cross-validation.
  - Robust baseline comparison (weighted form × fixture difficulty).
  - Generate evaluation report (`reports/model_eval.md`).

### Milestone 5: Mathematical Transfer Optimizer (MILP)
- [ ] Squad & Transfer MILP (`src/fpl_oracle/optimise/transfers.py`, `squad.py`):
  - PuLP integer linear programming over configurable multi-GW horizon (default 5-8 GWs).
  - Maximizes discounted expected points $\sum_t \gamma^t \text{xP}_t$ minus transfer hits.
  - Full FPL constraints: 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD), max 3 per club.
  - Accurate budget accounting: purchase price, selling price with 50% rise rule.
  - Banked transfers (up to 5 free transfers, 4 pts per extra).
  - Valid formations: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD (11 starters, 4 bench).
  - Candidate plans evaluation: roll transfer, 1 transfer, 2 transfers (-4 hit vs gain), multi-hit options.
  - What-if constraints (force keep, force sell, team exclusion, budget cap).
  - Transfer Roadmap output (firm vs. news-contingent).
- [ ] Price Change Predictor (`src/fpl_oracle/optimise/price_change.py`):
  - Momentum & net transfer velocity heuristics; urgency flag ("buy before tonight's rise").

### Milestone 6: Chip Strategy Engine (Critical 2026/27 Rules)
- [ ] Calendar & Blank/Double GW Detection (`src/fpl_oracle/chips/calendar.py`):
  - Live detection of fixture reschedules, blanks, doubles across all 38 GWs.
- [ ] Joint Chip Optimizer (`src/fpl_oracle/chips/planner.py`, `simulate.py`):
  - Set 1 (GW1-19) deadline hard constraint (Jan 2, 2027); Set 2 (GW20-38).
  - Search joint space of chips (Wildcard, Free Hit, Triple Captain, Bench Boost) using dynamic programming / beam search.
  - Account for chip synergies: Wildcard setup into Bench Boost, Free Hit on major blank GWs, Triple Captain on premier double GWs.
  - Chip Plan table: Recommended GW, expected gain, confidence, alternative GW, trigger conditions, opportunity cost of unused Set 1 chips.

### Milestone 7: Lineup & Captaincy Optimizer
- [ ] Lineup & Captain Selection (`src/fpl_oracle/optimise/lineup.py`):
  - Starting XI vs Bench order with autosub simulation.
  - Captain & Vice-captain ranking with P10/P50/P90, ceiling vs floor, safe vs high-differential options.
  - Vice-captain kickoff-time awareness.

### Milestone 8: Mini-League Intelligence & Win Probability
- [ ] Standings & Rival Analyzer (`src/fpl_oracle/league/standings.py`, `rivals.py`):
  - Full standings pagination, squad extraction, chip tracker, bank/value, effective ownership (EO).
  - Template vs differential classification.
- [ ] Monte Carlo Win-Probability Simulation (`src/fpl_oracle/league/montecarlo.py`):
  - Simulate remainder of season with projection distributions and rival transfer heuristics.
  - Compute $P(\text{finish 1st})$ and $\Delta P(\text{finish 1st})$ per strategic option.
- [ ] Strategy Mode (`src/fpl_oracle/league/strategy.py`):
  - Chasing mode (differential captain, high-variance picks, counter-timing).
  - Leading mode (template protection, low-variance picks, mirroring rival threats).

### Milestone 9: News Signals & LLM Integration
- [ ] News Ingestion & Processing (`src/fpl_oracle/news/ingest.py`, `extract.py`, `analyse.py`):
  - Official FPL news/chance_of_playing, verified RSS feeds (BBC, Sky, Guardian, PL).
  - Structured signal extraction (injury status, expected return, minutes probability, quotes).
  - Source quality weighting; strictly no hallucinated news.
- [ ] Provider-Agnostic LLM Layer (`src/fpl_oracle/llm/provider.py`, `gemini.py`, `anthropic.py`, `openai.py`):
  - Default Gemini via `GEMINI_API_KEY`, swappable Anthropic/OpenAI.
  - Function / tool calling layer (`src/fpl_oracle/llm/tools.py`).
  - Conversational Agent (`src/fpl_oracle/llm/agent.py`) with memory and graceful fallback if no key is provided.

### Milestone 10: Weekly Workflow Automation & Briefing
- [ ] Gameweek Briefing (`src/fpl_oracle/briefing/weekly.py`, `review.py`):
  - One-click briefing markdown & UI card (deadline countdown, news alerts, transfers, captain, lineup, chip status, rivals).
  - Post-gameweek review and accuracy tracker.
- [ ] Notification hooks (`src/fpl_oracle/briefing/notify.py`):
  - Optional Discord/Telegram webhook support.

### Milestone 11: Web Application & CLI
- [ ] FastAPI Backend (`src/fpl_oracle/server/main.py`, `routes/`, `jobs.py`):
  - REST endpoints for squad, projections, optimization, chips, league, chat, briefing.
  - APScheduler for periodic background data refresh and model retraining.
- [ ] Single-Page Web Frontend (`web/`):
  - Dark mode responsive UI with interactive pitch view, next 5 fixtures, projected points, chip timeline, mini-league intelligence, transfer optimizer with what-if controls, and chat assistant.
- [ ] CLI (`src/fpl_oracle/cli.py`):
  - Typer/Rich CLI supporting `fpl analyze`, `fpl squad`, `fpl optimize`, `fpl chips`, `fpl chat`, `fpl briefing`.

### Milestone 12: Backtesting, Verification & Documentation
- [ ] Backtest Harness (`tests/backtest.py`, `reports/backtest.md`):
  - Replay past seasons to evaluate projection accuracy, captaincy uplift, optimizer value vs naive baseline.
- [ ] Automated Verification (`make verify` / `python run.py verify`):
  - Live API smoke test, schema validation, unit tests, miniature backtest.
- [ ] Complete Documentation:
  - `README.md`, `SETUP.md`, `DECISIONS.md`, `reports/model_eval.md`, `reports/backtest.md`, `reports/chat_examples.md`.
