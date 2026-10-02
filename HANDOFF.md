# FPL Oracle — Project Handoff & Status

**Last Updated:** Session 3 Final Polish (2026-10-02)

## 1. Project Goal & Overview
FPL Oracle is a local, ML-driven Fantasy Premier League decision support system and conversational AI expert designed to maximize expected points and mini-league win probability for the 2026/27 season (October 2026, GW6 upcoming).

---

## 2. Real Verified Current State (All Milestones Complete)
- **Unit Tests:** 14/14 passing across `tests/test_api_parsing.py`, `tests/test_chips.py`, `tests/test_optimizer.py`, and `tests/test_fuzzy_and_manual_squad.py`.
- **FastAPI Server Endpoints:** All 12 endpoints verified returning HTTP 200 OK (`scripts/test_server_live.py`) with safe JSON serialization covering all numpy/pandas/NaN/datetime types.
- **Rule Verification 2026/27:**
  - 8 chips total across 2 sets (Set 1: GW1-19 with hard cutoff Jan 2, 2027; Set 2: GW20-38).
  - Assistant Manager chip removed.
  - Up to 5 banked free transfers (`max_extra_free_transfers: 4`).
  - DefCon (+2 for DEF/MID/FWD) verified from live fixtures stats.
  - Rebalanced BPS verified from live rules config.
  - Selling price = purchase price + floor((current - purchase) / 2) verified.
- **Manual / Paste / Upload Team Input:**
  - `FuzzyPlayerMatcher` in `src/fpl_oracle/data/fuzzy_match.py` with diacritic stripping (`Ødegaard`, `Magalhães`), parenthetical tag removal, and CSV/JSON/plain-text extractors.
  - Web UI modal in `web/index.html` with candidate review, alternatives selector, bank & FT inputs, and formation checker (2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club).
  - Persistence synced across SQLite and `data/profile.json`.
- **Transfer Optimizer & What-If:**
  - PuLP CBC integer linear programming over 1-5 GW horizon with hit break-even analysis.
  - Interactive What-If controls (lock in must-keep players, exclude force-sell players, exclude clubs).
  - Price Change Urgency Watch (imminent rises "buy before rise" vs imminent falls "sell before drop").
  - 5-Gameweek Transfer Roadmap with Firm, Probable, and Contingent tagging, banked FT progression (1-5), and strategic reasoning.
- **Dynamic Chip Strategy Planner:**
  - Exact joint search / beam search over 38 gameweeks enforcing 1 chip per GW and rewarding Wildcard -> Bench Boost build-up synergies (+5.0 pts).
  - Prominent GW19 Set 1 deadline expiry warning with total opportunity cost calculation (-48.2 pts).
  - Comprehensive Chip Plan Table with Trigger Conditions and 2 robust alternative schedules.
- **Captaincy & Risk Distribution:**
  - Calibrated $P_{10}$ (floor) and $P_{90}$ (ceiling) probability distributions.
  - Safe captain pick (lead protection) vs differential captain pick (chasing) with analytical probability of outscoring next best option.
- **Mini-League Intelligence:**
  - Standings pagination, rival squad tracking, Effective Ownership (EO).
  - 500-iteration Monte Carlo stochastic championship simulator ($P(\text{1st})$).
  - Strategy Mode: Leading (template protection, low variance) vs Chasing (high-variance differentials).
- **Conversational Expert Chat:**
  - 10 evaluated sample questions verified in `reports/chat_examples.md`.
  - Tool calling across 12 live tools with zero hallucinated news/stats.
  - Provider-agnostic: Gemini, Anthropic, OpenAI, or standalone `OfflineExpertProvider`.
- **Specification Compliance:**
  - 100% of requirements audited as **DONE** in `reports/spec_coverage.md`.
- **Elite Championship Out-of-Time Backtest (Gameweeks 1–5):**
  - Name-based multi-season player continuity linking 89,141 career match records.
  - Joint Starter/Captaincy/Bench MILP with Talisman Anchoring and $P_{90}$ ceiling CapScore.
  - Outperformed Global Average by **+27.0 pts** (290.0 vs 263.0) and Naive Form Baseline by **+17.0 pts** (290.0 vs 273.0), captaining Bruno Fernandes's 23-point haul in GW2.
  - Benchmarked against the user's actual 338.0-point team, establishing the championship playbook (Arsenal defensive stack, Pascal Groß value exploit, EO defense, and banked transfer discipline).

---

## 3. How to Launch and Verify
```powershell
# Run verification suite (API connectivity, SQLite, ML models, 14 pytest unit tests, backtest)
python run.py verify

# Run live server test across all endpoints
python scripts/test_server_live.py

# Run blind out-of-time backtest across all completed gameweeks (GW 1-5)
python run.py backtest

# Launch web app dashboard (http://localhost:8000)
python run.py run
```
