# FPL Oracle — Master Project Handoff & Progress

**Status:** In Active Execution (Phased Overhaul)  
**Current Phase:** Phase 5 (UI Rebuild — React / Vite / Modern Pitch Dashboard)  
**Last Updated:** 2026-10-03 05:30 UTC  
**Repository:** https://github.com/AnshX01/fpl-oracle.git  
**Working Directory:** `C:\Users\anshw\Documents\fpl-expert`

---

## 1. Project Overview & Architecture
FPL Oracle is an autonomous, machine-learning-driven Fantasy Premier League decision engine and conversational AI expert built for the 2026/27 season. Running 100% locally on Windows / Python 3.11, it combines:
1. Decomposed ML components (Minutes, Attacking returns, Clean sheets, DefCon, Bonus, Cards/Saves) into calibrated $P_{10}/P_{50}/P_{90}$ distributions.
2. Exact mathematical optimization (PuLP MILP for squad, starting XI, captaincy, and multi-GW transfer trajectory with banked transfer and hit penalty logic).
3. Live FPL API integration with caching, fallback snapshots, and fuzzy matching for manual team inputs.
4. Dynamic chip scheduling (Set 1 GW1-19 expiry warning, Set 2 planning, beam search).
5. Mini-league Monte Carlo simulation and game-theoretic risk modes (Lead Protection vs Chasing).
6. Grounded conversational agent with zero hallucination guarantee.
7. Modern responsive UI dashboard with real-time SSE progress, Pitch View, Transfer Workbench, Chip Roadmap, and Contingency Engine.

---

## 2. Phase-by-Phase Roadmap & Status

| Phase | Description | Status | Verification Gate |
|---|---|:---:|---|
| **Phase 0** | Orient, Secure, Baseline & Security Audit | **DONE** | Clean secrets check, baseline report in `reports/baseline.md` |
| **Phase 1** | Correctness & Robustness (Fault injection, safe JSON, error boundaries, GameState) | **DONE** | All 13 endpoints pass live + 5 fault injection tests; ruff & mypy 0 errors |
| **Phase 2** | Data, ML & Optimizer Integrity (Leakage test, rolling backtest, live rules check, uncertainty, hypothesis) | **DONE** | Leakage test 3/3 pass, rules check 4/4 pass, hypothesis 3/3 pass, model_eval.md updated |
| **Phase 3** | Live "My Money" Flow & Background Analysis Job (SSE progress, overrides, bank sync) | **DONE** | End-to-end 1-click sync & override verification; 15 live endpoints pass 100% |
| **Phase 4** | Expert Behaviour & Contingency Engine (Plan B/C, panic button, pre-deadline checklist, news citations) | **DONE** | Grounded chat test pass (4/4), contingency test pass (4/4), all 18 endpoints 100% |
| **Phase 5** | UI Rebuild (Modern Pitch & Squad, Workbench, Chip Roadmap, Mini-League, Contingency) | **NEXT** | Complete editorial UI rebuild, responsive desktop/mobile |
| **Phase 6** | Scheduling & Automation (APScheduler, live GW polling, auto-retrain rollback) | **PENDING** | Scheduler jobs verified, idempotency checked |
| **Phase 7** | Tests & CI (Complete test suite, GitHub Actions workflow with offline fixtures) | **PENDING** | Full pytest green, offline fixture tests |
| **Phase 8** | Docs & Final Acceptance Scenario | **PENDING** | All acceptance tests, reports generated, secrets review |

---

## 3. Milestone State Tracking

### Done
- **Phase 4 (Expert Behaviour & Contingency Engine)**:
  - Plan B (Injury/Press Conference Pivot) and Plan C (Differential/Price Rise Pivot) precomputation engine in `src/fpl_oracle/optimise/contingency.py`.
  - Injury & Rotation Contingency Matrix (`compute_injury_matrix`) evaluating auto-sub outcome vs direct emergency transfer vs `TRUST_BENCH` vs `EXECUTE_TRANSFER` with wait-vs-commit tradeoff.
  - Panic Button crisis re-optimizer (`panic_button_reoptimize`) handling natural language queries or explicit player IDs, zeroing points/minutes, auto-promoting bench reserve, selecting revised captain, and finding best market replacement.
  - Pre-deadline checklist generator (`generate_pre_deadline_checklist`) assessing Starters Fitness, Vice-Captain Failsafe, Bench Order, Chip Set 1 GW19 Expiry, and Deadline Lock countdown.
  - Built `src/fpl_oracle/briefing/review.py` (`PostGameweekReviewer`) for post-deadline review and performance diagnostics.
  - Grounding and anti-hallucination test suite (`tests/test_chat_grounding.py`) passing 4/4 tests.
  - Contingency test suite (`tests/test_contingency.py`) passing 4/4 tests.
  - Server endpoints live verification (`scripts/test_server_live.py`) 100% passing across 18 endpoints and 5 fault injections!
- **Phase 0 (Orient, Secure, Baseline)**:
  - Full codebase review against `fpl_expert_antigravity_goal_prompt.md`.
  - Secrets and security audit (`scripts/check_secrets.py`) verified clean; .env and data files gitignored.
  - Baseline execution recorded in `reports/baseline.md`.
- **Phase 1 (Correctness & Robustness)**:
  - `GameweekPhase` state machine (`PRE_DEADLINE`, `LIVE`, `BONUS_PENDING`, `FINISHED`, `BETWEEN_GWS`) with countdown.
  - Persistent pooled `httpx.AsyncClient` with jittered backoff, in-flight coalescing, polite batching, and transport mock injection.
  - Structured rotating logger to `logs/fpl_oracle.log`.
  - Global error boundaries returning `{error, code, hint, fallback_used}` on all endpoints.
  - Comprehensive `/api/health` and `/api/game-state` endpoints with data freshness attributes (`data_as_of`, `stale`).
  - Hermetic fault-injection test suite (`tests/test_fault_injection.py`) passing 5/5 scenarios.
  - 100% live server test suite (`scripts/test_server_live.py`) passing across all endpoints and fault injections.
  - `mypy` passing with 0 errors across source files; `ruff` passing with 0 errors.
- **Phase 2 (Data, ML & Optimizer Integrity)**:
  - Live Rules Verification Engine (`src/fpl_oracle/api/rules_checker.py`) verifying 2026/27 official rules against live API and exposed in `/api/health` (tested in `tests/test_rules_checker.py`, 4/4 passing).
  - Strict Leakage Audit test suite (`tests/test_leakage.py`) proving zero lookahead or contemporaneous data leakage (3/3 passing).
  - Hypothesis Property-Based Testing (`tests/test_optimizer_properties.py`) verifying budget limits, 15-player structure, club limits, lineup formations, and monotonic selling prices across randomized cases (3/3 passing).
  - Uncertainty Calibration and Rolling-Origin Time-Series Cross-Validation implemented in `src/fpl_oracle/ml/eval.py` and documented in `reports/model_eval.md`.
  - All 29 unit and property tests passing in `pytest`!
- **Phase 3 (Live "My Money" Flow & Background Analysis Job)**:
  - Built `src/fpl_oracle/server/pipeline.py` (`SyncPipeline`): 9-stage orchestrated analysis pipeline with live SSE streaming (`GET /api/sync/stream`) and status polling (`GET /api/sync/status`). Decouples client disconnects from background job execution.
  - Implemented `calculate_banked_free_transfers` in `src/fpl_oracle/optimise/transfers.py` dynamically tracking 1 to 5 banked transfers across manager gameweek history and chip plays.
  - Updated `POST /api/profile` and `GET /api/profile` to support instant overrides for `bank`, `free_transfers`, `risk_preference`, `llm_provider`, `manager_id`, and `manual_squad`.
  - Enriched `GET /api/squad` to return `total_squad_value`, `total_selling_value`, `total_team_value`, `free_transfers`, `available_transfers`, and `bank_millions`.
  - All live server endpoints and background pipeline tests passing 100% with 0 errors and 0 warnings!

### In Progress
- **Phase 4 (Expert Behaviour & Contingency Engine)**:
  - Plan B & Plan C precomputation with explicit triggers and xP delta vs Plan A.
  - Injury / Rotation contingency matrix and wait-vs-commit threshold calculation.
  - "Panic Button" endpoint (`POST /api/contingency/panic`) for instant re-optimization under late team news.
  - Pre-deadline checklist generator and post-deadline review explanation.
  - Grounded chat verification test suite (`tests/test_chat_grounding.py`).

### Next
- **Phase 5 (UI Rebuild)**:
  - Sleek editorial React + Vite + TypeScript + Tailwind + Framer Motion dashboard (gated on Phases 1–4 passing).
- **Phases 6–8 (Automation, CI & Acceptance)**.

### Known Issues & Technical Debt
1. PuLP deprecation warnings regarding `LpVariable` directly in v4 and `PULP_CBC_CMD` (purely informational deprecation warnings from upstream PuLP 3.x library).
