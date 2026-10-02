# FPL Oracle — Master Project Handoff & Progress

**Status:** In Active Execution (Phased Overhaul)  
**Current Phase:** Phase 3 (Live "My Money" Flow & Background Analysis Job)  
**Last Updated:** 2026-10-03 01:21 UTC  
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
| **Phase 3** | Live "My Money" Flow & Background Analysis Job (SSE progress, overrides, bank sync) | **IN PROGRESS** | End-to-end 1-click sync & override verification |
| **Phase 4** | Expert Behaviour & Contingency Engine (Plan B/C, panic button, pre-deadline checklist, news citations) | **PENDING** | Grounded chat test pass, panic button scenario pass |
| **Phase 5** | UI Rebuild (React + Vite + TypeScript + Tailwind + Framer Motion, sleek editorial design) | **PENDING** | Lighthouse perf/a11y ≥ 90, Playwright screenshots |
| **Phase 6** | Scheduling & Automation (APScheduler, live GW polling, auto-retrain rollback) | **PENDING** | Scheduler jobs verified, idempotency checked |
| **Phase 7** | Tests & CI (Complete test suite, GitHub Actions workflow with offline fixtures) | **PENDING** | Full pytest green, offline fixture tests |
| **Phase 8** | Docs & Final Acceptance Scenario | **PENDING** | All acceptance tests, reports generated, secrets review |

---

## 3. Milestone State Tracking

### Done
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
  - 100% live server test suite (`scripts/test_server_live.py`) passing across all 13 endpoints and fault injections.
  - `mypy` passing with 0 errors across 51 source files; `ruff` passing with 0 errors.
- **Phase 2 (Data, ML & Optimizer Integrity)**:
  - Live Rules Verification Engine (`src/fpl_oracle/api/rules_checker.py`) verifying 2026/27 official rules against live API and exposed in `/api/health` (tested in `tests/test_rules_checker.py`, 4/4 passing).
  - Strict Leakage Audit test suite (`tests/test_leakage.py`) proving zero lookahead or contemporaneous data leakage (3/3 passing).
  - Hypothesis Property-Based Testing (`tests/test_optimizer_properties.py`) verifying budget limits, 15-player structure, club limits, lineup formations, and monotonic selling prices across randomized cases (3/3 passing).
  - Uncertainty Calibration and Rolling-Origin Time-Series Cross-Validation implemented in `src/fpl_oracle/ml/eval.py` and documented in `reports/model_eval.md`.
  - All 29 unit and property tests passing in `pytest`!

### In Progress
- **Phase 3 (Live "My Money" Flow & Background Analysis Job)**:
  - SSE background progress stream `/api/sync/stream` reporting staged pipeline steps.
  - Immediate user overrides and selling price math sync.

### Next
- **Phase 4 (Expert Behaviour & Contingency Engine)**:
  - Plan B/C contingency strategies.
  - Panic button for 1-click last-minute lineup repairs.
  - Pre-deadline checklist and verified news citations.

### Known Issues & Technical Debt
1. PuLP deprecation warnings regarding `LpVariable` directly in v4 and `PULP_CBC_CMD` (purely informational deprecation warnings from upstream PuLP 3.x library).
