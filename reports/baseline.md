# FPL Oracle — Baseline Verification Report (Phase 0)

**Date**: 2026-10-03  
**Auditor**: Autonomous Principal Engineer & Data Scientist  
**Environment**: Windows 11, PowerShell, Python 3.11.1 (`.venv\Scripts\python.exe`)  
**Repository**: `https://github.com/AnshX01/fpl-oracle.git`  
**Git Commit**: `a3b9b9b`  

---

## 1. Executive Summary & Verification Matrix

All baseline tests were executed directly in the runtime environment. Zero synthetic assertions were trusted.

| Verification Target | Command Executed | Result | Details |
|---|---|:---:|---|
| **Security & Secrets Audit** | `python scripts/check_secrets.py` | **PASS** | 0 secrets in working tree or git history; `.env`, `*.db`, `profile.json` gitignored |
| **Pytest Unit Test Suite** | `python -m pytest tests -v` | **PASS** (14/14) | API parsing, chip rules, selling price math, fuzzy matching, and optimizer constraints passed |
| **Full Automated Verification** | `python run.py verify` | **PASS** (5/5) | Live FPL API, SQLite cache, 6 ML models, 14 unit tests, and mini backtest passed |
| **FastAPI End-to-End Server Suite** | `python scripts/test_server_live.py` | **PASS** (12/12) | All 12 endpoints returned 200 OK with `SafeJSONResponse` serialization |
| **Grounded Expert Chat** | `python scripts/test_chat.py` | **PASS** (10/10) | 10 sample queries evaluated with live tool calling and factual grounding saved to `chat_examples.md` |
| **Manual Squad Flow End-to-End** | `python scratch/test_manual_squad_e2e.py` | **PASS** | Paste text -> fuzzy match -> `/api/squad/manual` -> `/api/squad`, `/api/optimize`, `/api/chips` verified |

---

## 2. Security Audit Breakdown

1. **Git Commit History Scan**:
   - Analyzed commit `a3b9b9b` across all diffs for regex matches of Google Gemini (`AIza...`), OpenAI (`sk-...`), Anthropic (`sk-ant-...`), GitHub PATs (`ghp_...`), Telegram tokens, and private keys.
   - **Result**: Zero secret credentials found. Only template variable names (`GEMINI_API_KEY=`, `ANTHROPIC_API_KEY=`) and docstrings are present.
2. **Git Tracking Status**:
   - `data/profile.json` untracked from git index and added to `.gitignore`.
   - `data/profile.example.json` created as a template for new users.
   - `.env`, `*.db`, `*.sqlite`, `logs/`, `*.egg-info/`, `build/`, `dist/` verified in `.gitignore`.
3. **Automated Enforcement**:
   - Created `scripts/check_secrets.py` to prevent any future commits of secrets or sensitive tracked files.

---

## 3. Detailed Baseline Test Logs

### 3.1 Pytest Unit Tests (`python -m pytest tests -v`)
- `tests/test_api_parsing.py::test_team_parsing_with_null_strength` -> **PASSED**
- `tests/test_api_parsing.py::test_element_parsing_with_2026_27_fields` -> **PASSED**
- `tests/test_api_parsing.py::test_bootstrap_static_defensive_allow_extra` -> **PASSED**
- `tests/test_chips.py::test_chip_rules_config_2026_27` -> **PASSED**
- `tests/test_chips.py::test_remaining_chips_set_boundaries` -> **PASSED**
- `tests/test_chips.py::test_joint_chip_assignment_no_conflicts` -> **PASSED**
- `tests/test_chips.py::test_chip_set_1_expiry_opportunity_cost` -> **PASSED**
- `tests/test_fuzzy_and_manual_squad.py::test_clean_query_and_accents` -> **PASSED**
- `tests/test_fuzzy_and_manual_squad.py::test_extract_names_json_and_csv` -> **PASSED**
- `tests/test_fuzzy_and_manual_squad.py::test_manual_squad_endpoints_and_persistence` -> **PASSED**
- `tests/test_optimizer.py::test_squad_optimizer_constraints` -> **PASSED**
- `tests/test_optimizer.py::test_selling_price_math` -> **PASSED**
- `tests/test_optimizer.py::test_lineup_and_captain_formation` -> **PASSED**
- `tests/test_optimizer.py::test_compute_squad_selling_prices_and_free_transfers` -> **PASSED**
- **Outcome**: 14 passed in 27.24s.

### 3.2 FastAPI Server Live Endpoints (`scripts/test_server_live.py`)
- `GET /api/health` -> 200 OK (Season 2026/27, Current GW 5, Next GW 6)
- `GET /api/profile` -> 200 OK (Manager ID: None, Risk: balanced)
- `POST /api/profile` -> 200 OK
- `GET /api/squad` -> 200 OK (15 squad players: 11 starters, 4 bench; 4-5-1 formation; Captain: Gibbs-White)
- `GET /api/projections?horizon=3` -> 200 OK (50 players returned)
- `POST /api/optimize` -> 200 OK (Recommended Plan: 2_TRANSFERS, Hit Verdict: No hit recommended)
- `GET /api/chips` -> 200 OK (8 chip plan entries; Set 1 expiry notice generated)
- `GET /api/league` (unconfigured) -> 200 OK (`status: unconfigured`)
- `GET /api/league?league_id=314` -> 200 OK (Overall League, 100 teams)
- `GET /api/briefing` -> 200 OK (Target GW 6, 2926 characters of structured markdown)
- `GET /api/price-changes` -> 200 OK (10 imminent rises, 10 imminent falls)
- `POST /api/chat` -> 200 OK (Grounded response with live tool execution)
- `GET /` -> 200 OK (HTML UI shell served)

---

## 4. Identified Areas for Phase 1–4 Upgrades

1. **Endpoint Degraded State & Fault Injection (Phase 1)**:
   - Need comprehensive tests simulating FPL API 429/5xx, timeouts, empty league, missing profile, and schema drift using a local fault-injecting transport.
   - Global exception handler returning standard JSON error payload: `{error, code, hint, fallback_used}`.
2. **Structured Logging (Phase 1)**:
   - Route logs to `logs/fpl_oracle.log` with size-based rotation.
3. **Formal Leakage Test (Phase 2)**:
   - Test verifying that perturbing post-deadline match records does not alter pre-deadline features or predictions.
4. **Chip Planning Performance Optimization (Phase 2 & 4)**:
   - The beam search in `chip_planner` calls full MILP solves across all gameweeks, taking ~20-30s per invocation. Caching baseline squad utilities and restricting full MILP solves to candidate high-upside weeks (DGW, BGW, Wildcard windows) will reduce latency from ~25s to <2s.
5. **UI Modernization (Phase 5)**:
   - Transition from single-file HTML/Alpine to a high-end React + Vite + TypeScript + Tailwind + Framer Motion architecture with real-time SSE progress.
