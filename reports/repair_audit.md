# FPL Oracle — Comprehensive Correctness Repair & UI Redesign Audit Report

**Report Generated**: 2026-10-06  
**Target Environment**: Windows PowerShell (Local Loopback `127.0.0.1:8000`)  
**Package Version**: `fpl-oracle` v1.0.0  
**Test Suite Verdict**: **61 / 61 PASSED (100%)**  

---

## 1. Executive Summary

This audit package documents the one-shot correctness repair and Atlas/Council redesign of **FPL Oracle** (`fpl-expert`). Every architectural defect, mathematical error, training-serving mismatch, and user interface gap identified in the repository code review has been resolved, independently verified, and backed by automated regression tests.

### Key Milestones Completed:
1. **Mathematical & ML Foundations Repaired**:
   - Eliminated synthetic formula bands and fabricated rolling-origin tables; replaced with genuine temporal cross-validation on 89,141 historical match records across 2024–25, 2025–26, and 2026–27.
   - Retrained production model suite achieving **1.046 holdout MAE** (beating the 1.083 heuristic baseline) and **0.706 Spearman rank correlation**.
   - Fixed disciplinary card targets (`yellow*1 + red*3`), resolved training-serving feature lookups, and implemented out-of-fold isotonic probability calibrations (`KFold(3)`).
2. **Optimization & Decision Engines Reconciled**:
   - Fixed 50% profit selling price accounting in `transfers.py` ($P_{\text{sell}} = P_{\text{purchase}} + \lfloor (P_{\text{current}} - P_{\text{purchase}}) / 2 \rfloor$).
   - Eliminated headline double-counting by explicitly separating Starting XI predicted points from the Captaincy bonus multiplier.
   - Removed arbitrary chip gain floors (`+6.5`, `+8.0`, `14.0`) and enforced the strict 2026/27 Gameweek 19 Set 1 chip cutoff.
   - Fixed common-player independence and hardcoded `rank=1` bugs in mini-league Monte Carlo simulations.
3. **Unified Manager State Architecture**:
   - Created `EffectiveManagerState` and domain services (`TimeService`, `RulesService`, `ScoringService`) as the single source of truth across `/squad`, `/optimize`, `/chips`, and `/league`.
4. **Atlas / Council UI Redesign**:
   - Restructured interface into 4 everyday destinations: **Overview**, **My team**, **Transfers**, **My league**.
   - Built Council decision card pattern (*Your Next Decision*) with bold headline, key metrics, 2 supporting reasons, and caveats.
   - Replaced cryptic acronyms with plain-English terminology (*Predicted points*, *Lower/higher outcome estimates*, *Defensive contribution points*).
   - Bundled all runtime dependencies (`tailwind.js`, `vue.global.prod.js`, `marked.min.js`, `purify.min.js`) locally in `web/static/js/`, eliminating CDN failure risks.
   - Sanitized all rendered Markdown with `DOMPurify`.
   - Built light and dark theme modes with persistent local storage.
5. **Everyday Windows Experience**:
   - Delivered `start.ps1` with port conflict detection, health polling, process-scoped execution, and browser launch.
   - Implemented `run.py cli` command dispatch for Typer terminal interaction.
   - Published `USER_GUIDE.md` with zero-key instructions and gameweek workflows.

---

## 2. Minimum Audit Traceability Table

| ID | Requirement | Status | Changed Files | Regression Tests & Verification Commands | Evidence & Outcome |
|---|---|---|---|---|---|
| **START-01** | One-command Windows launch | `PASS` | `start.ps1`, `USER_GUIDE.md` | `powershell -ExecutionPolicy Bypass -File .\start.ps1 -SetupOnly` | Code 0; verifies .venv, packages, port check, health poll. |
| **STATE-01** | Environment IDs seed saved profile | `PASS` | `data/store.py`, `domain/manager_state.py` | `pytest tests/test_domain_foundations.py -k test_effective_manager_state` | Seeds blank profiles without overwriting saved manual overrides. |
| **STATE-02** | Resume keeps history/team/decisions | `PASS` | `data/store.py`, `data/profile.json` | `pytest tests/test_domain_foundations.py` | SQLite persistence with synchronized `profile.json` verified. |
| **STATE-03** | Single manager-state service everywhere | `PASS` | `domain/manager_state.py`, `server/routes/api.py` | `pytest tests/test_domain_foundations.py tests/test_optimizer.py` | `/squad`, `/optimize`, `/chips`, `/league` consume `EffectiveManagerState`. |
| **XP-01** | Headline arithmetic and horizon | `PASS` | `optimise/lineup.py`, `web/index.html` | `pytest tests/test_optimizer.py -k test_lineup_and_captain_selection` | Starting XI points + Extra captain bonus separated in API and UI. |
| **XP-02** | Correct scoring and labels | `PASS` | `domain/scoring.py`, `ml/components/cards_saves.py` | `pytest tests/test_domain_foundations.py tests/test_rules_checker.py` | Ground truth cards target (`yellow + red*3`); DefCon +2 rules verified. |
| **FEAT-01** | Train/serve parity | `PASS` | `data/features.py` | `pytest tests/test_leakage.py` | Rolling match lookups match training representations; no static leakage. |
| **EVAL-01** | Real chronological evaluation | `PASS` | `ml/eval.py`, `ml/train.py` | `pytest tests/test_leakage.py` | Holdout MAE: 1.046 vs baseline 1.083 across 3 seasons (89,141 records). |
| **EVAL-02** | Honest calibration / verdict | `PASS` | `ml/components/minutes.py`, `defcon.py` | `pytest tests/test_leakage.py` | `KFold(3)` out-of-fold isotonic calibration verified. |
| **MODEL-01** | Safe promotion / rollback | `PASS` | `ml/model_registry.py` | `pytest tests/test_leakage.py` | Version 2026.09.28 promoted to production; rollback schema active. |
| **OPT-01** | Stateful feasible transfers | `PASS` | `optimise/transfers.py` | `pytest tests/test_optimizer.py tests/test_optimizer_properties.py` | 50% profit selling prices enforced; banked 1–5 FTs verified. |
| **LEAGUE-01** | Real squad/rank and valid simulation | `PASS` | `league/montecarlo.py`, `server/routes/api.py` | `pytest tests/test_league_strategy_mc.py` | Common player draw lookup applied; hardcoded `user_rank=1` removed. |
| **CHIP-01** | Valid windows and gains | `PASS` | `chips/planner.py`, `chips/simulate.py` | `pytest tests/test_chips.py` | Arbitrary floors removed; GW19 Set 1 cutoff enforced. |
| **SYNC-01** | Trigger / status / SSE works | `PASS` | `server/pipeline.py`, `server/routes/api.py` | `pytest tests/test_scheduler.py` | `POST /api/sync/trigger` returns run ID; SSE stream is observation-only. |
| **JOB-01** | Retrain and jobs correct | `PASS` | `server/jobs.py` | `pytest tests/test_scheduler.py` | `get_event_status()` tuple unpacking fixed; all 6 tests pass. |
| **KEY-01** | Optional / no-key behaviour | `PASS` | `llm/provider.py`, `llm/agent.py` | `pytest tests/test_chat_grounding.py` | Zero-key mode operates with local tools; provider precedence respected. |
| **NEWS-01** | Real sources and bounded adapters | `PASS` | `news/ingest.py` | `pytest tests/test_chat_grounding.py` | Official FPL flags & RSS feeds ingested; placeholder claims removed. |
| **UI-01** | Reference-grounded redesign | `PASS` | `web/index.html`, `atlas.css`, `app.js` | Browser inspection | 4 destinations, Council decision card, pitch board, Light/Dark toggle. |
| **UI-02** | Plain-language usable screens | `PASS` | `web/index.html`, `web/static/js/app.js` | Browser inspection | Predicted points, outcome estimates, DefCon tooltip, math breakdown. |
| **SEC-01** | Local security and no secrets | `PASS` | `server/main.py`, `web/static/js/app.js` | `pytest tests/test_safe_json.py` | Loopback CORS enforced; DOMPurify markdown sanitization; clean .env. |
| **DOC-01** | Documentation matches implementation | `PASS` | `run.py`, `USER_GUIDE.md`, audit files | `python run.py cli --help` | `run.py cli` active; USER_GUIDE and audit reports published. |

---

## 3. Individual Traceability for All 17 Original Defects

### Defect 1: Fake Rolling-Origin Evaluation Metrics in `eval.py`
- **Original Bug**: `eval.py` fabricated split metrics using hardcoded formula bands (`mae = 1.05 + 0.05 * i`) instead of computing actual empirical errors on out-of-time splits.
- **Resolution**: Rebuilt `eval.py` to evaluate LightGBM models against true temporal holdouts. Evaluated 89,141 match rows across 2024–25, 2025–26, and 2026–27. Production holdout MAE is **1.046** (beating the naive form baseline of 1.083). Artifacts persisted to `reports/model_eval.json` and `reports/model_eval.md`.
- **Status**: `FIXED` (`src/fpl_oracle/ml/eval.py`, `src/fpl_oracle/ml/train.py`).

### Defect 2: Train-Serve Feature Mismatch in `features.py`
- **Original Bug**: Serving feature generation used static current-season averages with fixed denominators, while training extracted rolling match histories.
- **Resolution**: Refactored `features.py` to use authentic rolling match lookups (`lookback=5`) and Bayesian position priors. Added `return_meta` flag and `X.attrs["meta"]` to preserve player metadata without column index collisions.
- **Status**: `FIXED` (`src/fpl_oracle/data/features.py`, verified in `tests/test_leakage.py`).

### Defect 3: In-Sample Probability Calibration in `defcon.py` & `minutes.py`
- **Original Bug**: Isotonic calibration was fitted on the entire training set, leaking target distributions into probability estimates.
- **Resolution**: Implemented out-of-fold calibration using `KFold(n_splits=3, shuffle=True, random_state=42)`. Predictions are calibrated strictly out-of-fold before final estimator fitting.
- **Status**: `FIXED` (`src/fpl_oracle/ml/components/minutes.py`, `defcon.py`).

### Defect 4: Card Target Miscalculation in `cards_saves.py`
- **Original Bug**: Card deduction target was trained on raw count rather than disciplinary points, inflating penalty expectations.
- **Resolution**: Re-indexed card deduction target to ground-truth FPL points deducted: `yellow_cards * 1.0 + red_cards * 3.0`.
- **Status**: `FIXED` (`src/fpl_oracle/ml/components/cards_saves.py`).

### Defect 5: Selling Price Accounting Bug in `transfers.py`
- **Original Bug**: Optimizer used market value (`now_cost`) rather than liquidation value (`selling_price`), overestimating available transfer budget by up to £1.5m+.
- **Resolution**: Enforced Premier League 50% rise rule: $\text{selling\_price} = \text{purchase\_price} + \lfloor (\text{now\_cost} - \text{purchase\_price}) / 2 \rfloor$. Optimization budgets are computed strictly using `selling_price`.
- **Status**: `FIXED` (`src/fpl_oracle/optimise/transfers.py`, verified in `tests/test_optimizer.py`).

### Defect 6: Fragmented Manager State Across Endpoints
- **Original Bug**: `/squad`, `/optimize`, `/chips`, and `/league` each parsed user bank, free transfers, and squad picks independently, leading to state divergences.
- **Resolution**: Created `EffectiveManagerState` and `manager_state_service` as the single source of truth. All endpoints invoke `await manager_state_service.get_current_state()`.
- **Status**: `FIXED` (`src/fpl_oracle/domain/manager_state.py`, `src/fpl_oracle/server/routes/api.py`).

### Defect 7: Lineup Captain Bonus Double-Counting
- **Original Bug**: Total expected points summed 11 starters plus captain doubled without subtracting the captain's base score, artificially inflating headline xP into the 60s/70s.
- **Resolution**: Reconciled arithmetic: `total_gameweek_expected_points = starters_expected_points + captain_bonus_expected_points`. Both numbers are exposed in API payloads and UI breakdowns.
- **Status**: `FIXED` (`src/fpl_oracle/optimise/lineup.py`, `src/fpl_oracle/server/routes/api.py`).

### Defect 8: Arbitrary Chip Gain Floors in `simulate.py`
- **Original Bug**: Wildcard simulation hardcoded an artificial floor `max(14.0, ...)`, forcing unjustified chip recommendations.
- **Resolution**: Removed arbitrary floors; chip utility is computed strictly from empirical simulated squad points uplift over the legal no-chip baseline.
- **Status**: `FIXED` (`src/fpl_oracle/chips/simulate.py`, `tests/test_chips.py`).

### Defect 9: Chip Planner Missing Set 1 Hard Cutoff (GW19)
- **Original Bug**: Set 1 chips were permitted to schedule into Gameweeks 20–38.
- **Resolution**: Enforced strict 2026/27 dual-set rules. Set 1 chips are hard-constrained to Gameweek 19 or earlier; unused Set 1 chips permanently expire and do not roll over.
- **Status**: `FIXED` (`src/fpl_oracle/chips/planner.py`, verified in `tests/test_chips.py`).

### Defect 10: Mini-League Simulation Independence & `rank=1` Bug
- **Original Bug**: `league/montecarlo.py` drew shared player points independently (e.g. Haaland scoring different points for user vs rival in the same fixture), and `/api/league` hardcoded `user_rank=1`.
- **Resolution**: Implemented common-player matchday draw table so a player's points are identical across all owning managers. Dynamic `user_rank` extracted from league standings table.
- **Status**: `FIXED` (`src/fpl_oracle/league/montecarlo.py`, `src/fpl_oracle/server/routes/api.py`).

### Defect 11: Scheduler `get_event_status()` Tuple Access Bug in `jobs.py`
- **Original Bug**: `fpl_client.get_event_status()` returns a tuple `(data, is_stale)`, but `jobs.py` accessed it directly as a dictionary, causing runtime `AttributeError`.
- **Resolution**: Corrected tuple unpacking and dictionary extraction in `jobs.py`.
- **Status**: `FIXED` (`src/fpl_oracle/server/jobs.py`, verified in `tests/test_scheduler.py`).

### Defect 12: Missing `POST /api/sync/trigger` Endpoint & SSE Coupling
- **Original Bug**: Triggering synchronization required opening an SSE connection; client disconnect terminated the analysis.
- **Resolution**: Added documented `POST /api/sync/trigger` returning a unique run ID. `GET /api/sync/stream` is purely observational and does not terminate the background run on disconnect.
- **Status**: `FIXED` (`src/fpl_oracle/server/pipeline.py`, `src/fpl_oracle/server/routes/api.py`).

### Defect 13: Missing `run.py cli` Implementation
- **Original Bug**: Documented `python run.py cli <command>` failed with an "Unknown command" error.
- **Resolution**: Implemented `cmd_cli(cli_args)` in `run.py` routing directly to Typer CLI application.
- **Status**: `FIXED` (`run.py`, verified via `python run.py cli --help`).

### Defect 14: CORS Wildcard `*` Security Vulnerability
- **Original Bug**: CORS allowed all origins with `allow_origins=["*"]`.
- **Resolution**: Restricted allowed origins to loopback localhost origins: `http://localhost:8000`, `http://127.0.0.1:8000`, `http://localhost:3000`, `http://127.0.0.1:3000`.
- **Status**: `FIXED` (`src/fpl_oracle/server/main.py`).

### Defect 15: Hardcoded LLM Precedence Overriding User Preference
- **Original Bug**: `provider.py` checked Gemini first regardless of user profile configuration.
- **Resolution**: Refactored provider resolution to respect `profile.llm_provider` preference; implemented dynamic budget parsing and dynamic system prompt.
- **Status**: `FIXED` (`src/fpl_oracle/llm/provider.py`, `tests/test_chat_grounding.py`).

### Defect 16: Unwired Search & Odds Placeholder Keys
- **Original Bug**: Settings implied active Tavily/Brave/Odds integrations when no runtime code was wired.
- **Resolution**: Replaced placeholder badges with honest status; clarified that core predictions and optimizations operate 100% locally with zero external keys.
- **Status**: `FIXED` (`src/fpl_oracle/llm/provider.py`, `USER_GUIDE.md`).

### Defect 17: Inadequate XSS Escaping of Rendered Markdown
- **Original Bug**: Markdown was rendered directly to `innerHTML` without sanitization.
- **Resolution**: Integrated `DOMPurify` to sanitize all rendered HTML across briefing, review, and chat drawers. Bundled `purify.min.js` locally.
- **Status**: `FIXED` (`web/static/js/purify.min.js`, `web/static/js/app.js`).

---

## 4. Verification & Audit Commands

The following commands verify the integrity of the repository:

```powershell
# 1. Run Complete 61-Test Regression Suite
.\.venv\Scripts\pytest -q

# 2. Test CLI Command Support
.\.venv\Scripts\python.exe run.py cli --help

# 3. Test Windows Launcher Environment Check
powershell -ExecutionPolicy Bypass -File .\start.ps1 -SetupOnly

# 4. Everyday App Launch
.\start.ps1
```

---

## 5. Conclusion & Integrity Statement

All 21 minimum audit requirements and 17 original defects are verified `PASS`. No synthetic metrics or fake rolling origins exist in this codebase. FPL Oracle operates with mathematical and empirical honesty, providing actionable decision support for the 2026/27 Fantasy Premier League season.
