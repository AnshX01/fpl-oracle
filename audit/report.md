# FPL Oracle: Exhaustive Adversarial Audit Report

**Date of Audit:** October 2026  
**Auditor:** Adversarial Senior Reviewer (Antigravity Agentic Audit)  
**Target Codebase:** `fpl-oracle` (Python 3.14.2 on Windows 11)  
**Scope:** 425 files, 1,139 functions, 461 unit tests, 90,052 historical records  
**Audit Directory:** [`audit/`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/)

---

## 1. Executive Summary

`fpl-oracle` presents an impressive, mathematically rigorous core for deterministic FPL rules, historical scoring parity (100% match across 90,052 records), and PuLP MILP squad constraints. However, **it is currently unsafe for real-money or competitive mini-league deployment** due to three critical defects: (1) live Google Gemini API keys are written in plaintext to persistent log files via unmasked `httpx` request URLs; (2) the mini-league Monte Carlo engine contains an asymmetric tie-breaking defect that awards 100% win probability to the user whenever squads are tied; and (3) the news reconciliation pipeline is vulnerable to fabricated RSS quotes and prompt injection, allowing untrusted sources to zero out player availability. Furthermore, the transfer optimizer cannot evaluate moves of more than 2 players at once despite official rules permitting up to 5 banked free transfers, and chips nearing the Gameweek 19 deadline are forfeited for 0 points due to rigid static gain gating. A systematic remediation plan of 4 phases is required before this system can be trusted competitively.

---

## 2. Top 10 Issues Table

| # | ID | Title | Severity | Area | One-Line Summary |
|---|---|---|---|---|---|
| 1 | **F-001** | Live Plaintext API Key Leak in Logs | **CRITICAL** | 3.10 Security | `httpx` logs Gemini API key in persistent disk logs across thousands of lines. |
| 2 | **F-002** | Monte Carlo 100% Win Rate on Tied Squads | **CRITICAL** | 3.5 Simulation | Strict `>` check gives user 1st place in 100% of trials against identical template rivals. |
| 3 | **F-003** | Gated-Active Prompt Injection / RSS Fabrication | **CRITICAL** | 3.8 News | Fabricated quotes pass substring check and set `effective_chance_of_playing = 0.0`. |
| 4 | **F-004** | Truncation Collapse in `expected_floor_div` | **HIGH** | 3.1 Scoring | Hardcoded `max_k = 20` causes Poisson expectations to collapse toward 0 for $\lambda > 10$. |
| 5 | **F-005** | Beam Search Incapable of >2 Transfers | **HIGH** | 3.4 Optimisation | Cannot evaluate 3–5 transfer moves in a single gameweek despite 5 banked FTs. |
| 6 | **F-006** | False Claims of Rollback on Corrupt Bundle | **HIGH** | 3.3 Model Quality | SHA256 mismatch crashes with unhandled `ValueError` rather than restoring version. |
| 7 | **F-007** | Expiring Chips Forfeited at GW19 Deadline | **HIGH** | 3.4 Optimisation | Static gain thresholds cause unplayed Set 1 chips to expire unused for 0 points. |
| 8 | **F-008** | In-Process Lock Only (`BUNDLE_LOCK`) | **HIGH** | 3.9 Architecture | `threading.RLock()` offers zero concurrency protection across multi-process workers. |
| 9 | **F-009** | Arbitrary 5x Weights Reduce Sample Size by 26k | **MEDIUM** | 3.3 Model Quality | Step multiplier drops Kish effective sample size from 90,303 to 63,548. |
| 10 | **F-010** | Entity Resolution Collision on Shared Surnames | **MEDIUM** | 3.8 News | Surnames (e.g. "Timber") resolve to the first candidate with 1.0 confidence. |

---

## 3. What is Solid (Architectural Strengths)

- **100% Historical Scoring Parity:** Verified across all 90,052 active player records in `master_history.csv` across 4 seasons (2023/24 to 2026/27) with exactly 0 calculation mismatches against the official FPL API's `total_points`.
- **Mathematical Solver Formulations:** Mixed Integer Linear Programming (MILP) with PuLP and COIN-OR CBC cleanly encodes 15 squad slots, 11 starters, formation bounds, 3-per-club caps, integer budget tenths, and 50% profit selling halving.
- **Safe JSON Serialization:** Default `SafeJSONResponse` middleware recursively strips `NaN`, `+Infinity`, `-Infinity`, and NumPy/pandas types across all 22 FastAPI endpoints.
- **Forward Holdout Pre-Deadline Freezing:** Pre-deadline snapshots automatically freeze player predictions 15 minutes before the deadline and score them against finalized match results, providing a genuine forward tracking record.
- **Clean Scoring Rules Centralization:** Consolidated rule constants in [`src/fpl_oracle/rules/scoring.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/rules/scoring.py) correctly implement 6 points for goalkeeper goals (rejecting architectural doc errors claiming 10 points).
- **Clean Dependency Audit:** `pip-audit` reports **0 known CVEs** across all installed packages.

---

## 4. Full Findings Grouped by Severity

### Critical Findings
- **F-001: Live Plaintext Google Gemini API Key Leaked into Persistent Application Logs**  
  *Location:* `logs/fpl_oracle.log:16592-20062` | *Area:* 3.10 Security | *Status:* `REPRODUCED`  
  `httpx` logs request URLs at `INFO` level including the Gemini query string `?key=...`, persisting raw credentials to disk.
- **F-002: Monte Carlo Symmetric Tie-Breaking Defect (100% Win Probability on Tied Squads)**  
  *Location:* `src/fpl_oracle/league/montecarlo.py:414` | *Area:* 3.5 Simulation | *Status:* `REPRODUCED`  
  User trial rank uses `s > trial_user_pts`; tied rivals evaluate to `False`, awarding the user rank 1 on 100% of trials (`win_prob = 100.0%`).
- **F-003: Gated-Active Prompt Injection / RSS Fabrication Vulnerability**  
  *Location:* `src/fpl_oracle/news/reconcile.py:214-222` | *Area:* 3.8 News | *Status:* `REPRODUCED`  
  Fabricated quotes in RSS feeds pass substring containment and set `effective_chance_of_playing = 0.0` in Gated-Active mode without publisher allowlisting.

### High Severity Findings
- **F-004: Truncation Collapse in `expected_floor_div` for Event Rates > 10**  
  *Location:* `src/fpl_oracle/domain/scoring.py:95-107` | *Area:* 3.1 Scoring | *Status:* `REPRODUCED`  
  Hardcoded `max_k = 20` truncates Poisson probability mass for rates above 10, collapsing expected floor points toward 0.
- **F-005: Beam Search Incapable of >2 Transfers (Cannot Exploit Banked 3–5 FTs)**  
  *Location:* `src/fpl_oracle/optimise/sequential.py:183-220` | *Area:* 3.4 Optimisation | *Status:* `READ`  
  Search branches strictly on 0, 1, or 2 transfers, preventing discovery of hit-free 3-, 4-, or 5-transfer moves when FTs are banked.
- **F-006: False Claims of Automatic Model Rollback on Corrupt Bundle**  
  *Location:* `src/fpl_oracle/ml/predict.py:59-71` | *Area:* 3.3 Model Quality | *Status:* `REPRODUCED`  
  SHA256 checksum mismatches raise an unhandled `ValueError` crashing inference rather than recovering from versions directory.
- **F-007: Expiring Chips Forfeited at GW19 Deadline Due to Rigid Static Gain Gates**  
  *Location:* `src/fpl_oracle/chips/planner.py:356-358` | *Area:* 3.4 Optimisation | *Status:* `REPRODUCED`  
  Static gates (e.g. 4.0 for 3xC) evaluate to False at GW19 if expected gain is below gate (e.g. 3.8 xP), allowing chips to forfeit for 0 pts.
- **F-008: In-Process Thread Lock Only (`BUNDLE_LOCK`) Offers Zero Cross-Process Concurrency Protection**  
  *Location:* `src/fpl_oracle/ml/bundle_lock.py:6` | *Area:* 3.9 Architecture | *Status:* `READ`  
  Uses `threading.RLock()` which cannot synchronize multi-process Uvicorn workers or background scheduler containers.

### Medium Severity Findings
- **F-009: Arbitrary 5x Weights on 2026/27 Rows Reduce Effective Sample Size by ~26k Rows**  
  *Location:* `src/fpl_oracle/ml/components/bonus.py:26` | *Area:* 3.3 Model Quality | *Status:* `REPRODUCED`  
  Kish effective sample size drops from 90,303 to 63,548 with no validation tuning.
- **F-010: Fuzzy Match Entity Resolution Collision on Shared-Surname Players**  
  *Location:* `src/fpl_oracle/data/fuzzy_match.py:61-75` | *Area:* 3.8 News | *Status:* `REPRODUCED`  
  Surname queries (e.g. "Timber") award 1.0 confidence to the first matching candidate without club verification.
- **F-011: Starting Lineup Selection Completely Insensitive to Risk Modes**  
  *Location:* `src/fpl_oracle/optimise/lineup.py:67` | *Area:* 3.4 Optimisation | *Status:* `REPRODUCED`  
  Risk mode modulates captaincy only; starter XI decision variables strictly use expected points without variance penalties.
- **F-012: Architectural Claims of 'Dixon-Coles Model' Contradicted by Fixed Poisson Implementation**  
  *Location:* `DECISIONS.md:88`, `src/fpl_oracle/data/features.py:568` | *Area:* 3.3 Model Quality | *Status:* `READ`  
  Code implements fixed independent Poisson with 1.35 mean; rho correlation and time decay are absent.
- **F-013: Non-Deterministic Wall-Clock Timing Assertion in Test Suite**  
  *Location:* `tests/test_league_strategy_mc.py:370` | *Area:* 3.11 Tests | *Status:* `REPRODUCED`  
  `assert elapsed < 2.0` fails under coverage tracing or on loaded CI workers.

### Low & Informational Findings
- **F-014: 20 Ruff Linting Errors Across Core Modules** (`rules/scoring.py`, `optimise/squad.py`) — Unused variables and imports.
- **F-015: 7 Static Type Checking Errors in Source Code** (`mypy src`) — Incompatible returns and untyped cache variables.
- **F-016: Feature Count Discrepancy Between Documentation (57) and Schema (62)** — Documentation desync.

---

## 5. Area-by-Area Scorecard (1 to 5 Scale)

| # | Audit Area | Score | Justification & Verdict |
|---|---|:---:|---|
| **3.1** | **Scoring rules & points formula** | **4 / 5** | Historical scoring parity is flawless (0 mismatches across 90k rows). Centralized rules are clean. Deducted 1 point for Poisson truncation in `expected_floor_div` (F-004). |
| **3.2** | **Data leakage & train/serve parity** | **4 / 5** | Strict chronological `shift(1)` on rolling features; expanding temporal windows; feature schema locked by SHA256 hash. Deducted 1 point for 57 vs 62 feature count doc drift (F-016). |
| **3.3** | **Model quality & statistical validity** | **2 / 5** | Claims of "Dixon-Coles" are false (code uses fixed Poisson); arbitrary 5x sample weights destroy 26k effective samples; claims of automatic rollback on corrupt bundle are false (F-006, F-009, F-012). |
| **3.4** | **Optimisation (lineup, squad, transfers, chips)** | **3 / 5** | PuLP squad MILP is excellent. However, transfer beam search is hardcoded to $\le 2$ transfers (F-005), GW19 chips forfeit under rigid gates (F-007), and lineup solver ignores risk modes for starters (F-011). |
| **3.5** | **Simulation (mini-leagues, rivals, forward)** | **2 / 5** | Critical asymmetric tie-breaking flaw awards 100% win probability to the user on tied rosters (F-002), severely compromising competitive mini-league guidance. |
| **3.6** | **Calibration & uncertainty** | **3 / 5** | Pinball evaluation and quantile metrics are implemented, but intervals rely on pooled normal z-scores rather than true per-stratum conformalized quantiles. |
| **3.7** | **API integration & data pipeline** | **4 / 5** | Resilient async client with retry logic, memory/disk cache, and polite rate limiting. Cache TTLs are sensible (60s–3600s), though documentation falsely claims a 12-hour cache. |
| **3.8** | **News, injuries & reconciliation** | **2 / 5** | Vulnerable to fabricated quotes and prompt injection in Gated-Active mode (F-003); fuzzy entity resolution collides on shared surnames without club disambiguation (F-010). |
| **3.9** | **Architecture & runtime boundaries** | **3 / 5** | Clean modular separation and fast startup. Deducted 2 points for in-process `threading.RLock()` which offers zero protection across multi-process deployments (F-008). |
| **3.10** | **Security, privacy & side effects** | **1 / 5** | Critical failure: Plaintext Google Gemini API key logged to disk in application logs across thousands of lines via unmasked `httpx` query strings (F-001). |
| **3.11** | **Tests & verification strategy** | **4 / 5** | 461 tests covering rules, solver bounds, and API models. Deducted 1 point for 20 Ruff linting errors (F-014), 7 mypy errors (F-015), and 1 flaky wall-clock time assertion (F-013). |
| **3.12** | **Product behaviour scenarios** | **3 / 5** | Scenarios 1, 5, 6 passed cleanly. Scenarios 2, 4 failed (broken squad rolls transfer; GW19 chips forfeit); Scenarios 3, 7, 8 exhibited partial or decoupled behaviors. |
| **3.13** | **Documentation honesty & drift** | **2 / 5** | Numerous doc claims contradict the code: "Goalkeeper goals 10 pts" (code is 6), "57 features" (code is 62), "Dixon-Coles model" (code is fixed Poisson), "Automatic rollback" (absent). |

---

## 6. Audit Confidence Level & Verification Conditions

- **Current Audit Confidence Level:** **HIGH (95%)**
- **Evidence Base:**
  - 100% of tracked/untracked files reviewed (398 FULL, 27 SKIM on raw data fixtures, 0 NONE).
  - 100% of public AST functions reviewed (1,139 / 1,139 accounted for in function ledger).
  - 461 tests executed and validated in local virtual environment.
  - 8 product behavior scenarios executed against the live Python codebase.
  - 90,052 historical match rows audited for points parity.
  - Line and branch coverage calculated across all 89 source packages.
- **Conditions That Would Raise Confidence to 100%:**
  1. **Full Season Forward Tracking:** Evaluating live model recommendations across a complete 38-gameweek cycle against real-time commercial benchmarks (Fantasy Football Hub, FPL Review).
  2. **Multi-Process Concurrency Stress Testing:** Running high-volume distributed load tests with multiple Uvicorn workers and background retrain jobs on Linux containers with active database locks.

---

## 7. Overall Verdict

**VERDICT: NOT PRODUCTION READY (CONDITIONAL APPROVAL PENDING PHASE 1 & 2 FIXES)**

`fpl-oracle` has an enviable mathematical and domain scoring backbone, but it must **NOT** be deployed in its current state. The plaintext API key leak in application logs must be patched and rotated immediately, the Monte Carlo tie-breaking bug must be corrected to prevent distorted win odds, and the news injection vulnerability must be closed before feeding external RSS feeds into player availability models. Once Phases 1 and 2 of [`audit/fix_plan.md`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/fix_plan.md) are completed, the system will be technically robust, secure, and ready for competitive play.
