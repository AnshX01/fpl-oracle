# Commands Run - FPL Oracle Exhaustive Audit

This document logs every command executed during the audit, the environment, working directory, and the outcome/output summary, ensuring full reproducibility.

## Environment & System Information
- **OS**: Windows 11
- **Shell**: PowerShell
- **Python**: `.venv\Scripts\python.exe` (Python 3.14.2)
- **Base Repo**: `C:\Users\anshw\Documents\fpl-oracle`

---

## 1. Tool Setup & Audit Tooling Installation

### 1.1 Python and Environment Inspection
- **Command**:
  ```powershell
  python --version; where.exe python; pip list
  ```
- **Exit Code**: 0
- **Summary**: Identified system python 3.14 and virtual environment `.venv\Scripts\python.exe`.

### 1.2 Tool Availability Check in Virtualenv
- **Command**:
  ```powershell
  .venv\Scripts\python.exe -c "import sys; ... check pytest, pytest_cov, ruff, mypy, pip_audit, mutmut, flake8, pulp, highspy"
  ```
- **Exit Code**: 0
- **Summary**: `pytest`, `ruff`, and `pulp` available. `pytest-cov`, `mypy`, `pip-audit`, and `mutmut` missing from `.venv`.

### 1.3 Installation of Missing Audit Tools in `.venv`
- **Command**:
  ```powershell
  .venv\Scripts\pip.exe install pytest-cov mypy pip-audit
  ```
- **Exit Code**: 0
- **Summary**: Installed `pytest-cov` (7.1.0), `mypy` (2.4.0), `pip-audit` (2.10.1), `coverage` (7.16.2), and dependencies.

### 1.4 Installation of `mutmut` for Mutation Testing
- **Command**:
  ```powershell
  .venv\Scripts\pip.exe install mutmut
  ```
- **Exit Code**: 0
- **Summary**: Installed `mutmut` (3.8.0) and dependencies (`libcst`, `textual`).

---

## 2. Test Suite Execution

### 2.1 Full Test Suite Run (Baseline)
- **Command**:
  ```powershell
  .venv\Scripts\pytest.exe -q
  ```
- **Exit Code**: 0
- **Summary**: 461 passed in 509.65s (8m 29s). 0 failures, 0 skipped. Note that documentation claims "426 tests"; test count has grown to 461.

---

## 3. Static Analysis & Linters

### 3.1 Linter (`ruff check .`)
- **Command**:
  ```powershell
  .venv\Scripts\ruff.exe check .
  ```
- **Exit Code**: 1
- **Summary**: 20 lint errors found across codebase:
  - `src/fpl_oracle/domain/scoring.py:7:1`: I001 unsorted import block.
  - `src/fpl_oracle/llm/agent.py:53:17`: I001 unsorted import block.
  - `src/fpl_oracle/llm/tools.py:298:9`: F841 unused local variable `starters_clean`.
  - `src/fpl_oracle/llm/tools.py:299:9`: F841 unused local variable `bench_clean`.
  - `src/fpl_oracle/optimise/sequential.py:3:1`: I001 unsorted import block.
  - `src/fpl_oracle/rules/__init__.py:21:5`: F401 `PENALTY_MISSED_POINTS` imported but unused.
  - `tests/rules/test_scoring_contract.py`: multiple unused imports (`APPEARANCE_GE60`, `APPEARANCE_LT60`, `GOALS_CONCEDED_PER_MINUS1`, `MAX_CHIPS_PER_GW`), and unnecessary f-string.

### 3.2 Type Checking (`mypy src`)
- **Command**:
  ```powershell
  .venv\Scripts\mypy.exe src
  ```
- **Exit Code**: 1
- **Summary**: 7 errors across 4 files (checked 89 source files):
  - `src/fpl_oracle/optimise/squad.py:88, 90, 151, 161`: Value of type `Any | None` is not indexable.
  - `src/fpl_oracle/news/gemini_extractor.py:108`: Need type annotation for `_evidence_cache`.
  - `src/fpl_oracle/llm/tools.py:248`: Argument 1 to `float` has incompatible type `Any | None`.
  - `src/fpl_oracle/llm/agent.py:40`: Argument 1 to `set` of `ContextVar` has incompatible type `dict[Never, Never]`; expected `None`.

---

## 4. Security & Dependency Auditing

### 4.1 Dependency Vulnerability Audit (`pip-audit`)
- **Command**:
  ```powershell
  .venv\Scripts\pip-audit.exe
  ```
- **Exit Code**: 0
- **Summary**: No known vulnerabilities found. `fpl-oracle` skipped as local package not on PyPI.
---

## 5. Test Coverage & Branch Analysis

### 5.1 Pytest Coverage Run
- **Command**:
  ```powershell
  .venv\Scripts\pytest.exe --cov=src --cov-branch --cov-report=json:audit/coverage.json
  ```
- **Exit Code**: 1 (460 passed, 1 failed due to timing assertion `assert elapsed < 2.0` in `test_league_strategy_mc.py:370` under coverage overhead)
- **Summary**: Total statements: 11,306; missed: 2,785; line coverage: 75.36%; branch coverage: 73.0%. 54 files below 90% line coverage analyzed and recorded in `audit/scratch/coverage_analysis.json`.

---

## 6. Adversarial Verification & Reproduction Scripts

### 6.1 Historical Scoring Parity Verification
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/recompute_historical_points.py
  ```
- **Exit Code**: 0
- **Summary**: Recomputed deterministic points across 90,303 rows in `data/historical/master_history.csv`. 100.0% match rate across all 90,052 player rows (0 mismatches). 251 mismatches identified as 2024/25 trial manager rows.

### 6.2 Poisson Truncation & Probability Coherence Test
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/test_poisson_and_probability.py
  ```
- **Exit Code**: 0
- **Summary**: Proved truncation collapse in `expected_floor_div`: for rate=15, expectation is 6.33 vs 7.25 exact; for rate=20, 4.57 vs 9.75 exact; for rate=25, 1.63 vs 12.25 exact.

### 6.3 Monte Carlo Symmetric Tie-Breaking Bug Reproduction
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/test_symmetric_tie.py
  ```
- **Exit Code**: 0
- **Summary**: Proved that when user and rival hold identical squads, `simulate_league_ranks` evaluates `s > trial_user_pts` to False for all rivals, returning `user_win_probability_pct = 100.0%` instead of 50.0%.

### 6.4 Gated-Active Prompt Injection / RSS Fabrication Test
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/test_prompt_injection.py
  ```
- **Exit Code**: 0
- **Summary**: Proved that fabricated quotes in unverified RSS feeds pass substring containment and set `effective_chance_of_playing = 0.0`, benching healthy starters.

### 6.5 Corrupt Model Bundle Rollback Failure Test
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/test_corrupt_bundle.py
  ```
- **Exit Code**: 0
- **Summary**: Proved that corrupted model weights raise an unhandled `ValueError` rather than recovering from versions checkpoint directory.

### 6.6 Effective Sample Size Weighting Probe
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/check_effective_sample_size.py
  ```
- **Exit Code**: 0
- **Summary**: Proved that static 5x sample weight on 1,538 rows of 2026/27 reduces Kish effective sample size from 90,303 to 63,548 (-26,755 samples).

### 6.7 Entity Resolution Ambiguity Test
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/test_same_surname.py
  ```
- **Exit Code**: 0
- **Summary**: Proved that queries matching shared surnames award 1.0 confidence to the first matching candidate without checking team or position context.

---

## 7. Product Behaviour Scenarios (Section 3.12)

### 7.1 Scenario Runner Execution
- **Command**:
  ```powershell
  .venv\Scripts\python.exe -u audit/scratch/run_scenarios.py
  ```
- **Exit Code**: 0
- **Summary**: Successfully executed all 8 scenarios; results recorded in `audit/scratch/scenario_results.json` and documented in `audit/scenario_results.md`.

---

## 8. Ledger & Register Generation

### 8.1 Claims Register Generation
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/build_claims_register.py
  ```
- **Exit Code**: 0
- **Summary**: Generated `audit/claims_register.csv` with evaluated verdicts and evidence.

### 8.2 Coverage & Function Ledger Cross-Referencing
- **Command**:
  ```powershell
  .venv\Scripts\python.exe audit/scratch/annotate_ledgers.py
  ```
- **Exit Code**: 0
- **Summary**: Annotated `audit/coverage_ledger.csv` and `audit/function_ledger.csv` with finding IDs and pass numbers.
