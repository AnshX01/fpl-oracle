# Section 3 Audit Checklists: Area-by-Area Answers & Evidence

This document provides explicit, evidence-backed answers to every question posed in Section 3 of the audit prompt.

---

## 3.1 Scoring rules and the points formula

- **Where is every points value defined? Are there duplicates or literals scattered in code?**
  - **Answer:** Centralized in [`src/fpl_oracle/rules/scoring.py`](file:///C:/Users/anshw/Documents/fpl-oracle/src/fpl_oracle/rules/scoring.py) and loaded via [`config/rules.yaml`](file:///C:/Users/anshw/Documents/fpl-oracle/config/rules.yaml).
  - **Evidence:** Evaluated via [`audit/scratch/check_scattered_points.py`](file:///C:/Users/anshw/Documents/fpl-oracle/audit/scratch/check_scattered_points.py). Found minor unused literal imports in `rules/__init__.py` (F-014), but no conflicting scattered points definitions.
- **Verify each against official FPL scoring table: appearance, goals by position, assists, clean sheets, goals conceded, saves, penalty saves/misses, own goals, cards, bonus, DefCon.**
  - **Answer:** Verified against official Premier League 2026/27 rules.
  - **Evidence:** `APPEARANCE_LT60=1`, `APPEARANCE_GE60=2`, `GOAL_POINTS={"GKP":6, "DEF":6, "MID":5, "FWD":4}`, `ASSIST_POINTS=3`, `CLEAN_SHEET_POINTS={"GKP":4, "DEF":4, "MID":1, "FWD":0}`, `DEFCON_POINTS=2`.
  - **Goalkeeper Goal Points Check:** Architecture doc claimed 10 points. Code correctly defines GKP goals as 6 points (`src/fpl_oracle/rules/scoring.py:24`). Verified against Premier League official handbook where GKP and DEF share 6 points. Finding F-012/Claims Register.
- **Recompute `total_points` for stored historical player-gameweeks from raw event columns.**
  - **Answer:** Recomputed across 90,303 rows in `data/historical/master_history.csv`.
  - **Evidence:** 100.0% match rate across all 90,052 active player records (0 mismatches). Exactly 251 mismatches found belonging exclusively to 2024/25 trial manager records with 0 minutes. Checked rule changes between seasons (DefCon added in 2026/27).
- **Is anything in the expected-points ensemble missing? Decomposition sum check.**
  - **Answer:** Decomposition sums additively to final xP across components (`appearance + goals + assists + clean_sheet + defcon + bonus - cards - concessions`).
  - **Evidence:** Tested with random vectors in `audit/scratch/test_poisson_and_probability.py`. Sums match within floating-point tolerance $10^{-7}$.
- **Check exact Poisson floor expectations numerically against brute-force summation.**
  - **Answer:** **FAILED for rates $\lambda > 10$.**
  - **Evidence:** Finding **F-004**. `expected_floor_div` truncates summation at hardcoded `max_k = 20`. For rate=20, computes 4.57 vs exact 9.75; collapses toward 0 for high rates.
- **Probability coherence: $P(\min \ge 60) > P(\text{play})$, leave $[0, 1]$, $E[\text{minutes}]$ contradict probabilities, negative xP.**
  - **Answer:** Checked, no issue. $P(\min \ge 60)$ is bounded by $P(\text{starts}) \le P(\text{play})$ via clipping in `features.py`. Probabilities stay in $[0, 1]$. Negative xP is allowed mathematically for disciplinary hazard players (e.g. $-2.0$ for cards/OG).
- **Double gameweeks and blank gameweeks: sum logic, quantile handling, 3 fixtures, postponements.**
  - **Answer:** BGW fixture count = 0 outputs 0.0 xP. DGW fixture sum adds per-fixture expectations. Quantiles in DGW flag `uncertainty_status="dgw_marginal_sum_unvalidated"`. Postponed fixtures default to event None.

---

## 3.2 Data leakage and train/serve parity

- **Trace every feature from raw data to model input: rolling statistics lag check.**
  - **Answer:** Checked, no issue.
  - **Evidence:** Traced in `src/fpl_oracle/data/features.py:350-410`. `df.groupby('element').shift(1)` is strictly applied sorted by `kickoff_time` and `round`. A match's outcome cannot enter its own features.
- **Team-level features: point-in-time snapshots or season-end backfills?**
  - **Answer:** Point-in-time expanding rolling windows computed up to round $T-1$. Newly promoted clubs use documented neutral priors (`PRIOR_TEAM_XG = 1.30`).
- **Look-ahead in price, ownership, fixture-difficulty, and news?**
  - **Answer:** Checked, no issue. Features use snapshot values at deadline freeze.
- **Out-of-fold calibration and stacking weights.**
  - **Answer:** Stacking weights are fit on temporal validation folds; interval quantiles evaluated on held-out test origins.
- **Chronological train/test splits.**
  - **Answer:** Chronological expanding origin splits by gameweek; no future-to-past shuffling.
- **Training-mode vs serving-mode feature values.**
  - **Answer:** Parity verified via `test_live_context_cache.py`. Serving mode reconstructs the same 62 feature vector as training.
- **Feature schema hash check.**
  - **Answer:** Verified. `FEATURE_SCHEMA_HASH` locks columns, order, and dtypes. Found feature count discrepancy: architecture doc claims 57 features, while schema has 62 (Finding **F-016**).

---

## 3.3 Model quality and statistical validity

- **Evidence of projection accuracy (backtests, baselines, calibration)?**
  - **Answer:** Projections demonstrate baseline MAE of ~2.1 pts on forward holdouts. Accuracy is unproven against paid commercial services.
- **Component models: targets, loss, sample weights, class imbalance, zero-inflation.**
  - **Answer:** Component models use LightGBM regressor/classifier. Zero inflation handled via two-stage hurdle models for goals/assists.
  - **Sample weighting flaw:** Finding **F-009**. Hardcoded 5x sample weight on 2026/27 reduces Kish effective sample size by 26,755 samples without tuning.
- **Team goals model: Dixon-Coles or independent Poisson?**
  - **Answer:** **Falsely claimed as Dixon-Coles (Finding F-012).** Uses independent Poisson with fixed 1.35 mean and static home multipliers. Low-score rho correction and time decay are absent.
- **Quantiles: z-values estimation, position pooling, nominal coverage.**
  - **Answer:** Quantiles use pooled z-multipliers ($-0.806$ / $1.009$) across positions rather than per-stratum conformalized intervals (Claims Register).
- **Model promotion gate: noise, margin, significance, rollback verification.**
  - **Answer:** Gate requires mean gain $> 0$ and degradation $\le 0.05$. Rollback on corrupt bundle is broken (Finding **F-006**; crashes with `ValueError`).
- **Cold start: new signings, promoted teams, low minutes.**
  - **Answer:** Handled via Bayesian shrinkage to position priors (`neutral_team_metrics`).

---

## 3.4 Optimisation (lineup, squad, transfers, chips)

- **Official rules constraint re-derivation: squad shape, 3-per-club, budget, formation bounds.**
  - **Answer:** Checked, no issue. Fully satisfied by PuLP CBC MILP formulations in `squad.py` and `lineup.py`.
- **Free transfers banking formula (max 5, hit deductions, WC/FH behaviour).**
  - **Answer:** Checked, no issue. `next_free_transfers` caps at 5, reduces to 1 on transfers without hit, resets on Wildcard/Free Hit per 2024-27 rules.
- **Selling price rounding and integer-tenths handling.**
  - **Answer:** Checked, no issue. `(now_cost - purchase_price) // 2` correctly uses integer division on tenths without floating-point error.
- **Beam search multi-transfer capability with 3–5 banked FTs.**
  - **Answer:** **FAILED (Finding F-005).** Beam search only enumerates 0, 1, or 2 transfers; cannot evaluate hit-free 3-, 4-, or 5-transfer moves.
- **Hit policy and cooldown.**
  - **Answer:** Cooldown blocks repeat hits for 6 weeks after a failed hit (`hit_policy.py:20`).
- **Chip rules: GW19 deadline, 1 per GW, FH not in GW1, no FH in GW20 after GW19.**
  - **Answer:** Constraints encoded in `rules.yaml` and `planner.py`.
  - **Deadlock Flaw:** Finding **F-007**. At GW19, rigid static thresholds cause unplayed Set 1 chips to forfeit for 0 points.
- **Intervention preference: minimal interventions vs Wildcard diagnosis.**
  - **Answer:** Evaluated in Scenario 1 (PASS: minimal 1 FT upgrade) and Scenario 2 (FAIL: rolls transfer on broken squad).
- **Solver robustness: timeouts, infeasibility, ties.**
  - **Answer:** PuLP solvers set 10s timeout and return status codes. Lineup solver risk modes only alter captaincy (Finding **F-011**).

---

## 3.5 Simulation (mini-league Monte Carlo)

- **Simulated mean equals model xP? Clipping bias.**
  - **Answer:** Clipping `max(0, xP - App - CS)` introduces $+0.12$ pt upward bias on low-xP assets.
- **Single Gamma draw limitations.**
  - **Answer:** Single draw loses negative points (cards), discrete haul distributions (e.g. 2 goals = 10 pts vs continuous density), and teammate covariance.
- **Seeding, reproducibility, precision.**
  - **Answer:** Uses NumPy seeded RNG (`np.random.default_rng(seed)`).
- **League win probability sanity: symmetric cases give symmetric answers?**
  - **Answer:** **FAILED (Finding F-002).** Asymmetric tie-breaking bug (`s > trial_user_pts`) awards 100% win probability to the user on tied squads.

---

## 3.6 News ingestion, availability and the LLM layer

- **Single-reconciler invariant.**
  - **Answer:** Confirmed. Only `ReconciledAvailability` in `reconcile.py` modulates player availability features.
- **Prompt injection and RSS fabrication.**
  - **Answer:** **VULNERABLE (Finding F-003).** Fabricated quotes in RSS feeds pass substring check and zero out availability in Gated-Active mode.
- **Entity resolution on shared surnames, accents, nicknames.**
  - **Answer:** **VULNERABLE (Finding F-010).** Surnames (e.g. "Timber") award 1.0 confidence to the first matching candidate without checking club or position.
- **Cache TTL, invalidation, poisoning.**
  - **Answer:** 300s–3600s TTLs. Documented claim of 12-hour cache is FALSE.
- **Assistant numbers origin and offline fallback.**
  - **Answer:** Tool results provide player IDs and stats. Offline fallback identifies itself as offline.

---

## 3.7 Server, API and background jobs

- **Endpoint validation, auth, error responses.**
  - **Answer:** 22 FastAPI endpoints with Pydantic validation and `SafeJSONResponse`.
- **Scheduled jobs and overlap protection.**
  - **Answer:** 7 background jobs managed by APScheduler (`jobs.py:30-101`). Overlap prevented via execution tracker flags.
- **Gameweek finalisation handling.**
  - **Answer:** Triggered when FPL API sets `event.finished = true` and `event.data_checked = true`.
- **Forward holdout pre-deadline freeze.**
  - **Answer:** Confirmed. `holdout_forward` job freezes predictions 15 minutes before deadline and evaluates post-GW.

---

## 3.8 Data layer and storage

- **Schema, indices, timestamp handling, player ID shifts mid-season.**
  - **Answer:** Handled in SQLite `data_store.py` with unique element IDs.
- **Data quality: units (tenths), negative minutes, price jumps.**
  - **Answer:** Prices stored in integer tenths (£5.0m = 50). Minutes clipped $\ge 0$.
- **Concurrency: simultaneous reads and writes.**
  - **Answer:** **DEFECTIVE (Finding F-008).** Uses `threading.RLock()`; offers zero multi-process concurrency protection.
- **Data provenance.**
  - **Answer:** Official FPL endpoints and public RSS feeds.

---

## 3.9 Security and privacy

- **Secrets in repo, history, logs, config.**
  - **Answer:** **CRITICAL FINDING F-001.** Plaintext Google Gemini API key logged via `httpx` query parameter `?key=...` in `logs/fpl_oracle.log:16592-20062`.
- **Dependency audit (`pip-audit`).**
  - **Answer:** Confirmed clean: 0 known CVEs.
- **Model bundle integrity & tampering.**
  - **Answer:** SHA256 hashes checked on load. Tampered bundle crashes with `ValueError` without rollback (Finding **F-006**).

---

## 3.10 Tests

- **Meaningfulness of assertions: mock mirroring, formula copying.**
  - **Answer:** High quality in rules contract tests; however, `test_league_strategy_mc.py:370` asserts wall-clock time `< 2.0s` which fails under profiling (Finding **F-013**).
- **Mutation testing (`mutmut`).**
  - **Answer:** `mutmut` refuses native Windows execution (WSL prompt). Targeted mutation test revealed **25.0% survival rate** on scoring contract: Poisson truncation mutant survived due to test bounds $\lambda \le 3.0$.
- **Test count check ("426 tests").**
  - **Answer:** Currently **461 tests** collected and passing.

---

## 3.11 Code quality, performance and operability

- **Linters and type safety.**
  - **Answer:** `ruff check .` found 20 errors (Finding **F-014**); `mypy src` found 7 errors (Finding **F-015**).
- **Performance.**
  - **Answer:** Core MILP solves execute in 50–200ms. Beam search takes 15–25s on 5-GW horizons.
- **Logging and configuration provenance.**
  - **Answer:** Logging is detailed but suffers from secret leaks (F-001).

---

## 3.12 Product behaviour scenarios

- **Results for the 8 canonical scenarios:**
  1. Healthy core, 1 target: **PASS** (1 FT upgrade, 0 hits).
  2. Broken squad: **FAIL** (rolls transfer; Wildcard retains 0 xP players).
  3. Large blank gameweek: **PARTIAL** (recommends Wildcard instead of Free Hit).
  4. Chips near expiry: **FAIL** (0 chips recommended at GW19; forfeited).
  5. Marginal hit proposal: **PASS** (strictly rolls transfer).
  6. Small mini-league lead: **PASS** (DEFENDING_LEAD postures correctly).
  7. Imminent price rise: **PARTIAL** (flags rise, but optimizer timing decoupled).
  8. Unresolved injury doubt: **PARTIAL** (risk modes alter captaincy only; starter XI identical).

---

## 3.13 Documentation honesty and drift

- **Discrepancies found:**
  - Doc claims GKP goals = 10 pts; Code is 6 pts (Official rule).
  - Doc claims 57 features; Code has 62 features.
  - Doc claims Dixon-Coles model; Code uses fixed Poisson.
  - Doc claims automatic rollback on corrupt bundle; Code raises crashing `ValueError`.
  - Doc claims 12-hour cache; Code uses 60s–3600s TTLs.
