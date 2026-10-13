# Remediation Log: Adversarial Audit Hotfixes

This log tracks every remediation commit across Phase H0, H1, H2, and handover to the improvement spec.
Per finding: commit hash, test name, before and after execution output, configuration flags added, and unexpected discoveries.

---

## F-001: Live Plaintext API Key Leak in Logs

- **Commit**: `b8b880d` (`F-001: Redact API keys and auth headers, mute httpx loggers`)
- **Test Name**: `tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions`, `tests/test_f001_api_key_logging.py::test_mocked_llm_call_logging_redaction`
- **Flags Added**: None (pure security bugfix)
- **Before Output**:
  ```text
  FAILED tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions - AssertionError: httpx logger level is 0, expected >= WARNING
  assert 0 >= 30
   +  where 0 = <Logger httpx (INFO)>.level
   +  and   30 = logging.WARNING
  ```
- **After Output**:
  ```text
  tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions PASSED [ 50%]
  tests/test_f001_api_key_logging.py::test_mocked_llm_call_logging_redaction PASSED [100%]
  2 passed in 0.18s
  ```
- **Actions Taken**:
  1. Configured `httpx` and `httpcore` loggers to `logging.WARNING` at import time and within `setup_logging()`.
  2. Enhanced `SecretRedactingFilter` to redact query params (`key=...`, `api_key=...`), `Authorization` / `Bearer` headers, Google API key regex `AIza[0-9A-Za-z-_]{35}`, and environment secrets, across messages, exc_text, and exception tracebacks.
  3. Ensured `.gitignore` includes `logs/`, `.env`, `*.log`, and `data/cache/`.
  4. Verified git tracking: `logs/` is not tracked (`git ls-files logs` returned 0 entries). Logged commits touching `key=` in `docs/needs_human.md`.
  5. Created redacted log copy `logs/fpl_oracle.redacted.log` (0 unredacted Google API key patterns remaining).
- **Surprises / Notes**:
  `logs/` was already properly gitignored on disk and not tracked in git; the leak was purely local disk log persistence.

---

## F-008: In-Process Thread Lock Only (Zero Cross-Process Protection)

- **Commit**: `2b1162b` (`F-008: Replace thread-only lock with cross-process FileLock`)
- **Test Name**: `tests/test_f006_f008_bundle_recovery_and_lock.py::test_cross_process_bundle_lock_mutual_exclusion`
- **Flags Added**: None (system infrastructure)
- **Before Output**:
  ```text
  Only threading.RLock() was used in bundle_lock.py; concurrent OS processes had zero mutual exclusion on bundle operations.
  ```
- **After Output**:
  ```text
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_cross_process_bundle_lock_mutual_exclusion PASSED [100%]
  ```
- **Actions Taken**:
  1. Updated `src/fpl_oracle/ml/bundle_lock.py` to use `filelock.FileLock` anchored to `MODELS_DIR / "bundle.lock"`.
  2. Implemented canonical path caching (`_LOCK_CACHE`) to preserve re-entrancy for nested locked method calls within the same process while guaranteeing strict OS-level exclusion across processes.
- **Surprises / Notes**:
  Cross-process file locking must be reentrant on the same process/thread to allow nested `@bundle_locked` methods (e.g. `verify_weight_integrity` calling `quarantine_and_recover`).

---

## F-006: Automatic Checkpoint Rollback on Corrupt Bundle

- **Commit**: `6dfcf32` (`F-006: Automatic quarantine and rollback on corrupt model bundle`)
- **Test Name**: `tests/test_f006_f008_bundle_recovery_and_lock.py::test_corrupt_bundle_auto_recovers_to_previous_version`, `tests/test_f006_f008_bundle_recovery_and_lock.py::test_no_valid_version_raises_clear_error_not_raw_valueerror`, `tests/test_f006_f008_bundle_recovery_and_lock.py::test_tampered_manifest_auto_recovers`
- **Flags Added**: None (reliability bugfix)
- **Before Output**:
  ```text
  ValueError: SHA256 mismatch for minutes_model.pkl: expected 8ebdc..., got a7c8e...
  Corrupt file remains corrupted on disk and inference pipeline crashes with unhandled ValueError.
  ```
- **After Output**:
  ```text
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_corrupt_bundle_auto_recovers_to_previous_version PASSED
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_no_valid_version_raises_clear_error_not_raw_valueerror PASSED
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_tampered_manifest_auto_recovers PASSED
  audit/scratch/test_corrupt_bundle.py output:
  [ModelIntegrity] AUTOMATIC RECOVERY SUCCESSFUL: Quarantined bad files, restored verified version 'v1.0.0'.
  ```
- **Actions Taken**:
  1. Added `ModelBundleCorruptError` exception class.
  2. Implemented `quarantine_and_recover()` in `ModelRegistry`: moves bad `.pkl` and calibration files into `quarantine/<timestamp>/`, locates the latest verified backup from `versions/`, restores all files, updates active version, and writes `recovery_event.json`.
  3. Integrated verification into `predict.py:load_or_train()`: never loads unverified weights; automatically quarantines and recovers on corruption.
  4. Exposed recovery event in `get_model_status()` and `/health` FastAPI endpoint.
- **Surprises / Notes**:
  `audit/scratch/test_corrupt_bundle.py` asserted that `verify_weight_integrity()` would raise an exception; with automatic self-healing recovery, it now recovers and succeeds cleanly, matching operational resilience requirements.


---

## F-010: Shared Surname Entity Resolution Ambiguity

- **Commit**: `c8fa2ae` (`F-010: Disambiguate shared surnames using club and position context`)
- **Test Name**: `tests/test_f010_entity_resolution.py::test_shared_surname_requires_club_disambiguation`
- **Flags Added**: None (entity resolution correctness)
- **Before Output**:
  ```text
  res_ambiguous['confidence'] = 0.95; unmatched query "Timber" or "Silva" arbitrarily resolved to first candidate with high confidence despite multiple players sharing the identical surname.
  ```
- **After Output**:
  ```text
  tests/test_f010_entity_resolution.py::test_shared_surname_requires_club_disambiguation PASSED [100%]
  ```
- **Actions Taken**:
  1. Updated `FuzzyPlayerMatcher.match_single_name()` in `src/fpl_oracle/data/fuzzy_match.py` to accept optional `club` and `position` hints.
  2. Detected duplicate surname collisions across candidate elements.
  3. When duplicate surname collisions occur without disambiguating club/position hints, applied an ambiguity penalty (`confidence = 0.40`) or returned `matched = None` with `ambiguity_detected = True`.
  4. When club or position hint matches, boosted candidate confidence to 0.95.

---

## F-003: News Override Safety: Allowlist, Corroboration, and Injection Defenses

- **Commit**: `57f89f8` (`F-003: News override safety with publisher allowlist and corroboration`)
- **Test Name**: `tests/test_f003_news_override_safety.py` (5 test cases)
- **Flags Added**: `trusted_publishers` in `config/news_sources.yaml`
- **Before Output**:
  ```text
  Untrusted RSS feeds and single uncorroborated sources could override official FPL availability to 0% in Gated-Active mode. Quotes containing hidden prompt injection HTML comments passed verification.
  ```
- **After Output**:
  ```text
  tests/test_f003_news_override_safety.py::test_untrusted_rss_source_cannot_override_in_gated_active PASSED [ 20%]
  tests/test_f003_news_override_safety.py::test_single_allowlisted_source_capped_not_zeroed PASSED [ 40%]
  tests/test_f003_news_override_safety.py::test_two_independent_allowlisted_sources_permit_zero_cop PASSED [ 60%]
  tests/test_f003_news_override_safety.py::test_duplicate_syndicated_copies_count_as_single_source PASSED [ 80%]
  tests/test_f003_news_override_safety.py::test_hidden_text_and_prompt_injection_variants_rejected PASSED [100%]
  ```
- **Actions Taken**:
  1. Added `trusted_publishers` allowlist to `config/news_sources.yaml` (BBC, Guardian, Premier League, The Athletic, etc.).
  2. Implemented domain extraction and publisher matching in `AvailabilityReconciler` (`src/fpl_oracle/news/reconcile.py`). In Gated-Active mode, evidence from untrusted domains is rejected.
  3. Added quote sanitization filtering hidden HTML tags/comments (`<!-- ... -->`), prompt injections, and zero-width unicode characters.
  4. Enforced corroboration requirement: overriding chance of playing to 0.0 requires >=2 independent allowlisted publishers with non-identical quotes; single allowlisted sources are capped at reduced probability (25.0) and flagged.

---

## F-002: Monte Carlo Tie Handling & Fractional Win Credit

- **Commit**: `0a80408` (`F-002: Fractional win credit and symmetric tie handling in Monte Carlo league simulation`)
- **Test Name**: `tests/test_f002_monte_carlo_ties.py` (4 test cases)
- **Flags Added**: None (Monte Carlo mathematical correctness)
- **Before Output**:
  ```text
  Ties resulted in 0 win credit for both users (win_counts strictly required total_pts > max_rival_pts).
  Two identical squads in a 2-manager league produced win probabilities of 0% / 0% instead of 50% / 50%,
  causing total win probabilities to fail summing to 100%.
  ```
- **After Output**:
  ```text
  tests/test_f002_monte_carlo_ties.py::test_two_identical_squads_yield_exact_50_50_win_probability PASSED [ 25%]
  tests/test_f002_monte_carlo_ties.py::test_three_way_tie_yields_one_third_credit PASSED [ 50%]
  tests/test_f002_monte_carlo_ties.py::test_win_probabilities_sum_to_100_percent PASSED [ 75%]
  tests/test_f002_monte_carlo_ties.py::test_symmetric_swap_yields_symmetric_results PASSED [100%]
  4 passed in 0.52s
  ```
- **Actions Taken**:
  1. Updated `simulate_league_outcomes()` in `src/fpl_oracle/league/strategy.py` to award fractional win credit `1.0 / (1 + tied_count)` when user score matches leader score.
  2. Implemented midrank-based fractional podium placement calculation (`finish_probs` and `podium_probs`) for accurate league placement probability distributions.
  3. Ensured symmetrical squads produce exactly symmetric 50.0% / 50.0% win probabilities and all outcome probabilities sum to 100.0%.

---

## F-004: Dynamic Poisson Summation Bound for Expected Floor Division

- **Commit**: `2a78852` (`F-004: Dynamic Poisson summation bound for expected floor division`)
- **Test Name**: `tests/test_f004_poisson_truncation.py` (2 test cases)
- **Flags Added**: None (numerical correctness)
- **Before Output**:
  ```text
  Static truncation at max_k = 30 in scoring.py:expected_floor_div() discarded >80% of Poisson probability mass
  for arrival rates >= 30, producing non-monotonic expectations and severe underestimates.
  ```
- **After Output**:
  ```text
  tests/test_f004_poisson_truncation.py::test_expected_floor_div_monotonic_high_rates PASSED [ 50%]
  tests/test_f004_poisson_truncation.py::test_expected_floor_div_matches_brute_force_dynamic_bound PASSED [100%]
  2 passed in 0.65s
  ```
- **Actions Taken**:
  1. Replaced static `max_k = 30` with dynamic summation bound:
     `max_k = min(500, max(40, int(rate + 6.0 * math.sqrt(rate))))`.
  2. Implemented loop termination once residual Poisson mass falls below `1e-12`.
  3. Verified monotonicity and numerical equivalence against high-precision brute-force summation across rates 0.01 to 60.0 for divisors 2 and 3.

---

## F-007: Use-It-Or-Lose-It Chip Recommendation at Deadline & Set 1 Warning

- **Commit**: `10dd17e` (`F-007: Use-it-or-lose-it chip recommendation at deadline and API-based set 1 warning`)
- **Test Name**: `tests/test_f007_chip_expiry.py` (3 test cases)
- **Flags Added**: None (chip planning correctness)
- **Before Output**:
  ```text
  At GW19 (Set 1 expiry deadline), chips with positive expected gain below static thresholds (e.g. 3.5 xP for 3xC)
  evaluated is_recommended = False, causing chips to expire for 0 gain.
  ```
- **After Output**:
  ```text
  tests/test_f007_chip_expiry.py::test_gw19_forces_chip_usage_on_sub_threshold_positive_gain PASSED [ 33%]
  tests/test_f007_chip_expiry.py::test_penultimate_gw_lowers_threshold_by_half PASSED [ 66%]
  tests/test_f007_chip_expiry.py::test_set_1_warning_active_from_gw17_with_real_deadline PASSED [100%]
  3 passed in 0.81s
  ```
- **Actions Taken**:
  1. Implemented use-it-or-lose-it threshold rule in `src/fpl_oracle/chips/planner.py`: at the final window of a chip set (GW19 for Set 1, GW38 for Set 2), threshold is lowered to `0.0 xP` (any positive expected gain is recommended).
  2. At the penultimate window (GW18 / GW37), thresholds are halved (`threshold * 0.5`).
  3. Added `set_1_warning` field active from GW17 onwards if any Set 1 chip remains unplayed, reading official deadline time from `bootstrap.events[19].deadline_time`.

---

## F-011: Risk-Aware Starter Selection with Downside Penalty for Conservative Mode

- **Commit**: `f88d76d` (`F-011: Risk-aware starter selection with downside penalty for conservative mode`)
- **Test Name**: `tests/test_f011_risk_lineup.py` (2 test cases)
- **Flags Added**: `risk_aware_lineup: true` in `config/flags.yaml`
- **Before Output**:
  ```text
  Starter objective coefficients were strictly df.loc[i, 'expected_points'] * starter[i],
  producing 100% identical starting lineups across conservative, balanced, and aggressive modes (Scenario 8).
  ```
- **After Output**:
  ```text
  tests/test_f011_risk_lineup.py::test_conservative_mode_benches_doubtful_high_variance_starter PASSED [ 50%]
  tests/test_f011_risk_lineup.py::test_balanced_mode_identical_to_baseline PASSED [100%]
  2 passed in 0.45s
  ```
- **Actions Taken**:
  1. Configured feature flag `risk_aware_lineup` (default `true`) in `config/flags.yaml`.
  2. Updated `select_lineup_and_captain()` in `src/fpl_oracle/optimise/lineup.py` to accept `downside_risk_lambda: float = 0.20`.
  3. In conservative mode, starter coefficients penalize quantile spread `(p90 - p10)` and availability doubts (`(100 - chance_of_playing) / 100 * 3.0`).
  4. In aggressive mode, starter coefficients reward upside `(p90 - expected_points) * 0.10`.
  5. In balanced mode, starter coefficients remain 100% numerically identical to baseline expected points.

---

## F-013: Non-Deterministic Wall-Clock Timing Assertion in Test Suite

- **Commit**: `b1e758f` (`F-013: Remove non-deterministic wall-clock timing assertion in test suite`)
- **Test Name**: `tests/test_league_strategy_mc.py::test_side_by_side_plan_win_prob_comparison`
- **Flags Added**: None
- **Before Output**:
  ```text
  AssertionError: Simulation took too long: 3.45s (assert elapsed < 2.0 failed under pytest-cov tracing).
  ```
- **After Output**:
  ```text
  tests/test_league_strategy_mc.py passed cleanly under coverage tracing.
  ```
- **Actions Taken**:
  Removed brittle wall-clock elapsed time assertion from `tests/test_league_strategy_mc.py`, preserving algorithmic correctness verification.

---

## F-014: 20 Ruff Linting Errors Across Core Modules

- **Commit**: `01653c9` (`F-014: Resolve ruff lint and import formatting errors across repo`)
- **Flags Added**: None
- **Before Output**:
  ```text
  ruff check . reported 20 errors across 6 modules (unused variables, unsorted imports, blank lines).
  ```
- **After Output**:
  ```text
  ruff check . exited with 0 errors. All checks passed!
  ```
- **Actions Taken**:
  1. Cleaned unused local variables (`starters_clean`, `bench_clean`, unused imports).
  2. Formatted and sorted import blocks across `scoring.py`, `squad.py`, `gemini_extractor.py`, `agent.py`, etc.

---

## F-015: 7 Static Type Checking Errors in Source Code

- **Commit**: `d2b9407` (`F-015: Resolve type checking errors in mypy src`)
- **Flags Added**: None
- **Before Output**:
  ```text
  mypy src reported 7 errors:
  - squad.py: Incompatible return type tuple vs SolveSquadResult
  - gemini_extractor.py: "dict" has no attribute "extracted_signals"
  - tools.py: Dict[str, Any] incompatible with List[Dict[str, Any]]
  - agent.py: Argument 1 to "dumps" incompatible
  ```
- **After Output**:
  ```text
  mypy src reported: Success: no issues found in 47 source files.
  ```
- **Actions Taken**:
  Resolved all 7 type errors: properly unpacked `SolveSquadResult` in `squad.py`, corrected attribute/dict access in `gemini_extractor.py`, ensured correct types in `tools.py` and `agent.py`.

---

## F-016 & Doc Drift: Feature Table Parity & Independent Poisson Clarification

- **Commit**: `25009da` (`F-016: Synchronize 62-column feature table, clean doc links, and clarify Poisson match modeling`)
- **Test Name**: `tests/test_feature_table_doc_parity.py` (2 test cases), script `scripts/generate_architecture_feature_table.py`
- **Flags Added**: None
- **Before Output**:
  ```text
  Architecture documentation claimed 61 features but listed 39 in an incomplete table.
  Contained local file:///C:/ links and unprovable operational claims.
  Claimed bivariate Dixon-Coles model while code implemented independent Poisson.
  ```
- **After Output**:
  ```text
  tests/test_feature_table_doc_parity.py passed cleanly.
  Architecture documentation synchronized with all 62 FEATURE_COLUMNS.
  Local file:/// links removed; Poisson match modeling accurately documented.
  ```
- **Actions Taken**:
  1. Created `scripts/generate_architecture_feature_table.py` to automatically generate markdown table from `FEATURE_COLUMNS`.
  2. Updated `fpl_oracle_system_architecture.md` with complete 62-column table and accurate feature group breakdown.
  3. Reworded match modeling section to describe independent Poisson expectations with attack/defence multipliers, noting bivariate Dixon-Coles is scheduled for spec WI-3.1.
  4. Added CI regression test `tests/test_feature_table_doc_parity.py` to prevent future documentation drift.

---

## Scenario 7: Price-Rise Timing Warning Attached to Recommended Transfers

- **Commit**: `699ea1b` (`Scenario 7: Price-rise timing warning and Scenario 2/3 audit artifact tests`)
- **Test Name**: `tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_7_price_rise_timing_warning_attached`
- **Flags Added**: None
- **Before Output**:
  ```text
  Price change predictions were computed in a separate module but disconnected from the weekly briefing transfer recommendation table.
  ```
- **After Output**:
  ```text
  tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_7_price_rise_timing_warning_attached PASSED
  ```
- **Actions Taken**:
  In `WeeklyBriefingGenerator` (`src/fpl_oracle/briefing/weekly.py`), transfers in the recommended plan whose incoming player is flagged in `imminent_rises` now have ` ⚠️ (Price rises tonight, act before 23:59 UK)` attached directly to their transfer line.

---

## Scenarios 2 & 3: Audit Harness Artifact Verification

- **Commit**: `699ea1b` (`Scenario 7: Price-rise timing warning and Scenario 2/3 audit artifact tests`)
- **Test Name**: `tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_2_broken_squad_fixture_artifact_vs_fixed`, `tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_3_blank_gameweek_fixture_artifact_vs_fixed`
- **Flags Added**: None
- **Before Output (Flawed Audit Harness Fixtures)**:
  ```text
  - Scenario 2: squad players set to 0.0 xP but player_pool_df had same players at >4.5 xP. Optimizer retained them because pool claimed they were healthy.
  - Scenario 3: blanking players set to 0.0 xP in pool but current_squad_df passed with unadjusted baseline, resulting in 0.0 xP gain.
  ```
- **After Output (Corrected Synchronized Fixtures)**:
  ```text
  tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_2_broken_squad_fixture_artifact_vs_fixed PASSED
  tests/test_h2_scenarios_and_audit_artifacts.py::test_scenario_3_blank_gameweek_fixture_artifact_vs_fixed PASSED
  - Scenario 2: With synchronized pool, 0 broken players are retained in Wildcard.
  - Scenario 3: With synchronized squad, Free Hit gain is +11.75 xP (>= 10.0 threshold) and Free Hit is recommended.
  ```
- **Actions Taken**:
  Per working rules Rule 8, verified that audit scenario failures 2 and 3 were harness artifacts caused by desynchronized mock fixtures in `audit/scratch/run_scenarios.py`. Added regression tests verifying both the artifact mechanism and proper behavior under synchronized inputs. No ad-hoc constants patched into production code; root-cause chip horizon improvements remain in spec Phase 8.

---

## F-009: Arbitrary 5x Weights on 2026/27 Rows

- **Commit**: `a705281` (`F-009: Add temporal holdout sample weight comparison script`)
- **Script**: `scripts/compare_sample_weights.py`
- **Flags Added**: None (deferred to spec WI-0.1)
- **Holdout Evaluation Results**:
  ```text
  Weight Multiplier  Kish ESS  ESS Reduction  DefCon LogLoss  DefCon Brier  DefCon AUC  Bonus MAE  Bonus RMSE  Bonus Spearman
               1.0x   88323.0            0.0          0.3535        0.1093      0.9179     0.1717      0.4483          0.2561
               2.0x   87153.4         1169.6          0.3545        0.1099      0.9187     0.1736      0.4494          0.2533
               3.0x   83939.0         4384.0          0.3601        0.1120      0.9163     0.1740      0.4496          0.2525
               5.0x   73726.2        14596.8          0.3643        0.1137      0.9135     0.1702      0.4484          0.2553
  ```
- **Actions Taken**:
  1. Created `scripts/compare_sample_weights.py` evaluating 1.0x, 2.0x, 3.0x, and 5.0x weights on temporal holdout.
  2. Confirmed finding: 5.0x weighting reduces Kish ESS by ~14.6k rows and degrades DefCon log loss from 0.3535 to 0.3643.
  3. Preserved weights for hotfix phase per instructions; weighting policy will be tuned via out-of-fold temporal CV in Backtest Harness (spec WI-0.1).

---

## F-005: Beam Search Incapable of >2 Transfers (Banked 3-5 FTs)

- **Commit**: `6cc698a` (`F-005: Add strict xfail test for 3-transfer beam candidate generation`)
- **Test Name**: `tests/test_f005_banked_transfers_beam.py::test_beam_finds_optimal_3_transfer_hit_free_plan`
- **Status**: `XFAIL(strict=True)`
- **Reference**: Spec WI-2.1 (Multi-period MILP optimizer)
- **Execution Output**:
  ```text
  tests\test_f005_banked_transfers_beam.py x [100%]
  1 xfailed in 0.69s
  ```
- **Actions Taken**:
  Constructed scenario where 3 FTs are banked and a 3-player transfer yields +21.0 xP with 0 hit cost vs at most +14.0 xP for 2 transfers. Added strict `xfail` test marking beam candidate limitation to be solved by multi-period MILP optimizer (WI-2.1).

---

## F-012: Architectural Claims of 'Dixon-Coles Model'

- **Commit**: `e11eb70` (`F-012: Add strict xfail test for bivariate Dixon-Coles model`)
- **Test Name**: `tests/test_f012_dixon_coles_bivariate.py::test_dixon_coles_fits_low_score_rho`
- **Status**: `XFAIL(strict=True)`
- **Reference**: Spec WI-3.1 (True bivariate Dixon-Coles model)
- **Execution Output**:
  ```text
  tests\test_f012_dixon_coles_bivariate.py x [100%]
  1 xfailed in 0.19s
  ```
- **Actions Taken**:
  Constructed test asserting bivariate low-score correlation parameter $\rho$ estimation and non-zero $(0, 0)$ and $(1, 0)$ joint score adjustments. Marked with strict `xfail` linking to spec WI-3.1.
