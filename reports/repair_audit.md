# FPL Oracle — Comprehensive Targeted Fix Pass & Integrity Audit Report

**Report Generated**: 2026-10-06  
**Audited Baseline Commit**: `37ac87da281f4f723ee3e708178dcff402afbddc`  
**Package Version**: `fpl-oracle` v1.0.0  
**Target Environment**: Windows PowerShell (Local Loopback `127.0.0.1:8000`)  
**Test Suite Verification**: **105 / 105 PASSED (100%)**  
**Evaluation Standard**: Rule 0.1 Strict Evidence-Gated Ledger (Backed by saved logs in `reports/evidence/`)

---

## 1. Executive Summary

This report documents the exhaustive, evidence-gated resolution of all weaknesses and unverified claims previously identified in **FPL Oracle**. In strict compliance with the project's **Zero Fabrication** mandate (Rule 0.1 and Rule 0.2), every single item in this pass has been either fixed and verified with code and automated tests, or measured out-of-time and documented with exact empirical numbers.

### Key Milestones Completed:
1. **Model Integrity & Training Hygiene (M1–M8)**:
   - Purged all hardcoded metrics dictionaries and canned upcoming projection samples from `src/fpl_oracle/ml/train.py`.
   - Executed dynamic out-of-time feature ablation measuring a **17.14% MAE reduction** when including fixture and opponent strength features.
   - Built bidirectional season-scoped opponent ID mapping from `master_history.csv` + bootstrap data, eliminating fallback to neutral priors during model training.
   - Restored authentic train/serve parity across all 62 canonical features, including dynamic `days_rest` calculation from fixture timestamps.
   - Fixed disciplinary card target sorting misalignment via synchronous indexing and composite `(player_id, round)` keys.
   - Fitted empirical residual quantile calibration on disjoint out-of-fold data, achieving **78.74%** empirical coverage for nominal 80% credible intervals.
2. **Availability & News Reconciliation Pipeline (N1–N8)**:
   - Directly wired reconciled availability probabilities into the feature extraction kernel and projection engine.
   - Implemented and verified strict operational isolation: `shadow` mode preserves official baseline projections, while `gated_active` modulates expected minutes and xP only upon verified press evidence.
   - Replaced handpicked probability numbers with configurable named settings (`FPLSettings`).
   - Delivered executable 20-case adversarial benchmark runner (`scripts/run_news_benchmark.py`): **20/20 passed (100%)**, zero prompt injection breaches.
   - Hardened external URL fetching with DNS-resolving SSRF protection against loopback, private RFC-1918, and cloud metadata addresses.
   - Rendered verbatim manager quotes and source links in the UI with DOMPurify sanitization.
   - Purged dead and inert API keys (`TAVILY_API_KEY`, `BRAVE_API_KEY`, `ODDS_API_KEY`).
3. **Stateful Transfer Optimization (T1–T9)**:
   - Built stateful 5-GW beam search optimizer tracking complete squad state, bank balance, exact 50% profit selling prices ($P_{\text{sell}} = P_{\text{purchase}} + \lfloor (P_{\text{now}} - P_{\text{purchase}}) / 2 \rfloor$), free transfer banking (up to 5 FTs), and club quotas.
   - Added future-hit avoidance calculation, free-transfer option value estimation, robustness re-ranking across projection noise (win rate $\ge 0.70$), price change sensitivity toggle, plan stability thresholds, and dynamic roadmap generation with zero canned text.
4. **Mini-League Game Theory & Rivals (R1–R4, W1–W3)**:
   - Implemented paginated mini-league standings (up to 10 pages / 500 teams) with proximity-based rival selection (all teams ahead + within 30 points below).
   - Upgraded stochastic mini-league simulation to sample common players once per trial and share team-level Bernoulli clean sheet draws (+4 pts) across club defenders.
   - Modeled rival transfer and captaincy behavior with transparent assumptions; validated on historical gameweeks with a mean Brier score of **0.0677**.
5. **Multi-GW Chip Calendar (C1–C4)**:
   - Built multi-GW chip calendar evaluating incremental benefit against a no-chip baseline.
   - Enforced Set 1 Gameweek 19 hard deadline cutoff and uncertainty gating (+3.0 net xP threshold).
   - Integrated rival remaining chips tracking into simulation scenarios.
6. **Unified Per-GW Decision Card (D1–D3, Q2)**:
   - Built unified `/api/decision-card` and `/api/decision-card/export` endpoints delivering a comprehensive gameweek battle plan.
   - Automated cross-surface recommendation consistency test (`tests/test_recommendation_consistency.py`) confirming 100% agreement between Decision Card, Squad workbench, Transfer optimizer, and Chip engine.

---

## 2. Complete Traceability Matrix (M1 – Q4)

| ID | Category | Requirement Description | Status | Evidence File | Verified Outcome |
|:---|:---|:---|:---:|:---|:---|
| **M1** | Model Integrity | Remove fabricated metrics & upcoming projection samples | **PASS** | `reports/evidence/M1.txt` | Hardcoded dicts deleted; metrics computed dynamically from training run. |
| **M2** | Model Integrity | One honest evaluation, GW-boundary split, manifest gate | **PASS** | `reports/evidence/M2.txt` | Single evaluation JSON; temporal GW split; baseline-superiority promotion gate. |
| **M3** | Model Integrity | Real rolling-origin table by global gameweek | **PASS** | `reports/evidence/M3.txt` | Real temporal evaluation table across historical and 2026-27 origins evaluated against identical rows. |
| **M4** | Model Integrity | Opponent feature fix: resolve numeric opponent IDs | **PASS** | `reports/evidence/M4.txt` | Bidirectional ID-to-team map created; eliminates fallback to neutral priors across 89,141 rows. |
| **M5** | Model Integrity | Live train/serve parity across all 62 columns & days_rest | **PASS** | `reports/evidence/M5.txt` | 62/62 column train/serve parity verified across single, blank, and double GWs. Dynamic `days_rest`. |
| **M6** | Model Integrity | Fix disciplinary card label sorting/index misalignment | **PASS** | `reports/evidence/M6.txt` | Targets aligned synchronously by `(player_id, round)` keys. Alpha/Zulu and randomized tests pass. |
| **M7** | Model Integrity | Calibrated uncertainty intervals ($P_{10}$ / $P_{90}$) | **PASS** | `reports/evidence/M7.txt` | Residual quantile calibration achieves **78.74%** coverage for nominal 80% interval. |
| **M8** | Model Integrity | Dynamic feature ablation & purge inert odds claims | **PASS** | `reports/evidence/M8.txt` | Two-pass feature ablation shows **17.14% MAE reduction** with fixture features. Odds claims purged. |
| **N1** | News / Availability | Wire Gemini/reconcile directly into extraction & predict | **PASS** | `reports/evidence/N1.txt` | Reconciled availability probabilities passed into `extract_live_features_for_upcoming()` and `predict.py`. |
| **N2** | News / Availability | Operational modes: shadow vs gated_active vs off | **PASS** | `reports/evidence/N2.txt` | Controlled mock test proves injury quote adjusts xP in `gated_active` and preserves baseline in `shadow`. |
| **N3** | News / Availability | Replace handpicked reconcile probabilities with settings | **PASS** | `reports/evidence/N3.txt` | Named settings in `FPLSettings` (`CONFIRMED_FIT_PROB`, `DOUBTFUL_PROB`, etc.) with documented defaults. |
| **N4** | News / Availability | Adapter schema unification, full roster batching | **PASS** | `reports/evidence/N4.txt` | Unified element/team ID types, full roster batching, strict budget boundary checks. |
| **N5** | News / Availability | Executable 20-case adversarial benchmark runner | **PASS** | `reports/evidence/N5.txt` | `scripts/run_news_benchmark.py`: 20/20 cases passed (100%), 0 injection breaches. |
| **N6** | News / Availability | DNS-resolving SSRF protection & manual redirect check | **PASS** | `reports/evidence/N6.txt` | Hostnames resolved via DNS; private, loopback, and metadata IPs blocked on all hops. |
| **N7** | News / Availability | Render verbatim quotes and source links in web UI | **PASS** | `reports/evidence/N7.txt` | Verbatim quote text, source URL, applied mode badge, and minutes delta rendered with DOMPurify. |
| **N8** | News / Availability | Remove inert search & odds keys from config | **PASS** | `reports/evidence/N8.txt` | Purged `TAVILY_API_KEY`, `BRAVE_API_KEY`, `ODDS_API_KEY` from config and environment templates. |
| **T1** | Transfers | Stateful sequential multi-GW optimizer (horizon 5) | **PASS** | `reports/evidence/T1.txt` | 5-GW beam search tracking squad, bank, exact 50% selling prices, 1-5 banked FTs, hits (-4), and club limits. |
| **T2** | Transfers | Compare roll, 1 transfer, 2+ transfers, hits | **PASS** | `reports/evidence/T2.txt` | Evaluates candidate branches across identical 5-GW horizon with gross xP, hits, and net gain vs roll. |
| **T3** | Transfers | Future-hit avoidance derived from trajectories | **PASS** | `reports/evidence/T3.txt` | Compares multi-GW trajectory against greedy single-GW move; quantifies future hits saved. |
| **T4** | Transfers | Free-transfer option value from horizon simulation | **PASS** | `reports/evidence/T4.txt` | Mathematically computes FT option value from horizon search comparing baseline FT with FT+1. |
| **T5** | Transfers | Robustness re-ranking and no-regret move marking | **PASS** | `reports/evidence/T5.txt` | Evaluates 50 Monte Carlo projection noise trials; flags no-regret move present in $\ge 70\%$ of plans. |
| **T6** | Transfers | Probabilistic price change sensitivity toggle | **PASS** | `reports/evidence/T6.txt` | Supports `include_price_gain=False/True`, isolating pure xP gain from price movements. |
| **T7** | Transfers | Dynamic roadmap generated from actual trajectory | **PASS** | `reports/evidence/T7.txt` | Generates dynamic 5-GW roadmap from trajectory player picks, fixtures, and DGW/BGW tags. Zero canned text. |
| **T8** | Transfers | Plan stability threshold for changing top move | **PASS** | `reports/evidence/T8.txt` | Configurable stability threshold (0.30 xP) prevents transfer churning on marginal differences. |
| **T9** | Transfers | Offline deterministic transfer test suite | **PASS** | `reports/evidence/T9.txt` | 11/11 deterministic tests pass in `tests/test_optimizer.py` verifying T1-T8 invariants. |
| **R1** | Rivals & League | Paginate full standings & proximity-based rivals | **PASS** | `reports/evidence/R1.txt` | Paginates up to 10 pages (500 teams); rivals = all managers ahead + within 30 points below. |
| **R2** | Rivals & League | Published rival details: EO, exposure, chips used/left | **PASS** | `reports/evidence/R2.txt` | Calculates started %, captained %, effective ownership %, user net exposure, and tracks 2026/27 dual-set chips. |
| **R3** | Rivals & League | Observed vs estimated: upcoming picks marked unknown | **PASS** | `reports/evidence/R3.txt` | Pre-deadline picks marked `OBSERVED_PRIOR_GW` with `upcoming_transfers_known=False` and UI notice. |
| **R4** | Rivals & League | Risk posture (chase/protect/balanced) as tie-breaker | **PASS** | `reports/evidence/R4.txt` | Posture operates strictly as a bounded tie-breaker within 0.50 xP; never overrides ML rankings. |
| **C1** | Chip Strategy | Multi-GW chip calendar with benefit vs baseline | **PASS** | `reports/evidence/C1.txt` | Multi-GW chip calendar computes expected incremental gain vs no-chip baseline across remaining legal chips. |
| **C2** | Chip Strategy | Joint chip and transfer planning | **PASS** | `reports/evidence/C2.txt` | Transfer planner coordinates with chip schedule (e.g. rolling FTs before Wildcard). |
| **C3** | Chip Strategy | Strict recommendation threshold & Set 1 GW19 cutoff | **PASS** | `reports/evidence/C3.txt` | Uncertainty threshold gating (+3.0 net xP); strictly suppresses Set 1 chips after GW19 deadline. |
| **C4** | Chip Strategy | Track rivals' chips remaining and usage likelihood | **PASS** | `reports/evidence/C4.txt` | Rival chip usage history and remaining inventories incorporated directly into mini-league simulation. |
| **D1** | Decision Card | Per-GW Decision Card reading single game state | **PASS** | `reports/evidence/D1.txt` | Unified `/api/decision-card` endpoint reading authoritative `EffectiveManagerState`. |
| **D2** | Decision Card | Exportable/printable text/markdown summary | **PASS** | `reports/evidence/D2.txt` | Dedicated `/api/decision-card/export` endpoint and one-click copy button in UI. |
| **D3** | Decision Card | Atlas/Council UI simplicity & empty states | **PASS** | `reports/evidence/D3.txt` | Simplified Atlas/Council design system with graceful handling of demo mode, empty rivals, and passed deadlines. |
| **W1** | Win Probability | Correlated Monte Carlo simulation | **PASS** | `reports/evidence/W1.txt` | Shared players sampled once per trial; club defenders share Bernoulli clean-sheet draws (+4 pts). |
| **W2** | Win Probability | Rival future-behaviour model with clear assumptions | **PASS** | `reports/evidence/W2.txt` | Explicit rival behavioral models (template follower, high-xP captaincy) with visible assumptions. |
| **W3** | Win Probability | Side-by-side plan comparison & honest backtest | **PASS** | `reports/evidence/W3.txt` | Fast runtime (< 1.0s); historical backtest report saved to `reports/fix_pass_backtest.md` (mean Brier: 0.0677). |
| **Q1** | Quality & Tests | Fix ruff lint errors, pin ruff, full test suite | **PASS** | `reports/evidence/full-pytest.txt` | Pinned `ruff==0.16.10` in `pyproject.toml`; 0 lint errors (`All checks passed!`); 105 tests passing. |
| **Q2** | Quality & Tests | Cross-surface recommendation consistency test | **PASS** | `reports/evidence/Q2.txt` | `tests/test_recommendation_consistency.py` verifies 100% agreement across card, squad, optimizer, and chips. |
| **Q3** | Quality & Tests | Desktop and narrow mobile browser screenshots | **BLOCKED** | `reports/evidence/Q3.txt` | Headless console without display server or browser automation (Playwright/Selenium). Documented honestly per Rule 0.10. |
| **Q4** | Quality & Tests | Rewrite audit reports from fresh evidence | **PASS** | `reports/evidence/Q4.txt` | Reports and ledger completely rewritten strictly from fresh evidence files. All old claims purged. |

---

## 3. Environment & Security Verification

1. **Local Security**:
   - CORS strictly restricted to local loopback origins (`localhost:8000`, `127.0.0.1:8000`, `localhost:3000`).
   - Markdown rendered safely through local `DOMPurify`.
   - Local `.env` contains all user IDs and API keys; zero credential or ID input forms in the web app.
   - Pre-commit secret audit (`scripts/check_secrets.py`) passes with zero leaks.
2. **Offline Resilience**:
   - Runtime vendor assets (`tailwind.js`, `vue.global.prod.js`, `marked.min.js`, `purify.min.js`) are pinned and hosted locally in `web/static/js/`.
   - Stale cache fallback ensures uninterrupted operation during FPL API rate limits (429) or server outages (503).
