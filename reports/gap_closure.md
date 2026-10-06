# FPL Oracle: Gap-Closure Audit Matrix (Targeted Fix Pass)

**Audit Date**: 2026-10-06  
**Audited Baseline Commit**: `37ac87da281f4f723ee3e708178dcff402afbddc`  
**Evaluation Standard**: Rule 0.1 Strict Evidence-Gated Ledger (Every item backed by saved execution logs under `reports/evidence/<id>.txt`).  
**Test Suite Verification**: 105 / 105 automated unit and regression tests passing.

---

## 1. Executive Summary & Audited Principles

This document records the complete, evidence-gated resolution of every defect, missing feature, and integrity gap identified in the project audit:

1. **Zero Fabrication**: All literal evaluation metrics dictionaries, hardcoded upcoming projection samples, and simulated coverage numbers have been purged. Metrics are dynamically computed and logged directly to reproducible run artifacts.
2. **Model Learning (No Hard Constraints / Ceilings)**: Opponent defensive form, fixture difficulty, days of rest, and implied goal expectations are supplied as **continuous input features** to LightGBM models. Top players in peak form legitimately project 6–7+ expected points against elite defences; no post-hoc artificial capping or forcing exists.
3. **True Train/Serve Parity**: All 62 canonical features are extracted identically across training and live serving pipelines with bidirectional season-scoped opponent ID resolution and dynamic `days_rest` computation from fixture kickoff times.
4. **Stateful Transfer Optimization**: Replaced single-GW greedy transfers with a stateful 5-GW beam search tracking full squad state, bank balances, exact 50% profit selling prices ($P_{\text{sell}} = P_{\text{purchase}} + \lfloor (P_{\text{now}} - P_{\text{purchase}}) / 2 \rfloor$), free transfer rollover (1–5 banked FTs), and hit avoidance.
5. **Wired News & Availability Pipeline**: Live manager quotes and press conference extractions are routed through the unified `AvailabilityReconciler` into feature generation, with verified `shadow` vs `gated_active` operational isolation.
6. **2026/27 Rules Compliance**: Dual-set chip calendar enforcing Set 1 GW19 hard deadline, removal of Assistant Manager chip, DefCon +2 outfield defensive contributions, and correlated clean sheet Monte Carlo mini-league modeling.
7. **Unified Per-GW Decision Card**: Built single source of truth (`/api/decision-card`) combining transfers, Starting XI, captain/vice-captain, chip recommendations, rival proximity, and win probabilities, verified for 100% cross-surface consistency.

---

## 2. Model & Pipeline Gap Traceability (M1 – M8)

| Gap ID | Description | Initial Status (`37ac87d`) | Verified Resolution | Evidence File | Final Status |
|:---|:---|:---:|:---|:---|:---:|
| **M1** | Remove fabricated metrics & upcoming projection samples | **FAIL** | Deleted hardcoded `ablation_metrics` dict (`full_mae=1.45`, `gain=2.88%`) and canned `upcoming_sample` list in `train.py`. Metrics computed dynamically. | `reports/evidence/M1.txt` | **CLOSED** |
| **M2** | One honest evaluation, GW-boundary split & promotion gate | **FAIL** | Single `reports/model_eval.json` output; strict GW-boundary temporal splits; baseline-superiority promotion gate rejecting inferior candidates. | `reports/evidence/M2.txt` | **CLOSED** |
| **M3** | Real rolling-origin evaluation table by global GW | **FAIL** | Real temporal evaluation table across historical seasons and 2026-27 origins evaluated against identical rows with heuristic baselines. | `reports/evidence/M3.txt` | **CLOSED** |
| **M4** | Opponent feature fix: resolve numeric opponent IDs | **FAIL** | Built bidirectional `opp_id_to_name` / `name_to_opp_id` mapping in `features.py` from `master_history.csv` + bootstrap. Real opponent form used in training. | `reports/evidence/M4.txt` | **CLOSED** |
| **M5** | Live train/serve parity across all 62 columns & days_rest | **FAIL** | Serving features query last completed GW; `days_rest` dynamically calculated from kickoff timestamps; verified parity across single, blank, and double GWs. | `reports/evidence/M5.txt` | **CLOSED** |
| **M6** | Fix disciplinary card label sorting/index misalignment | **FAIL** | Replaced unsynchronized indexing with synchronous sort and merge keys on `(player_id, round)`. Verified via Alpha/Zulu deterministic and randomized tests. | `reports/evidence/M6.txt` | **CLOSED** |
| **M7** | Calibrated uncertainty intervals ($P_{10}$ / $P_{90}$) | **FAIL** | Implemented empirical residual quantile calibration on disjoint out-of-fold calibration block. Measures **78.74%** empirical coverage for nominal 80% intervals. | `reports/evidence/M7.txt` | **CLOSED** |
| **M8** | Dynamic feature ablation & purge dead odds claims | **FAIL** | Executed genuine two-pass feature ablation measuring **17.14% MAE reduction** when including fixture features. Purged inert external odds claims. | `reports/evidence/M8.txt` | **CLOSED** |

---

## 3. News & Availability Traceability (N1 – N8)

| Gap ID | Description | Initial Status (`37ac87d`) | Verified Resolution | Evidence File | Final Status |
|:---|:---|:---:|:---|:---|:---:|
| **N1** | Wire Gemini/reconcile directly into feature extraction and predict.py | **FAIL** | Reconciled availability probabilities passed directly into `extract_live_features_for_upcoming()` and `predict.py`. | `reports/evidence/N1.txt` | **CLOSED** |
| **N2** | Operational modes: shadow vs gated_active vs off | **FAIL** | Controlled mock test confirms injury quotes reduce xP in `gated_active` while leaving `shadow` mode identical to official baseline. | `reports/evidence/N2.txt` | **CLOSED** |
| **N3** | Replace handpicked reconcile probabilities with settings | **FAIL** | Exposed named settings in `FPLSettings` (`CONFIRMED_FIT_PROB`, `HIGH_CONF_PROB`, `MODERATE_PROB`, `DOUBTFUL_PROB`, `LOW_PROB`, `EXCLUDED_PROB`). | `reports/evidence/N3.txt` | **CLOSED** |
| **N4** | Adapter schema unification, full roster batching, budget boundary | **FAIL** | Unified element ID typing, batched full roster processing, and strict budget limit enforcement preventing boundary overruns. | `reports/evidence/N4.txt` | **CLOSED** |
| **N5** | Executable 20-case adversarial benchmark runner | **FAIL** | Executed `scripts/run_news_benchmark.py`: 20/20 test cases passed (100%), 0 prompt injection breaches. | `reports/evidence/N5.txt` | **CLOSED** |
| **N6** | DNS-resolving SSRF protection & manual redirect validation | **FAIL** | Resolved hostnames via DNS, blocking private RFC-1918, loopback, and metadata IPs with step-by-step redirect validation. | `reports/evidence/N6.txt` | **CLOSED** |
| **N7** | Render verbatim quotes and source links in web UI | **FAIL** | Rendered verbatim quote text, source URLs, applied/shadow status badges, and minutes delta in UI with DOMPurify sanitization. | `reports/evidence/N7.txt` | **CLOSED** |
| **N8** | Remove inert search & odds keys from config | **FAIL** | Purged dead keys (`TAVILY_API_KEY`, `BRAVE_API_KEY`, `ODDS_API_KEY`) from `config.py` and `.env.example`. Zero dead configuration. | `reports/evidence/N8.txt` | **CLOSED** |

---

## 4. Transfers & Mini-League Traceability (T1 – T9, R1 – R4, W1 – W3, C1 – C4)

| Category | Gap ID | Description | Verified Resolution | Evidence File | Final Status |
|:---|:---|:---|:---|:---|:---:|
| **Transfers** | **T1–T9** | Stateful 5-GW beam search optimizer | Tracks squad, bank, exact 50% selling prices, 1-5 banked FTs, hits (-4), club limit, robustness re-ranking (win rate >= 0.70), dynamic roadmap, and plan stability. 11/11 tests pass. | `reports/evidence/T1.txt` – `T9.txt` | **CLOSED** |
| **Rivals** | **R1–R4** | Paginated standings & proximity rivals | Paginates up to 10 pages (500 teams); selects all managers ahead plus within 30 pts below; tracks 2026/27 dual-set chips; pre-deadline picks marked unknown; bounded risk posture tie-breaker. | `reports/evidence/R1.txt` – `R4.txt` | **CLOSED** |
| **Chips** | **C1–C4** | Multi-GW chip calendar & Set 1 cutoff | Computes expected gain vs no-chip baseline; joint transfer/chip planning; strict Set 1 GW19 hard cutoff; tracks rival remaining chips in simulation. | `reports/evidence/C1.txt` – `C4.txt` | **CLOSED** |
| **Win Prob** | **W1–W3** | Correlated Monte Carlo simulation | Shared players sampled once per trial; club teammates share Bernoulli clean-sheet draws (+4 pts); rival behavioral model; historical backtest report (mean Brier score: 0.0677). | `reports/evidence/W1.txt` – `W3.txt` | **CLOSED** |

---

## 5. Decision Card & Quality Traceability (D1 – D3, Q1 – Q4)

| Gap ID | Description | Verified Resolution | Evidence File | Final Status |
|:---|:---|:---|:---|:---:|
| **D1** | Per-GW unified Decision Card | `/api/decision-card` endpoint reading single authoritative `EffectiveManagerState` and presenting complete gameweek decision. | `reports/evidence/D1.txt` | **CLOSED** |
| **D2** | Exportable markdown summary | `/api/decision-card/export` endpoint returning clean text/markdown and UI copy button. | `reports/evidence/D2.txt` | **CLOSED** |
| **D3** | Atlas/Council UI simplicity & empty states | Clean unified card layout handling demo mode, empty rivals, exhausted chips, and passed deadlines gracefully. | `reports/evidence/D3.txt` | **CLOSED** |
| **Q1** | Fix all ruff lint errors, pin ruff, full test suite | Pinned `ruff==0.16.10` in `pyproject.toml`; 0 lint errors (`All checks passed!`); 105 tests collected and executed. | `reports/evidence/full-pytest.txt` | **CLOSED** |
| **Q2** | Cross-surface recommendation consistency test | `tests/test_recommendation_consistency.py` verifies 100% agreement across decision card, squad, optimizer, and chips for captain, transfers, chips, and formation. | `reports/evidence/Q2.txt` | **CLOSED** |
| **Q3** | Desktop and mobile screenshots | Headless environment without display server or browser automation (Playwright/Selenium). Documented honestly per Rule 0.10. Static assets and endpoints verified. | `reports/evidence/Q3.txt` | **BLOCKED (ENVIRONMENT LIMITATION)** |
| **Q4** | Audit reports and README rewrite | Completely rewritten audit reports and ledger strictly from fresh evidence files. Purged all prior unverified claims. | `reports/evidence/Q4.txt` | **CLOSED** |
