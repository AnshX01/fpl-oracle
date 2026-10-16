# FPL Oracle 2026/27: Improvement Specification (v1)

> **Audience:** an autonomous coding agent (Antigravity CLI) working inside the `fpl-oracle` repository.
> **Input document:** `fpl_oracle_system_architecture.md` (the "Architecture Doc"). This spec modifies that system; it does not replace it.
> **Written:** 10 Oct 2026. Rules facts were verified against 2026/27 announcements (sources in Appendix B).
> **Hard constraint:** the existing 426 tests must stay green at every commit unless a work item explicitly says a test encodes a wrong rule (then fix the test in the same commit and say so in the commit message).

---

## 0. How the agent must work (read first)

### 0.1 Operating protocol
1. **Start by reading, not writing.** Read the Architecture Doc, then the real code under `src/fpl_oracle/`. Paths in this spec come from the Architecture Doc; if a path or symbol does not exist, search for the closest equivalent and record the mapping in `docs/improvement_log.md`.
2. **Work in the order given in §11.** Do not start a phase before its dependencies pass their acceptance criteria.
3. **One work item = one branch/commit series.** Name commits `WI-<id>: <summary>`. Keep each diff reviewable (<600 changed lines where possible).
4. **Every work item has acceptance criteria. A work item is done only when they are demonstrably met** (test output, metric file, or both). Write results to `docs/improvement_log.md` (what changed, metrics before/after, anything surprising).
5. **Feature-flag behavioural changes** that alter recommendations (`config/flags.yaml`, default OFF until the work item's acceptance gate passes, then ON). Pure bug fixes need no flag.
6. **Never silently change numbers users see.** If a fix changes projections, log the before/after on a fixed fixture set (`tests/fixtures/golden_gw/`).
7. **Determinism:** every stochastic component takes an explicit seed; every run artifact records `git_sha`, `config_hash`, `data_hash`, `feature_schema_hash`, `rules_version`.
8. **Don't trust this spec blindly either.** Where it says "verify", write a test or a script that verifies against live FPL API data or official rules text, and record the result.

### 0.2 Stop-and-ask conditions (halt, write the question into `docs/open_questions.md`)
- A rules fact in §1 contradicts live FPL API data.
- A change would make the test suite fail for a reason other than a wrong-rule test.
- A work item needs a paid API key or a data source whose terms you cannot confirm allow this use.
- Acceptance metrics get worse than baseline after two honest attempts.

### 0.3 Non-goals
- No UI redesign. No new chips or game modes. No auto-submitting transfers to the FPL site.
- No replacing LightGBM with deep nets. Model class changes only where a work item says so.

---

## 1. Verified 2026/27 rules and corrections to the Architecture Doc

### 1.1 Facts verified from 2026/27 announcements
| # | Fact | Implication |
|---|------|-------------|
| R1 | Two sets of four chips (WC, FH, TC, BB); one chip per GW; Set 1 must be played before the **GW19 deadline: 13:30 GMT, Sat 2 Jan 2027**; Set 2 starts GW20 (6 Jan 2027). No carry-over. | Read the deadline from the API (`events[].deadline_time`), never hard-code. |
| R2 | **Free Hit cannot be played in GW1.** If the first FH is played in GW19, the second FH cannot be played in GW20. | Add as hard constraints in the chip planner/MILP. |
| R3 | Max **5 banked free transfers** (confirmed unchanged). No extra AFCON transfers this season. | The 5-FT design in the Architecture Doc is correct. |
| R4 | **Banked FTs are preserved through a Wildcard or Free Hit.** (Exact rule about whether a chip week still adds +1 FT is described loosely by sources.) | Make the chip-week FT rule a single configurable function and verify empirically (WI-1.3). |
| R5 | DefCon: **DEF 2 pts at >=10 CBIT** (clearances, blocks, interceptions, tackles); **MID/FWD 2 pts at >=12 CBIRT** (adds ball recoveries). Max 2 pts per match. Unchanged for 2026/27. | Position-specific thresholds; model count of actions, not just a binary flag. |
| R6 | **BPS changes 2026/27:** no BPS deduction for being tackled; CBI earns 1 BPS per **3** actions (was per 2); GK: 2 BPS for any save, +1 for a save inside the box (no more outside-box bonus), +1 for saving a big chance; penalty save 7 BPS (+1 big-chance = 8 total). | Bonus model trained on older seasons has a **rules-era shift**; handle explicitly (WI-3.6). |
| R7 | Points, rank and mini-leagues update **live**; projected bonus is added after 20 minutes of each match; the **gameweek locks at 09:00 UK time the day after the last match**. | Retrain/ingestion jobs must key on finalised data, not on "last match ended" (WI-7.2). |
| R8 | The Assistant Manager chip is **not** in 2026/27. | Grep the codebase for any residual reference. |
| R9 | FPL now ships an official price-change predictor. | `price_change_watch` can use it as a benchmark if exposed. |
| R10 | Fantasy Football Hub's chip guide states no blank or double gameweeks are expected before the GW19 deadline. (Treat as a claim from one source; derive from the live fixtures feed.) | The Set 1 chip thresholds in the Architecture Doc were written for DGW/BGW weeks and are probably mis-set for Set 1 (see WI-5.2). |

### 1.2 Defects found in the Architecture Doc (fix list)
| ID | Where | Problem | Required fix |
|----|-------|---------|--------------|
| D1 | §2 Scoring Ensemble, `Rate_goal` | Goalkeeper goal value is `10`. Official value is **6 for GKP and DEF**, 5 MID, 4 FWD (verify against the official scoring table). | Set GKP=6, add a rules-contract test (WI-1.1). |
| D2 | §4.2 FT banking formula | `min(5, Banked - TransfersMade + 1)` goes negative after a hit. | `ft_next = min(5, max(0, ft - free_used) + 1)` where `free_used = min(transfers, ft)`. Chip weeks per R4. |
| D3 | §2 ensemble | Omits own goals (-2) and penalties missed (-2) though features track them. | Add expectation terms (WI-1.2). |
| D4 | §3 header and table | Says "57-column schema" but the feature table lists ~55. | Make the doc generated from `FEATURE_SCHEMA` so it cannot drift (WI-7.3). |
| D5 | §3 "Dixon-Coles" | What is described is an independent Poisson attack x defence model with hard-coded league mean 1.35 and fixed home factors. Dixon-Coles proper adds a low-score dependence correction (rho) and time-decay weighting, with parameters fitted by likelihood. | Rename or implement properly (WI-3.1). |
| D6 | §4.4 hit policy | "Net horizon gain >= 4.0" does not say whether the -4 is already subtracted. Fixed thresholds 0.3 xP and option value 0.4-0.8 xP sit below projection noise. | Define precisely, replace with probability-of-improvement (WI-2.5). |
| D7 | §5 chip thresholds | 4.0/8.0/10.0/8.0 xP gates ignore opportunity cost and the GW19 expiry. | Joint chip optimisation (WI-5.1/5.2). |
| D8 | §6 Monte Carlo | Single Gamma for all "other" points, with `max(0, xP - E[App] - E[CS])` clipping. This loses negative events, discrete haul structure, teammate correlation, and biases the mean when the clip binds. | Scoreline-based simulator (WI-4.1). |
| D9 | §2 quantiles | One pooled set of z-quantiles (-0.806 / 1.009) for all positions and xP levels. | Per-stratum conformalised quantiles (WI-3.8). |
| D10 | §2 DefCon | "Outfield" +2 only, one binary model. | Per-position thresholds, count model (WI-3.5). |
| D11 | §9 / architecture | Hard-coded "GW5 Current Position" and "GW12-18 peak DGW" in the chip timeline are assertions about the future. | Remove; compute from fixtures. |
| D12 | §8 | Claims of "100% uptime", "never fails", "100% compliant" are unprovable. Absolute `file:///C:/Users/...` links break outside one machine. | Reword; use repo-relative links (WI-7.3). |

---

## 2. Phase 0: Evaluation harness (blocks everything else)

**Why:** 426 passing tests show the code does what was intended, not that projections are good. Nothing below can be accepted without numbers.

### WI-0.1 Walk-forward backtest runner
**Files:** `src/fpl_oracle/eval/backtest.py`, `scripts/run_backtest.py`, `reports/eval/` (output).
**Spec:**
- Walk forward by gameweek: for each GW `g` in a test window, train only on data finalised before the `g` snapshot, predict `g` (and `g+1..g+3` for multi-horizon), compare to realised points.
- Use the same serving feature path as production (serving mode, not training mode), so train/serve skew shows up in the numbers.
- Cover at least the last 2 completed seasons plus the live 2026/27 GWs. Retraining every GW is expensive; allow `--retrain-every k` with `k=1` as the reference setting and `k=4` for fast iteration.
**Metrics (per position, per horizon, per fixture type single/DGW):**
- MAE, RMSE, Spearman rank correlation, top-decile precision (does the model's top 10 per position contain the realised top performers).
- Stratified error by realised-return bucket: zeros (did not play), blanks (<=2), tickers (3-4), haulers (>=5).
- Probabilistic: pinball loss at 0.1/0.5/0.9, empirical coverage of the P10-P90 band, CRPS if full samples exist, reliability tables for `P(play)`, `P(min>=60)`, `P(CS)`, `P(DefCon)`.
**Two evaluation views, always both reported:** (a) all player-GW rows where the player was available at snapshot time; (b) only rows where the player played. (Reason: predicting zeros for the bench inflates MAE in view (a) and hides unfair comparisons.)
**Acceptance:** `python scripts/run_backtest.py --seasons 2024-25 2025-26 --window last-20-gw` produces `reports/eval/<date>/summary.json` and `summary.md`, runs in <30 min with `--retrain-every 4`, and is bit-reproducible with a fixed seed.

### WI-0.2 Baselines
Implement and report these side by side with the model:
1. Zero/hold-out mean (season-to-date points per match).
2. Rolling mean of last 5 appearances.
3. Official FPL `ep_next` (only available as live snapshots; see below).
4. Position-and-price-band mean (cold-start prior).
**Add `ep_next` snapshotting now:** extend the `holdout_forward` job to store `ep_next` and `ep_this` for every player at each pre-deadline snapshot into `data/snapshots/`. This becomes a free, honest benchmark within weeks. Also snapshot Understat-based or odds-based inputs when they exist (WI-3.2).
**Acceptance:** report table shows model vs all baselines; if the model does not beat baseline 2 on RMSE and Spearman for every position, open an issue in `docs/open_questions.md` before continuing.

### WI-0.3 Noise-floor context (reporting only)
Include in the report a "reference scale" note. FPL Review's analysis of a simulated perfect model (knows true expected value, faces random outcomes) found RMSE about 2.81 and MAE about 1.96 per player-GW on a pooled sample (R2 about 0.15). Meaning: weekly points are mostly noise, so small MAE changes are not evidence. All model comparisons must use paired significance tests (WI-3.9), not raw deltas. The OpenFPL paper (arXiv 2508.09992) is a good open reference on stratified evaluation (zeros/blanks/tickers/haulers) and position-specific ensembles; use it for methodology, not as a numeric target.

### WI-0.4 CI gate
**Spec:** `make eval-smoke` runs a 4-GW backtest on a frozen fixture; fails CI if RMSE degrades >2% or coverage leaves [75%, 85%] for the nominal 80% band. Update thresholds only through a reviewed commit.

---

## 3. Phase 1: Rules correctness (small, fast, high value)

### WI-1.1 Centralised rules module and contract tests
**Files:** `src/fpl_oracle/rules/scoring.py`, `config/rules_2026_27.yaml`, `tests/rules/test_scoring_contract.py`.
**Spec:** one source of truth for every scoring value and rule constant. Each entry carries `value`, `source_url`, `verified_on`.
```yaml
points:
  appearance_lt60: 1
  appearance_ge60: 2
  goal: {GKP: 6, DEF: 6, MID: 5, FWD: 4}   # D1: GKP was 10 in the Architecture Doc
  assist: 3
  clean_sheet: {GKP: 4, DEF: 4, MID: 1, FWD: 0}   # needs >=60 min
  goals_conceded_per_minus1: {GKP: 2, DEF: 2}
  saves_per_plus1: 3
  penalty_save: 5
  penalty_miss: -2
  own_goal: -2
  yellow: -1
  red: -3
  bonus: [3, 2, 1]
  defcon: {points: 2, cap: 2, threshold: {DEF: 10, MID: 12, FWD: 12, GKP: null}}
transfers:
  max_bank: 5
  hit_cost: 4
chips:
  set1_last_gw: 19          # read the actual deadline from API; this is a fallback
  free_hit_forbidden_gw: [1]
  max_chips_per_gw: 1
```
- Every module that uses a points value must import from here. Add a `grep`-based test that fails if the literals `10 * ` / goal-rate dictionaries reappear outside this module.
- **Contract test:** for a set of ~50 historical player-GW rows (fixture file, covering every position and event type), recompute `total_points` from raw event columns using only `rules/scoring.py` and assert it equals the FPL API's `total_points`. This catches D1 and any other value error. (Include DefCon and bonus rows. If a column is missing from the stored history, extend ingestion.)
**Acceptance:** contract test passes on all stored historical seasons (measured as exact-match rate; any mismatches listed and explained, e.g. historical rule eras handled via `rules_version` per season).

### WI-1.2 Ensemble completeness
**File:** `src/fpl_oracle/ml/ensemble.py`.
- Fix D1 via the rules module.
- Add `- 2 * E[own_goals]` and `- 2 * E[penalties_missed]` terms. Penalty-miss expectation = `P(takes penalty) * P(penalty awarded) * miss_rate`; if too sparse, shrink to a positional/league prior.
- Verify conceded deduction applies to GKP/DEF only; clean-sheet requires >=60 mins; DefCon never applies to GKP (verify against live `element_stats` that GKPs show no DefCon points; record the result).
**Acceptance:** back-computed `xP` decomposition sums exactly to the final `xP` (property test); contract test from WI-1.1 passes; golden-fixture diff logged.

### WI-1.3 Free-transfer state machine
**Files:** `src/fpl_oracle/rules/transfers.py`, `tests/rules/test_transfers.py`.
**Spec:** a pure function `next_free_transfers(ft, transfers_made, chip, *, config) -> (ft_next, hit_points)`.
- Normal week: `free_used = min(transfers_made, ft)`, `hit_points = 4 * max(0, transfers_made - ft)`, `ft_next = min(5, ft - free_used + 1)`.
- WC/FH week: no hit; banked FTs preserved (R4). **Open detail:** whether the following week adds +1. Implement the rule as `config.chip_week_ft_rule in {"preserve", "preserve_plus_one"}`, default `"preserve"`.
- **Empirical verification step:** from the FPL API entry history endpoints for a sample of managers who played WC/FH in 2025/26 (and any 2026/27 chip weeks so far), compute the FT counts before and after chip weeks (`entry/{id}/history/` and picks metadata) and determine which rule the data supports. Record the evidence in `docs/improvement_log.md` and set the default accordingly.
- Also enforce R2 (no FH in GW1; no FH in GW20 if FH used in GW19) and max one chip per GW.
**Tests:** table-driven cases including: 1 FT and 3 transfers (hit 8, next FT = 1); 5 FT and 0 transfers (next = 5); 5 FT and 2 transfers (next = 4); chip weeks.
**Acceptance:** all table cases pass; `optimise_transfers` and any simulator use this single function (grep-enforced).

### WI-1.4 Selling-price oracle
`sell_price(purchase, current) = purchase + floor((current - purchase) * 10 / 2) / 10` when `current > purchase`, else `current`. (Work in integer tenths-of-a-million to avoid float error.) Unit-test with: +0.1 rise (sell at purchase), +0.2 (+0.1), +0.3 (+0.1), price falls (full fall passes through).

### WI-1.5 Pre-commit rules audit
Run a one-off script that, for the current season's finished GWs, recomputes points for **all** players and reports the mismatch rate vs the API. Target: 0 unexplained mismatches.

---

## 4. Phase 2: Multi-period transfer optimiser (replace beam search as primary)

**Problem:** with up to 5 banked FTs, 3-5 transfer weeks can be hit-free, yet the beam only expands 0/1/2 transfers; chips, FT banking, and selling prices cannot be optimised jointly. A multi-period MILP is the standard exact approach and is used in open FPL tools and the literature (see Appendix B).

### WI-2.1 Formulation
**Files:** `src/fpl_oracle/optimise/multiperiod.py`, `tests/optimise/test_multiperiod.py`. Keep the existing beam search as `optimise/transfers_beam.py` (fallback and cross-check).
**Solver:** keep PuLP as the modelling layer; add **HiGHS** as the default backend (`pulp.HiGHS` / `highspy`) and keep CBC selectable. Benchmark both (WI-2.4).

**Sets and data**
- `T = {t0, ..., t0+H-1}` horizon (default H=8, configurable 1-12).
- `P` candidate players (pruning in §WI-2.3), `G` clubs, `pos(p)`, `club(p)`.
- `xP[p,t]` expected points for GW t (DGW already summed), `xPtc[p,t]` same (used for captain terms).
- Initial squad `x0[p]` in {0,1}; `sell0[p]` current selling price; `bank0`; `ft0`; chips available per half (from API).
- `buy[p,t]` purchase price assumed at t (default: current price held constant; optional price-trend adjustment).

**Decision variables (binary unless noted)**
- `x[p,t]` in squad after transfers in GW t; `s[p,t]` starts; `c[p,t]` captain; `b[p,t]` selected bench slots optional (see bench weight).
- `in[p,t]`, `out[p,t]` transfers.
- `ft[t]` integer in [1,5] free transfers available at start of GW t; `h[t]` >= 0 integer, hits count (each costs 4).
- `wc[t], fh[t], bb[t], tc[t]` chip binaries.
- `bank[t]` continuous >= 0.
- For FH weeks: auxiliary squad `xfh[p,t]`, `sfh[p,t]`, `cfh[p,t]` used for that week's scoring only; the persistent squad `x[p,t]` passes through unchanged.

**Constraints (for every t)**
1. Squad flow: `x[p,t] - x[p,t-1] = in[p,t] - out[p,t]` with `x[p,t0-1] = x0[p]`. For FH weeks the persistent squad is unchanged (`x[p,t] = x[p,t-1]`), and the FH squad is free: `sum_p xfh[p,t] = 15`.
2. Squad shape: `sum_p x[p,t] = 15`; 2 GKP, 5 DEF, 5 MID, 3 FWD; `<= 3` per club.
3. Lineup: `s[p,t] <= x[p,t]`; `sum s = 11`; 1 GKP; DEF in [3,5]; MID in [2,5]; FWD in [1,3]; `c[p,t] <= s[p,t]`; `sum_p c[p,t] = 1`.
4. Budget: `bank[t] = bank[t-1] + sum_p sell[p,t]*out[p,t] - sum_p buy[p,t]*in[p,t]`, `bank[t] >= 0`. `sell[p,t]` uses the oracle (WI-1.4) for players owned at t0; for in-horizon purchases assume `sell = buy` (conservative; document it).
5. Transfers and hits (no chip): `u[t] = sum_p in[p,t]`; `h[t] >= u[t] - ft[t]` (only enforced when no WC/FH that week: use Big-M with `M = 15`: `h[t] >= u[t] - ft[t] - 15*(wc[t] + fh[t])`).
6. FT dynamics: `ft[t+1] <= 5`; `ft[t+1] <= ft[t] - (u[t] - h[t]) + 1` for normal weeks; for WC/FH weeks apply the configured chip-week rule (WI-1.3) via Big-M. All bounds point in the direction the objective already prefers, so inequalities suffice. `ft[t0] = ft0` (read from the entry; do not assume).
7. Chips: `sum_chip chip[t] <= 1` per GW; each chip type at most once per half (Set 1: t <= 19; Set 2: t >= 20, using actual availability from the API, not assumptions); `fh[GW1] = 0`; `fh[19] + fh[20] <= 1`.
8. Chip effects in the objective: BB adds all 15 players' points (bench values `xP` weighted by `bb[t]`); TC replaces captain bonus `1x` with `2x` (use a linearised product: `tcc[p,t] <= c[p,t]`, `tcc[p,t] <= tc[t]`, `tcc[p,t] >= c[p,t] + tc[t] - 1`).

**Objective**
```
maximise  sum_t gamma^(t-t0) * [ sum_p xP[p,t]*(s[p,t] + c[p,t] + tcc[p,t])
                                 + bench_value[t] + BB_extra[t]
                                 - 4*h[t] ]
          + V_ft(ft[T+1])        # concave terminal value of banked FTs
          + lambda_bank * bank[T]  # small weight; default 0.05 xP per 1.0m
```
- `gamma` default 0.95, set by backtest (WI-2.4), not by taste. (An open-source FPL planner re-tuned discount on four seasons and found the best value was not obvious; treat it as a hyperparameter.)
- `bench_value[t]`: expected auto-sub points, `sum_k w_k * xP[bench_k,t]` for bench slots k with weights `w_k` **estimated empirically** (regress realised auto-sub points on bench xP and starters' `1-P(play)`). Until estimated, use small placeholders and flag them in the log.
- `V_ft(n)`: concave piecewise-linear terminal value for the n-th banked FT, **measured** by comparing plans with n vs n+1 FTs in a rolling-horizon backtest (replaces the hand-set +0.4 to +0.8 xP).

### WI-2.2 Plan extraction and alternatives
- Return the week-1 actions (the only ones to act on) plus the provisional later-week plan.
- Produce the comparison set required by the assistant: **roll** (force 0 transfers GW t0), **best 1-transfer**, **best with hit**, **best chip-using plan** per chip. Do this by re-solving with extra constraints (not by post-hoc filtering).
- Return top-K distinct week-1 moves using no-good cuts on `in/out` variables of GW t0.
- Keep the existing tool JSON schema; add fields, don't remove.

### WI-2.3 Candidate pruning
Keep per position: top N by horizon-weighted xP (default N=45 DEF/MID, 25 FWD, 15 GKP), plus all currently owned players, plus the cheapest 5 enablers per position by xP/price. Report in the log how often the unpruned optimum differs from the pruned one (target: never, tested on 20 historical GW states using a larger N as reference).

### WI-2.4 Benchmark and tune
- Runtime target: H=8, 600-player pool pruned to ~250, 1 FT, no chip fixed: **<60 s** on a laptop with a MIP gap <= 0.5%. Report CBC vs HiGHS times.
- Rolling-horizon backtest (act on week 1, re-solve each GW) over the last two seasons comparing: (a) existing beam, (b) multi-period MILP, (c) one-week greedy. Report total realised points and a paired comparison. Tune `H`, `gamma`, `V_ft`, `bench weights` on seasons not used for final reporting. The MILP's objective value must be >= the beam's on every shared test instance (it searches a superset).
**Acceptance:** all of: property tests below pass; runtime target met; realised-points comparison logged (no requirement that MILP wins in realised points, but a result that is worse than the beam by a paired CI excluding zero blocks the flag from turning on).

### WI-2.5 Hit policy: replace fixed thresholds
- Compute, for each candidate plan vs the roll plan, `Delta = J(plan) - J(roll)` with hit costs **already included** in `J`.
- Use scenario samples from the simulator (WI-4.1) to estimate `P(Delta > 0)` and the mean/quantiles of `Delta`.
- Verdicts: `JUSTIFIED` if `P(Delta>0) >= 0.7` and mean `Delta >= 2` (tune on backtest), `MARGINAL` if between, `AVOID` otherwise. Print the numbers used.
- Stability threshold: express as a shrinkage towards "no change", not a flat 0.3 xP; remove the constant, replace by the probability criterion above.

### Property tests required (all in `tests/optimise/`)
1. **Equivalence at H=1, no transfers:** equals the existing lineup optimiser on 50 random squads.
2. **Dominance:** MILP objective >= beam objective on 30 historical states.
3. **Rules oracle agreement:** replay the MILP's chosen plan through `rules/transfers.py`; hits, FT counts and bank match exactly.
4. **Feasibility:** squad composition, club cap, formation, budget at every GW.
5. **Chip logic:** never two chips in a GW; FH not in GW1; no FH in both GW19 and GW20; Set 1 chips never scheduled after GW19.
6. **Brute force:** on a toy league (3 clubs, 4 positions reduced, H=2) enumerate all legal plans and match the MILP optimum.

---

## 5. Phase 3: Projection model upgrades

### WI-3.1 Proper Dixon-Coles team model (or honest renaming)
**Files:** `src/fpl_oracle/ml/teamgoals.py`, `data/features.py`.
- Fit attack `alpha_i`, defence `delta_i`, home advantage `gamma`, and low-score parameter `rho` by weighted likelihood: `P(x,y) = tau_rho(x,y;lambda,mu) * Pois(x;lambda) * Pois(y;mu)` with `tau(0,0)=1-lambda*mu*rho`, `tau(0,1)=1+lambda*rho`, `tau(1,0)=1+mu*rho`, `tau(1,1)=1-rho`, else 1.
- Time-decay weights `w = exp(-xi * days_ago)`. The original paper used about 0.0065/day (half-life about 107 days); a recent re-estimate on Serie A found about 0.002/day. **Do not hard-code either:** choose `xi` by grid search on a validation period using ranked probability score (RPS).
- Constrain `rho` so `tau > 0` for realistic `lambda, mu` (bounds around [-0.5, 0.4] work; clip each `tau` to a small positive floor when building the grid).
- Prefer a simple regularisation (ridge toward 0 on attack/defence) so newly promoted teams don't produce extreme values. Promoted teams: initialise from a pooled "promoted team" prior.
- Output per fixture: `lambda`, `mu`, full scoreline grid (0..10 x 0..10, renormalised), `P(CS_home)`, `P(CS_away)`, `E[GC]`, `P(win/draw/loss)`.
- **Replace** the hard-coded `1.35` and fixed 1.10/0.90 home multipliers; keep the old calculation as a feature (`legacy_implied_*`) for A/B.
**Acceptance:** out-of-sample RPS and log-loss on 1X2 beat the legacy implied-rate approach on the last 2 seasons; calibration table for `P(CS)` shown.

### WI-3.2 Market-implied features (bookmaker odds)
**Why:** closing market prices incorporate team news and information a rolling-xG model cannot see; the odds-based goal expectancy is widely used in FPL tooling (clean-sheet odds, anytime-scorer odds). One public study of Serie A explicitly asks whether a structural model adds anything to the closing price, so default to the market as the strongest prior and let the structural model earn its weight.
**Data (ToS-aware, provider-abstracted):**
- Historical match-level odds (1X2, over/under 2.5, Asian handicap) for England are available free from football-data.co.uk (bookmaker columns B365, PS, etc., with closing odds in recent seasons). Use via CSV download or the `soccerdata` package's MatchHistory source. These train the match-level features.
- **Player-level odds (anytime scorer) and clean-sheet markets have no free deep history.** Therefore: build `OddsProvider` (interface) with adapters for whichever free or paid API the user chooses; start a scheduled `odds_snapshot` job **now** that stores pre-deadline snapshots (timestamped, raw), so a training history accumulates. Until history exists, use player-odds only in a post-hoc **shrinkage prior** (below), not as a trained feature.
- Check each provider's licence/terms before wiring (record in `docs/data_sources.md`). Some paid history starts only from late 2025.
**Method:**
1. De-vig 1X2 and totals using the Shin method or the power method (implement both; choose by validation log-loss).
2. Invert to implied `(lambda_home, mu_away)` via least-squares on the scoreline grid; with 1X2 + O/U 2.5 jointly, solve for goals and `rho` together (the open-source `penaltyblog.goal_expectancy_extended` does this; reimplement or depend on it after checking its licence).
3. Add features: `mkt_lambda`, `mkt_mu`, `mkt_cs_prob`, `mkt_win_prob`, `mkt_total_goals`; training rows use historical closing odds as a proxy for pre-deadline odds (document the bias: closing is later than the FPL deadline; also store a "pre-deadline" odds snapshot going forward and compare).
4. Blend: `lambda_final = w * lambda_market + (1-w) * lambda_structural`, with `w` fitted on validation per horizon (market weight decays for far-future fixtures where odds are missing; fall back to structural only).
5. Player-level shrinkage (once snapshots exist): blend model `P(player scores)` with de-vigged anytime-scorer probability, weight chosen on forward-collected data after >= 8 GWs.
**Acceptance:** (a) adding market features reduces out-of-sample team-goal log-likelihood and `P(CS)` Brier score vs without; (b) player-level xP RMSE for DEF/GKP (clean-sheet-driven) improves or is unchanged within CI; (c) missing-odds fallback exercised by a test.

### WI-3.3 Player share model and reconciliation
**Goal:** make player-level expectations coherent with team-level expectations.
- Model each player's share of team goals and assists given on-pitch (conditional on playing), e.g. Dirichlet-multinomial or a softmax over teammates with features: shot share, xG share (Understat), penalty/set-piece roles, position, minutes expectation.
- **Reconciliation step:** after producing `E[goals_p]`, rescale so that `sum_p E[goals_p] * (1 / P(player on pitch share))` aligns with team `lambda` (within tolerance); log the scaling factors. Same for assists (assist rate = fraction of team goals that are assisted, estimated).
- Features from the FPL API: `penalties_order`, `direct_freekicks_order`, `corners_and_indirect_freekicks_order`.
**Acceptance:** for every fixture the sum of player `E[goals]` is within 5% of team `lambda` (property test on live data); player-level RMSE for MID/FWD not worse than before.

### WI-3.4 Minutes and availability additions
- Add features: midweek European fixture within 4 days (needs fixtures source), `manager_changed_recently`, `days_since_return_from_injury`, `chance_of_playing_next_round` (API), squad-depth proxy (number of same-position teammates with >=60% starts).
- Keep the **single-reconciler invariant**: only `ReconciledAvailability` modulates features; add a test that scans the pipeline for secondary discounts.
**Acceptance:** `P(starts)` Brier score and `E[minutes]` MAE improve or stay equal on the backtest; reliability plot for `P(play)` within +/- 3 pp per decile (with >= 200 samples per decile).

### WI-3.5 DefCon model rebuild (D10)
- Target: `defensive_contribution_actions` per match (CBIT for DEF, CBIRT for MID/FWD). If the API exposes `defensive_contribution` per match use it; if not, reconstruct from per-match clearances, blocks, interceptions, tackles, recoveries, noting any gap.
- Model counts with a negative-binomial (or Poisson with a validated dispersion) conditional on minutes; `P(DefCon) = P(N >= threshold_pos)` using R5 thresholds from the rules module. Cross-check against the existing binary classifier; keep the better-calibrated one per position.
- Features: opponent possession share / expected shots against (more defending in games where the team is under pressure), team style, role (full-back vs centre-back), minutes, home/away.
- Remove the uniform 5x weight; instead use `is_2026_27` as a feature plus time-decayed weights, and compare.
- GKP: DefCon is 0 by rule; verify against data and document.
**Acceptance:** per-position reliability (decile) for `P(DefCon)` within +/- 3 pp; Brier score improves vs the current model; expected DefCon points sum matches realised frequencies over the last 10 GWs within 5%.

### WI-3.6 Bonus model and the BPS rules-era shift (R6)
- Add `bps_rules_version` feature (2024-25, 2025-26, 2026-27). Where per-match raw actions exist, **recompute BPS under the 2026/27 weights** for history (tackled penalty removed; CBI per 3; GK save weights) to build a consistent training target-aligned feature set; where raw components don't allow exact recomputation, use the era flag and shrink historical estimates.
- Bonus is **zero-sum per match** (3/2/1 to the top three BPS, with tie rules). Predicting independent per-player `E[bonus]` breaks this. Handle in two ways: (a) post-hoc constrain `sum E[bonus]` per fixture <= 6 via proportional scaling, (b) within the simulator (WI-4.1) draw BPS and allocate bonus exactly.
- Evaluate separately for GKP, full-backs, and DEF/CB: R6 specifically changes their bonus prospects, and the live season now gives fresh evidence.
**Acceptance:** per-position mean predicted bonus vs realised over 2026/27 finalised GWs within 10% (report n); fixture-level sum constraint holds.

### WI-3.7 Cold start and shrinkage
- Rolling stats are noisy for few-minutes players. Replace raw rolling rates with empirical-Bayes shrunk rates: `rate_shrunk = (n*rate_player + k*rate_prior)/(n+k)` with `n` in minutes and `k` chosen per stat by validation; prior = position x price-band x team-strength group.
- New signings, promoted teams, and players returning from long injuries: separate flag features plus the prior.
**Acceptance:** error on players with < 450 minutes in the lookback window improves (stratified report).

### WI-3.8 Quantiles: per-stratum conformalised intervals (D9)
- Replace pooled `z10/z90` with **conformalised quantile regression** (CQR; Romano, Patterson, Candes 2019): fit quantile models (LightGBM quantile objective) at 0.1 and 0.9 then calibrate with split conformal on a **chronologically later** calibration set, per position (and single vs DGW).
- Because player-GW data are time-ordered and not exchangeable, add a rolling recalibration (recompute the conformal correction every GW from the last N finalised GWs) and report coverage drift.
- Keep an **analytic fallback** (the current method) and compare both in the backtest.
- For DGW: do **not** sum marginal quantiles. Use simulator samples (WI-4.1) or flag `uncertainty_status="dgw_marginal_sum_unvalidated"` as now until WI-4.1 lands.
**Acceptance:** nominal 80% band has empirical coverage in [77%, 83%] for each position, for single GWs, and for DGW separately (if enough DGW data; else report n and skip the gate); pinball loss at 0.1/0.9 improves vs current.

### WI-3.9 Promotion gate (D6-adjacent)
Replace "MAE and rank-correlation improve" with:
- Paired block-bootstrap over **gameweeks** (not rows) of the per-GW metric difference between candidate and production, 10,000 resamples.
- Promote only if: the 95% CI for RMSE difference excludes 0 in the good direction **or** the candidate is statistically indistinguishable (CI includes 0) **and** it improves a pre-registered secondary metric (e.g. calibration) **and** it is not worse for top-decile players.
- Log the CI, resample count and seed in the registry entry; keep rollback verified by checksum as already designed.

### WI-3.10 Stacking weights
The blend `(0.72, 0.04, 0.24)` is a global constant. Re-fit per position with non-negative least squares (weights sum to 1) on out-of-fold predictions only; confirm no leakage (weights fitted on data not used by the component models' calibration). Compare vs the constant on the backtest.

---

## 6. Phase 4: Scoreline-based simulator (D8)

**Files:** `src/fpl_oracle/sim/match_sim.py`, `src/fpl_oracle/league/montecarlo.py` (refactor to consume it).

### WI-4.1 Algorithm (vectorised NumPy; one call returns samples for all players)
For each GW and `n_sims` (default 2000):
1. **Fixtures:** for each fixture draw `(home_goals, away_goals)` from the Dixon-Coles scoreline grid (WI-3.1 + market blend).
2. **Minutes:** for each rostered player draw `starts ~ Bernoulli(P(starts))`, and minutes bucket from the minutes model (0 / <60 / >=60 / 90). Starters share a team-level latent "rotation/ injury" factor optionally; keep it independent initially.
3. **Goals:** for each team goal, choose the scorer by categorical over on-pitch teammates with probabilities from the player share model (WI-3.3); include own-goals as a small separate probability. Penalties: if the team is awarded a penalty (simulate count from a fixture-level Poisson), pick the taker from the penalty order, convert with prob `p_pen`, record misses/saves.
4. **Assists:** for each non-own, non-penalty goal, assisted with probability `p_assist`; assister by categorical over teammates (excluding scorer).
5. **Clean sheets and conceded:** CS iff opponent goals == 0 (for players with >=60 min). GC deduction `floor(goals_conceded_while_on_pitch / 2)`; if minutes < 90 and the player was subbed, scale goals on pitch by sampled minute (goal minute ~ uniform over match).
6. **Saves:** GK saves ~ NegBin with mean from opponent shots-on-target model minus goals conceded; penalty saves from step 3.
7. **DefCon:** counts from WI-3.5, thresholded.
8. **Cards:** Bernoulli per player for yellow, small for red.
9. **BPS and bonus:** build per-player BPS from simulated events with the 2026/27 weights plus a noise term for non-modelled actions; allocate 3/2/1 per fixture with tie handling (ties share).
10. **Points:** apply the rules module to each simulated event vector.
**DGW:** two independent fixture draws per player in the same GW; minutes in the two fixtures positively correlated through a shared latent availability variable.
**Joint draws across managers:** one draw matrix `S[player, sim]` shared by all squads, preserving cross-manager covariance (as the current design already intends).

### WI-4.2 Consistency and validation
- **Mean check:** for each position the simulated mean must equal the analytic `xP` within 1% (aggregate) and 0.05 points (per player, for players with >= 1000 draws). Differences must be explained, not forced: if the simulator and analytic path disagree, find the missing term; do **not** force the mean with a residual Gamma.
- **Distribution check:** PIT histogram and CRPS on the last 10 GWs; haul-rate (P(points >= 10)) predicted vs realised by position.
- **Structure checks:** teammates' clean sheets are perfectly correlated; sum of bonus per fixture <= 6; sum of goals per team equals the scoreline.
- **Speed:** n_sims = 2000 for ~700 players in < 5 s on a laptop.
**Acceptance:** all checks pass and results logged; league win-probability outputs change by a documented amount vs the old method on 10 historical mini-league snapshots.

### WI-4.3 Consumers
Wire the sample matrix into: captain choice (P(captain > alternative), floor and ceiling from real distribution), hit verdicts (WI-2.5), chip value distributions (WI-5.1), league win probability (rivals), and DGW quantiles (replacing the unvalidated marginal-sum flag when the checks above pass).

---

## 7. Phase 5: Chip planning

### WI-5.1 Joint chip optimisation
Chips are already binaries in the multi-period MILP (WI-2.1). The planner should:
- Solve with chips free, with chips forbidden (baseline), and with each chip pinned to each GW in its window; report `Delta U = J(chip at GW) - J(no chip)` for each, plus the distribution under simulator scenarios.
- Account for **opportunity cost** of the single-chip-per-GW rule and for interactions (e.g. WC then BB; FH covering a blank).
- **Use-it-or-lose-it:** Set 1 chips unused at the GW19 deadline have zero value. The planner must explicitly present, for each unused Set 1 chip, "latest sensible GW" and "expected loss if you never use it", and the default recommendation should be to schedule all four before GW19 unless the best available `Delta U` is genuinely negative or tied to a materially better future window. Backward-induction logic: value of waiting = expected best `Delta U` over remaining windows (use scenario sampling for form/injury uncertainty); use now iff `Delta U_now >= option value of waiting` (not a fixed xP threshold).
- Enforce R2 (no FH in GW1; no FH in GW20 after FH in GW19), max one chip per GW, WC preserves banked FTs per WI-1.3.

### WI-5.2 Re-derive thresholds by set and context
The Architecture Doc's gates (TC extra >= 4.0, BB bench >= 8.0, FH >= 10.0, WC >= 8.0) came from DGW/BGW reasoning. For Set 1 there are no DGW/BGW expected, so the realised distribution of `Delta U` is different. Using the backtest (WI-0.1) and past seasons' Set 1-like periods, compute the empirical distribution of best-available `Delta U` per chip per GW (hindsight and as-forecast). Replace fixed gates with data-driven percentiles per set (e.g. "recommend now if >= p75 of last season's as-forecast value for this GW range"), or drop gates and rely on WI-5.1's option-value rule. Keep the old gates behind a flag for comparison.
**Acceptance:** chip plan on 10 historical states reproduces sensible choices (e.g. FH recommended into a blank, BB after WC) and never recommends an illegal sequence; a report shows how often the old and new rules disagree and why.

---

## 8. Phase 6: News ingestion and assistant hardening

### WI-6.1 Prompt-injection and output validation (RSS to LLM)
RSS and article text is untrusted. Required controls (defence in depth; none is sufficient alone):
1. **Quarantined extractor:** the model that reads article text has **no tools** and returns **only** a strict JSON schema (player_id candidate, status enum, expected_return_date, confidence, supporting_quote_span). Reject any response that does not validate.
2. **Data/instruction separation:** untrusted text goes into a clearly delimited data slot; the system prompt states it is data. Cap input and output tokens.
3. **Typed extraction and allow-lists:** player names are resolved to FPL IDs by fuzzy match against the official list; unresolved names are dropped. Statuses limited to a fixed enum.
4. **Output is untrusted too:** never let extractor output trigger tool calls or alter prompts of the privileged assistant model; it flows only through `ReconciledAvailability`.
5. **Evidence requirement:** each extraction must include the quoted span; verify the span exists verbatim in the article text, otherwise discard.
6. **Tests:** a corpus of injected articles ("ignore previous instructions...", hidden text, homoglyph variants) must produce no schema-valid output that changes any player status beyond what the article text actually supports.

### WI-6.2 Statistically sound promotion gates
Gating on "Precision >= 85%, Recall >= 80%, false ruled-out <= 2%" using point estimates on small labelled sets is brittle.
- Use Wilson 95% intervals: promote only if the **lower bound** of precision >= 85%, recall lower bound >= 80%, and the **upper bound** of false-ruled-out rate <= 2%.
- State the minimum labelled-set size required for these bounds to be reachable (e.g. for a Wilson 95% upper bound under 2% even with zero errors you need roughly 190 or more negatives, since the bound is about 3.84/(n+3.84)); refuse promotion below it.
- Keep `SHADOW` as default; log all disagreements between news-adjusted and API-only availability.

### WI-6.3 Assistant grounding
- Have the engine emit a **structured recommendation object** (move, verdict, numbers, alternatives); the LLM narrates it. Add a post-check that every number in the final text appears in the tool outputs (allowing rounding); on failure, regenerate once, then fall back to a templated narration.
- Model names and fallbacks live in config, not code. Revisit whether the lightest model is adequate for multi-step justifications; make it an A/B config.
- Offline provider must clearly label itself as offline mode.

---

## 9. Phase 7: Operations, config and documentation

### WI-7.1 Config consolidation
All rule constants, thresholds, model names, solver settings, horizons and weights in `config/` with schema validation. Every threshold carries a comment pointing to the backtest that justified it (or `TODO: unvalidated`).

### WI-7.2 Job timing aligned with live scoring (R7)
- `history_refresh` and `retrain_trigger` must only act once a GW is **finalised** (`finished=true` and data_checked per API, which follows the 09:00 UK lock the next day), never at final whistle.
- Live projected-bonus: the cadence refresh can show projected bonus after 20 minutes, but the holdout scorer must use final points only.
- Add a test with a mocked API timeline that asserts no retrain before finalisation.

### WI-7.3 Documentation hygiene (D4, D11, D12)
- Generate the feature table in the Architecture Doc from `FEATURE_SCHEMA` (script `scripts/gen_arch_doc.py`), and fail CI if the committed doc is out of date.
- Replace absolute `file:///C:/...` links with relative links.
- Replace unprovable claims ("100% uptime", "never fails", "100% compliant") with measured statements and link to the backtest report and the rules contract test.
- Delete hard-coded "GW5" and "GW12-18 DGW" statements; generate chip calendar from the fixtures API.

### WI-7.4 Data-source resilience
- FBref lost its Opta-derived advanced stats (xG etc.), so do **not** depend on it. Use the FPL API plus Understat for xG, and keep adapters behind an interface so a source change doesn't break the pipeline.
- Add a data-quality report each run: missing-value rates by column, schema drift, duplicated player IDs, price anomalies.

---

## 10. Phase 8: Personal-expert advisor layer (product requirement, high priority)

**Product requirement from the owner:** the tool must behave like **a personal FPL expert the manager has hired**, not a calculator that prints the argmax of a points model. An expert (a) looks at the whole situation before acting, (b) prefers the cheapest sufficient fix, (c) compares *when* as well as *whether*, (d) knows this manager's league, risk appetite and preferences, (e) explains and defends advice, pushes back on weak ideas, and says what would change their mind, and (f) is accountable for past advice.

### 10.1 The failure this layer exists to fix (reference case S1)
The owner observed the following gap between what the optimiser did and what a real expert would do. **Treat the figures below as the owner's description, not verified facts; the agent must build the S1 fixture from the owner's real squad snapshot (see Appendix C, question 5) and must not hard-code any player names or prices in production code.**

| Factor | What the tool's optimiser did | What expert reasoning requires |
|---|---|---|
| Horizon comparison | Compared Wildcard in GW6 against *never* using it before the GW19 deadline | Compare Wildcard in GW6 against Wildcard in **each later window** (e.g. GW12 during winter fixture congestion, where the value may be materially higher), and against no-Wildcard plans that use banked free transfers |
| Purchasing power | Assumed a Wildcard was needed to afford two premium targets | Check what the bank and sell values already allow: one target may be affordable by a single free transfer (e.g. a like-for-like swap) with no hit |
| Squad health | Saw several low-projection players and replaced all of them at once | Count functional starters. If a healthy core remains (e.g. 11 playable starters including the premium captains), an emergency overhaul is unnecessary; fix only what is broken |

**Root causes in the current design:** (1) chip decisions are evaluated against a single baseline ("no chip") rather than against a *timeline* of alternatives; (2) the optimiser maximises horizon points with no preference for minimal intervention, so a many-transfer plan that is slightly better than a one-transfer plan always wins; (3) there is no squad-health diagnosis step, so a Wildcard looks attractive whenever the sum of weak slots is large; (4) nothing models what the manager can already afford.

### 10.2 Design principles (each is testable; see WI-8.9)
- **P1 Minimal sufficient intervention.** Recommend the cheapest plan (fewest transfers, no hit, no chip) that captures most of the available gain. Always show the **marginal value of each additional transfer** and of each chip, so the manager sees where returns flatten.
- **P2 Timing is part of the decision.** For any chip or large restructure, the output must be a **timeline table** over all legal windows (now, each later gameweek before the Set 1 expiry) with expected gain, probability of beating the alternatives, and the opportunity cost of using the chip now.
- **P3 Diagnose before prescribing.** Run a squad-health check first; only escalate to chips or multi-transfer plans when diagnosis shows structural problems.
- **P4 Affordability-aware.** For every target (the system's top targets and anything the manager names), compute the cheapest legal path to own him, using bank, sell values, free transfers and sidegrades.
- **P5 Knows the manager.** Uses a persistent manager profile (mini-league, rank, risk appetite, preferences, chips left, time to deadline) and states when advice depends on it.
- **P6 Honest and accountable.** Gives confidence, shows what would change the recommendation, and logs every recommendation to an advice ledger scored against realised results.
- **P7 Proactive and structured.** Delivers a short pre-deadline briefing without being asked, in a fixed, scannable structure.
- **P8 Consistent expert voice.** Decisive, concise, reasoned; challenges the manager's plan with numbers when it is worse than an alternative; asks a clarifying question only when the answer would change the recommendation.

### WI-8.1 Squad-health diagnosis
**Files:** `src/fpl_oracle/advisor/health.py`, `tests/advisor/test_health.py`.
- For each squad player output a status `KEEP | WATCH | REPLACE` with machine-readable reasons: minutes risk (`P(starts)` low or rotation signals), injury/availability (`ReconciledAvailability`), fixture run (opponent-strength-adjusted `xP` over the next N GWs relative to position benchmark), form-versus-underlying gap (expected vs actual output), price pressure (falling price: selling-value risk).
- Compute **functional starters**: the number of the 11 best-lineup slots filled by players with `P(play) >= p_min` and `xP >= position_floor` (both thresholds fitted from backtest, not hard-coded; default placeholders flagged in the log).
- Compute a **structural deficiency score** = expected points lost versus an ideal replacement set, restricted to slots that are actually deficient. The diagnosis feeds P1/P2: **emergency overhaul recommendations are allowed only when** functional starters are below a configured minimum, or deficiency score exceeds a calibrated threshold, or an available chip window has expected value that clearly exceeds the cost of fixing only the deficient slots (WI-8.2).
**Acceptance:** on 20 historical squad states labelled by a rule-based "obviously healthy / obviously broken" oracle (built from realised outcomes: e.g. healthy = at least 11 players who both started and scored above position median over the next 3 GWs), diagnosis agrees on at least 85% of clear cases; the S1 fixture is classified as "healthy core, targeted fixes".

### WI-8.2 Efficient frontier of plans (minimal-intervention planner)
**Files:** `src/fpl_oracle/advisor/frontier.py` (uses the multi-period MILP, WI-2.1).
- Solve the horizon problem repeatedly with the number of week-1 transfers constrained to `k = 0, 1, 2, ..., K` (K = min(5, banked FTs + 2)), and with `h[t]` hits explicitly priced. Produce the **frontier**: net expected gain over the roll plan versus `k`, with hit costs included and probability of improvement from simulator samples (WI-2.5).
- **Recommendation rule (knee-point selection):** choose the smallest `k` whose net gain is at least `rho` (default 85%, tune on backtest) of the best-frontier gain, or the first `k` where the marginal gain of the next transfer falls below its option-value cost (WI-2.1 `V_ft`) plus hit cost. Ties go to fewer transfers and no hit.
- Report for each point on the frontier the exact moves, the free-transfer balance afterwards and what is being given up.
**Acceptance:** property test: recommended `k` never exceeds the argmax `k` and never uses a hit unless the marginal net gain after hit cost is positive with probability above the configured level; on the S1 fixture the recommended plan uses at most the free transfers available and no Wildcard.

### WI-8.3 Affordability path finder
**Files:** `src/fpl_oracle/advisor/funding.py`.
- Input: a target player (or list). Output: the **cheapest legal path(s)** to own him: direct swap with a specific outgoing player and the bank left; two-step plans (sidegrade first, then upgrade) when the bank is short; the minimum number of transfers and hits; selling-price effects (WI-1.4).
- Implementation: solve MILP with `in[target, t0] = 1` forced and lexicographic objective (min hits, then min transfers, then max horizon xP); return top 3 distinct paths via no-good cuts.
- The "why can I afford this?" explanation must show bank, sell value of the outgoing player, the target price and the remaining bank.
**Acceptance:** for 30 random (squad, target) pairs the returned plan is legal under the rules oracle and no cheaper plan exists (checked by brute force on small candidate sets); S1: the "single free transfer swap" path is found if it exists in the data.

### WI-8.4 Chip timeline comparator (extends WI-5.1)
**Files:** `src/fpl_oracle/advisor/chip_timeline.py`.
- For each unused chip in the current set, build a table over **every legal gameweek** until the set's last gameweek (read the deadline from the API): expected gain versus the best no-chip plan, `P(gain > 0)` and a gain interval from simulator samples, the fixture/congestion context for that window (derived from the fixture list and known European/cup commitments, not hard-coded), value lost by waiting (squad rotting or injuries compounding), and **value of waiting** from backward induction (WI-5.1).
- For the Wildcard specifically, always compare three families: (a) Wildcard now, (b) Wildcard at each later window, (c) **no Wildcard: staged free-transfer plan** (roll banked transfers, then execute a multi-transfer week hit-free), since up to 5 FTs can be banked. The recommendation states which family wins and by how much, with probabilities.
- Never compare a chip window only against "never use it". "Never" is a legal alternative only before the expiry deadline and must be priced as zero residual value for first-set chips.
**Acceptance:** the timeline table exists for every chip query; the Wildcard analysis always includes at least three later windows (when they exist) plus the staged-FT alternative; a regression test fails if the comparison set degenerates to {now, never}.

### WI-8.5 Manager profile and context
**Files:** `src/fpl_oracle/advisor/profile.py`, `config/manager_profile.yaml` (schema-validated), conversation-state store.
- Profile fields: FPL entry ID, mini-leagues that matter and the user's position in each, objective weights (maximise expected points vs maximise probability of winning the mini-league vs chase overall rank), risk appetite (template-safe, balanced, differential-hungry), hard preferences ("never sell X", "avoid club Y", "keep captain options at Z"), tolerance for hits, how often they check the app.
- The advisor uses profile weights when ranking plans: expected points alone is the default; if the objective is a mini-league win, use league win-probability from the simulator (WI-4.3) and effective ownership of rivals.
- Preferences expressed in chat ("I'm not selling Haaland") are stored as profile entries only when the user asks for them to persist; one-off statements apply to the current conversation only.
**Acceptance:** two profiles (safe vs differential) produce different captain and transfer rankings on the same state in a test; a "never sell X" constraint is honoured by all planners.

### WI-8.6 What-if and plan critique tools
**Files:** `src/fpl_oracle/llm/tools.py` (new tools), `src/fpl_oracle/advisor/critique.py`.
- New tools: `evaluate_plan` (the manager proposes moves, chips, captain; the engine scores it against the frontier and the best alternatives and reports regret in expected points and in probability of beating the roll plan), `compare_options` (named alternatives side by side), `explain_why_not` ("why not Wildcard now?", "why not take a hit for X?"), and `affordability_path` (WI-8.3).
- Critique output must include: verdict (`GOOD | ACCEPTABLE | WORSE THAN ALTERNATIVE`), the strongest alternative, the numeric reason, and a concession when the manager's idea has merit the model cannot capture (e.g. an injury doubt not yet reflected).
**Acceptance:** for 15 scripted manager proposals (some good, some clearly worse) the verdicts match the expected labels; critique never contradicts the numbers in the tool output.

### WI-8.7 Advice ledger and accountability
**Files:** `src/fpl_oracle/advisor/ledger.py`, `data/advice_ledger.sqlite`.
- Persist every recommendation (state hash, advice, alternatives considered, predicted gain and probability, confidence, profile in force, timestamp). After the gameweek is finalised (WI-7.2), score it: realised points of the recommendation versus the best alternative in hindsight and versus the roll plan; track calibration of stated probabilities.
- Weekly "report card" in the briefing: record against the roll baseline, calibration of the stated probabilities, and the biggest miss with a one-line cause (injury, rotation, variance).
**Acceptance:** ledger entries are immutable; scoring job runs only on finalised gameweeks; report card produced for the last 8 GWs in a replayed simulation.

### WI-8.8 Briefing format, voice and response contract
**Files:** `src/fpl_oracle/llm/prompts/advisor_system.md`, `src/fpl_oracle/advisor/briefing.py`.
- Every recommendation is a structured object and the LLM narrates it (WI-6.3). Required fields: `decision` (one sentence), `why` (two to four numbered reasons with numbers), `alternatives_considered` (including at least the roll plan and the next best), `cost` (hits, transfers used, FT balance after), `risk` (what could go wrong, with probability), `confidence` (low/medium/high, tied to `P(improve)`), `what_would_change_this` (facts that flip the advice, such as an injury update or a price change), `deadline_and_next_checkpoint`.
- Pre-deadline briefing (hook into `deadline_alert` at -24h, -3h, -1h): top line decision, transfers, captain with floor/ceiling, chip outlook (timeline row), watchlist (injuries, price changes, rotation risks), and one "if you only read one line" sentence. Later alerts only mention what changed since the previous one.
- Voice rules: decisive, plain, no filler; push back when the manager's plan is worse; admit uncertainty without hedging every sentence; never present a probability the engine did not compute.
- Clarifying-question policy: ask at most one question and only if the answer would change the recommendation (for example "do you want to protect your mini-league lead or chase rank?"); otherwise state the assumption and proceed.
**Acceptance:** schema validation on 100% of recommendations; a lint test rejects responses missing alternatives, cost or confidence; numeric grounding check passes (WI-6.3); a human-readability review of 10 sample briefings is recorded in the log.

### WI-8.9 Expert scenario regression suite
**Files:** `tests/advisor/scenarios/` (fixtures as JSON snapshots of squad, bank, FTs, chips, fixtures, projections).
Each scenario encodes a situation and the properties the advice must satisfy. The suite must run in CI and block merges.
| ID | Situation | Required advisor behaviour |
|----|-----------|----------------------------|
| S1 | Owner's case: weak start, several low-projection players, healthy core with premium captain options, some bank, a premium target affordable with one free transfer, a Wildcard still available, Set 1 ends at GW19 | Does **not** recommend a Wildcard now; diagnoses healthy core; recommends the cheapest fix with no hit; shows the Wildcard timeline over later windows and the staged-FT alternative; states what would change the advice |
| S2 | Squad truly broken (several injured or benched starters, weak fixtures) | Does recommend an overhaul, and explains why the frontier plus health check justify a chip or hit |
| S3 | Free Hit candidate: large blank gameweek soon | Compares Free Hit now vs later windows; respects no-FH-in-GW1 and the GW19/GW20 rule |
| S4 | Set 1 deadline close with unused chips | Applies use-it-or-lose-it logic; explicit expected loss of never using each chip; picks the least-bad window |
| S5 | Manager proposes a hit for a marginal upgrade | Critique verdict `WORSE THAN ALTERNATIVE` with regret in expected points |
| S6 | Mini-league lead is small with a rival template-heavy | Safe-profile advice differs from differential-profile advice; shows effective-ownership reasoning |
| S7 | Price rise imminent for a target | Notes the cost of waiting versus transferring now, quantifies it, still obeys the minimal-intervention rule |
| S8 | Key player has an unresolved doubt | Does not over-commit; presents the decision tree (if he plays / if he does not) and what to monitor before the deadline |
**Acceptance:** all scenarios pass; each scenario has a golden JSON of the structured recommendation reviewed by the owner before being frozen; adding a new scenario requires only a fixture file.

### WI-8.10 Delivery order within Phase 8
Order: WI-8.1 and WI-8.3 (need the MILP) then WI-8.2 and WI-8.4, then WI-8.5, WI-8.6, WI-8.8, then WI-8.7 and WI-8.9. A minimal version of WI-8.1 and WI-8.4 (diagnosis plus the "timeline, never just now-vs-never" rule) should ship **as soon as the multi-period MILP exists**, because that is the defect the owner hit first.

---

## 11. Execution order and acceptance summary

| Order | Work item | Depends on | Effort | Main acceptance signal |
|------:|-----------|------------|--------|------------------------|
| 1 | WI-1.1 rules module + contract test | - | S | Contract test passes on stored seasons |
| 2 | WI-1.3 FT state machine + WI-1.4 selling price | 1.1 | S | Table tests pass; single function used everywhere |
| 3 | WI-1.2 ensemble completeness | 1.1 | S | Decomposition sums to xP; golden diff logged |
| 4 | WI-0.1 to 0.4 backtest harness, baselines, CI gate | 1.x | M | Report beats/loses to baselines clearly stated |
| 5 | WI-7.2 job timing, WI-7.3 doc hygiene | 0.x | S | Mock-API test; generated doc in CI |
| 6 | WI-2.1 to 2.4 multi-period MILP | 1.3, 1.4, 0.1 | L | Property tests + runtime target + backtest comparison |
| 7 | WI-3.1 Dixon-Coles team model | 0.1 | M | RPS/log-loss better than legacy |
| 8 | WI-3.2 odds features (+ start `odds_snapshot` job immediately in step 4) | 3.1 | M | Brier/log-loss improvement; fallback test |
| 9 | WI-3.3 share model + 3.4 minutes + 3.5 DefCon + 3.6 bonus + 3.7 shrinkage | 3.1, 3.2 | L | Per-component calibration gates |
| 10 | WI-3.8 CQR quantiles + 3.9 promotion gate + 3.10 stacking | 0.1 | M | Coverage in [77,83]% per position |
| 11 | WI-4.1 to 4.3 scoreline simulator | 3.1, 3.3, 3.5, 3.6 | L | Mean and structure checks; speed target |
| 12 | WI-2.5 hit policy using simulator | 2.x, 4.x | S | Verdicts with probabilities |
| 13 | WI-5.1, 5.2 chip planning | 2.x, 4.x | M | Legal sequences; option-value rule |
| 14 | WI-6.1 to 6.3 news/assistant hardening | - (can run in parallel from step 5) | M | Injection corpus passes; Wilson-bound gates |
| 15 | WI-7.1, 7.4 config and data resilience | any | S | Schema validation; data-quality report |
| 6b | **Early slice of Phase 8:** WI-8.1 health diagnosis + WI-8.4 chip timeline (no "now vs never" comparisons) + WI-8.3 affordability | 6 (MILP) | M | S1 scenario passes: no Wildcard, cheapest fix, timeline shown |
| 16 | WI-8.2 frontier, WI-8.5 profile, WI-8.6 what-if tools, WI-8.8 briefing/voice | 6, 11, 13 | L | Scenario suite S1-S8 plus schema-validated recommendations |
| 17 | WI-8.7 advice ledger, WI-8.9 full scenario suite | 16, 7.2 | M | Report card replay; all scenarios frozen and green |

**Note on step 4/8:** start the `ep_next` and odds snapshot jobs as early as possible; they only become useful once data accumulates, and the Set 1 chip deadline (2 Jan 2027) is a fixed external deadline for decisions that benefit from these inputs.

### Definition of done for the whole programme
1. 426 original tests still pass (minus any explicitly corrected wrong-rule tests), plus the new suites.
2. `reports/eval/<date>/summary.md` shows the final model vs all baselines on the last two seasons and live 2026/27 GWs, with paired CIs, coverage and calibration.
3. `docs/improvement_log.md` records every work item's before/after.
4. Architecture Doc regenerated, free of D1-D12.
5. Advisor scenario suite S1-S8 (WI-8.9) green, and a sample weekly briefing reviewed and approved by the owner as "reads like a hired expert, not a calculator".

---

## Appendix A: Reference formulas and snippets

**A.1 Dixon-Coles tau and log-likelihood weights**
```python
def tau(x, y, lam, mu, rho):
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0
# weight for a match played d days ago: w = exp(-xi * d)
# loglik = sum_m w_m * ( log tau + log Pois(x; lam) + log Pois(y; mu) )
```
**A.2 Poisson floor expectation (already in the repo; keep, and test against brute-force summation to k=40)**
`E[floor(X/m)] = sum_{k>=0} floor(k/m) * Pois(k; lam)`; for the scoreline-based path, use the sampled goals directly.

**A.3 Free-transfer step (normal week)**
```python
free_used = min(transfers, ft)
hit_points = 4 * max(0, transfers - ft)
ft_next = min(5, ft - free_used + 1)
```
**A.4 Selling price (tenths of a million)**
```python
def sell(purchase10, current10):
    return current10 if current10 <= purchase10 else purchase10 + (current10 - purchase10) // 2
```
**A.5 Wilson interval (for WI-6.2)**
`p_hat +/- z*sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))` all over `(1 + z^2/n)`, centre `(p_hat + z^2/(2n)) / (1 + z^2/n)`, `z=1.96`.

**A.6 Paired block bootstrap**
Resample gameweeks with replacement (block = GW); per resample compute metric difference; report percentile CI.

---

## Appendix B: Sources (verify again before relying on any figure)

**Rules 2026/27**
- Premier League, "All you need to know about changes to FPL for 2026/27": https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627
- Premier League, "What's happening with FPL chips in 2026/27": https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627
- Fantasy Football Scout, 5 rule changes (BPS table, 09:00 lockdown, 5 FTs): https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced
- Flashscore / Fantasy Football Hub on BPS changes: https://www.flashscore.com/news/soccer-premier-league-fantasy-premier-league-2026-27-all-rule-changes-and-new-features/KGmq7ts1/
- Fantasy Football Fix, 2026/27 changes (BPS, live scoring, chips): https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-new-rules/
- Defensive contribution thresholds (DEF 10, MID/FWD 12, cap 2): https://www.beatfpl.com/guides/fpl-defensive-contribution and https://onsidearena.com/fpl-defensive-contributions
- Banked FTs preserved through WC/FH (third-party; confirm empirically): https://full90fpl.com/fpl-transfers-explained/ and https://www.fplpulse.com/blog/fpl-chips-explained
- Chip strategy (no blanks/doubles expected before GW19): https://www.fantasyfootballhub.co.uk/articles/fpl-chip-strategy-guide

**Methods**
- Multi-period FPL MILP write-ups and open implementations: https://fpl.hashnode.dev/fpl-solver ; https://github.com/harrycjturner04/fpl-ai-coach/pull/2 ; https://fplcopilot.com/solver (HiGHS-based planner up to 10 GWs)
- Data-driven FPL team selection with MILP (Ramezani et al.): https://arxiv.org/abs/2505.02170
- OpenFPL (open forecasting; stratified evaluation): https://arxiv.org/abs/2508.09992
- FPL Review, "Ultimate Truth" (perfect-model noise floor): https://docs.fplreview.com/articles/ultimate-truth/
- Dixon and Coles (1997); Serie A decay re-estimate and market comparison: https://arxiv.org/pdf/2608.11505
- Goal expectancy from odds (penaltyblog): https://penaltyblog.readthedocs.io/en/latest/_sources/models/goal_expectancy.rst.txt
- Conformalized quantile regression (Romano, Patterson, Candes 2019): https://arxiv.org/abs/1905.03222
- Free historical odds (football-data.co.uk; via soccerdata MatchHistory): https://soccerdata.readthedocs.io/en/stable/datasources/MatchHistory.html
- Prompt-injection defences (typed extraction, quarantined model, output validation): https://jwatte.com/blog/ai-indirect-prompt-injection.md and https://github.com/luisalima/agentic-security
- FBref advanced-stats removal: https://www.theixsports.com/the-ix-soccer/fbrefs-loss-advanced-stats-womens-soccer-data-accessibility/

## Appendix C: Open questions for the human (answer before the dependent work item)
1. Which odds provider (free or paid) is acceptable for player-level markets? (blocks WI-3.2 step 5)
2. Is a minimum solve time budget of 60 s acceptable for the multi-period plan on the target machine? (WI-2.4)
3. Do you want a mini-league win-probability target or an overall-rank target as the primary objective for the planner? (changes the objective in WI-2.1/4.3: expected points vs rank-aware risk preference)
4. Should the system ever recommend differential risk-taking explicitly (rank-chasing), or stay EV-maximising by default?
5. For scenario S1: please provide your FPL entry ID or a squad snapshot (15 players with purchase and current prices, bank, free transfers, chips used) as it was when the tool recommended a Wildcard in GW6, plus the recommendation text the tool produced. (blocks WI-8.9 S1 and the WI-8.1 acceptance check)
6. Which mini-leagues matter to you and what is your objective in each: win the league, finish top N, or overall rank? (feeds WI-8.5)
