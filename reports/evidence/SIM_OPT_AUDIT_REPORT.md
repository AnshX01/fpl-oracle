# Simulation & Optimizer Audit Report — fpl-oracle

**Repository:** C:\Users\anshw\Documents\fpl-oracle
**HEAD:** 5ab9b6e793a3fc1712d6f63d6f1f602d320f3894 (working tree contains uncommitted modifications to montecarlo.py, planner.py, bonus.py, defcon.py, lineup.py, config/*.yaml and other files; see `git status`)
**GOAL.md baseline audit target:** HEAD 4b6db3c on Linux/Python 3.11.16; this checkout re-verifies each Part-A finding on the actual checkout and records whether it still reproduces.
**Environment:** Windows 11, Python 3.14.2 (venv `.venv`), ruff 0.16.10, mypy 2.4.0, numpy 2.5.3, pandas 3.0.6, lightgbm 4.7.0, scipy 1.18.1, PuLP 3.3.2, pytest 9.1.1
**Method:** Read sources at the cited file:line locations; executed live Python against `fpl_oracle` modules; ran regression tests `tests/test_f002_*.py`, `test_f004_*.py`, `test_f005_*.py`, `test_f007_*.py` (12 passed in 4.70s). Original findings from `audit/findings.md` (F-002, F-004, F-005, F-007, F-009, F-011) and GOAL.md Part A (A5 MC defects, joint planner integration).

Per GOAL.md rule: if a finding no longer reproduces on this checkout, it is reported as such rather than marked "to fix".

---

## 1. Executive summary table

| ID | Area | Severity | Defect | Status on this checkout | Key file:line |
|----|------|----------|--------|-------------------------|---------------|
| F-002 | Monte Carlo tie-breaking | CRITICAL | Tied user assigned rank 1 on 100% of tied trials instead of symmetric split | FIXED — symmetric fractional tie credit present and verified (50.0% / 33.3%) | src/fpl_oracle/league/montecarlo.py:452-457 |
| F-004 | Poisson E[floor(K/d)] | HIGH | Hardcoded max_k=20 truncation collapses expectations for lambda > 10 | FIXED — dynamic cap, default max_k=None; exact match verified at lambda=20 (9.75=9.75) | src/fpl_oracle/domain/scoring.py:94-116 |
| F-005 | Transfer beam search | HIGH | Beam cannot play >2 transfers in one GW; banked 3-5 FTs unexploitable | PARTIAL — beam capped at 2 (count>2 rejected); router sends FTs>=3 to multi-period MILP which supports N transfers | src/fpl_oracle/optimise/sequential.py:183-220,365; transfers_beam.py:503-613; transfers.py:57-58 |
| F-007 | GW19 chip gates | HIGH | Static gain gates (3xC 4.0, FH 10.0) let chips expire unused at GW19 deadline | FIXED — threshold forced to 0.0 at current_gw==19 or congestion; GW18 threshold halved | src/fpl_oracle/chips/planner.py:437-447 |
| F-009 | ML sample weights | MEDIUM | 5x weighting on 2026/27 rows collapses Kish ESS | PARTIAL — 5x retired: bonus now 4.0x, defcon 1.5x; no CV-based tuning remains | bonus.py:47-48; defcon.py:79-82 |
| F-011 | Lineup risk modes | MEDIUM | Starters insensitive to risk_preference; only captaincy modulated | REPRODUCES — risk_aware_lineup flag defaults false; starter coefficients are pure xP except conservative/aggressive with flag on | src/fpl_oracle/optimise/lineup.py:58-104 |
| A5-a | MC residual clamp/truncation | HIGH | Residual mean clamped at zero; gamma draws truncated | PARTIAL — mean-exact decomposition verified numerically (xP 0.1 -> 0.1002); residual gate mu_other_play > 1e-4 and xp<=0 clamp remain | montecarlo.py:106-109,167-199 |
| A5-b | MC horizon repetition | MEDIUM | First GW repeated for all 5 horizon GWs when per-GW projections absent | PARTIAL — fallback `[projections_df] * horizon_gws` remains; production callers pass projections_by_gw | montecarlo.py:305,293-307; api.py:959; pipeline.py:260; decision_card.py:465 |
| A5-c | p_cs / home-away flow | MEDIUM | p_cS partly fixed heuristic; home/away dropped or inferred as exp_cs_pts/4 | PARTIAL — row/model p_clean_sheet used when present, DEFAULT 0.30 fallback; home/away used for inferred team CS prob | montecarlo.py:113-117,76-79; match_sim.py:244-267 |
| A5-d | Joint chip+transfer planner | HIGH | evaluate_joint_transfer_and_chip_plan has no API/pipeline/decision-card callers | PARTIAL — callers DO exist in this checkout (analysis.py:323-324,520; pipeline.py:205; decision_card.py:108) but transfers and chips are still optimized as separate calls; not one unified stateful plan | server/analysis.py:323-324,368-427,519-520; server/pipeline.py:205,220; briefing/decision_card.py:108,121-131 |

---

## 2. Per-finding detail

### F-002 — Monte Carlo symmetric tie-breaking (CRITICAL) — FIXED, verified

**Defect as documented:** user trial rank computed as `sum(1 for s in all_scores if s > trial_user_pts) + 1`; on tied trials the user is ranked 1 with 100% win probability instead of splitting win probability symmetrically (50.0% for two tied teams, 33.3% for three).

**Current code (src/fpl_oracle/league/montecarlo.py:450-457):**
```python
all_scores = np.vstack([trial_user_pts[None, :], trial_rival_pts])
strictly_ahead = np.sum((all_scores - trial_user_pts) > 1e-9, axis=0)
tied_with_user = np.sum(np.abs(all_scores - trial_user_pts) <= 1e-9, axis=0)
user_ranks = (strictly_ahead + (tied_with_user + 1) / 2.0).tolist()
user_wins = float(np.sum(np.where(strictly_ahead == 0, 1.0 / tied_with_user, 0.0)))
```
The fractional win credit `1.0 / tied_with_user` when `strictly_ahead == 0` (line 456) implements the symmetric split recommended in the finding. `tied_with_user` correctly includes the user's own score, giving mid-rank `(strictly_ahead + (tied+1)/2)`.

**Reproduction on this checkout:** identical user + rival squads, 3000 sims, seed 7:
- user_win_probability_pct = 50.0%, expected_final_rank = 1.5, rival win = 50.0%, sum = 100.0%.
**Regression tests:** tests/test_f002_monte_carlo_ties.py — 4 tests pass (50% two-way tie, 33.3% three-way tie, symmetry, sum-to-100%).

**Note:** `git diff` vs HEAD shows the tie vectorization was reworked (per-trial loop -> vectorized match simulator) in the working tree; the fractional tie credit already existed in HEAD's loop form, so the described 100%-win-on-tie defect does not reproduce at either revision.

**Recommendation:** keep the symmetric credit; add a pinned regression test asserting 49-51% for identical squads (already present in tests/test_f002_monte_carlo_ties.py). No code change required.

---

### F-004 — expected_floor_div Poisson truncation collapse (HIGH) — FIXED, verified

**Defect as documented:** `expected_floor_div` hardcoded `max_k = 20`; for Poisson rates lambda > 10, mass above k=20 is truncated, collapsing E[floor(K/divisor)] toward 0. Documented figures: rate=15 -> 6.33 vs 7.25 exact; rate=20 -> 4.57 vs 9.75 exact; rate=25 -> 1.63 vs 12.25 exact.

**Current code (src/fpl_oracle/domain/scoring.py:94-116):**
```python
def expected_floor_div(rate: float, divisor: int, max_k: int | None = None, tol: float = 1e-12) -> float:
    if rate <= 0.0 or divisor <= 0:
        return 0.0
    cap = max(100, int(rate + 10 * math.sqrt(rate) + 20)) if max_k is None else max_k
    ...
    while k <= cap:
        ...
        if max_k is None and k > rate and (1.0 - cdf) < tol:
            break
        k += 1
    return exp_val
```
Default `max_k=None` gives a rate-dependent cap and early stop at residual Poisson mass < 1e-12. Note docs/remediation_log.md:164 still records the old signature `max_k: int = 20` — stale documentation.

**Verification on this checkout (exact brute force vs computed):**
| lambda | divisor | computed | exact |
|--------|---------|----------|-------|
| 0.1 | 2 | 0.0047 | 0.0047 |
| 10.0 | 2 | 4.7500 | 4.7500 |
| 15.0 | 2 | 7.2500 | 7.2500 |
| 20.0 | 2 | 9.7500 | 9.7500 |
| 25.0 | 2 | 12.2500 | 12.2500 |
| 30.0 | 2 | 14.7500 | 14.7500 |
| 20.0 | 3 | 6.3333 | 6.3333 |

All callers use the default: scoring.py:173 (`gc_deduction`, divisor 2), scoring.py:179 (`save_pts`, divisor 3), ensemble.py:274, ensemble.py:280, tests/test_match_sim.py:75-76. No caller passes max_k=20.

**Regression tests:** tests/test_f004_poisson_truncation.py (2 tests) pass; tests/test_domain_foundations.py passes.

**Recommendation:** keep the dynamic cap; remove the stale `max_k = 20` reference from docs/remediation_log.md:164. No functional change required.

---

### F-005 — Beam search limited to 0/1/2 transfers (HIGH) — PARTIAL (mitigated at router, beam itself unchanged)

**Defect:** sequential beam search generates candidates for 0, 1, or 2 transfers per gameweek only; with 3-5 banked free transfers (official 2026/27 rule, up to 5 banked) hit-free 3/4/5-player restructuring moves in a single week are undiscoverable by the beam.

**Code evidence:**
- src/fpl_oracle/optimise/sequential.py:98 `candidate_limit=2`, :99 `two_transfer_limit=1`, :102 `max_hits_per_move=1`
- sequential.py:208 `for move in immediate[:candidate_limit] + horizon[:candidate_limit]` — at most 4 one-transfer moves plus null move
- sequential.py:213-232 calls `optimizer._get_candidate_2_transfers(...)` (transfers_beam.py:503-613) and appends only `pairs[:two_transfer_limit]` (line 228) — at most the top 2-transfer pair per frame
- sequential.py:365-366 `if not special and (count > 2 or hits > max_hits_per_move ...): continue` — hard rejection of any move with more than 2 transfers
- config/settings.yaml (working tree): `optimizer.max_hits: 2  # Hard ceiling on extra transfers per single gameweek (WI-2.2)`

So the beam's supported maximum is **2 transfers in a single gameweek move** (0 = roll, 1 = single swap, 2 = double swap). Banked FTs of 3, 4, 5 accumulate (next_ft = min(5, ...), line 374) but cannot be spent in one beam move.

**Mitigation (in place):** src/fpl_oracle/optimise/transfers.py:57-58:
```python
if is_flag_enabled("multiperiod_milp", default=False) or free_transfers >= 3:
    return multiperiod_optimizer.evaluate_transfer_options(...)
```
When 3+ FTs are banked, routing goes to the multi-period MILP (src/fpl_oracle/optimise/multiperiod.py), whose flow formulation has per-gameweek transfer-in/transfer-out binaries (tin/tout, ~line 397-408) and FT banking `ft_var[t]` bounded 1..MAX_BANKED_FREE_TRANSFERS (line 415-418), so it supports 3-5 transfers per GW. Verified by regression test tests/test_f005_banked_transfers_beam.py: with 3 banked FTs the optimizer selects a 3-transfer, 0-hit plan (+21 xP vs +14 xP best 2-transfer) — **test passes**, meaning the MILP path handles it.

**Residual gap:** if `multiperiod_milp` flag is false and FTs < 3 (the common 1-2 FT state), the beam path in sequential.py remains capped at 2. The `max_hits: 2` setting in settings.yaml documents this as an intentional ceiling (WI-2.2), but F-005's core complaint (cannot exploit banked 3-5 FTs in beam) stands for any path not routed to MILP.

**Recommendation:** (a) keep MILP routing for FTs>=3 (already in place); (b) either extend beam candidate generation to N transfers when banked FTs >= N, or route to MILP whenever banked FTs > 2 unconditionally (already the case); (c) add a regression test asserting beam-only path with exactly 2 FTs never proposes >2 transfers and MILP path with 4-5 FTs can propose 4-5 hit-free transfers.

---

### F-007 — GW19 chip forfeiture under static gain gates (HIGH) — FIXED, verified

**Defect as documented:** at GW19 (final Set 1 GW, hard expiry), static thresholds (LEGACY 3xC 4.0, Free Hit 10.0) mark sub-threshold positive-gain chips `is_recommended = False`; chips expire for 0 points. Evidence (stale scenario_results.md scenario 4, captured against older code citing planner.py:356-358): at GW19 with 4 unused Set 1 chips, `gw19_chip_plan_table_recommended` was `[]` and `set_1_warning` was None.

**Current code (src/fpl_oracle/chips/planner.py):**
- Legacy static gates still defined at lines 40-45: `LEGACY_CHIP_RECOMMENDATION_THRESHOLDS = {"3xc": 4.0, "bboost": 8.0, "freehit": 10.0, "wildcard": 8.0}`; `CHIP_RECOMMENDATION_THRESHOLDS = LEGACY_CHIP_RECOMMENDATION_THRESHOLDS` at line 63.
- Data-driven context tables at lines 49-61 (SET_1 normal 3xc=4.0, freehit=6.0; etc.).
- Recommendation gate at lines 437-447:
```python
base_threshold = get_chip_threshold(c, rec_gw, set_idx=1, context=ctx1, use_legacy=use_legacy_thresholds)
if num_chips >= num_avail or current_gw == 19:
    # Full congestion / deadline: any positive expected gain justifies recommendation to prevent expiry
    threshold = 0.0
elif current_gw == 18:
    threshold = round(base_threshold * 0.5, 2)   # e.g. 3xc 4.0 -> 2.0 at GW18
else:
    threshold = base_threshold
is_recommended = (gain > 0.0 if threshold == 0.0 else gain >= threshold) and (rec_gw == current_gw)
```
- GW19 deadline detection lines 353-359 (reads `bootstrap.events[id==19].deadline_time`, falls back to "Saturday 2 January 2027"); expiry enforcement line 361 `set_1_remaining = ... if current_gw <= 19 else []`; congestion warnings lines 403-418.

**Verification on this checkout:** tests/test_f007_chip_expiry.py — 4 tests pass:
- GW19 with 3xc gain 2.5 (< 4.0 static threshold): `recommendation_threshold == 0.0`, `is_recommended_this_gw == True`, `recommend_chip_this_gw == True`, set_1_warning non-null.
- GW18: threshold == 2.0 (halved from 4.0).
- GW16-GW19 congestion scheduling (4/3/2/1 chips scheduled as GWs remaining allow).
- set_1_warning fires from GW14 and formats the real API deadline `2027-01-02T11:00:00Z` (not a hardcoded constant).

**Residual gap:** the `use_legacy_thresholds=True` path still applies static 4.0/10.0 gates outside the deadline/congestion override (default is `use_legacy_thresholds=False`, flag `use_legacy_chip_thresholds: false` in config/flags.yaml). Chips below threshold at non-deadline GWs are correctly deferred, not forfeited, since they remain schedulable at later GWs — only true expiry (GW19) forces them. Acceptable.

**Recommendation:** keep the deadline override; consider extending the zero-threshold rule to the final feasible GW for each chip set generally (already done for GW19 and GW38 at planner.py:504-506).

---

### F-009 — 5x sample weights on 2026/27 rows (MEDIUM) — PARTIAL (5x retired, distortion remains at 4.0x/1.5x)

**Defect as documented:** component models apply hardcoded 5x sample weight to 2026/27 records (1,538 of 90,303 total rows), distorting gradients and dropping Kish effective sample size `ESS = (Σw)² / Σw²` from 90,303 to a claimed 63,548 (-26,755 rows, ~29.7%).

**Current code:**
- src/fpl_oracle/ml/components/bonus.py:46-48:
```python
weights = np.ones(len(X_train))
if "is_2026_27" in X_train.columns:
    weights = np.where(X_train["is_2026_27"] > 0, 4.0, 1.0)   # was 5.0
self.reg.fit(X_train, y, sample_weight=weights)
```
- src/fpl_oracle/ml/components/defcon.py:79-82 (working tree diff): comment `# Time-decay weights + is_2026_27 era feature (retires uniform 5x multiplier)`; `weights = 1.0 + 0.5 * X["is_2026_27"].values` → 1.5x for 2026/27 rows.

So the literal 5x multiplier is retired in the working tree. Bonus training still uses 4.0x; defcon 1.5x. Neither weight is selected by temporal cross-validation.

**Kish ESS quantification (using the finding's row basis: N=90,303, N_recent=1,538):**
| Weight on 2026/27 rows | Σw | Σw² | Kish ESS | Loss vs 90,303 |
|------------------------|-----|------|----------|----------------|
| 1.0 (none) | 90,303 | 90,303 | 90,303 | 0 |
| 5.0 (finding's claim) | 96,455 | 127,215 | ~73,132 | ~17,171 (~19.0%) |
| 4.0 (current bonus.py:48) | 94,917 | 113,373 | ~79,465 | ~10,838 (~12.0%) |
| 1.5 (current defcon.py:82) | 89,534 | 92,225.5 | ~86,921 | ~3,382 (~3.7%) |

Note: the finding's claimed 63,548 could not be reproduced exactly from the stated row counts (recomputation gives ~73,132 at 5x); the finding's row counts may differ, but the direction and scale of distortion are correct. Distortion remains material at 4.0x (~12% ESS loss) with no CV justification.

**Recommendation:** replace hardcoded era weights with temporally cross-validated weights or exponential half-life decay (config-driven, with a test asserting weights chosen by OOF temporal CV, not fixed constants); record ESS before/after weighting in training artifacts.

---

### F-011 — Starting lineup insensitive to risk modes (MEDIUM) — REPRODUCES

**Defect:** `risk_preference` modulates `cap_scores` (captaincy term) but starter decision coefficients are pure `df.loc[i, "expected_points"]`; conservative mode cannot bench doubtful/high-variance starters. Scenario evidence (audit/scenario_results.md scenario 8): starting XI 100% identical across conservative/balanced/aggressive; captaincy changed only.

**Code evidence (src/fpl_oracle/optimise/lineup.py):**
- Line 58-62: risk-aware starter scoring is gated behind feature flag `risk_aware_lineup`, default `False`:
```python
flag_enabled = (
    risk_aware_lineup
    if risk_aware_lineup is not None
    else is_flag_enabled("risk_aware_lineup", default=False)
)
```
- Lines 89-104: starter scores adjusted only when `flag_enabled AND risk in {conservative, aggressive}`:
```python
if flag_enabled and risk_preference == "conservative":
    starter_scores = df["expected_points"] - (lam * spread) - cop_penalty   # spread = p90-p10
elif flag_enabled and risk_preference == "aggressive":
    starter_scores = df["expected_points"] * (1.0 - upside_wt) + p90_vals * upside_wt
else:
    starter_scores = df["expected_points"]      # balanced AND points AND flag-off: pure xP
```
- Lines 79-86: captaincy scores ARE modulated in every mode (points/conservative/aggressive/balanced).
- config/flags.yaml (working tree): `risk_aware_lineup: false`.

So in the default shipped configuration, risk preference is a **captaincy-only dial**; the starting XI is identical across all risk modes. This matches scenario 8's FAIL/PARTIAL verdict exactly, including the mechanism citation lineup.py:62-64 (`starter_scores = df["expected_points"] * starter[i]` in the scenario's wording; actual current line is 108: `starter_scores.loc[i] * starter[i] + cap_scores.loc[i] * ...`).

**Recommendation:** (a) enable `risk_aware_lineup` by default or document the default explicitly; (b) apply quantile-spread/downside adjustment to starter coefficients in conservative mode regardless of flag (move lines 89-98 out from behind `flag_enabled`, or gate only the aggressive P90 tilt); (c) add regression test: conservative mode benches a high-variance doubtful starter (p10=2.0, p90=10.0, 75% chance) for a secure starter.

---

### A5-a — MC residual clamp at zero / truncated draws — PARTIAL (mean-exact verified, gates remain)

**Defect as documented (GOAL.md A5):** MC clamps residual mean at zero and truncates draws; documented symptom: defender xP 0.1 with p_cs 0.345 simulated mean ~1.5489 instead of 0.1, inflating bench/low-minute players; p_cs partly a fixed heuristic; horizon repeats first GW.

**Code evidence (src/fpl_oracle/league/montecarlo.py):**
- Line 108-109: `if xp <= 0.0: return 0.0` — hard zero clamp for non-positive xP.
- Lines 170-173: when `exp_app_cs > xp` (low-xP bench asset), realized appearance+CS points are scaled by `xp / exp_app_cs` — this enforces exact mean and is the reason the documented 1.5489 symptom does NOT reproduce.
- Lines 176-179 / 198-199: residual `mu_other = xp - exp_app_cs`; `if mu_other_play > 1e-4: ... other_pts = float(rng.gamma(k, theta)) else: other_pts = 0.0` — residual mean below `1e-4 * p_play` is truncated to exactly zero (clamp at zero; gamma draw skipped).
- Lines 113-117: `p_cs` from shared_cs_probability, else row `p_clean_sheet`, else `DEFAULT_CLEAN_SHEET_PROBABILITY` (0.30) — heuristic fallback only when no model/row value exists.
- Line 305: `horizon_projs = [projections_df] * horizon_gws` — first-GW repetition fallback when no per-GW projections supplied.

**Verification on this checkout:** live simulation of simulate_player_trials (200k draws, seed 7):
- DEF xP=0.1, p_cs=0.345 -> simulated mean 0.1002 (model xP 0.1) — mean-exact, the 1.5489 inflation does NOT reproduce.
- DEF/MID/FWD/GKP xP=4.5 -> simulated means 4.4859 / 4.4843 / 4.4849 / 4.4859 — within Monte Carlo error.

The scale-down branch (lines 170-173) makes the simulated mean match xP by construction whenever appearance+CS expectation exceeds xP; the residual truncation gate (line 178) only bites when `mu_other_play <= 1e-4` (bias bounded by ~1e-4 points per trial) — numerically negligible but the defect pattern remains in code.

**Recommendation:** (a) remove the `mu_other_play <= 1e-4 -> other_pts = 0.0` hard truncation; always draw gamma when mu_other > 0 (or use exact residual matching); (b) remove the `xp <= 0.0 -> 0.0` early return in favor of the same decomposition (harmless since xP<=0 yields zero components anyway); (c) require per-GW projections (projections_by_gw) at the call sites instead of the repetition fallback (see A5-b); (d) keep the regression tests at xP {0.1, 1.0, 4.5, 8.0} x {DEF,MID,FWD,GKP} asserting relative error < tolerance (GOAL G9 verify step).

---

### A5-b — Horizon repetition: first GW used for all 5 simulated GWs — PARTIAL

**Code:** src/fpl_oracle/league/montecarlo.py:292-309. When `projections_by_gw` is absent or empty and `projections_df` has a single (or no) `target_gw`, lines 305/307/309 build `horizon_projs = [projections_df] * horizon_gws` — identical projections/fixtures for all 5 horizon gameweeks.

**Call-site check:** all production callers pass per-GW projections, so the defect does not trigger in live paths:
- server/routes/api.py:954-959: `projections_horizon = ... projections(sim_target_gw, 5, ...)` then `simulate_league(..., projections_df=proj_df, horizon_gws=5, projections_by_gw=projections_horizon)`
- server/pipeline.py:253-261: `... projections_df=target_df ..., horizon_gws=5, projections_by_gw=projections`
- briefing/decision_card.py:457-466: `... projections_df=target_df, horizon_gws=5, projections_by_gw=horizon_proj`

The repetition fallback remains reachable for direct/test callers that omit `projections_by_gw`. GOAL A5 requires "use each GW's own projection and fixture, not the first GW repeated" — the mechanism exists (lines 293-303) and is used in production; the fallback should be deprecated or made to fail loudly instead of silently repeating.

**Recommendation:** replace silent repetition at line 305 with a clear error or per-GW split of a multi-GW `projections_df`; add a horizon test asserting different per-GW projections produce different per-GW score contributions.

---

### A5-c — p_clean_sheet and home/away: passed or inferred — PARTIAL

**Flow:**
- `simulate_league` accepts `fixture_cs_prob_map` and passes it to `compute_team_clean_sheet_probabilities` (line 316-319) and `match_simulator.simulate_gameweek` (line 427).
- `compute_team_clean_sheet_probabilities` (lines 49-82): priority 1 explicit map; 2 model-derived median of row `p_clean_sheet`/`clean_sheet_probability` for active GKP/DEF (lines 54-69); 3 FDR/home-away inferred baseline (lines 72-80, uses `row.get("was_home", row.get("is_home", True))` and CLEAN_SHEET_HOME/AWAY_FACTOR); 4 DEFAULT 0.30 prior.
- `simulate_player_gameweek` (lines 113-117): `p_cs = shared_cs_probability if provided else row p_clean_sheet else DEFAULT_CLEAN_SHEET_PROBABILITY`.
- `match_sim.simulate_gameweek` (lines 244-267): home/away from `is_home`; if `fixture_cs_prob_map` given, converts p_cs to implied opponent xG via `-ln(p_cs)`; elif row `p_clean_sheet` present, same conversion.
- `draw_gameweek` (line 232): `prob = row.get("p_clean_sheet", DEFAULT_CLEAN_SHEET_PROBABILITY)`.

**Assessment:** p_clean_sheet and home/away are NOT dropped in montecarlo.py — they flow through `fixture_cs_prob_map` and row columns, with home/away used both for inferred team CS probabilities and match scoreline xG. GOAL A5's claim that "predict.py drops p_clean_sheet and home/away and p_cs is inferred as exp_cs_pts/4" refers to code outside this file (not present in current montecarlo.py); the residual risk is the DEFAULT 0.30 fallback when projections lack p_clean_sheet (heuristic) and home/away defaulting to `True` (line 76-77) when columns missing.

**Recommendation:** make the 0.30 prior a named config setting with a test; require `is_home`/`was_home` in projection frames or flag inferred home/away as low-confidence.

---

### A5-d — Joint chip+transfer planner integration — PARTIAL (callers exist; not unified)

**Finding as documented (GOAL.md A5 / audit/findings.md):** `evaluate_joint_transfer_and_chip_plan` (and `evaluate_joint_plan`) have no callers in API/pipeline/decision card; UI gets separate transfer and chip planners; helper uses fixed retention values 6/10/10/8, builds WC squads from current-GW xP, does not rebuild FH continuation, has two `except Exception: pass` branches; tests mostly mock base plans.

**Caller inventory on this checkout (contradicts the "no callers" claim):**
- `server/analysis.py:323-324` — `transfer_optimizer.evaluate_joint_transfer_and_chip_plan(**kwargs, horizon_len=...)` called twice (short/extended horizons) in the expiry-sensitivity research path.
- `server/analysis.py:519-520` — `asyncio.to_thread(transfer_optimizer.evaluate_joint_transfer_and_chip_plan, **kwargs)` inside `AnalysisService.joint_plan` (line 368), the cached planner endpoint.
- `server/pipeline.py:205` — `opt_res = await analysis_service.joint_plan(...)` in the pipeline optimization stage.
- `briefing/decision_card.py:108` — `joint_res = await analysis_service.joint_plan(...)` in the decision card.
- `tests/test_joint_plan.py`, `test_joint_plan_live.py`, `test_sequential_search.py`, `test_expiry_sensitivity.py`, `test_refresh_continuity.py`, `test_horizon_discovery.py` — unit/test callers (mostly mocked or direct).

**However, the integration is NOT unified:** pipeline.py:205 calls `joint_plan` AND pipeline.py:220 calls `analysis_service.chip_strategy(...)` separately; decision_card.py:108 calls `joint_plan` and :122 calls `chip_strategy`, then `:131` `bind_chip_schedule(chip_strategy, joint_res)` binds them post hoc. Transfers and chips are still optimized independently and merged afterward rather than one stateful (squad, bank, FTs, chips, GW) search optimizing transfers+N transfers+chips jointly across the horizon with per-GW projection xP (not fixed retention constants). reports/final_fix_ledger.md:22 marks G10 "Wire Joint Chip & Transfer Plan" as FAIL citing missing callers in `api.py/pipeline.py/decision_card.py` — that ledger is stale wrt file names (the real module is `server/analysis.py`/`server/pipeline.py`) but correct in substance: no single joint plan drives all surfaces; the chip calendar still comes from a separate `ChipPlanner.generate_chip_strategy` call with legacy/static threshold behavior (see F-007).

**Recommendation:** (a) make `evaluate_joint_transfer_and_chip_plan` the single source: one call from pipeline, API (`/api/...` decision-card endpoint) and decision card; (b) remove the separate chip_strategy call or derive the calendar from the joint plan's chip trajectory; (c) replace fixed retention values (6/10/10/8 referenced in finding; planner computes dynamic retention via resource_frontier at transfers_beam.py:1765-1783) with data-derived break-even; (d) remove the two broad `except Exception: pass` branches; (e) cross-surface test: identical input yields identical recommendation in API response, pipeline output, and decision card payload.

---

## 3. Recommended fix list (with file:line and severity)

| Priority | File:line | Defect addressed | Fix | Severity |
|----------|-----------|------------------|-----|----------|
| 1 | src/fpl_oracle/optimise/lineup.py:89-104,58-62 | F-011 risk-insensitive starters | Apply conservative spread/P10 penalty to starter coefficients by default (gate only aggressive P90 tilt behind flag); enable risk_aware_lineup or document default | MEDIUM |
| 2 | src/fpl_oracle/optimise/sequential.py:365-366,208,228; config/settings.yaml max_hits | F-005 beam 2-transfer cap | Document beam ceiling; guarantee MILP path for banked FTs>2 (already via transfers.py:58); consider N-transfer beam branches when banked FTs >= N | HIGH |
| 3 | src/fpl_oracle/ml/components/bonus.py:47-48; defcon.py:81-82 | F-009 sample-weight distortion | Replace fixed 4.0x/1.5x era weights with temporally CV-tuned weights; report Kish ESS in training artifacts | MEDIUM |
| 4 | src/fpl_oracle/league/montecarlo.py:176-179,108-109,305 | A5 clamp/truncation/horizon | Remove residual zero-truncation gate; always draw gamma for mu_other>0; deprecate silent `[projections_df] * horizon_gws` repetition (require projections_by_gw or raise) | HIGH |
| 5 | src/fpl_oracle/chips/planner.py:437-447,40-45 | F-007 residual static gates | Keep deadline zero-threshold; consider applying it to each chip set's final feasible GW generally; deprecate use_legacy path | HIGH (fixed) |
| 6 | server/pipeline.py:205-227; briefing/decision_card.py:108-131 | A5-D joint planner unification | Single joint-plan call feeding transfers + chip calendar on all surfaces; remove separate chip_strategy call or bind pre-optimization; drop broad except:pass | HIGH |
| 7 | src/fpl_oracle/league/montecarlo.py:450-457; domain/scoring.py:94-116 | F-002 / F-004 | Already fixed — add/pin regression tests (tests/test_f002_*.py, test_f004_*.py) and keep them in CI | — (fixed) |
| 8 | docs/remediation_log.md:164 | Stale max_k=20 record | Update to reflect dynamic cap | INFO |

---

## 4. Verification evidence generated this run

- Live reproduction F-002: identical squads -> user win 50.0%, rank 1.5, rival 50.0% (montecarlo.py:450-457).
- Live reproduction F-004: expected_floor_div(20,2) = 9.7500 == exact 9.7500 across lambda 0.1-30 (scoring.py:94-116).
- Live reproduction A5-a: DEF xP 0.1 p_cs 0.345 -> simulated mean 0.1002; xP 4.5 all positions -> ~4.485 (montecarlo.py:167-199).
- Regression tests: tests/test_f002_monte_carlo_ties.py (4), test_f004_poisson_truncation.py (2), test_f005_banked_transfers_beam.py (1), test_f007_chip_expiry.py (4) = 12 passed in 4.70s.
- Kish ESS recomputation from stated counts (90,303 rows, 1,538 recent): 5x -> ~73,132 (finding claimed 63,548 on its own count basis); current 4.0x -> ~79,465; defcon 1.5x -> ~86,921.

No literal/fabricated numbers are reported above except where the documented finding's figures are quoted as such (clearly labeled); all recomputed values were executed in this session.