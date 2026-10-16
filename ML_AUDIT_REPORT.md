# FPL Oracle — ML Model & Evaluation Audit Report

**Scope:** `src/fpl_oracle/ml/` (model_registry, train, ensemble, predict, eval), `src/fpl_oracle/data/` (features, historical, player_identity), `data/models/` bundle, `data/models/manifest.json`, `data/models/calibration.json`.
**State audited:** Working tree at `C:\Users\anshw\Documents\fpl-oracle` (git HEAD `5ab9b6e`; working tree **dirty** — 31 modified files, manifest `git_commit` `4b6db3cc` does not match HEAD). Line numbers cited below are for the files **as they exist on disk today**.

---

## 1. Executive Summary

| Area | Overall finding |
|------|-----------------|
| Hash / manifest integrity | **PASS for active bundle** — all 7 production files match active manifest `v2026.10.06.1713` exactly. GOAL.md Part A's claimed mismatch (`180cf122...` vs `8af274a4...`) is **not present in the current tree** (likely already remediated). |
| Rollback safety | **CRITICAL GAP** — `data/models/versions/` exists but is **EMPTY**: no verified restore points. `quarantine_and_recover()` would raise `ModelBundleCorruptError` if integrity ever fails. |
| Promotion gate | Logic is stricter than GOAL.md claims (mean-gain + latest-origin **are** enforced at lines 272-284), BUT the latest rolling origin fails the coverage band (71.28% < 75% min) and manifest-stored origins lack the `interval_80_coverage_pct` schema key the gate now requires. |
| Stacking methodology | Honest out-of-fold in `train.py` (fit on disjoint val/cal block, score on holdout). GOAL.md "in-sample stacking" claim **not borne out** by current code. |
| Calibration persistence | Honest: `calibrate()` is in-memory only; production `calibration.json` is written only via the atomic promotion path (`promote_candidate_bundle`). `save_calibration()` defaults to `scratch/` and is **never called**. |
| Prediction cache | Content-based key **is** present (model version + feature hash + availability hash), but **no TTL/age or size bound**; season hardcoded to `2026-27`. |
| Player identity | **Split personality**: live serving path uses the `PlayerIdentityResolver` ((season, element)-scoped — fixes the documented 841 ID collisions), but the **training feature path still pools history by `norm_name` across ALL seasons** (features.py lines 776, 811), so cross-season identity contamination remains in training data. |
| Interval coverage | Latest origin coverage **71.28%** vs 80% nominal (reports/model_eval.json); gate min is 75%. Two of three origins are inside [75, 85], the latest is not. |
| master_history.csv | **Not force-refreshed** — `train.py:165` calls `ensure_dataset_ready()` without `force_refresh=True`; `historical.py:77-78` short-circuits when the CSV exists. |

**Top risks:** (1) rollback cannot restore (empty versions/); (2) production model's latest-origin coverage is outside the gate band yet the model serves; (3) training features contaminated across player identities/seasons; (4) unbounded, never-expiring prediction cache.

---

## 2. Hash & Manifest Integrity Verification (Item 1)

I computed SHA256 of every file in `data/models/` independently with `hashlib.sha256` (65 KB streaming chunks — identical algorithm to `ModelRegistry.calculate_file_hash`, model_registry.py:368-375).

### 2.1 Active manifest vs actual files

Manifest active version: **`v2026.10.06.1713`** (manifest.json:150-212, status `production` at line 211).

| File | Manifest expects (`v2026.10.06.1713`) | Actual on disk | Match? |
|------|--------------------------------------|----------------|--------|
| `minutes_model.pkl` | `6fcb69189755b730618a983c3197c22c8ffbc1756af537f9dfa365ebe0ad3669` (line 159) | `6fcb6918…ad3669` | **YES** |
| `attacking_model.pkl` | `611091e62a65c6b5e1765c71713f69e1b5a77f4694494d1f0a84cc929f24597d` (line 160) | `611091e6…f24597d` | **YES** |
| `defending_model.pkl` | `7a07a777d0cc8c8bf144b5a1225430b5703ac8e4427da04b9b76b7bf8b303b8e` (line 161) | `7a07a777…b303b8e` | **YES** |
| `defcon_model.pkl` | `c0298907536f73e5f743013c120d15775fe970c600ef52973d450cc9a79e87f8` (line 162) | `c0298907…9e87f8` | **YES** |
| `bonus_model.pkl` | `abcb23681970ae376b3188848ec5188abba7d2a4098bb4d3d43cc3fb7682e2a8` (line 163) | `abcb2368…82e2a8` | **YES** |
| `cards_saves_model.pkl` | `d96518e17d97341729a99246cdb428ed459dda2f32f04d692c9c5a49ac979940` (line 164) | `d96518e1…9940` | **YES** |
| `calibration.json` | `4251387c9dad9ed85b5e6ab9edd8b5306c2059bfbc4dc7e730acb3a1c9806995` (line 165) | `4251387c…806995` | **YES** |

**Verdict:** Active bundle integrity **PASSES**. The release is NOT broken by a hash mismatch in the current tree.

### 2.2 Divergence from GOAL.md Part A (A1)

GOAL.md states: manifest expects `180cf122…`, actual calibration hashes to `8af274a4…`, `model_registry.py:203` raises `ValueError`.

None of these hold in the current tree:
- Active manifest expects `4251387c…` (line 165), actual is `4251387c…` — **match**.
- `180cf122fd3c5614f02b15e8ca6853063046f5402f07f2406086dba078f67ba1` appears only in the **archived** entry `v2026.10.06.1420` (manifest.json:75) — stale, not active.
- No hash `8af274a4…` appears anywhere in the manifest.
- `model_registry.py:203` is `active_secondary=bdata.get("active_secondary"),` (inside `paired_block_bootstrap_gw`) — **not** a `ValueError`. `ValueError` raises live at lines 409, 418, 561, 733, 743, 829.

Conclusion: A1's hash-mismatch release-breakage appears already remediated, **or GOAL.md describes an older state**. Do not rely on GOAL.md line numbers — they are stale relative to the dirty working tree.

### 2.3 Integrity-infrastructure caveats (still real)

1. **`data/models/versions/` is EMPTY** (verified by directory listing). No archived bundle directories exist, so `quarantine_and_recover()` (model_registry.py:422-524) cannot find any version with valid weights and raises `ModelBundleCorruptError` (line 492-498). If `verify_weight_integrity()` (line 527) ever detects mismatch on the production dir, recovery is impossible → total serving outage until manual retrain.
   - *Severity: CRITICAL.* Fix: repopulate `versions/<tag>/` for every promoted manifest entry, or change recovery to fall back to re-training.

2. `verify_weight_integrity(model_registry.py:560-561)` raises `ValueError(mismatch_reason)` when called with an explicit `weights_dir != self.models_dir` on mismatch — inconsistent behavior depending on caller (auto-recover on production dir, hard error elsewhere).

3. Manifest git_commit `4b6db3cc641e670c31ab8ac85446452acecdf7f7` (line 87/153) != git HEAD `5ab9b6e`; working tree dirty. Provenance metadata is stale.

4. **CQR files untracked by integrity:** `data/models/cqr/` contains 9 files (`cqr_DEF_q10/q90`, `FWD`, `GKP`, `MID`, `cqr_meta.json`) but **none** appear in any manifest `file_hashes`. `ScoringEnsemble._load_default_calibration()` (ensemble.py:45-55) loads CQR from this dir, yet integrity verification never covers them. A tampered/corrupt CQR file would pass `verify_weight_integrity`.

---

## 3. Calibration.json Content (Item 2)

`data/models/calibration.json` (1143 bytes, SHA256 `4251387c…`):

- `blend_weights: [0.72, 0.04, 0.24]` — **CONFIRMED** (lines 4-8). Matches the hard-coded default in `ScoringEnsemble.__init__` (ensemble.py:38) — the persisted blend equals the fallback default, so it is impossible to tell from the artifact whether `fit_stacking_weights` actually moved the weights or they stayed at the prior.
- `z10: -0.7798`, `z90: 1.052` (lines 2-3).
- `bucket_quantiles` for 12 position/minutes buckets with `q10/q90/n` (lines 9-69).

Note on honesty: `ensemble.py:get_calibration_dict()` (line 82-97) rounds to 4 decimals; persisted file has 4-decimal `z10`/`z90`. Fine.

---

## 4. Promotion Gate Logic (Item 3) — `check_promotion_gate`

File: `src/fpl_oracle/ml/model_registry.py:167-297`.

### 4.1 What keys it reads

| Check | Code | Key(s) read |
|-------|------|-------------|
| Bootstrap sub-gate (WI-3.9) | :187-206 | `bootstrap_data` OR `candidate_predictions`/`active_predictions`/`targets`/`gameweeks` |
| Candidate MAE | :211 | `candidate_metrics.get("ml_mae", candidate_metrics.get("mae"))` |
| Baseline MAE | :212 | `candidate_metrics.get("base_mae", candidate_metrics.get("best_baseline_mae"))` |
| Active MAE | :231 | `active_metrics.get("ml_mae", active_metrics.get("mae"))` |
| Rolling origins | :244 | `rolling_origins` param or `candidate_metrics.get("rolling_origins")` |
| Per-origin schema | :254-256 | `ROLLING_ORIGIN_SCHEMA_KEYS` = `["season","train_size","test_size","mae","rmse","spearman","best_baseline_mae","gain_vs_baseline","interval_80_coverage_pct"]` (eval.py:24-34) |
| Per-origin finiteness | :258-260 | `mae`, `best_baseline_mae`, `gain_vs_baseline`, `interval_80_coverage_pct` |
| Mean gain vs baseline | :272-277 | `gain_vs_baseline` per origin (or `best_baseline_mae - mae`) |
| Latest origin gain | :279-284 | `gains[-1]` |
| Coverage band | :287-295 | `interval_80_coverage_pct` must be in `[coverage_min, coverage_max]` = `[75.0, 85.0]` |

The gate helper itself is **tolerant on key names** (`.get("ml_mae", .get("mae"))` etc.).

### 4.2 Flaws

**Flaw G1 — Caller uses brittle direct key access.** `verify_and_promote()` (model_registry.py:931, 933) does `cand_mae = candidate_metrics["ml_mae"]` and `active_mae = active_metrics["ml_mae"]` — direct subscription, `KeyError` if any emitter uses `"mae"` only. The gate helper tolerates both; the caller does not. *Severity: MEDIUM.*

**Flaw G2 — GOAL.md claim "gate silently skips mean-gain and latest-origin" is FALSE for current code.** Lines 269-284 explicitly reject `mean_gain <= 0.0` and `latest_gain <= 0.0`. Both checks are present and enforced. (GOAL.md/A4 appears to describe an older revision.)

**Flaw G3 — Manifest-stored rolling_origins lack the coverage key the gate now requires.** The archived/production entries `v2026.10.06.1643` (lines 105-145) and `v2026.10.06.1713` (lines 170-209) record `mae`, `rmse`, `spearman`, `base_*`, `gain_vs_baseline` — but **no `interval_80_coverage_pct`**. If these origins were re-fed to `check_promotion_gate`, it would return `False, "Origin '...' missing interval_80_coverage_pct."` (lines 288-290). Schema recorded at promotion time does not match the schema the gate validates. *Severity: HIGH.*

**Flaw G4 — Latest origin fails the coverage band.** Current `reports/model_eval.json` rolling origins:
- Holdout 2024-25: MAE 1.012, gain 0.029, coverage **77.34%** (inside [75,85])
- Holdout 2025-26: MAE 0.948, gain 0.042, coverage **81.77%** (inside)
- Holdout 2026-27: MAE 1.895, gain 0.057, coverage **71.28%** (**BELOW** 75.0 min) — spearman collapsed to 0.424.

An honest gate run against the current evaluation would **REJECT** on the latest-origin coverage check (:287-295), yet `v2026.10.06.1713` is status `production` and serving. *Severity: HIGH.* (GOAL.md A4 cites "73.515% vs 80% nominal"; the current tree shows 71.28% overall / 71.28% latest-origin in model_eval.json:66-68 and 72.64-76.74 in sibling reports — same conclusion: coverage is under the 80% nominal and the latest origin is under the gate's 75% floor.)

**Flaw G5 — `evaluate_uncertainty_calibration` verdict band is too lenient.** eval.py:152 marks coverage "WELL-CALIBRATED (±3.0%)" for `abs(cov - 80) <= 3.0`, i.e. 77-83% passes. A 71.28% origin gets "NARROW INTERVAL" but no hard failure outside the promotion gate. *Severity: LOW.*

---

## 5. Stacking Fit Methodology (Item 4) — `train.py`

File: `src/fpl_oracle/ml/train.py`, `compute_rolling_origin_cv` (lines 31-158).

### 5.1 Out-of-fold analysis

- Lines 55-64: For each origin, a `val_mask` is carved **out of** `train_mask` (`fit_mask = train_mask & ~val_mask`). Models are fit on `X_tr` (train minus val) — line 79 `m.fit(X_tr, Y_tr)`.
- Lines 90-112: `ens.fit_stacking_weights(v_comps, X_v, Y_v)` (line 111) and `ens.calibrate(v_comps, X_v, Y_v)` (line 112) run on the **disjoint validation block** (`X_v`, strictly prior to the holdout test set `X_te`), then predictions are produced on `X_te` (line 115).
- `train_all_models` (lines 242-244): final candidate ensemble fits stacking + calibration on `X_cal` (the calibration block, disjoint from the final fitting block `X_fit`, lines 200-206) and is evaluated on the holdout `X_val` (lines 249-258).

**Verdict:** Stacking is fit **OUT-OF-FOLD** in both sites. GOAL.md A4's claim "`train.py` fits stacking on in-sample 2025-26 predictions" is **not accurate for the current code**. The comment at train.py:92 even asserts this: "Out-of-fold validation block strictly prior to holdout origin for calibration and stacking weights (G5)".

Residual concern: the 2026-27 origin test set is tiny (`test_size: 2054`, manifest line 200) with MAE 1.895 / spearman 0.424 / coverage 71.28% — the most recent origin is degrading, but the gate's mean/latest-gain checks (positive gains 0.042/0.057) still pass because they measure gain vs baseline, not absolute quality. *Severity: MEDIUM — consider an absolute-quality floor in addition to gain.*

---

## 6. Calibration Persistence Path (Item 5) — `ensemble.py`

File: `src/fpl_oracle/ml/ensemble.py`.

### 6.1 `calibrate()` (lines 111-149)

- Fits `calibrated_z10`/`calibrated_z90` (empirical residual percentiles, lines 126-127) and `bucket_quantiles` (lines 132-147) on the passed calibration set.
- **In-memory only** — line 148 comment: "In-memory fitting only; disk persistence is handled exclusively by promotion routine". It does **not** write any file.
- Returns `(z10, z90)` only (line 149); bucket quantiles live on `self.bucket_quantiles`.

### 6.2 `save_calibration()` (lines 99-109)

- **Defaults to `scratch/calibration.json`**, NOT the production path (`Path("scratch") / "calibration.json"`, lines 101-102). `scratch/calibration.json` does **not exist** in the repo.
- **Never called anywhere in `src/`** (grep found only the definition). Calibration reaches disk exclusively through `ModelRegistry.promote_candidate_bundle` (model_registry.py:736-743), which writes `calibration.json` into a staging dir and atomically swaps into `data/models/` via `.tmp` + `os.replace` (lines 857-862, 865-867) — the honest atomic path.

### 6.3 What is actually served

- Module singleton `scoring_ensemble = ScoringEnsemble()` (ensemble.py:567) → `_load_default_calibration()` (line 42-44) loads `MODELS_DIR / "calibration.json"` at import (ensemble.py:45-49), so served blend weights/z-scores come from the promoted production file. `predict.py:200` uses this singleton read-only (`aggregate_components` does not mutate calibration state).
- **No evaluation path mutates the production `calibration.json` or the served singleton.** `backtest.py:148` creates its own local `ScoringEnsemble()` and calibrates it at line 269, but that mutation is in-memory, local to the backtest process, and never persisted. GOAL.md A1's "Evaluation mutates active calibration outside atomic promotion" is **not borne out** by the current code.

**Verdict:** Calibration persistence is honest (atomic promotion only; scratch default unused). *Recommendation: remove or repoint the unused `save_calibration()` scratch default to avoid confusion.* *Severity: LOW.*

---

## 7. Prediction Cache Staleness Risk (Item 6) — `predict.py`

File: `src/fpl_oracle/ml/predict.py`, `predict_gameweek`, lines 152-171.

### 7.1 Cache key construction

```python
model_version = str(active_entry.get("version", "default"))   # line 155, from manifest active version
season = "2026-27"                                              # line 158 — HARDCODED
feature_hash = sha256(pd.util.hash_pandas_object(features_df[FEATURE_COLUMNS + [...]], index=True).values.tobytes())[:16]  # lines 159-166
avail_hash = sha256(str(sorted(reconciled_availabilities.items())).encode()).hexdigest()[:8]  # lines 167-170
cache_key = f"{model_version}:{season}:gw{target_gw}:{feature_hash}:{avail_hash}"  # line 171
```

- **Content-based: YES** — `feature_hash` hashes the actual feature values (with index), and `avail_hash` hashes availability overrides.
- **Model version: YES** — included via manifest `active_version`; promotion to a new version invalidates the cache (and `load_or_train` clears `self._cache` at lines 71, 82).
- **Age / TTL: NO** — no expiry, no max size, no last-access tracking. `self._cache: dict = {}` (line 41) grows unbounded for the process lifetime. Historical refresh clears it (`historical.py:163`), but a long-running API process accumulates entries forever.
- `season` is hardcoded `2026-27` (line 158) — multi-season support would collide keys across seasons.

**Verdict:** Staleness from *content change* is mitigated (content hash); staleness from *time drift / unbounded growth* is real. *Severity: MEDIUM.* Fix: add TTL/maxsize (e.g. `cachetools.TTLCache`) and include `season` dynamically instead of hardcoding.

---

## 8. Player Identity Cross-Season Collision Evidence (Item 7 + A3)

File: `src/fpl_oracle/data/features.py` around lines 776, 809-811 (training path); `src/fpl_oracle/data/player_identity.py` (live path fix); `src/fpl_oracle/data/historical.py` (master_history refresh).

### 8.1 What `player_identity.py` says (the documented problem)

The module docstring (player_identity.py:4-7): "Eliminates the **841 element integer ID collisions across seasons** in master_history.csv by establishing a stable canonical player identity table keyed by normalized player identity, season-scoped element IDs, and official FPL player codes." This confirms the GOAL.md A3 premise: FPL reuses element IDs each season (e.g. ID 1 maps to multiple people across seasons — Balogun/Vieira/Raya per GOAL.md), and the codebase itself acknowledges 841 such collisions.

### 8.2 Live serving path — FIXED

`features.py:1116-1118`:
```python
p_hist = player_identity_resolver.get_player_history(
    elem_id=elem.id, season="2026-27", full_name=full_name, web_name=elem.web_name, code=elem.code
)
```
`PlayerIdentityResolver.get_player_history` (player_identity.py:190-224) resolves via `resolve_canonical_id` (line 147-188): code → `(season, elem_id)` → full name → web name, then `_canonical_to_indices` (built at player_identity.py:98-99 keyed by `norm_name` per row) pulls only that canonical identity's rows. `season_elem_to_canonical` (line 98) scopes element IDs **by season**, so ID 1 in 2023-24 and ID 1 in 2026-27 are distinct identities. **Live serving identity handling is sound.**

### 8.3 TRAINING feature path — STILL CONTAMINATED

`FeatureEngineering.build_historical_features` (features.py:526-988):
- Line 776: `df_player = df.sort_values(by=["name", "season", "round"])` — sorted by **name**, not element.
- Lines 809-811:
```python
if "norm_name" not in df_player.columns:
    df_player["norm_name"] = df_player["name"].apply(normalize_player_name)
grouped = df_player.groupby("norm_name", group_keys=False)
```
- Lines 814-848: rolling/shifted features (`prev_points`, `season_points_mean`, `roll_minutes_3/5/8`, `roll_points_*`, `roll_xG_*`, `roll_xA_*`, `roll_starts_ratio_5`, ...) computed via `grouped[...].apply(lambda s: s.shift(1)...)`.

**All rolling history is pooled by `norm_name` across ALL seasons in the training matrix.** Two different players who normalize to the same name (name collisions, or the same name reused after a transfer/retirement) have their match histories concatenated and shifted into each other's features — cross-season leakage/contamination in the training data. The `PlayerIdentityResolver` fix exists but is **not applied to the training pipeline**. *Severity: HIGH.*

### 8.4 master_history.csv not force-refreshed

- `historical.py:77-78`: `ensure_dataset_ready(force_refresh=False)` returns the existing CSV without re-downloading/rebuilding.
- `train.py:165`: `historical_manager.ensure_dataset_ready()` — called **without** `force_refresh=True`.
- A `refresh_current_season()` method exists (historical.py:107-164) with atomic `.tmp` + `os.replace` (line 146) and sha256 metadata (line 158), plus cache invalidation (`projection_engine._cache.clear()`, line 163) — but the training pipeline never calls it. *Severity: MEDIUM.*

Also, the live-history cache in `features.py:1042-1053` keys on `(st_mtime_ns, st_size, bootstrap tuple)` — not a content hash, so a same-size/same-mtime rewrite of `master_history.csv` would serve stale history. *Severity: LOW.*

---

## 9. Interval Coverage Honesty (Item A4 residual)

- `reports/model_eval.json` (current served evaluation): overall calibration coverage **71.28%** ("NARROW INTERVAL (71.28% vs 80% nominal)", model_eval.json:66-72); per rolling origin: 77.34% / 81.77% / **71.28%** (model_eval.json:88-138). Latest origin below gate floor of 75%.
- `reports/eval/20261015_113259/summary.json`: `coverage_80_pct: 85.2` — slightly **above** the gate's 85% max band (would also fail the gate's upper bound).
- GOAL.md A4 cites 73.515% vs 80% nominal; current tree shows 71.28-77.34% across origins — same direction: intervals are miscalibrated (under-covered on the latest origin) and the gate band [75, 85] is being violated by production-serving origins.

---

## 10. Fix Recommendations (with file:line) and Severity Ratings

| # | Recommendation | File:lines | Severity |
|---|----------------|------------|----------|
| F1 | Populate `data/models/versions/<tag>/` for every promoted manifest entry (or add re-train fallback) so `quarantine_and_recover()` can restore after corruption. Currently `versions/` is empty → recovery raises `ModelBundleCorruptError`. | model_registry.py:422-524 | CRITICAL |
| F2 | Make `check_promotion_gate` reject coverage outside [75,85] **and** require `interval_80_coverage_pct` in manifest-stored origins; backfill coverage into existing manifest origins or re-evaluate before re-promotion. Latest origin 71.28% < 75% while model is `production`. | model_registry.py:244-295; manifest.json | HIGH |
| F3 | Apply `PlayerIdentityResolver` (season-scoped element identity) to the **training** feature pipeline instead of pooling by bare `norm_name` across all seasons, to close the 841-ID cross-season contamination. | features.py:776, 809-811; player_identity.py | HIGH |
| F4 | Call `ensure_dataset_ready(force_refresh=True)` (or `refresh_current_season()`) in the training pipeline so `master_history.csv` is not silently served stale. | historical.py:77-78; train.py:165 | MEDIUM |
| F5 | Add TTL/max-size to `ProjectionEngine._cache` (currently unbounded, never-expiring), include `season` dynamically instead of hardcoded `"2026-27"`, and consider hashing model file content (not just version string) into the cache key. | predict.py:41, 152-171 | MEDIUM |
| F6 | Make `verify_and_promote` use the same tolerant key access as `check_promotion_gate` (`get("ml_mae", get("mae"))`) instead of direct `["ml_mae"]` subscription to avoid `KeyError`. | model_registry.py:931, 933 | MEDIUM |
| F7 | Add CQR files under `data/models/cqr/` to manifest `file_hashes` so `verify_weight_integrity` covers them; they currently load via `_load_default_calibration` but are untracked. | model_registry.py:377-395; manifest.json | MEDIUM |
| F8 | Remove or repoint the unused `save_calibration()` scratch default (`scratch/calibration.json`); calibration is correctly persisted only via `promote_candidate_bundle`, so the scratch path is dead code that invites confusion. | ensemble.py:99-109 | LOW |
| F9 | Tighten `evaluate_uncertainty_calibration` verdict band or tie coverage breach to a hard gate failure; ±3% "well-calibrated" (77-83%) masks under-coverage. | eval.py:131-168, esp. :152 | LOW |
| F10 | Reconcile manifest provenance: `git_commit` (`4b6db3cc`) ≠ HEAD (`5ab9b6e`), working tree dirty (31 modified files). Pin provenance at promotion or regenerate manifest from the committed tree. | manifest.json:87, 153 | LOW |

### Bottom line
The most dangerous issues are operational-safety (F1: no rollback restore points) and evaluation-honesty (F2: production model serving while its latest-origin coverage is below the gate's own floor). The identity fix (player_identity.py) exists for live serving but has not been ported to the training feature pipeline (F3), which remains the main ML-integrity gap. None of the four GOAL.md Part A claims hold exactly as stated in the current tree — A1's hash mismatch and A4's "silently skips"/"in-sample stacking" claims are not reproduced — but the underlying risks they point at (coverage band violation, unstaged calibration writes, identity collisions) are real and documented above with exact locations.

---
*Prepared by ML Audit agent. Hashes computed independently via `hashlib.sha256`; all file:line references verified against the working tree at `C:\Users\anshw\Documents\fpl-oracle`.*