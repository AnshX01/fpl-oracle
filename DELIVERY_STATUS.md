# Delivery status, October 7, 2026

This is a usable repaired source package, not certification that the full original decision-quality goal is complete. It never submits transfers or chips to FPL.

## Start on Windows

1. Extract the ZIP into a new folder. Do not overwrite your existing `.env` or user data.
2. Copy your existing `.env` into the new project folder, or copy `.env.example` to `.env` and set your manager and target league IDs.
3. Open PowerShell in that folder and run `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
4. Optional: `start.ps1 -SetupOnly` verifies setup, and `start.ps1 -Port 8001` selects another loopback port.

The package contains no owner `.env`, profile, database, chat history or credentials. Free/no-key advice works without an LLM key; optional providers stay in `.env`. Startup fetches the official squad separately from expensive advice. Advice is unavailable rather than replaced by a demo when required data is missing.

## Included repairs

- Shared configured squad, zero-safe bank/FT, lineup/captain/chip advice across web, card, briefing, chat and CLI.
- Persistent session history and refresh provenance; coherent revision publication, cleared old advice on failure.
- Bounded sequential transfer/chip planning with bank, purchase/sale prices, free transfers/hits, chip sets/expiry, Free Hit reversion and permanent Wildcard.
- Balanced observed-rival roster/gap utility bounded to 0.5 discounted expected points loss, without guessed upcoming rival captains.
- Finalized data refresh, complete registered-player population, preservation of historical double-GW rows, durable candidate policy, gated model promotion and interrupted-swap recovery.
- Model weights and calibration are preserved with their manifest-verified bytes. No incidental user data is bundled.

## Evidence and limits

Scoped test partitions and actual delayed-response JavaScript checks passed. These overlap; no full-suite count is claimed. Lint and typing passed. A clean extracted-package startup/hash smoke test is recorded separately. Desktop/mobile visual checks use clearly labeled deterministic fixtures, not live manager screenshots.

Corrected-population training results are in `reports/corrected_population_*`. Exact active holdout MAE is0.959, candidate0.961, best baseline1.005 on13,973 rows. Candidate rejected for slight active-MAE regression. Exact active aggregate80% interval coverage is83.75%. Separately refitted temporal rolling-origin models have coverages81.21%,84.22%,76.74% and positive baseline gains; those are not all the exact served weights. The prior current-season coverage failure disappears after including the full registered-player population. No thresholds were changed and no candidate weights were promoted. Sequential search is pruned and does not prove a global optimum or future wins.

Full historical advisor/mini-league replay is not certified: the available older simulation uses synthetic league states, and complete historical pre-deadline squad/news/model inputs are not recovered. Windows PowerShell and real user laptop timing were not executed on this host. GitHub publication is not complete merely because this ZIP or local commits exist.


## Targeted Windows pending-advice repair

The first real Windows run loaded official team/bank/FT but left advice calculating. That is not an advice-ready result. The targeted repair reuses unchanged history/form across projected weeks, stores snapshot provenance without expensive nested pandas metadata copies, shares plan/chip/news work across surfaces, binds contingency Plan A to the same chip decision, separates auxiliary requests from atomic advice publication, and adds a 90-second unavailable state with cancellation and old-response rejection. Rival picks now use the latest released official event. Diagnostics no longer default to Passing or Operational without evidence.

Host measurements against the actual manager's public official squad and full live player pool are not Windows timing promises. Projection-only time fell from60.233s to16.246s. A final concurrent official-profile six-endpoint run (bank1.8m/FT1, no overrides) completed all endpoints without exception: league28.7s, squad61.2s, PlanA61.2s, chips67.1s, card71.2s, briefing87.3s. Returned JSON agreed on XI, captain,63.24 next-GW xP, Wildcard and13 transfer-ins. Selected transfer-count correction is tested. The user's laptop needs the updated code and a real startup readback before declaring its session fixed.


## Batched consistency follow-up

A finite concurrent case matrix andfocused chat/pipeline/price/hit/constraint/revision tests found andfixed missing manager/fixture stale flags, invalid projections reaching solvers, market-based chat captain divergence, pipeline selected-chip schedule divergence, special-chipFT-used counts, andchanged POST advice bypassing the snapshot guard. See `reports/consistency_coverage.md` for tested cases andremaining gaps. Final scoped partition45 tests passed97.35s;no full-suite,exhaustive orWindows execution claim. The user must still pull/restart andverify the actual Windows result.
