# Rolling decision audit - 7 October 2026

## Decision result
Do not endorse the earlier 13-player Wildcard from the old screen. Its +24.86 discounted five-week edge was inflated by asymmetric future-chip pruning. Fixing that reduces it to +12.43. Extending the supported projection search to eight weeks changes the candidate to 10 transfers, retains Haaland, and gives +7.46 versus ordinary transfers now with later chip use. These are model comparisons, not a proven whole-season optimum or instruction to activate a chip.

An ordinary no-chip baseline is White -> Tarkowski, one FT, no hit: 54.64 model GW6 points versus 51.46 for an unchanged squad with an optimized XI. The user's current squad includes productive players; wholesale replacement is not justified by past-screen numbers.

## Implemented
- Symmetric future restructuring opportunities for chip-conserving states.
- Search the eight contiguous loaded projection weeks, not only five.
- Keep legal trajectories that preserve each chip through the horizon. Report point cost and later-value break-even rather than inventing unknown later chip value.
- Recompute each deadline; future moves are contingent, not commitments.
- Relevant-rival exposure tie-break considers observed rosters and hypothetical legal one-transfer-per-week evolution, worst-case exposure without invented behaviour probabilities. Maximum projected-points sacrifice remains 0.5.
- On-demand /api/league/scenarios stress diagnostics: held rosters, hypothetical non-price-increasing same-position moves, optimistic restructure envelope. These are not calibrated title odds and do not select the primary plan.
- All screens share selected plan scope; checklist, contingency, chip schedule, transfer prices and chat are aligned. Official flags/news attached to incoming players.
- Unknown overlap is labeled, not a fake 72%. Review no longer invents calibration, autosub or risk compliance. Price momentum does not imply a confirmed tonight change.
- Offline chat returns nonempty answers/errors, keeps newest history, reassesses on new news and distinguishes selected-plan captain from current owned team.
- Removed duplicate obsolete ordinary beam/Monte Carlo work from joint solver.

## Resource frontier from current model
Preserving through GW13 costs: WC 15.27 discounted points, FH 1.95, TC 5.88, BB 8.95. Corresponding undiscounted value at the horizon end that could reverse the choice: 23.02, 2.95, 8.86, 13.49. Later value is unknown. These thresholds are conditional on this bounded search and projections.

## Verification
49 selected behavioral tests passed in 27.98 seconds. Lint clean; type check passes 75 source files. Live official-source six-surface batch: league 30.76 seconds, core 72.16-74.10 seconds. Direct matrix/checklist/chat readback succeeded. Browser replay inspected at 1429x804 and 390x844, no page errors, visible prices/matrix/review/chat and readable verdicts. Replay is not native Windows runtime certification. No full-suite, exhaustive or global-optimality certification.

## Open limits
- No validated whole-season resource/squad tail value, calibrated evolving-rival behaviour or plan-level championship probability.
- P10/P90 Normal stress draws are uncalibrated, do not model team correlation and must not be sold as true odds.
- Hidden rival bank, buying basis, FT and chip choices remain uncertain. Refresh latest released observations each deadline.
- Minutes/form/projection model quality still determines the output. Loaded fixture projections are not proof of eight-week predictive accuracy.
- Native Windows readback and durable publication still pending for this batch.

Sources: https://fantasy.premierleague.com/api/bootstrap-static/ ; https://fantasy.premierleague.com/api/fixtures/ ; current official manager picks/history read through the project's API client. No FPL transfer or chip was submitted.

## October 7 usability/rules checkpoint

The user's native Windows report showed completed squad/chips/plans/briefing requests being discarded when a six-surface UI barrier hit 90 seconds. The core publication now runs as one coalesced background job. The UI polls progress and publishes squad, card and primary plan together only after captain/source/profile validation. Detailed league simulation, separate chip-calendar search and conditional backups are not required to display the ready core. Unmeasured/deferred results stay labeled, not fabricated. A real official-input HTTP run completed in 51.51 seconds on the development host; this does not certify Windows laptop timing.

Official rules corrections: the initial unlimited deadline grants one FT for the following GW, not a banked extra FT; Wildcard/Free Hit retain saved FTs unchanged; consecutive Free Hits are forbidden including GW19/20. Source: https://www.premierleague.com/en/news/4661029 . These rules changed the eight-week model candidate edge to about +7.3 discounted xP, still conditional on unresolved value beyond GW13.

Forward league evidence utilities reject post-deadline freezes, synthetic provenance and non-finalized outcomes. Counterfactual plans are not scored against the actual manager outcome. Reliability bins/Brier are descriptive, never sufficient for promotion. No calibrated claim is enabled and this utility is not yet wired into automatic deadline capture/scoring. Legacy GW6 prediction freeze remains unchanged.

Validation:26 joint/live/screenshot/rules/forward/rival/solver tests passed19.18s;29 surface/fault/job tests passed63.87s;JavaScript durable publication/hung-auxiliary/stale-generation regression passed;ruff clean;mypy77files passed. Desktop/mobile browser pixel replay inspected. Full-suite attempt exceeded bounded execution time without completion;no full-suite certification. Native Windows readback, automated forward evidence, and supported season-tail opportunity evaluation remain open.

## Local forward/expiry research leg after bbd3d53

Automatic league evidence is now a separate immutable store, invoked by the existing 15-minute forward job even when a legacy player freeze fails. It records one forecast inside the last24hours while the local server is running. The target is strict cumulative leadership after the next GW among a fixed observed manager set, not season victory or the whole league. Prior published XI/captain are hold assumptions; hidden changes become forecast error. Finalized official cumulative manager history supplies the outcome without subtracting transfer hits twice. Baseline Brier comparison remains descriptive; calibration promotion is disabled. Fresh rival picks, model manifest, deadline, snapshot and schema checks are required. No finalized production evidence exists yet.

An opt-in expiry-window comparison uses identical current inputs for short and extended searches. Actual GW6 inputs:8-GWedge7.35,throughGW19edge7.42,Wildcard candidate in both. Runtime106seconds on this host. The comparison assumes current price/minutes/form extrapolation, cannot know future news/postponements and is a pruned search. It does not establish tail calibration or global optimality and is deliberately excluded from normal startup. UI exposes the research separately and diagnoses evidence counts.22 focused forward/holdout/expiry/joint/rival/job tests pass10.17s;ruff and mypy78files pass;desktop/mobile research and diagnostics pixels inspected. This leg is not yet published.

## Final hardening after local b8d7e92

- Forward index reuse now checks SHA256 content, owner/league/GW/schema and predeadline timestamps. Concurrent capture is serialized, index publication atomic, source/profile/model changes rejected. Fresh published rival picks required. A legacy player-freeze failure still runs independent league evidence, covered by regression. Standings/rival stale flags preserved instead of ignored.
- Actual previous Free Hit is supplied to a fresh search, preventing GW20 after actual GW19 Free Hit. Legacy FT helper now shares the initial-unlimited-deadline replay. Future roadmap is conditional, never marked firm merely by being first.
- Whole-plan changing-rival stress is exposed separately. Actual model draws favored WC against holding/weekly-transfer rivals but FH against optimistic restructure. This scenario dependence is explicitly not a recommendation or calibrated winning probability.
- Plan B/C are available on demand without discarding ready core. Plan C now evaluates a legal same-position/budget/club-constrained complete squad, actual XI/captain and hit-adjusted net points. Removed fabricated -0.5 delta, unrelated primary transfer-outs and incorrect hit count. Hold fallback uses owned XI, not WC lineup. Behaviour regression verifies legal trade and points math.
- Actual core HTTP readback52.22seconds on this host, XI/captain agreement; detailed API47.06seconds; stress67.76seconds. Actual forward capture/retry succeeds, finalized scoring remains pending. Browser1429/390 inspected for core, alternatives, research and diagnostics;12dark/light/missing/failure fixture captures rerun. These are not Windows performance promises.
- A tracked full run passed322tests340.29seconds before the latest few corrections. Final stable-tree run is separately tracked; don't conflate the two. The independent synthetic proxy test passed123.77seconds and remains synthetic. Thresholds unchanged, no new weights promoted, no calibrated league probability enabled. Main untouched, no FPL actions.


## Final frozen-tree regression, October7

Commit59fe81e6fcab06937c586a7e06a6aed85e9f5c48:324passed350.21s,
0failures/errors/skips,source/tests/scripts/web unchanged during run. Includes
legalPlanC,XI-consistency,priorFH,immutableforwardforecast and cachedscore
provenance regressions. JavaScript durable publication/hungauxiliary/stalegeneration
checks pass. Model/legacyholdout bytes preserved;test-generated calibration/proxy
outputs excluded. NativeWindows,complete historical advisor replay,empirical
tail-value calibration and reliable winning odds remain unverified.
