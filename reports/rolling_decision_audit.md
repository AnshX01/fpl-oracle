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
