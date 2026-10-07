# Cross-surface consistency coverage

October 7, 2026. This is a finite test matrix, not exhaustive certification or proof of future performance.

## Completed evidence

The published startup repair is commit03ddd3a. A concurrent official-profile host readback used bank£1.8m/1FT with overrides disabled. Squad/card/briefing agreed on XI, captain,63.24 next-GW expected points; chip surfaces agreed on Wildcard; PlanA/card/briefing agreed on13 transfer-ins and0hits. All six endpoints returned. Host times were28.7-87.3 seconds. No Windows timing promise follows from that run.

The follow-up matrix runs real planning/lineup engines against deterministic source fixtures, with six advice operations concurrently: squad, card, optimizer, chip strategy, contingency PlanA, and weekly briefing. It compares bank/FT, overrides, XI/bench, captain/vice, formation, points, transfers-in/out, selected count, hits, chip, nextFT, and stale state.

Cases:
- Zero bank and zeroFT.
- Bank18/1FT.
- Zero bank/max5FT with stale manager state.
- Manual override.
- Triple Captain available.
- Bench Boost available.
- Wildcard available atGW19.
- Free Hit available atGW20.
- Missing squad.
- Empty projections.
- Partial projections missing an owned player.
- Non-finite projection values.

Invalid source cases must produce unavailable advice or409, never a template squad or numerical solver failure. These cases do not force every chip to be selected, every transfer count, or every bank/FT combination.

The stale case found a real discrepancy: card, optimizer and PlanA did not include manager-state staleness. The invalid-projection cases found a real discrepancy: invalid scores could reach lineup/chip solvers. The follow-up patch propagates stale state and guards missing/invalid projection inputs.

JavaScript behavioral checks cover atomic publication for matching revisions, rejection of mismatched revisions, hung auxiliary requests not blocking core publication,90-second failure, and old generation responses not replacing a newer successful retry. CLI tests verify shared operations rather than a separate template.

## Still open

- The user's actual Windows startup after pulling03ddd3a.
- Native Python3.14 dependency execution, not reproduced on this Python3.11 host.
- Full pipeline/chat parity across every case: focused captain, offline captain reply, missing squad, other-GW captain, selected pipeline action and forced transfer cases now pass, but are not the full matrix.
- UI pixels and revision middleware tied to every matrix case, not merely fixture overview/pending/error captures.
- Locked-in/out constraints and exclusion rules across every surface. Focused purchase/sale-price and zero/oneFT single-transfer hit cases now pass across card, optimizer, PlanA, briefing and chat.
- Mid-request state/model/history changes, concurrent forced refresh and failed upstream sources at every stage.
- Rival/partial-league/news-provider failure variants across all surfaces.
- Complete historical pre-deadline advisor replay with real squads/news/model versions.

Test partitions overlap. Do not add their counts or claim the full test suite passed. A passing matrix bounds known cases, not every possible FPL state.

## Additional focused follow-up

Chat captain advice previously ranked the whole market and could recommend an unowned unaffordable player. It now uses configured canonical lineup/captain/vice. Offline chat no longer offers a separate speculative differential captain. Explicit different-GW captain requests and missing squad return unavailable.

Pipeline selected chip schedule now comes from the joint trajectory; independent chip calendar is retained separately as context. Focused pipeline summary compares selected action, captain, bank/FT and target chip.

Forced single-transfer tests use a profitable sale price and compare0FT (4-point hit) with1FT (no hit), bank after, move list and hits across card, optimizer, PlanA, briefing and chat. Cardft_used now means free transfers consumed, not total wildcard/free-hit changes. These tests do not cover every price history or constraints combination.

Latest overlapping partition:29 tests passed76.73s (matrix, chat/pipeline,CLI andrefresh).74 source files type-check clean. No full-suite or Windows execution claim.


## Final batch validation

Final selected partition:45 tests passed97.35s, covering expanded surface matrix, pinned-read/revision guard, refresh continuity, joint planning andCLI. JavaScript atomic publication, timeout, superseded response and auxiliary nonblocking scenarios pass. Lint clean;74 source files type-check clean. This is not the full suite.

Constraint cases verifykeep-owned,exclude-buy andexclude-team cannot pollute the unconstrained cached default;chat locked-in/out mirrors constrained optimizer. GET andadvice POST revision changes both return409. Fixture staleness is now propagated across all six advice surfaces even when manager state is fresh. Full arbitrary news/rival/provider failure and all concurrent state mutations remain unverified.
