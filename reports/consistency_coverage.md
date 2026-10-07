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
- End-to-end pipeline summary and chat tool output parity across the case matrix.
- UI pixels and revision middleware tied to every matrix case, not merely fixture overview/pending/error captures.
- Changed purchase/sale prices, locked-in/out constraints, exclusion rules, and explicit transfer-hit cases across every surface.
- Mid-request state/model/history changes, concurrent forced refresh and failed upstream sources at every stage.
- Rival/partial-league/news-provider failure variants across all surfaces.
- Complete historical pre-deadline advisor replay with real squads/news/model versions.

Test partitions overlap. Do not add their counts or claim the full test suite passed. A passing matrix bounds known cases, not every possible FPL state.
