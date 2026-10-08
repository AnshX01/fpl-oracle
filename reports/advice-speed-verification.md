# Advice calculation speed checkpoint

Base: main 0d392eb. Full GW19 horizon, chip/date searches and search-quality settings unchanged. Done lock and Undo unchanged. No .env changes.

## Parity

Controlled 60-player market with a legal owned squad, varied prices and projections, GW18-19 through chip expiry, 2 FT, all four available chips, measured chip/date searches. Ran the exact same script against base and optimised code in separate processes.

- Base: 41.34 seconds.
- Optimised: 0.65 seconds.
- Entire result JSON byte-identical after dropping chip_measurement_seconds (runtime metadata). This includes recommended_plan, chip comparisons and baseline trajectories.
- Script: scripts/verify_advice_speed_parity.py OUTPUT_JSON [RISK]. Select the base or current source tree with PYTHONPATH.
- This is controlled shortened-horizon parity, not full live whole-plan parity or a live runtime promise.

Earlier checkpoint evidence (reported by previous run): full 14-GW advice including chip measurement completed in 2m43s on public test manager 777777 in a local clone. Base did not complete after 14+ minutes. Piecewise comparisons: candidate_1 1104 calls, squad 53 solves, candidate_2 1232 calls, six sequential searches, no differences. Previous full suite: 397 passed. These were local tests, not CI.

## Progress display

Status comes from actual search events, not a timer or made-up percentage:

- Comparing transfers through the final gameweek.
- Comparing chip dates with the actual count.
- Checking each chip against saving it.
- Checking the final chip plan.
- Preparing advice and Ready still handled by atomic publication.

Worker updates return to the publication event loop, are scoped to the job, and cannot overwrite a different or finished job. Targeted worker-to-publication test passes. Visually inspected real rendered pixels at 1440/390/360/320px using mocked APIs. No horizontal overflow; longest status wraps at 320px. Done and Undo remain visible.
