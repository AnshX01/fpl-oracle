# FPL Oracle repair checkpoint

This is unfinished implementation work, not a final release or accuracy certification.

Implemented repairs include shared projection/joint-plan caching, structured-news gates, stale/missing-data states, actual configured squad and financial state across advice/briefing/chat, no implicit serving retrain, finalized history refresh, disjoint temporal calibration, durable finalized-GW candidate training, and session-history persistence without restoring stale advice.

Completed scoped runs (overlap, do not sum):
- 54 domain/refresh/holdout/news/promotion/legal-rule tests passed.
- 36 model-gap/parity/leakage/calibration/evidence tests passed.
- 22 retraining/promotion/calibration/provenance tests passed.
- 17 decision-card/optimizer/cross-surface tests passed.
- 40 short component/API/parsing tests passed.
- Source lint passed; 70 source files passed type checking.
- 12 deterministic HTTP-fixture UI captures passed assertions. Desktop dark failed-advice and mobile light populated pixels inspected; fixtures are not live-user evidence.

Not complete:
- Full suite: broad runs timed out. Remaining network/scheduler/backtest groups need bounded reconciliation.
- Windows startup/basic-team latency not measured. Linux test execution is not Windows acceptance.
- Coherent refresh publication across every concurrent surface needs additional regression coverage.
- Full-advisor historical replay, sequential multi-chip planning and mini-league decision-quality validation remain open. Limited beam search is not a global-optimality proof.
- Model weights bundled in the original repository were not retrained or recertified by this checkpoint. Original reports and manifest claims are historical artifacts, not new validation.
- GitHub source publication blocked: repository access exists but no reliable supported source-push route was available. Main remains unchanged.

Secrets, runtime profiles, local database, caches and logs must not be included in a handoff bundle. Public model/history fixtures are not the user's current FPL team.
