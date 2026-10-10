# Open Questions

Record of open architectural or rules questions requiring human confirmation.

---

1. **Odds Provider for Player-Level Markets (WI-3.2):**
   - Free historical match-level odds are available from football-data.co.uk. For player-level odds (anytime goalscorer), which provider should be used? (Default: provider-abstracted with historical snapshotting).
2. **Multi-Period MILP Time Budget (WI-2.4):**
   - Is a maximum runtime of 60 seconds acceptable on the target machine for H=8 horizon optimization? (Targeting HiGHS backend).
3. **Primary Planner Objective (WI-2.1 / WI-4.3):**
   - Expected points maximization (default) vs. mini-league win probability target vs. overall rank percentile.
