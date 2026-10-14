# Open Questions

Record of open architectural, modeling, or rules questions requiring owner confirmation.

---

### 1. Odds Provider for Player-Level Markets (WI-3.2)
- **Question:** Free historical match-level odds are available from football-data.co.uk. For player-level odds (anytime goalscorer, first goalscorer), which odds provider API should be used in production?
- **Default Answer:** `odds provider = football-data.co.uk for match-level odds plus a snapshot job, defer player-level odds`. Match-level probabilities will be converted to player arrival rates using team Poisson attack/defence ratios with deterministic fallback.

### 2. Multi-Period MILP Time Budget (WI-2.4)
- **Question:** Is a maximum runtime of 60 seconds acceptable on the target machine for H=8 horizon multi-period transfer optimization with integer programming solvers?
- **Default Answer:** `60 s solve time is acceptable`. Enforce a 60-second hard timeout using HiGHS/CBC backend, returning the best incumbent integer solution with logged optimality gap.

### 3. Primary Planner Objective (WI-2.1 / WI-4.3)
- **Question:** Should the primary planner objective be raw expected points maximization, mini-league win probability targeting, or overall rank percentile?
- **Default Answer:** `objective = expected points by default with a mini-league mode driven by the manager profile`. Default to rolling expected net points over the horizon; mini-league chase/defend modes will be selectable based on manager profile and standings context.

### 4. GW6 Wildcard Squad Snapshot & Recommendation Context
- **Question:** What is the owner's FPL entry ID or squad snapshot from when the tool recommended a Wildcard in GW6, along with the exact recommendation text and reasoning provided by the tool at that time?
- **Default Answer:** Pending human input. In the interim, proxy scenarios (Scenario 2 and 3) are evaluated with synchronized fixtures.

### 5. Target Mini-League Priorities & Desired Outcomes
- **Question:** Which mini-leagues matter most to the owner, and what specific outcome does the owner want from each (e.g. win outright, top 3 / podium, prize threshold, or overall rank maximization)?
- **Default Answer:** Pending human input. Default behavior optimizes for top-1 win probability in head-to-head / small mini-leagues and overall rank percentile in broad public leagues.
