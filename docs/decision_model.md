# FPL Oracle decision model

What the tool decides each gameweek, the formula behind every step, and the evidence for it.
Goal: win mini-league 435956 over the season. Every recommendation answers one question:
*which legal action now maximises my chance of finishing first?*

## 0. Data basis

Before every advice run the tool pulls each gameweek FPL has finalized (finished and data-checked, bonus
confirmed) into the match history and retrains on it if it is new, so form and weights include the latest
completed gameweek. In-progress gameweeks are never used (their numbers can still change). The card shows
"form through GW x, model trained through GW y".

## 1. Player projection (per player i, gameweek t)

For each fixture f of player i in gameweek t (0 fixtures = blank, 2 = double):

    xP(i,f) = P(play) · app_pts
            + goal_pts(pos) · E[G(i,f)] + 3 · E[A(i,f)]
            + cs_pts(pos) · P(60+ min) · P(CS | 60+)          (P(CS | 60+) trained on 60+ minute rows only)
            + 2 · P(play) · P(DefCon | play)                   (trained on labelled seasons, played rows only)
            + E[bonus] + E[saves pts] − E[GC deduction] − E[cards] − E[own goal] − E[pen miss]

    E[G(i,f)] = λ_team(f) · share_G(i)       λ_team from the walk-forward Dixon-Coles team model
    Σ_team availability · share = 1          goals of an absent player go to his team-mates

    xP(i,t) = Σ_f xP(i,f) · availability(i,t)   availability from official status/chance; a one-week doubt decays
                                                 by half per further gameweek

Scoring constants come from the live `game_config.scoring` (GKP goal 10, DEF 6, MID 5, FWD 4, DefCon 2).

**Evidence (held-out 2025-26 GW27–38 + 2026-27 GW1–6):** mean xP 1.183 vs actual 1.182; calibration within
±0.2 pts in every position × predicted-xP bin with n ≥ 100; by price band 12.0m+ 4.62 vs 4.60; attacking share
player-level slope 1.03 (no elite compression); DefCon calibrated per bin up to p ≈ 0.45.
DefCon uses logit-Platt calibration (held-out Brier 0.1088; the isotonic version plateaued at 0.80 live).

## 2. Plan objective (multi-period MILP, horizon H = clamp(weeks to chip expiry, 6, 12))

    J(plan) = Σ_t γ^(t−t0) · [ Σ_XI xP + xP(captain) · (1 + TC_t) + BB_t · Σ_bench xP + w_b · Σ_bench xP
                               + FH_t · (FH squad value − owned value) − 4 · hits_t ]
            + γ^H · V_FT(free transfers left) + λ_bank · bank_H

γ = 0.95 per week (later forecasts are less certain). Constraints: 15-man squad 2/5/5/3, ≤ 3 per club,
budget with selling price = purchase + ⌊(now − purchase)/2⌋, FT banking ≤ 5, one chip per week,
chips per set (GW1–19, GW20–38), Free Hit reverts, chips at least `min_gameweeks_between_chips` apart.

Solve: portfolio of restricted solves (roll, 1 and 2 transfers, no-chip, each chip now, chip next week)
+ fix-and-optimise polishing of each; the unrestricted plan starts from the best one found. Reported gap is
against the full solve's bound, never claimed proven unless met.

## 3. Decision rule under uncertainty (the step an expert does by instinct)

A plan's gain over the status quo is a sum of projection differences. Each projection carries an
**estimation error** (the model's uncertainty about the player's true rate, not match luck), larger for
players with few minutes of evidence. Recommend action a over baseline b only if

    E[J(a) − J(b)] ≥ max( m_min , z · σ_est(J(a) − J(b)) )

    σ_est(Δ)² = Σ_players changed ( Σ_t γ^t · σ_est(i,t) )²    (same player's errors are correlated across weeks)

σ_est is fitted at training time on the latest held-out block:
σ_est² = Var_i(mean residual of player i) − mean_i(within-player residual variance / n_i)
(regular players, n ≥ 6, mean xP ≥ 2). Measured: 0.37 xP per player-gameweek, with no reliable dependence on
minutes of evidence, so one value is used. z = Φ⁻¹(0.75) = 0.674 (act on a 3-to-1 edge), m_min = 0.3 xP.
Limit: errors shared by team-mates (a mis-rated team) are not modelled, so σ_Δ is a lower bound for plans that
stack one team.
Baseline b: for transfers, rolling; for a chip now, the same chip next week (or the best no-chip plan).
This is what stops a 13-player Wildcard built on small samples from beating a two-transfer fix by a
margin smaller than its own uncertainty.

## 4. Mini-league layer

For each tracked rival r (managers above you, and within RIVAL_POINTS_WINDOW below):

    mu_r  = (my points − r's points today) + Σ_horizon (my held XI xP − r's held XI xP) + (J_plan − J_best)
    var_r = Σ_weeks Σ_players (w_plan − w_r)² · sd², scaled for the weeks after the horizon
    P(finish above r) = Φ(mu_r / √var_r),   P(win) ≈ Π_r P(finish above r)

Symmetric assumption: rivals improve their squads as well as our planner improves ours, so only today's
squad quality, today's gap, how far a plan falls below our best plan, and how many players we share enter.
Shared players cancel. sd per player-gameweek = (p90 − p10) / 2.563.

Use: expected points decide (sections 2–3). Among plans the forecasts cannot tell apart from the
recommendation (it does not beat them by its own confidence margin), the tool picks the one with the higher
P(win) and says so on the card. Over many remaining weeks var is large and this rarely changes the move;
near the end of the season it matters more, as it should. GW8 2026-27: P(finish above leader) ≈ 41%.

## 5. Captaincy and bench

Captain = argmax xP among starters (vice = best remaining starter, preferring a different kickoff).
Bench order = expected autosub value given each starter's non-play probability and formation legality.
