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

    E[G(i,f)] = P(play) · E[G | plays]      λ_team: the fixture's bookmaker odds (margin removed, Poisson fit
    E[G | plays] = λ_team(f) · s_G(i) · g(pos) · c(i)^0.5 · (E[min | plays] / 90) · exp(tree correction)
                                              to 1X2 and over/under 2.5), else team ratings fitted to recent
                                              market prices, else the walk-forward Dixon-Coles model
    s_G(i)  = player's share of his team's xG while on the pitch, last 30 matches, shrunk to the position
              (FWD 0.29, MID 0.11, DEF 0.037) with 1.5 team-xG of prior
    g(pos)  = goals per xG by position (DEF 0.85, MID 1.00, FWD 0.98: xG over-rates defenders' headers)
    c(i)    = his own goals per xG (last 30 matches, 3 xG of league-average prior); half of it carries forward
    tree    = Poisson LightGBM on the log of the above as offset, early-stopped on the latest 15% of matches:
              it corrects the formula and shrinks towards "no correction", not towards the average player
    Assists: same with s_A (share of team xG as xA) and 1.35 FPL assists per xA.
    Team totals: per match, Σ availability · E[G] = 0.963 · λ_team (own goals, 3.7% of goals, score for
    nobody) and Σ availability · E[A] = 0.879 · λ_team; goals of an absent player go to his team-mates.
    Cards: recent seasons weigh more (half per season back), since cards per appearance fall each season.
    Clean sheets: P(CS | 60+) = 0.5 x model + 0.5 x bookmaker-implied chance rescaled to the observed rate
    (held-out Brier better in all three seasons: 0.1768 -> 0.1725, 0.1907 -> 0.1850, 0.1957 -> 0.1946).
    Availability: FPL return dates ("Expected back", "Suspended until") give 0 until the first gameweek after the
    date, then 50/75/100% (a served ban 100% at once); unknown dates stay 0; press news needs two publishers or an
    FPL flag to rule a player out, otherwise 75%.
    Training data excludes 2024-25 managers (Assistant Manager chip), who scored with zero minutes.

    xP(i,t) = Σ_f xP(i,f) · availability(i,t)   availability from official status/chance; a one-week doubt decays
                                                 by half per further gameweek

Market ratings: log λ_home = μ + home + att[h] − def[a], log λ_away = μ + att[a] − def[h], ridge least squares
on the last 240 days of market prices with a 60-day half-life. Held-out against each match's own later prices:
correlation 0.95 one week ahead, 0.90 twelve weeks ahead (the Dixon-Coles model: 0.58–0.83 for the same week).
Odds are free from football-data.co.uk (season files priced around the FPL deadline; fixtures.csv for upcoming
games) and optionally The Odds API (ODDS_API_KEY).

Next-gameweek goalscorer odds (kickoffscore.com, averaged over several bookmakers, keyed by FPL ids): a priced
player's goals-if-he-plays comes from his anytime price. Anytime prices carry a large margin, most of it on
long shots, removed per team-match by the power method: chances p^k with k set so the priced players' goals,
-ln(1 - p^k) weighted by their chance of playing, equal the team's market goals (own goals excluded) less the
model's goals for unpriced players. GW8 2026-27: Haaland at Villa raw 58%, after margin 42% (the source's own
model 44%; this model alone 39%). Not yet validated on our own data (no free price history); the weekly track
record measures it from now on. With odds in place, held-out RMSE improved 1.965 → 1.956.

Scoring constants come from the live `game_config.scoring` (GKP goal 10, DEF 6, MID 5, FWD 4, DefCon 2).

**Evidence (rolling-origin holdouts 2024-25, 2025-26, 2026-27 GW1–7; each season predicted by models trained
only on earlier seasons):** actual / predicted goals and assists checked in 31 groups (position, the player's
xG rate, the size of the forecast, named players); 2 groups differ by more than chance (|z| > 2; about 1.4
expected by luck alone): assists forecast at 0.2–0.35 run 16% high, and Mbeumo out-scores his chances (34 goals
from 21.6 expected). Haaland 1.07, B.Fernandes 0.97, defenders' goals within chance. Goal deviance 0.1680 vs
0.1723 for the previous model, assist deviance 0.1789 vs 0.1853. Points: forecasts of 4–6 points are right on
average (0.99–1.00; the previous model was 10–14% high there).
DefCon uses logit-Platt calibration (held-out Brier 0.1088; the isotonic version plateaued at 0.80 live).

## 2. Plan objective (multi-period MILP, horizon H = clamp(weeks to chip expiry, 6, 12))

    J(plan) = Σ_t γ^(t−t0) · [ Σ_XI xP + xP(captain) · (1 + TC_t) + BB_t · Σ_bench xP + w_b · Σ_bench xP
                               + FH_t · (FH squad value − owned value) − 4 · hits_t ]
            + γ^H · V_FT(free transfers left) + λ_bank · bank_H

γ = 0.95 per week (later forecasts are less certain). Constraints: 15-man squad 2/5/5/3, ≤ 3 per club,
budget with selling price = purchase + ⌊(now − purchase)/2⌋, FT banking ≤ 5, one chip per week,
chips per set (GW1–19, GW20–38), Free Hit reverts, chips at least `min_gameweeks_between_chips` apart.

Solve: a portfolio of restricted solves, each run to the proven best: roll, plan_1, plan_2 (one and two
transfers), opt (unrestricted), hold (no chip now) and chip:X (each chip now); then, when needed, opt_nohit and
hold_nohit (no first-week hit), cover:p (league cover) and keep:e (keep an owned player). Fix-and-optimise
polishing runs only if a solve hit the safety cap, and the unrestricted plan then starts from the best plan found.
The reported gap is against the full solve's bound, never claimed proven unless met. HiGHS runs single-threaded
with a fixed seed, so a model always ends on the same plan. Ties in plan value (to the cent) are broken
canonically: fewest hits, fewest transfers, lowest player ids.

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
A hit has a second test first: the transferred players' own expected returns must cover the hit cost
(raw gain ≥ 4 per hit), before the confidence test above is applied.
This is what stops a 13-player Wildcard built on small samples from beating a two-transfer fix by a
margin smaller than its own uncertainty.

**Search and order of decisions.** Every solve runs to the proven best plan (relative gap 0.01%, about 0.05
points), so the same inputs give the same advice on any machine; time-limited searches stopped wherever the clock
and machine load left them and flipped close calls between back-to-back runs. The one-hour cap per solve is a
safety net only (a full 12-week plan proves out in about 6 minutes alone); a run takes about 30 minutes on a 10-core Mac.
Every margin above still gains 1 point (SEARCH_TOLERANCE), a fixed status-quo margin. The chip-vs-keep and
move-vs-roll decisions are made first; keep checks then run on the surviving plan with its own chip, and a keep may not undo the decision that plan
passed. The league tie-break only chooses among plans with the same chip, scoring title chances net of hits.
GW8 2026-27: Wildcard now no longer clears its margin; two transfers gain 1.18 against 1.70 needed; the plan
rolls the transfers and plays the Wildcard in GW9, keeping Haaland, Calafiori and Gross.

**Keep what you own unless selling is clearly better.** For the four dearest players the best plan sells this
week (only players good enough to start: at least the plan's weakest starter's forecast), the planner solves the
best plan that keeps him, same chip, same polish. Greedy from the dearest: a player is kept when the plan that keeps
him (and those kept so far) is within the confidence margin of the best plan. GW8 2026-27 without manual keeps:
Calafiori, Saka and Palmer kept; Haaland and Groß still sold.

**Team value at the plan's end.** Expected price change from this week's net transfers per owner: about 2.4 x £0.1m
per unit of the ratio over the next 4-8 gameweeks (2023-24, 2024-25; R2 0.12). A player held at the end is worth half
of a rise and all of a fall, priced at 0.25 points per £1m per remaining gameweek, discounted like the plan. Effects
are small: a strongly bought player adds about +0.1 point.

**Known gap: the value of waiting.** Chip-now vs keep-it compares both plans on today's forecasts, so a kept
Wildcard gets no credit for reacting to injuries first. A scenario version (absence spells per player, both arms
re-planned) was built and withdrawn: re-planning a 3-week window with every absence revealed at once rewards
the squad with more rotation risk (hindsight, not waiting), its estimate flipped sign across variants, and the
honest version (reveal a week, re-plan, repeat) needs about three times the solves of a 30-minute run. The
+0.9 figure once quoted here was measured while Wildcard plans could not be polished (a Free Hit ban bug) and is
withdrawn. Absences measured for any later attempt: regular starters start one in 8.6% of weeks; 53% last one
match, 25% four or more.

**Free Hit**: planned only in a blank or double gameweek, or the last week of its set (GW19 / GW38: use it or
lose it). In a normal week it does what one more transfer does.

## 4. Mini-league layer

Tracked rivals: every manager above you, and every manager below who can still finish above you with at
least 5% probability: gap ≤ 1.645 · 13 · √(weeks left) points (13 = per-week spread of the gap between two
managers, matching var_r below; 120 points at GW8, 38 at GW36). RIVAL_POINTS_WINDOW overrides it. For each:

    mu_r  = (my points − r's points today) + Σ_horizon (my held XI xP − r's held XI xP) + (J_plan − J_best)
    var_r = Σ_weeks Σ_players (w_plan − w_r)² · sd², scaled for the weeks after the horizon
    P(finish above r) = Φ(mu_r / √var_r)
    cov(D_r, D_s) = Σ (w_plan − w_r)(w_plan − w_s) · sd²     (the gaps share our own players)
    P(win) = P(D_r > 0 for every r) under that joint normal, from 20,000 fixed draws

Symmetric assumption: rivals improve their squads as well as our planner improves ours, so only today's
squad quality, today's gap, how far a plan falls below our best plan, and how many players we share enter.
Shared players cancel. sd per player-gameweek = (p90 − p10) / 2.563. Each rival's squad is their real one
(last released gameweek); a rival's first week keeps last week's captain if he is still in their XI, later
weeks use their top-xP captain.

League cover candidates: the three players we do not own who score most for the tracked rivals,
Σ_r Σ_t w_r,t(p) · xP_t(p), are each locked into week 1 of an extra solve with this week's chip unchanged.
They enter the same comparison as every other plan. Owning a rival's player lowers var_r; that helps
against a rival you lead (mu_r > 0) and hurts against one you chase (mu_r < 0), so the formula, not a
rule of thumb, decides whether covering is worth it. The card lists each cover checked with its P(win).

Use: expected points decide (sections 2–3). Among plans the forecasts cannot tell apart from the
recommendation (it does not beat them by its own confidence margin), the tool picks the one with the higher
P(win), by more than 0.5 percentage points, and says so on the card. Inside that tie the point gap is noise, so P(win) is compared with every plan
set to the recommendation's predicted points (only who owns whom counts), and a switch may never lower the real
P(win). Before this, a tied plan won on its extra predicted points alone, which undid the move-vs-roll rule
(GW8 2026-27 with Calafiori, Groß and Haaland kept: two transfers +1.24 against 1.75 needed, then chosen anyway). Over many remaining weeks var is large and this rarely changes the move;
near the end of the season it matters more, as it should. GW8 2026-27 (7 rivals in reach, all captaining Haaland): P(win) ≈ 22%, P(finish above leader) ≈ 28%;
keeping Haaland as a cover costs 5.4 points and lowers P(win) to ≈ 19%.

## 5. Captaincy and bench

Captain = argmax xP among starters (vice = best remaining starter, preferring a different kickoff). With league
data, every starter whose extra armband points are within max(m_min, z · σ · √2 · (multiplier − 1)) of the top
pick is compared by P(win) at equal predicted points, as in the tie-break; a near-equal captain replaces the top
pick only if it raises that P(win) by more than 0.5 percentage points and does not lower the real P(win). The top pick becomes
vice. Chasing a rival who shares your team, the less-owned captain is the only way to gain on him; leading,
matching his captain protects the lead.
The XI, captain and vice are chosen by exact enumeration of the legal formations (ties by player id), so the same
squad always gives the same lineup.
Bench order = expected autosub value given each starter's non-play probability and formation legality.
