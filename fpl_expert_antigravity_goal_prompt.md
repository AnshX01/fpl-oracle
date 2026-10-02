# GOAL: Build "FPL Oracle" — a local, ML-driven Fantasy Premier League expert that helps me win my mini league

You are an autonomous senior engineer + data scientist + FPL analyst. Build the complete tool described below, end to end, in one run. Plan first, then build, then test, then document. Do not stop to ask me questions; make sensible decisions, record them in `DECISIONS.md`, and only stop for things only I can provide (API keys), which you must list clearly at the end and in `SETUP.md`. Do not leave stubs, TODOs or mock data in the final product. Everything must actually run locally on my machine.

The season is **2026/27**. Today's date is in the first week of October 2026, so the season is already underway. Never assume the gameweek: always derive it live from the API.

---

## 1. PRODUCT VISION

One local web app (localhost) plus a CLI that behaves like a world-class FPL analyst I can talk to at any time. I give it my FPL team (Manager ID, or a pasted/uploaded squad). It pulls everything live, runs an ML projection engine and a mathematical optimizer, reads the latest news, and tells me exactly:

1. Who to transfer in/out this gameweek (and whether to roll the transfer or take a -4 hit).
2. Who to captain, vice-captain, start, and bench.
3. The long-term plan: a multi-gameweek transfer roadmap with reasoning.
4. **When to play each chip** (Wildcard, Free Hit, Triple Captain, Bench Boost), across both chip sets.
5. How to beat my specific mini league rivals (differentials, template protection, risk strategy based on my rank vs rivals).
6. Anything else I ask, via a conversational expert chat grounded in live data.

The success criterion: its recommendations should maximise my expected points over the rest of the season and my probability of finishing first in my mini league. Optimise for expected points AND for mini-league win probability (these differ when I am chasing or leading).

It may be slow. A full analysis taking several minutes is fine. Quality over speed. Cache aggressively so repeat queries are fast.

---

## 2. TECH STACK (use these unless you have a strong reason, and justify deviations in DECISIONS.md)

- **Backend:** Python 3.11+, FastAPI, uvicorn, `httpx` (async) with retries/backoff, `pydantic` v2 models, SQLite (via SQLAlchemy) for caching/history, `apscheduler` for background refresh.
- **ML:** pandas, numpy, scikit-learn, LightGBM (primary), optional XGBoost, `scipy`, `statsmodels`. Optional: a Poisson/Dixon-Coles style team strength model.
- **Optimiser:** `PuLP` (CBC solver) or `OR-Tools` for integer linear programming.
- **LLM layer:** provider-agnostic abstraction (`llm/provider.py`) supporting Google Gemini (default, via `GEMINI_API_KEY`), plus Anthropic and OpenAI as optional swappable providers via env vars. Tool/function-calling so the chat agent can call my analysis functions.
- **News:** `feedparser` + `trafilatura` (or `beautifulsoup4`) for RSS/article extraction; optional web-search API (Tavily or Brave) if a key is provided; graceful fallback if not.
- **Frontend:** a clean, fast single-page UI (React + Vite + Tailwind, or HTMX + Jinja if simpler). Dark mode, mobile-friendly. Charts via Recharts or Plotly.
- **Packaging:** `pyproject.toml`, `.env.example`, `Makefile` (or `justfile`) with `make setup`, `make run`, `make train`, `make test`. One command to start everything: `make run` -> opens `http://localhost:8000`.
- **Tests:** pytest, with unit tests for the optimiser, chip logic, data parsing and a backtest harness.

---

## 3. DATA SOURCES

### 3.1 Official FPL API (no key required for public endpoints) — PRIMARY
Base URL: `https://fantasy.premierleague.com/api/`

| Endpoint | Use |
|---|---|
| `bootstrap-static/` | All players (`elements`), teams, positions (`element_types`), gameweeks (`events`, incl. `is_current`, `is_next`, `deadline_time`), chip definitions (`chips`), game settings/rules (`game_settings`), scoring settings. **Single source of truth for rules, prices, ownership, form, status, news, injuries, xG-style fields.** |
| `fixtures/` and `fixtures/?event={gw}` | All fixtures, difficulty (FDR), kickoff times, results, stats. Use to detect blank and double gameweeks. |
| `element-summary/{player_id}/` | Per-player match-by-match history this season, upcoming fixtures, previous seasons' history. |
| `event/{gw}/live/` | Live points/stats during a gameweek. |
| `entry/{manager_id}/` | Manager summary, leagues I'm in (classic + h2h), overall rank. |
| `entry/{manager_id}/history/` | My past gameweeks, chips used (`chips` list with the gameweek each was played), season-by-season history. |
| `entry/{manager_id}/event/{gw}/picks/` | My squad for a gameweek, captain, multiplier, bench order, active chip, transfers, `entry_history` (bank, value, transfers cost). |
| `entry/{manager_id}/transfers/` | My transfer history (needed to compute free transfers available and purchase prices / selling prices). |
| `leagues-classic/{league_id}/standings/` | Mini-league table (paginate with `?page_standings=N`). |
| `leagues-h2h/{league_id}/standings/` | Head-to-head leagues if applicable. |
| `event-status/` | Whether bonus points are processed / gameweek finished. |
| `dream-team/{gw}/` | Team of the week. |
| `team/set-piece-notes/` | Set-piece taker notes (use as a model feature and for news context). |

Requirements for the FPL client:
- Send a realistic `User-Agent` header. Cache `bootstrap-static` for ~5-10 minutes (and ~60s during live gameweeks). Respect rate limits: concurrency cap (e.g. 5), exponential backoff on 429/5xx, and a polite delay when fetching many `element-summary` calls. Persist raw JSON snapshots to SQLite/disk so I can re-run offline and so the ML pipeline can learn from snapshots over time.
- The API is unofficial and can change. Parse defensively with pydantic (`extra="allow"`), log schema drift, and never crash the app if one field is missing.
- **Do not require my FPL login or password.** Only use public endpoints keyed by Manager ID. If I want, provide an optional manual-entry path (see 4.1) for when the API is down or the season state is odd.
- During the Premier League's pre-deadline or API-update windows the API may briefly return errors; handle gracefully with cached data and a visible "data may be stale (timestamp)" banner.

### 3.2 Historical training data
- Use the open-source community dataset `vaastav/Fantasy-Premier-League` on GitHub (raw CSVs of per-gameweek player data for past seasons, `merged_gw.csv`, `players_raw.csv`, fixtures, teams). Download via `raw.githubusercontent.com` / `git clone` at setup. Verify the repo structure and available seasons when you fetch it; adapt to what is actually there. Seasons from 2016/17 onward exist; some older ones lack xG fields. Handle missing columns per season.
- Also ingest **this season's** data from the live API into the same schema, so the model trains on everything up to the latest finished gameweek.
- Be careful about **data leakage**: features for a gameweek may only use information available before that gameweek's deadline.

### 3.3 News and qualitative signals
Build `news/` ingestion that periodically (and on demand) fetches:
- Official FPL `news` / `news_added` / `chance_of_playing_next_round` / `status` fields from bootstrap-static (authoritative for injuries and suspensions).
- Premier League press-conference news and injury roundups via RSS (BBC Sport football, Sky Sports, The Guardian football, Premier League official site, club sites where an RSS exists). Verify each feed URL works at build time and keep a configurable list in `config/news_sources.yaml`; drop dead feeds automatically.
- Optional (only if a key is provided): web search API for "[player] injury", "[team] press conference", "[manager] team news", "Fantasy Premier League price changes".
- An LLM **news-analysis pass** that turns raw articles into structured signals per player/team: injury severity and expected return, rotation risk, "minutes expectation" (0-100%), penalty/set-piece duties, manager quotes about form or role change, transfer-window arrivals/departures, suspension risk (yellow card accumulation), international-break fatigue. Each signal must include source URL, timestamp, and a confidence score. Store them and feed them (a) into the ML model as features and (b) into explanations. **Never invent news.** If no source supports a claim, say so.
- Source-quality weighting: official club/PL/FPL sources > established press > everything else. Flag conflicting reports.

### 3.4 Optional extra data (only if reachable and legal to use; otherwise skip silently)
- Betting-implied team goals / clean sheet probabilities if a free odds API key is provided (`ODDS_API_KEY`, optional). Otherwise derive team attack/defence strength from results and FPL xG-style fields.
- Do **not** scrape sites whose terms forbid it. Prefer APIs and RSS.

---

## 4. FEATURES

### 4.1 Team input (multiple ways)
1. **Manager ID** (main path): fetch my current squad automatically, plus bank, squad value, free transfers, chips used/available, transfers history.
2. **Manual / paste / upload**: JSON or CSV or plain-text list of 15 player names; fuzzy-match to FPL player IDs with confirmation UI; ask for bank and free transfers.
3. **Mini-league ID(s)**: list my leagues from `entry/{id}/` and let me pick the one I care about (store as my "target league").
4. Persist my profile locally (`data/profile.json`): Manager ID, target league ID, risk preference, LLM provider.

### 4.2 Dashboard
- My squad on a pitch view with each player's: next 5 fixtures (colour-coded), projected points for the next gameweek and next 5, ownership, price, price-change probability, injury/news flag, minutes-risk.
- Next deadline countdown, current gameweek, blank/double gameweek calendar for the rest of the season.
- My rank trajectory and a comparison to my mini-league rivals.
- Data freshness timestamps.

### 4.3 ML projection engine (the core)
Predict expected FPL points per player for each of the next N gameweeks (N up to 8, then a coarser rest-of-season view), and a distribution (not just a mean).

**Decompose points** into components, each modelled separately, then aggregate with the actual scoring rules read from the API/config:
- P(starts), expected minutes (e.g. P(60+), P(<60), P(0)).
- Expected goals, assists (position-aware), clean sheet probability (team-level, then player-level given minutes), goals conceded effect (for GK/DEF -1 per 2 goals), saves (GK), bonus points, yellow/red cards, own goals, penalties won/saved/missed, **defensive contribution points** (DefCon: modelled from the player's rate of CBIT/CBIRT-style actions, with the position-specific thresholds read from the current rules; verify the thresholds from the API/rules page at build time rather than hard-coding), appearance points.
- Use the **current 2026/27 scoring**: verify at build time and put every scoring constant in `config/scoring.yaml` with a comment of where it was verified. Notably, the Bonus Points System was rebalanced for 2026/27 to reduce overlap with DefCon, so do not assume last season's bonus behaviour; train bonus on the most recent data, weight recent gameweeks, and include a feature flagging pre/post-rule-change.
- **Features** (all strictly pre-deadline): rolling form (last 3/5/8 GWs; minutes-weighted), season-to-date per-90 stats, xG/xA/xGI/xGC fields from FPL, ICT index components, BPS tendencies, shots / shots in box / big chances (where available), set-piece and penalty duties, team attack and defence strength (own model: Dixon-Coles-like or Elo with home/away), opponent strength, home/away, fixture congestion and rest days (European competitions, cup ties), international-break effects, manager rotation tendencies, player age, price, ownership, transfer momentum, FPL `chance_of_playing`, news-derived signals, position, team-mates' availability (e.g. injured striker raises a backup's start probability), playing-time trend, double gameweek flag (sum of two fixtures with correct per-match modelling), blank gameweek flag.
- **Models:** LightGBM gradient boosting per component (or one multi-output setup), with calibrated probabilities (isotonic/Platt) for start/clean-sheet. Train with time-aware cross-validation (expanding window by gameweek/season). Add a simple, robust baseline (e.g. weighted recent form × fixture multiplier) and an ensemble. Report MAE/RMSE/rank-correlation and calibration plots vs baseline in `reports/model_eval.md`. The ML model must beat the baseline in backtests; if it doesn't, say so and use the better one.
- **Uncertainty:** produce P10/P50/P90 or per-player variance (quantile regression or bootstrap) so the optimiser and captaincy logic can reason about risk (ceiling for differentials, floor for protecting a lead).
- **Retraining:** `make train` does the full pipeline; a scheduled job retrains after each gameweek's data finalises (bonus points are finalised the day after the last match of the gameweek, so wait for `event-status` to confirm). Version models and keep metrics history.
- Handle **new players, transfers in/out of the league, and positional changes** (cold-start priors from price/position/team).

### 4.4 Transfer optimiser
- Integer linear program over the full player pool for a multi-gameweek horizon (default 5-8 GWs, configurable) maximising the sum of discounted expected points (discount factor configurable, e.g. 0.95/GW), subject to **all real rules**, read from the API where possible: 15-man squad (2 GK, 5 DEF, 5 MID, 3 FWD), max 3 per club, £100m starting budget -> use my actual bank + selling prices (selling price = purchase price + half of the rise rounded down; compute from my transfer history), valid starting XI formations (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD), captain and vice-captain, bench ordering with autosub awareness, free transfers (banking up to 5), -4 per extra transfer, and chip effects.
- Evaluate and rank candidate plans: roll the transfer, 1 transfer, 2 transfers (with/without hit), etc. Show net expected gain after hits. Explicit "hit worth it?" verdict with break-even horizon.
- Include **price-change awareness**: estimate which players are likely to rise/fall tonight from net transfers and ownership momentum (heuristic model from the API's `transfers_in_event` / `transfers_out_event` and ownership), and show "buy before rise" urgency vs "can wait".
- Support **what-if** mode: user fixes certain players ("keep Salah", "no Arsenal players", "budget cap £X") and re-optimises.
- Support **injury-aware** recommendations: if a player has an uncertain status with a deadline before press conferences, explain the wait/commit tradeoff.
- Output a **Transfer Roadmap**: planned transfers for the next 4-6 gameweeks (not just this one), explaining sequencing, banking free transfers, upcoming price rises, fixture swings, and planned chip gameweeks. Mark which steps are firm vs. contingent on news.

### 4.5 CHIP STRATEGY ENGINE (critical — I explicitly want this)
2026/27 chip rules (verify against `bootstrap-static` `chips` data and the official rules at build time, and keep every constant in `config/rules.yaml`; **never hard-code without a verification comment**):
- Four chips: **Wildcard, Free Hit, Triple Captain, Bench Boost**.
- **Two sets of chips** in the season, one set available in each half. The **first set must be used before the Gameweek 19 deadline** (reported as 13:30 GMT on Saturday 2 January 2027) and **cannot be carried over**; the **second set unlocks after that** and runs through the end of the season. That means 8 chips total.
- The **Assistant Manager chip was removed** for 2026/27. Do not include it.
- Only one chip may be played per gameweek.
- Wildcard changes are permanent; Free Hit is for a single gameweek only and the squad reverts afterwards.
- Up to five free transfers can be banked; each additional transfer costs 4 points. There are no extra AFCON-style free transfers this season.
- My already-used chips come from `entry/{id}/history/` (`chips` array). Compute exactly which chips I still have in each set and the hard deadlines for them.

Engine requirements:
- For every remaining chip in the current set, and for the next set, compute the **best gameweek(s) to play it** with expected gain in points, using the projection engine and actual fixture data:
  - **Bench Boost:** maximise the sum of the bench's expected points; identify double gameweeks where all 15 have two fixtures, plan squad build-up ahead of it.
  - **Triple Captain:** best captain candidates in double gameweeks (two fixtures, high floor and ceiling), or a premium with elite fixture in a single GW; compare expected gain over regular captaincy.
  - **Free Hit:** blank gameweeks (where many of my players don't play) or extreme fixture swings; optimiser builds the best one-week 15 within budget; compute gain vs. my best non-chip plan.
  - **Wildcard:** squad-restructure timing: when injuries, price drops, and fixture swings make a full rebuild worth more than incremental transfers; before big double gameweeks; before the GW19 first-set deadline if unused; lay out the target team.
- Detect and calendar **blank and double gameweeks** from `fixtures/` and warn me ahead of time. Note that fixture lists are subject to rescheduling (cups, European ties, postponements): re-detect on every refresh and flag changes that invalidate a plan.
- Produce a **Chip Plan table**: chip -> recommended GW -> expected gain -> confidence -> alternative GW -> "trigger conditions that would change this". Update it every refresh. Explicitly handle the first-set GW19 deadline: warn me early, and show the opportunity cost of letting a chip expire unused. Search the joint space of chip assignments (don't pick each chip greedily in isolation, because chips interact, e.g. WC before BB, FH on a blank, one chip per GW).
- Simulate chip plans with a **dynamic-programming or beam-search** over gameweeks and report the best joint schedule with its total expected gain, plus 2-3 robust alternatives.
- Explain every recommendation in plain English with the numbers.

### 4.6 Captaincy and lineup
- Rank captain options by expected points with P10/P50/P90 and probability of outscoring the next-best option. Account for double gameweeks and the benching/autosub rule. Provide a safe vs. high-ceiling pick, and recommend based on my mini-league position.
- Recommend starting XI and bench order, with explicit "bench risk" for any starter with doubt.
- Recommend vice-captain with reasoning (kickoff order matters: vice only activates if the captain doesn't play).

### 4.7 Mini-league intelligence
- Pull my mini league standings (all pages), and for each rival the current squad, captain, chips used/remaining, bank/value, recent transfer history, and form.
- Compute **ownership within my league** (effective ownership by captaincy multiplier) per player, and classify: template (owned by most rivals), differentials (owned by few), my unique picks.
- **Win-probability model:** Monte Carlo simulation of the remaining season (or the next K gameweeks) using the projection engine's distributions, simulating each rival's squad (assume their current squad with simple heuristic transfers) to estimate my probability of finishing 1st, and how each candidate decision (transfer, captain, chip) changes it.
- **Strategy mode:** if I am **leading**, recommend risk-reducing moves (mirror the rival's key players, protect against their differentials, lower variance); if **chasing**, recommend high-variance differential moves (low-owned high-ceiling players, captain differentials, chip timing that diverges from rivals). State clearly which mode is active and why.
- Track which chips my rivals have burned and when they likely plan to play them; recommend counter-timing.
- Head-to-head view against the top rivals: points gap, upcoming fixtures swing, and where I can gain.

### 4.8 The FPL Expert Chat (talk to it anytime)
A chat interface (web + CLI `fpl chat`) powered by the LLM with **tool calling** into the app's functions, so answers are grounded in live data, not the model's memory. Tools should include: `get_my_team`, `get_projections(player|team|position, horizon)`, `get_fixtures(team|gw)`, `get_news(player|team)`, `optimise_transfers(constraints)`, `plan_chips()`, `captain_options(gw)`, `league_analysis()`, `simulate_win_probability(decision)`, `price_change_watch()`, `compare_players(a,b,...)`, `player_deep_dive(id)`, `refresh_data()`.
- It must support questions like: "Should I take a -4 for Haaland?", "When should I use my Bench Boost?", "Who's the best differential midfielder under 6.5?", "Is it time to wildcard?", "What happens if I Free Hit in the blank GW?", "How do I catch the person above me?", "Explain why you picked X over Y".
- Always cite the underlying numbers and the news sources (with links and timestamps) used. Distinguish **facts from the data**, **model estimates**, and **opinion**. Say "I don't know" or "no reliable news found" rather than guessing. Never fabricate stats, injuries, prices or fixtures.
- Long-running analyses should stream progress ("fetching 500 player summaries… training… optimising…") and be allowed to take minutes. Use background jobs with a status endpoint; the chat should say when it's still working.
- Keep conversation memory per session; allow saving "decisions I made" so the tool can review them later (post-gameweek reviews: what you recommended vs. what I did vs. what happened).

### 4.9 Weekly workflow automation
- A one-click **"Gameweek Briefing"** generated before every deadline: deadline countdown, injuries/news changes, my recommended transfers, captain, lineup, chip decision, price changes to watch, mini-league situation, and a short written summary. Export as Markdown and show in the UI.
- Optional notifications: a configurable local notification or Telegram/Discord webhook (only if I provide a token) N hours before each deadline and on late injury news.
- **Post-gameweek review:** how my team scored vs. projections, what the model got wrong, rank change, league change, and lessons. Use it to log model accuracy over time.

---

## 5. BACKTESTING AND VALIDATION (non-negotiable)

- Build a **backtest harness** replaying past gameweeks from the historical dataset: at each GW, use only the data available then, produce projections, run the optimiser with the real rules, and compare the total points achieved to (a) a naive baseline strategy, (b) the average manager score where available, (c) hindsight-optimal. Report in `reports/backtest.md`.
- Evaluate: projection error by position, captain pick success rate vs. baseline, transfer-plan value, chip-timing value (replay a past season's double gameweeks).
- Include a **self-check command** `make verify` that: hits the live API, validates the schema, reports API health, runs unit tests, runs a tiny backtest, and prints a green/red summary.
- Fix the obvious sources of bias: survivorship, leakage, price changes affecting purchase prices in backtests.

---

## 6. ARCHITECTURE AND REPO LAYOUT

```
fpl-oracle/
  README.md  SETUP.md  DECISIONS.md  Makefile  pyproject.toml  .env.example
  config/ rules.yaml scoring.yaml news_sources.yaml settings.yaml
  data/ (gitignored: cache, snapshots, sqlite db, models)
  src/fpl_oracle/
    api/ fpl_client.py models.py cache.py
    data/ historical.py features.py store.py
    news/ ingest.py extract.py analyse.py
    ml/ train.py predict.py components/ ensemble.py calibration.py eval.py
    optimise/ squad.py transfers.py lineup.py price_change.py
    chips/ calendar.py planner.py simulate.py
    league/ standings.py rivals.py montecarlo.py strategy.py
    llm/ provider.py gemini.py anthropic.py openai.py tools.py agent.py prompts/
    briefing/ weekly.py review.py notify.py
    server/ main.py routes/ jobs.py
    cli.py
  web/ (frontend)
  tests/
  reports/
```
- Clear module boundaries, type hints, docstrings, logging (structured), config over constants.
- Every recommendation endpoint returns structured JSON (`decision`, `expected_gain`, `confidence`, `reasoning`, `sources`, `data_timestamp`) so the UI and chat can both use it.

---

## 7. REASONING AND EXPLANATION STANDARD

Every recommendation (transfer, captain, chip, hold) must include:
1. **The decision** in one line.
2. **The numbers:** projected points over the horizon for the options compared, net of hits.
3. **The why:** fixtures, form, minutes, news, price trend, league context.
4. **The risk:** what could go wrong, with probabilities where possible.
5. **What would change the call** (e.g. "if X isn't in the XI per the press conference, switch to Y").
6. **Long-term effect** on chip plan, bank, and future flexibility.
Keep it readable: lead with the answer, then support.

---

## 8. SECURITY, ETHICS, ROBUSTNESS

- Run on localhost only by default (bind 127.0.0.1). Keys only via `.env`, never logged or committed.
- No automation that logs into my FPL account or makes transfers for me. Advice only. (The tool recommends; I click.)
- Respect robots.txt/terms; use APIs and RSS; throttle requests.
- Everything degrades gracefully: if the LLM key is missing, the numeric engine, optimiser, and dashboard must still work, and chat falls back to a templated explainer. If the news feeds fail, say so in the UI.
- Include a clear disclaimer in the UI that projections are probabilistic and FPL has inherent variance; the tool can raise the odds, not guarantee a win.

---

## 9. DELIVERABLES AND DEFINITION OF DONE

By the end of the run, all of the following must be true:
1. `make setup && make run` starts the app at `http://localhost:8000` without errors on a fresh machine (Windows/macOS/Linux notes in SETUP.md).
2. The historical dataset is downloaded, features built, models trained, and evaluation reports generated (`reports/model_eval.md`, `reports/backtest.md`) showing results vs. baselines, with honest numbers.
3. Entering my Manager ID loads my real squad, my leagues, and shows a full analysis: transfer plan, captain, lineup, **chip plan**, mini-league analysis and win probability, and a Gameweek Briefing.
4. The expert chat works with tool calling and cites live data/news. Test it with at least 10 sample questions and save the transcript to `reports/chat_examples.md`.
5. `make verify` passes; unit tests cover the optimiser constraints (budget, 3-per-club, formation, selling-price math), chip logic (set boundaries, GW19 deadline, one chip per GW), and API parsing.
6. `SETUP.md` lists **exactly** what I need to provide, step by step: the required `GEMINI_API_KEY` (or alternative provider key), my FPL Manager ID (how to find it in the FPL site URL), my mini league ID, and optional keys (search API, odds API, Telegram). `.env.example` is complete.
7. `README.md` explains how to use it each gameweek (a 5-minute weekly routine) and how to retrain.
8. `DECISIONS.md` documents assumptions, rule verifications (with sources), known limitations, and what you would improve next.
9. Do a final self-review pass: run the app, hit every endpoint, click through the UI, fix bugs, and remove dead code before declaring completion.

## 10. WORKING METHOD

- Start by writing a plan (`PLAN.md`) with milestones: data layer -> historical dataset + features -> ML + backtests -> optimiser -> chip engine -> league intelligence -> news + LLM -> chat agent -> UI -> briefing/automation -> verification.
- Verify live API behaviour by actually calling it before coding against assumed schemas; adapt to what you observe (field names, new rules in `game_settings` and `chips`).
- Verify the 2026/27 rules (chips, scoring, DefCon thresholds, bonus system, deadlines) from official FPL sources at build time and encode them in config with source notes. If sources conflict, prefer the live API, then the official FPL site.
- Commit in logical steps. Test as you go. If something is impossible (an endpoint disappeared, a feed is dead), implement the best fallback, record it, and keep going.
- When finished, print a concise summary: what was built, how to run it, what keys I must add, and the top 5 known limitations.
