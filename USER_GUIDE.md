# FPL Oracle 2026/27 — User Guide

Welcome to **FPL Oracle**, your personal, local Fantasy Premier League decision-support engine.

FPL Oracle runs 100% locally on your machine. It analyzes your squad, fixture runs, player performance metrics, and mini-league rivals to suggest your optimal starting XI, captaincy, transfer moves, and chip timing.

> **Important Boundary**: FPL Oracle is an **advisory tool only**. It does not log into your official FPL account and **never submits transfers or changes on your behalf**. All recommendations must be confirmed and executed by you on the official FPL website or app.

---

## 1. Quick Start: Everyday Launch

To start FPL Oracle, open Windows PowerShell in the project directory and run:

```powershell
.\start.ps1
```

If PowerShell execution policy is restricted on your machine, run with process-scoped permissions:
```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

### What this does:
1. Verifies Python 3.11+ and the virtual environment (`.venv`).
2. Confirms that local mathematical and machine learning models are loaded.
3. Automatically opens your browser to **`http://127.0.0.1:8000`**.
4. Shuts down cleanly when you press `Ctrl+C` in the terminal.

---

## 2. Four Everyday Destinations

FPL Oracle is structured around four simple destinations:

### 1. Overview
- **Your Next Decision (Council Decision Card)**: The single highest-priority decision for the upcoming gameweek (e.g. *Save your free transfer* or *Transfer Alexander-Arnold → Gabriel*), with two supporting reasons, cost, and press-conference caveats.
- **Predicted Points Headline & Math Breakdown**: Your team's forecasted points for the upcoming gameweek. Click **Show Math** to inspect the exact arithmetic:
  $$\text{Total Points} = \text{11 Starting Players Points} + \text{Captaincy Bonus (+1x)}$$
  *(Bench points are excluded unless an autosub is activated).*
- **Captain & Vice-Captain**: Expected points, floor (P10 lower outcome estimate), and ceiling (P90 higher outcome estimate).
- **Squad Availability Radar**: Live injury and doubt alerts from official Premier League press conferences.

### 2. My Team
- **Pitch View**: Visual formation (e.g. 3-5-2) showing your 11 starters with purchase cost, selling price, predicted points, and Fixture Difficulty Rating (FDR).
- **Interactive Autosub Simulation**: Click **Out?** on any starter to simulate them missing the match. Watch how your first bench reserve is automatically promoted to the starting XI in real time!
- **Bench Dugout**: View your 4 bench reserves ranked in order of autosub priority.
- **Injury & Rotation Contingency Matrix**: A precomputed table evaluating whether to rely on your bench or execute an emergency transfer if a player is ruled out.
- **Paste / Custom Squad**: If you don't want to use an FPL Manager ID or want to test a hypothetical squad, click *Paste / Custom Squad* to paste 15 player names with fuzzy matching.

### 3. Transfers
- **Plan A (Primary)**: The mathematically optimal multi-gameweek transfer plan (balancing points gain, banked transfers, and hit costs).
- **Plan B (Injury Pivot)**: An alternative plan if a key player suffers a late training ground injury.
- **Plan C (Differential / Chasing Pivot)**: An alternative plan targeting low-ownership players to gain rank quickly in your mini-league.
- **Price Change Radar**: Predictions of players likely to rise (+£0.1m) or fall (-£0.1m) at midnight.

### 4. My League
- **Actual Standings**: Your real rank, total points, and exact points gap to the leader.
- **Game-Theoretic Strategy Mode**: Automatically switches between:
  - *Template Shield*: When leading, protects your lead by matching key rival picks.
  - *Differential Chaser*: When trailing, targets high-upside differentials to close the point gap.
- **Monte Carlo Season Simulator**: 1,000 season runs simulating rest-of-season outcomes to project your championship win probability and expected final rank.

---

## 3. Recommended Gameweek Workflow

Follow this 5-step workflow each week before the deadline:

1. **Check Your Team State**: Launch `.\start.ps1` and verify your bank, banked free transfers (1–5), and current published squad. Use the **Settings** modal to correct any numbers if you made transfers on FPL earlier.
2. **Review "Your Next Decision"**: Look at the Council decision card on the Overview tab. Does it suggest saving your free transfer or making a transfer?
3. **Wait for Friday Press Conferences**: Do not rush transfers early in the week unless you face an imminent price rise you cannot afford. Press conferences on Thursday/Friday provide critical team news.
4. **Run Pre-Deadline Audit**: Click the **Audit** button in the top bar to run the 5-point failsafe check (confirming deadline time, valid formation, captain set, vice-captain failsafe, and bench order).
5. **Execute in FPL**: Manually apply your chosen lineup, captain, and transfers in the official FPL app or website.

---

## 4. Plain-English Terminology Guide

FPL Oracle replaces confusing technical jargon with clear football language:

| Technical Acronym | Plain-English Term | What It Means |
|---|---|---|
| **xP** | **Predicted points** | The statistical average expected points for a player in this fixture based on minutes, form, opponent difficulty, and bookmaker odds. |
| **P10 / P90** | **Lower / Higher outcome estimates** | A realistic range of outcomes. A player with 6.0 predicted points might have a P10 floor of 2.0 pts (appearance only) and a P90 ceiling of 14.0 pts (goals + bonus). |
| **DefCon** | **Defensive contribution points** | The 2026/27 rule granting +2 bonus points to defenders, midfielders, and forwards who achieve tackle, clearance, and block thresholds. |
| **Net xP delta** | **Predicted gain after hit points** | The projected points gain over a 5-week horizon minus any 4-point transfer hit deductions. |
| **EO** | **Rival ownership** | The percentage of your mini-league rivals who own a particular player. |
| **MILP / Optimization** | **Transfer Workbench** | The mathematical solver that calculates the highest-scoring squad within budget and rule constraints. |

---

## 5. API Keys & Zero-Key Operation

**Core Advice Works 100% Offline with Zero API Keys!**
- All machine learning predictions (LightGBM suite) run locally on your CPU.
- The mathematical transfer optimizer (CBC MILP) runs entirely locally.
- Mini-league Monte Carlo simulations run locally.
- Chat works out of the box in **No-Key Mode**, answering questions using local rule trees and live data tools.

### Optional AI Keys (Settings):
If you wish to enable flexible conversational natural language explanations via an external language model:
- **Google Gemini**: Set `GEMINI_API_KEY` in `.env`.
- **OpenAI**: Set `OPENAI_API_KEY` in `.env`.
- **Anthropic**: Set `ANTHROPIC_API_KEY` in `.env`.

> *Note: External AI keys only rephrase explanations. They do not alter or "improve" the mathematical transfer recommendations or player projections.*

---

## 6. Frequently Asked Questions

**Q: Why does my predicted points total differ from other FPL websites?**  
A: FPL Oracle uses a dedicated decomposed 6-model LightGBM suite trained on over 89,000 match performances, factoring in out-of-fold calibrated minutes, defensive actions (+2 DefCon rule), and temporal rolling form rather than naive historical averages.

**Q: What is the Set 1 Chip Deadline in 2026/27?**  
A: Under 2026/27 Premier League rules, chips are split into two halves (Set 1: GW1–19, Set 2: GW20–38). Unused Set 1 chips (Wildcard, Free Hit, Triple Captain, Bench Boost) expire permanently at the Gameweek 19 deadline and **do not roll over** to the second half of the season.

**Q: Can I run FPL Oracle on mobile?**  
A: Yes! As long as `.\start.ps1` is running on your computer, you can open your mobile browser to `http://<your-pc-ip>:8000` on the same Wi-Fi network. The interface is fully responsive down to 360px mobile screens.
