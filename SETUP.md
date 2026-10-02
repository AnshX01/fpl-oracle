# FPL Oracle — Setup & Configuration Guide

Welcome to **FPL Oracle**, a local, machine-learning-driven Fantasy Premier League decision engine and conversational AI expert designed specifically for the **2026/27** season.

---

## 1. Quickstart (Get Running in Under 2 Minutes)

### Prerequisites
- **Python 3.11** (Recommended 3.11.x for pre-built C-extension wheels for `lightgbm`, `scipy`, `pandas`).
- Windows, macOS, or Linux.
- Internet connection (to query the public FPL API on first run).

### One-Command Setup & Launch
Clone or enter the directory and run:

**On Windows (PowerShell / CMD):**
```powershell
# 1. Automatic environment setup & dependency installation
python run.py setup

# 2. Verify all models, database, and solvers
python run.py verify

# 3. Launch the local web server and open browser
python run.py run
```

**On macOS / Linux (using Make):**
```bash
make setup
make verify
make run
```

The web dashboard is now live at: **`http://localhost:8000`**

---

## 2. API Keys & Environment Configuration

Copy the example environment file if you haven't already:
```bash
cp .env.example .env
```

Open `.env` in any text editor. The application runs immediately **without any external API keys** thanks to its built-in `OfflineExpertProvider` rule-based expert engine. However, connecting an LLM provider unlocks deep multi-turn conversational analysis.

### 2.1 Primary LLM Provider (Choose One)
| Provider | Environment Variable | Where to Get It | Notes |
|---|---|---|---|
| **Google Gemini (Default)** | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) | **Recommended**. High rate limits, excellent tool-calling support, generous free tier. |
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) | Set `LLM_PROVIDER=anthropic` in `.env`. Uses `claude-3-5-sonnet-20241022`. |
| **OpenAI GPT** | `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/) | Set `LLM_PROVIDER=openai` in `.env`. Uses `gpt-4o`. |
| **Offline Expert** | *(None required)* | Built-in | Set `LLM_PROVIDER=offline`. Fully operational locally with zero network calls or credentials. |

### 2.2 Optional News & Search Enhancement Keys
| Key | Purpose | Fallback if Missing |
|---|---|---|
| `TAVILY_API_KEY` | Real-time web search for breaking press conferences and team news | Gracefully falls back to official FPL injury statuses and top football RSS feeds. |
| `BRAVE_API_KEY` | Alternative web search provider | Falls back to RSS news extraction. |
| `ODDS_API_KEY` | Implied clean sheet & goalscorer betting odds from The Odds API | Falls back to Dixon-Coles Poisson team attacking/defensive strength model. |

### 2.3 Optional Webhook Alerts (Pre-Deadline Notifications)
| Key | Purpose |
|---|---|
| `DISCORD_WEBHOOK_URL` | Sends automated Gameweek Briefing and transfer recommendations to a Discord channel. |
| `TELEGRAM_BOT_TOKEN`<br>`TELEGRAM_CHAT_ID` | Sends automated Gameweek Briefing to a personal Telegram chat or channel. |

---

## 3. Finding Your FPL Manager ID and Mini-League ID

No passwords or private credentials are ever requested or stored. FPL Oracle only reads public game data.

### 3.1 How to Find Your Manager ID (Entry ID)
1. Open your browser and log into [Fantasy Premier League](https://fantasy.premierleague.com/).
2. Click on the **"Points"** tab or the **"Gameweek History"** tab.
3. Look at your browser's address bar. The URL will look like:
   ```
   https://fantasy.premierleague.com/entry/1234567/event/5
   ```
   or
   ```
   https://fantasy.premierleague.com/entry/1234567/history
   ```
4. The numerical digits (e.g., `1234567`) are your **Manager ID**.
5. Put this in your `.env` file:
   ```ini
   FPL_MANAGER_ID=1234567
   ```
   *(Or enter it dynamically on the Web UI dashboard!)*

### 3.2 How to Find Your Mini-League ID
1. On the FPL website, click on **"Leagues & Cups"**.
2. Click on the specific classic mini-league you want to win.
3. Look at the URL in your browser:
   ```
   https://fantasy.premierleague.com/leagues/987654/standings/c
   ```
4. The number between `/leagues/` and `/standings/` (e.g., `987654`) is your **Mini-League ID**.
5. Put this in your `.env` file:
   ```ini
   FPL_TARGET_LEAGUE_ID=987654
   ```

### 3.3 Manual Squad Entry (No Manager ID Required)
If you do not have an active Manager ID or the FPL site is undergoing server maintenance, you can input your squad directly:
1. Click the **"📋 Paste / Upload Squad"** button in the Web UI header.
2. Paste 15 player names as plain text, CSV, or JSON (or click "Upload .txt / .csv / .json").
3. Specify your available bank (£m) and banked free transfers (1 to 5).
4. Click **"🔍 Match Players"**: FPL Oracle uses an accent-insensitive fuzzy matcher to map names to official 2026/27 player elements.
5. Review the matched starting XI & bench candidates, select any alternatives if ambiguous, and click **"✅ Confirm & Save Squad"**.
6. The squad is saved to `data/profile.json` and automatically powers projections, transfer optimization, captaincy, and chip strategies.

---

## 4. Running FPL Oracle

### 4.1 Local Web Dashboard
```bash
python run.py run
# Or: uvicorn src.fpl_oracle.main:app --host 127.0.0.1 --port 8000 --reload
```
Features available in the UI:
- **Pitch View**: Projected starting XI, captaincy badges, DefCon probability flags, fixture difficulty.
- **Transfer Planner**: Optimal transfer roadmap for the next 1–5 GWs, banked transfers tracker, -4 hit break-even analysis.
- **Chip Strategy Timeline**: 38-gameweek visual schedule with Blank/Double GW tags and Set 1 GW19 hard cutoff.
- **Mini-League Intelligence**: Monte Carlo championship simulation ($P(\text{1st})$), rival ownership, differential recommendations.
- **Gameweek Briefing**: One-click comprehensive report for the upcoming gameweek.
- **Conversational Expert Chat**: Live interactive assistant with tool-calling capabilities.

### 4.2 Command Line Interface (CLI)
You can run any analysis instantly from your terminal:

```bash
# Display full squad analysis, projections, and captaincy
python run.py cli squad --manager-id 1234567

# Calculate optimal transfers for the next 3 gameweeks
python run.py cli optimize --manager-id 1234567 --horizon 3

# Compute optimal chip roadmap across all 38 gameweeks
python run.py cli chips --manager-id 1234567

# Analyze mini-league standings, rival differentials, and win probability
python run.py cli analyze --league-id 987654 --manager-id 1234567

# Generate full weekly gameweek briefing report
python run.py cli briefing --manager-id 1234567

# Start interactive conversational AI terminal session
python run.py cli chat
```

### 4.3 Training and Data Refresh
To re-train the 6 component ML models on latest gameweek data:
```bash
python run.py train
```
This updates all model weights in `data/models/` and generates an updated `reports/model_eval.md`.

### 4.4 Verification and Self-Check
Run the automated verification suite to ensure all subsystems are functioning:
```bash
python run.py verify
```
Checks:
- [x] Live FPL API connectivity & 2026/27 rule compatibility.
- [x] SQLite persistence and snapshot storage.
- [x] LightGBM model weights integrity.
- [x] PuLP MILP solver operation.
- [x] Gameweek backtesting harness.

---

## 5. Troubleshooting & FAQ

### Port 8000 is already in use
Specify a different port via environment variable or CLI argument:
```powershell
$env:PORT="8001"
python run.py run
```
Or edit `PORT=8001` in `.env`.

### FPL API returns 503 or 429 during game updates
Around deadlines or matchday point updates, the official FPL servers occasionally lock out queries. FPL Oracle automatically handles this:
- Cached responses from `data/fpl_oracle.db` are served seamlessly.
- An alert banner indicates the timestamp of the cached snapshot.
- Backoff retries prevent spamming the official servers.

### Windows Terminal Emoji Rendering
If your Windows PowerShell terminal shows characters like `?` or box symbols instead of status badges:
- Windows Terminal (the modern tabbed app) supports full UTF-8.
- FPL Oracle automatically detects Windows console encoding and configures UTF-8 output streams.

### Offline Mode
If you do not have a `GEMINI_API_KEY` or lose internet access after the initial API cache is created:
- Set `LLM_PROVIDER=offline` in `.env`.
- All features, projections, MILP optimization, chip strategies, and briefing generators will function completely offline using the SQLite cached database and local LightGBM models.
