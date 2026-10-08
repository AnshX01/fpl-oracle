# FPL Oracle

Local FPL advice for your current team, transfers, captain, bench and chips.

## Windows setup

Python 3.11 or newer is required. From the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Open http://127.0.0.1:8000. The launcher checks the tested PuLP/CBC solver before starting.

Keep your existing `.env`. Only you edit it. For a new install, copy `.env.example` to `.env` and add your manager and league IDs. No key is needed for public FPL data and offline chat. Optional Gemini needs your own key and account-tier confirmation; it stays disabled when the configuration or budget checks fail. Never commit `.env` or share its keys.

## Before the deadline

1. Open the app and update data.
2. Confirm your current team, bank, remaining free transfers and chips.
3. Read the plan at the top. Check the team, captain and bench.
4. Use Emergency check if a player is ruled out. Choose how long they are out.
5. Make any changes yourself on FPL, then tap Done if you followed the plan. Choose "I did something different" for another change.

Done stops new advice until the deadline. Undo reopens the local plan. None of these buttons makes transfers or logs into FPL.

Public FPL picks can lag changes before a deadline, so the local confirmation is the current planning baseline. After a deadline, checked public picks can reconcile it. Conflicts ask for a new confirmation instead of silently replacing your team.

## Planning

- Legal XI, goalkeeper-only bench cover, valid formation and no reused reserve.
- Ordinary plans use at most two transfers and one four-point hit. Saving free transfers is a candidate too.
- Hits must cover their cost in projected incoming-minus-outgoing points. A settled failed hit blocks another hit for six following gameweeks. Unsettled results block repeats.
- Search keeps current-week and future-value transfer candidates. It tracks squad, bank, buying/selling prices, FTs and chips week by week through the current chip-set expiry.
- Wildcard and Free Hit dates are compared with replanned ordinary transfers. Free Hit restores the permanent squad next week.
- League standings choose the strategy automatically. Rival coverage only breaks close point scores.
- Report, chat, emergency and home use the same verified planning inputs. Missing or stale required data blocks current advice.

## Checks and limits

See [DELIVERY_STATUS.md](DELIVERY_STATUS.md) for the release checks.

The search is bounded, not an exhaustive proof of the global best plan. Future forecasts use the information available now. Unannounced doubles, future injuries, price changes and rival moves are not known. Double-week means sum fixtures; summed uncertainty intervals are not calibrated joint intervals. Simulation rates are scenario results, not measured season-winning odds.

Forward model scoring needs future pre-deadline freezes and checked results. Synthetic tests do not prove prediction accuracy. Gemini news stays inactive until real held-out responses pass its gate. Startup never retrains or replaces your weights implicitly.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe scripts\check_secrets.py
node tests\js\test_atomic_publish.cjs
node tests\js\test_ui_polish.cjs
node tests\js\test_autosubs.cjs
node tests\js\test_current_league_rank.cjs
```

Model training and promotion are separate explicit tasks, not routine setup steps.
