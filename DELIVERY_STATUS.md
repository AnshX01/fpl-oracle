# Release checks - October 8, 2026

## Local repairs

- Clear home plan and shorter FPL labels. Mobile spacing checked at 390, 360 and 320 pixels; desktop checked at 1440 pixels.
- Local Confirm team, Done, Undo and deadline reset. The saved actual team drives later advice.
- Consistent bank, FT, chip, lineup and source-age checks across home, chat, report and emergency.
- Legal bench cover and formation; ordinary two-transfer/one-hit cap; banking and checked hit outcomes.
- Future-value transfer discovery and pruning, club-limit shortlist fix, and every-date Wildcard/Free Hit replanning through chip-set expiry.
- Current league rank separated from overall rank; automatic league strategy.
- Price momentum shown as a watch signal, not a promised price move.

## Verification

Recovered release checks: four JavaScript tests pass; lint and secret scan pass; 83 source files type-check cleanly. Screens and modals were recaptured and visually inspected at 390, 360, 320 and 1440 pixels using deterministic fixtures, with no console errors or horizontal overflow. Clean no-profile/no-.env mocked startup returned 200 for home, health, profile, squad basic and confirmation. PuLP 3.3.2/CBC solve passed. Frozen final regression: 389 tests passed in 1083.54 seconds (18 minutes 3 seconds). Source, tests, scripts and UI hashes were unchanged during that run. Done, reload lock, Undo and different-team browser flows passed at 1440 and 390 pixels. Exclusive synthetic search runtime: 5 gameweeks took 24.50 seconds with score 388.9317; 14 gameweeks took 88.76 seconds with score 946.8773. Scores matched the prior reference. These timings measure one bounded search, not the full date-comparison workflow, startup or a Windows laptop. The wider search is slower. No five-second advice or speed improvement is claimed.

Final source was rechecked after regression and runtime with no hash drift. Four JavaScript checks, lint and secret scan passed again. Final publication uses this unchanged source plus these release notes; GitHub commit and branch readback are reported with delivery.

## Evidence limits

- Forward forecast scoring has no completed future gameweek yet. This does not prove superiority over an FPL expert.
- Double-week intervals and rival scenario rates have no demonstrated live calibration.
- Real Gemini held-out news responses were not measured here; production news remains gated.
- Windows launcher and user install history were inspected. Native Windows execution was not performed in this environment.
- No FPL login, account write or transfer was attempted.
- Existing .env, historical data and model weights are preserved. Test-generated synthetic report files are not new forecasting evidence.

## Update on Windows

Stop the app with Ctrl+C first. In your existing folder:

```powershell
# Open PowerShell in your existing FPL Oracle folder first.
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -e .
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

If Git reports local edits or a conflict, stop instead of resetting or deleting them. Keep your current .env; do not overwrite it. Open http://127.0.0.1:8000 and confirm the current team.
