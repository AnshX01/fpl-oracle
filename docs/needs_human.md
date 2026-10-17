# Items Requiring Human Intervention

This document tracks all remediation steps that require manual human action (such as credential rotation, git history rewriting, or external service authorization).

---

## 1. Finding F-001: Google Gemini API Key Compromise & Secret Scan Audit

### Symptoms & Root Cause
Prior to remediation, `httpx` and `httpcore` logged full HTTP request URLs including the query parameter `?key=...` at INFO level. This caused active Google Gemini API keys to be written in plaintext to `logs/fpl_oracle.log` on disk. A filtered redacted copy was generated at `logs/fpl_oracle.redacted.log`.

### Secret Scanner Audit of Entire Git History
A comprehensive regular-expression secret scan was executed across all branches and commits in the git repository:
```bash
git log --all -S"AIza" --oneline
```
Output:
- `b8b880d` (`F-001: Redact API keys and auth headers, mute httpx loggers`)
- `fd00d00` (`Complete G12: production-grade news extraction, deterministic fallback, and benchmark gating`)
- `c54deb1` (`feat(core): Phase 1 complete - resilience, gameweek phase machine, fault injection, strict typing`)

Inspection of diffs via `git grep -n "AIza" b8b880d fd00d00 c54deb1`:
1. `scripts/check_secrets.py`: Static scanner regex pattern definition `r"AIza[0-9A-Za-z-_]{35}"`.
2. `src/fpl_oracle/utils/logging.py`: Redaction filter regex pattern.
3. `scripts/verify_gemini.py`: Placeholder CLI documentation `GEMINI_API_KEY=AIzaSy...`.
4. `tests/test_f001_api_key_logging.py`: Fake test fixture tokens (`AIzaSyD1234567890123456789012345678901`, `AIzaSyZ9876543210987654321098765432109`).
5. `reports/baseline.md`: Documentation referring to secret audit regex.

**Confirmed Findings:**
- Zero real Google API keys have ever been committed to the git repository history.
- The `logs/` directory has never been tracked in git (`git ls-files logs` returned 0 files; `git log --all -- logs/` is empty).
- Git history rewriting (via `git filter-repo` or BFG) is **not necessary** because no real keys exist in git history.

### Status of Human Actions
> [!IMPORTANT]
> **STATE: NOT DONE.** The following manual steps require the human account owner and CANNOT be performed autonomously by the agent:

### Required Human Actions:
1. **Rotate Key in Google AI Studio (NOT DONE)**:
   - Navigate to [Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey).
   - Revoke/delete the existing Gemini API key that was in use prior to this remediation.
   - Generate a new API key and update your local `.env` file (`GEMINI_API_KEY=<new_key>`).
2. **Delete Original Unredacted Local Log File (NOT DONE)**:
   - Once rotated, permanently delete `logs/fpl_oracle.log` from your local machine.
   - The safe redacted version `logs/fpl_oracle.redacted.log` can be retained if historical debugging is needed.
3. **Git History Rewrite Decision (DECISION: NOT NEEDED)**:
   - Executed full scan: `git log --all -p -G'AIza[0-9A-Za-z_-]{35}'` -> 0 commits found.
   - Executed full scan: `git log --all -p -G'AQ\.[A-Za-z0-9_-]{20,}'` -> 0 commits found.
   - Result: 0 live secrets in git history; no history rewrite or force push required.

---

## 2. Finding F-011: Risk-Aware Lineup Feature Flag & Risk Modes (Owner Decision)
- **Status:** Owner Decision Recorded — Keep Default-Off (`risk_aware_lineup: false`).
- **Mechanism:** In `src/fpl_oracle/optimise/lineup.py`, the `risk_aware_lineup` feature flag is default-off (`is_flag_enabled("risk_aware_lineup", default=False)`).
- **Consequence:** The risk modes (`conservative`, `balanced`, `aggressive`) are currently **inert** across all optimization runs (all 3 modes solve standard linear expectation maximization, as empirically verified across 30 replay runs in C2 and D3).
- **Owner Directive (D7):** Do **NOT** enable `risk_aware_lineup` by default. It remains `false` in production config.
- **Superseding Architecture Note:** Item 4's simulator-based mini-league mode (which simulates joint rival lineups and stochastic match distributions to select tactical chasing vs defending strategies) is designed to supersede this heuristic quadratic variance penalty (`lambda * sigma^2`). Any activation or overhaul is deferred to the mini-league simulator integration.

