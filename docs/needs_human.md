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

### Required Human Actions:
1. **Rotate Key in Google AI Studio**:
   - Navigate to [Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey).
   - Revoke/delete the existing Gemini API key that was in use prior to this remediation.
   - Generate a new API key and update your local `.env` file (`GEMINI_API_KEY=<new_key>`).
2. **Delete Original Unredacted Local Log File**:
   - Once rotated, permanently delete `logs/fpl_oracle.log` from your local machine.
   - The safe redacted version `logs/fpl_oracle.redacted.log` can be retained if historical debugging is needed.

---

## 2. Finding F-011: Risk-Aware Lineup Feature Flag Approval
- The feature flag `risk_aware_lineup` in `config/flags.yaml` defaults to `false`.
- A golden fixture diff was executed comparing `balanced`, `conservative`, and `aggressive` modes.
- Conservative mode benches doubtful starters (e.g. 75% flag with high spread) in favor of secure defenders, while aggressive mode prioritizes P90 upside.
- **Required Action:** Repository owner should review the golden fixture diff in `docs/remediation_log.md` (F-011) and decide whether to approve enabling `risk_aware_lineup: true` by default.
