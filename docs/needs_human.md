# Items Requiring Human Intervention

This document tracks all remediation steps that require manual human action (such as credential rotation, git history rewriting, or external service authorization).

---

## 1. Finding F-001: Google Gemini API Key Compromise

### Symptoms & Root Cause
Prior to remediation, `httpx` logged full HTTP request URLs including the query parameter `?key=...` at INFO level. This caused active Google Gemini API keys to be written in plaintext to `logs/fpl_oracle.log` on disk. A filtered redacted copy has been created at `logs/fpl_oracle.redacted.log`.

### Required Human Actions:
1. **Rotate Compromised API Key in Google AI Studio**:
   - Navigate to [Google AI Studio (API Keys)](https://aistudio.google.com/app/apikey).
   - Revoke/delete the existing Gemini API key that was in use prior to this remediation.
   - Generate a new API key and update your local `.env` file (`GEMINI_API_KEY=<new_key>`).
2. **Delete Original Unredacted Log File**:
   - Once the new key is configured, permanently delete `logs/fpl_oracle.log`.
   - The safe redacted version `logs/fpl_oracle.redacted.log` can be retained if historical debugging is needed.
3. **Decide on Git History Rewrite**:
   - A search of git commit history (`git log -S"key=" --all --oneline`) shows the following commit hashes touched lines containing `"key="`:
     ```
     48a533a e056cb0 1cb6d98 bc9f54d bcbee11 ec620f1 39f0678 bbd3d53 b39d8b8 3454a4a 0108427 2fe419f f9109a4 889ffb9 3f85569 465bf31 f4e2bf9 a4b73ae d76ce2f 4ed9beb 4dcfe9b ebfdf0d a3b9b9b
     ```
   - Note: `logs/` directory itself has never been tracked by git (`git ls-files logs` returned 0 files).
   - If a live key was ever committed to any file in git history, the repository owner must decide whether to run `git filter-repo` or BFG Repo-Cleaner to rewrite history, or simply treat the key as invalidated by rotation (Step 1 above).
