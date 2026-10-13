# Remediation Log: Adversarial Audit Hotfixes

This log tracks every remediation commit across Phase H0, H1, H2, and handover to the improvement spec.
Per finding: commit hash, test name, before and after execution output, configuration flags added, and unexpected discoveries.

---

## F-001: Live Plaintext API Key Leak in Logs

- **Commit**: (Pending commit) `F-001: Redact API keys, auth headers, and mute httpx/httpcore loggers`
- **Test Name**: `tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions`, `tests/test_f001_api_key_logging.py::test_mocked_llm_call_logging_redaction`
- **Flags Added**: None (pure security bugfix)
- **Before Output**:
  ```text
  FAILED tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions - AssertionError: httpx logger level is 0, expected >= WARNING
  assert 0 >= 30
   +  where 0 = <Logger httpx (INFO)>.level
   +  and   30 = logging.WARNING
  ```
- **After Output**:
  ```text
  tests/test_f001_api_key_logging.py::test_no_plaintext_keys_or_auth_in_logs_and_exceptions PASSED [ 50%]
  tests/test_f001_api_key_logging.py::test_mocked_llm_call_logging_redaction PASSED [100%]
  2 passed in 0.18s
  ```
- **Actions Taken**:
  1. Configured `httpx` and `httpcore` loggers to `logging.WARNING` at import time and within `setup_logging()`.
  2. Enhanced `SecretRedactingFilter` to redact query params (`key=...`, `api_key=...`), `Authorization` / `Bearer` headers, Google API key regex `AIza[0-9A-Za-z-_]{35}`, and environment secrets, across messages, exc_text, and exception tracebacks.
  3. Ensured `.gitignore` includes `logs/`, `.env`, `*.log`, and `data/cache/`.
  4. Verified git tracking: `logs/` is not tracked (`git ls-files logs` returned 0 entries). Logged commits touching `key=` in `docs/needs_human.md`.
  5. Created redacted log copy `logs/fpl_oracle.redacted.log` (0 unredacted Google API key patterns remaining).
- **Surprises / Notes**:
  `logs/` was already properly gitignored on disk and not tracked in git; the leak was purely local disk log persistence.

