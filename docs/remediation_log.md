# Remediation Log: Adversarial Audit Hotfixes

This log tracks every remediation commit across Phase H0, H1, H2, and handover to the improvement spec.
Per finding: commit hash, test name, before and after execution output, configuration flags added, and unexpected discoveries.

---

## F-001: Live Plaintext API Key Leak in Logs

- **Commit**: `b8b880d` (`F-001: Redact API keys and auth headers, mute httpx loggers`)
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

---

## F-008: In-Process Thread Lock Only (Zero Cross-Process Protection)

- **Commit**: `2b1162b` (`F-008: Replace thread-only lock with cross-process FileLock`)
- **Test Name**: `tests/test_f006_f008_bundle_recovery_and_lock.py::test_cross_process_bundle_lock_mutual_exclusion`
- **Flags Added**: None (system infrastructure)
- **Before Output**:
  ```text
  Only threading.RLock() was used in bundle_lock.py; concurrent OS processes had zero mutual exclusion on bundle operations.
  ```
- **After Output**:
  ```text
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_cross_process_bundle_lock_mutual_exclusion PASSED [100%]
  ```
- **Actions Taken**:
  1. Updated `src/fpl_oracle/ml/bundle_lock.py` to use `filelock.FileLock` anchored to `MODELS_DIR / "bundle.lock"`.
  2. Implemented canonical path caching (`_LOCK_CACHE`) to preserve re-entrancy for nested locked method calls within the same process while guaranteeing strict OS-level exclusion across processes.
- **Surprises / Notes**:
  Cross-process file locking must be reentrant on the same process/thread to allow nested `@bundle_locked` methods (e.g. `verify_weight_integrity` calling `quarantine_and_recover`).

---

## F-006: Automatic Checkpoint Rollback on Corrupt Bundle

- **Commit**: (Pending commit) `F-006: Add automatic model quarantine and rollback recovery`
- **Test Name**: `tests/test_f006_f008_bundle_recovery_and_lock.py::test_corrupt_bundle_auto_recovers_to_previous_version`, `tests/test_f006_f008_bundle_recovery_and_lock.py::test_no_valid_version_raises_clear_error_not_raw_valueerror`, `tests/test_f006_f008_bundle_recovery_and_lock.py::test_tampered_manifest_auto_recovers`
- **Flags Added**: None (reliability bugfix)
- **Before Output**:
  ```text
  ValueError: SHA256 mismatch for minutes_model.pkl: expected 8ebdc..., got a7c8e...
  Corrupt file remains corrupted on disk and inference pipeline crashes with unhandled ValueError.
  ```
- **After Output**:
  ```text
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_corrupt_bundle_auto_recovers_to_previous_version PASSED
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_no_valid_version_raises_clear_error_not_raw_valueerror PASSED
  tests/test_f006_f008_bundle_recovery_and_lock.py::test_tampered_manifest_auto_recovers PASSED
  audit/scratch/test_corrupt_bundle.py output:
  [ModelIntegrity] AUTOMATIC RECOVERY SUCCESSFUL: Quarantined bad files, restored verified version 'v1.0.0'.
  ```
- **Actions Taken**:
  1. Added `ModelBundleCorruptError` exception class.
  2. Implemented `quarantine_and_recover()` in `ModelRegistry`: moves bad `.pkl` and calibration files into `quarantine/<timestamp>/`, locates the latest verified backup from `versions/`, restores all files, updates active version, and writes `recovery_event.json`.
  3. Integrated verification into `predict.py:load_or_train()`: never loads unverified weights; automatically quarantines and recovers on corruption.
  4. Exposed recovery event in `get_model_status()` and `/health` FastAPI endpoint.
- **Surprises / Notes**:
  `audit/scratch/test_corrupt_bundle.py` asserted that `verify_weight_integrity()` would raise an exception; with automatic self-healing recovery, it now recovers and succeeds cleanly, matching operational resilience requirements.


