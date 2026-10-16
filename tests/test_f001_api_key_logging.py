import logging
import re

import httpx

from fpl_oracle.utils.logging import RedactingFormatter, SecretRedactingFilter, setup_logging


def test_redaction_filter_and_formatter_catch_all_formats(caplog, monkeypatch):
    """
    S1-1b: Assert redaction filter and formatter catch:
    - AIza... prefix keys
    - AQ.... prefix keys
    - ?key=... in URLs
    - x-goog-api-key headers
    - Authorization headers
    - exact value of any secret loaded from the environment
    - formatted exception text and tracebacks
    - preserves exc_info intact (not None)
    """
    test_env_secret = "super_custom_secret_token_987654"
    monkeypatch.setenv("CUSTOM_API_SECRET_TOKEN", test_env_secret)

    setup_logging()
    test_logger = logging.getLogger("fpl_oracle.test_redaction_comprehensive")

    formatter = RedactingFormatter()
    caplog.handler.setFormatter(formatter)
    caplog.handler.addFilter(SecretRedactingFilter())

    fake_aiza = "AIza" + "SyD123456789012345678901234567890123"
    fake_aq = "AQ." + "AbCdEfGhIjKlMnOpQrStUvWxYz_123456789"

    with caplog.at_level(logging.INFO):
        # 1. AIza key in URL
        test_logger.info(f"Connecting to https://generativelanguage.googleapis.com/v1beta/models?key={fake_aiza}")
        # 2. AQ key in message
        test_logger.info(f"Gemini OAuth token is {fake_aq}")
        # 3. x-goog-api-key header
        test_logger.info(f"Headers: x-goog-api-key: {fake_aiza}")
        test_logger.info(f"Headers dict: {{'x-goog-api-key': '{fake_aq}'}}")
        # 4. Authorization header
        test_logger.info("Headers: Authorization: Bearer secret_bearer_token_xyz")
        # 5. Exact value of environment secret
        test_logger.info(f"Loaded config secret: {test_env_secret}")

        # 6. Exception with traceback containing fake key
        try:
            req = httpx.Request("GET", f"https://api.example.com/v1?key={fake_aq}&other=1")
            resp = httpx.Response(403, request=req)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            test_logger.exception("HTTP call failed")

    log_text = caplog.text

    # Assert no secrets in captured logs
    assert fake_aiza not in log_text, "Found unredacted AIza key in logs"
    assert fake_aq not in log_text, "Found unredacted AQ key in logs"
    assert test_env_secret not in log_text, "Found unredacted env secret in logs"
    assert "secret_bearer_token_xyz" not in log_text, "Found unredacted Bearer token in logs"

    # Assert redaction markers are present
    assert "[REDACTED" in log_text


def test_exc_info_preserved_and_traceback_scrubbed(caplog):
    """
    S1-1c: Keep exc_info=True (tracebacks do not print locals),
    but make the redacting formatter scrub the fully formatted traceback.
    Test with a mocked failing HTTP call whose exception message contains a fake key.
    """
    setup_logging()
    test_logger = logging.getLogger("fpl_oracle.test_exc_info")

    formatter = RedactingFormatter()
    caplog.handler.setFormatter(formatter)
    filter_ = SecretRedactingFilter()
    caplog.handler.addFilter(filter_)

    fake_key = "AQ." + "SecretKeyForHttpCallFailure_9988776655"

    record_captured = None

    class CaptureFilter(logging.Filter):
        def filter(self, record):
            nonlocal record_captured
            record_captured = record
            return True

    test_logger.addFilter(CaptureFilter())

    try:
        req = httpx.Request(
            "POST", f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro?key={fake_key}"
        )
        resp = httpx.Response(500, request=req)
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        test_logger.exception("Failing HTTP call")

    # Assert exc_info is NOT None on the log record (preserved for debuggers / APM)
    assert record_captured is not None
    assert record_captured.exc_info is not None, "record.exc_info was stripped or set to None"
    assert record_captured.exc_info[0] is httpx.HTTPStatusError

    # Assert formatted text scrubs the key from the traceback
    formatted_output = formatter.format(record_captured)
    assert fake_key not in formatted_output, "Raw key leaked in formatted traceback output"
    assert "[REDACTED" in formatted_output


def test_app_startup_path_logging_and_warnings(caplog):
    """
    S1-1d: Entry points: call setup_logging() first, before other imports that configure
    logging. httpx and httpcore loggers at WARNING. Prove with a test that runs the real
    app startup path and asserts no key-shaped string in captured logs.
    """
    setup_logging()
    import fpl_oracle.server.main as srv_main

    assert srv_main.app is not None

    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    assert httpx_logger.level >= logging.WARNING
    assert httpcore_logger.level >= logging.WARNING

    # Ensure no key-shaped strings in logs during startup
    key_pattern = re.compile(r"AIza[0-9A-Za-z-_]{35}|AQ\.[A-Za-z0-9_-]{20,}")
    assert not key_pattern.search(caplog.text)
