import logging
import re
import pytest
from fpl_oracle.utils.logging import SecretRedactingFilter, setup_logging


def test_no_plaintext_keys_or_auth_in_logs_and_exceptions(caplog):
    """
    F-001: Assert that logging config redacts:
    - query parameters (?key=..., &api_key=...)
    - Authorization headers
    - Google API key regex (AIza...)
    - Exceptions and tracebacks containing API keys
    - httpx and httpcore loggers are set to WARNING
    """
    setup_logging()

    # Check logger levels for httpx and httpcore
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    assert httpx_logger.level >= logging.WARNING, f"httpx logger level is {httpx_logger.level}, expected >= WARNING"
    assert httpcore_logger.level >= logging.WARNING, f"httpcore logger level is {httpcore_logger.level}, expected >= WARNING"

    fake_key = "AIzaSyD1234567890123456789012345678901"
    test_logger = logging.getLogger("fpl_oracle.test_security")

    # Add filter to caplog handler as well
    redacting_filter = SecretRedactingFilter()
    caplog.handler.addFilter(redacting_filter)

    with caplog.at_level(logging.INFO):
        # 1. Direct log with URL containing key=
        test_logger.info(f"Request sent to https://generativelanguage.googleapis.com/v1beta/models?key={fake_key}")
        # 2. Direct log with Authorization header
        test_logger.info("Headers: {'Authorization': 'Bearer secret_token_abc123'}")
        # 3. Exception containing URL with key
        try:
            raise ValueError(f"HTTP 403 Forbidden: URL https://api.google.com/test?key={fake_key}")
        except Exception:
            test_logger.exception("LLM call failed with exception")

    log_text = caplog.text
    google_key_pattern = re.compile(r"AIza[0-9A-Za-z-_]{35}")

    assert not google_key_pattern.search(log_text), f"Found unredacted Google API key in log text: {log_text}"
    assert fake_key not in log_text, "Found raw API key string in log text"
    assert "secret_token_abc123" not in log_text, "Found raw Authorization token in log text"
    assert "[REDACTED" in log_text, "Expected [REDACTED] in sanitized log text"


def test_mocked_llm_call_logging_redaction(caplog, monkeypatch):
    """
    Simulate a mocked LLM call producing error logs or exceptions with Google API keys,
    asserting that the logging filter strips the key from output.
    """
    setup_logging()
    test_key = "AIzaSyZ9876543210987654321098765432109"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)

    caplog.handler.addFilter(SecretRedactingFilter())

    logger = logging.getLogger("fpl_oracle.llm.mock")
    with caplog.at_level(logging.INFO):
        # Simulate httpx request info log or failed LLM query
        logger.info("POST https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=%s", test_key)
        try:
            raise RuntimeError(f"Failed to connect to https://generativelanguage.googleapis.com/v1beta/models?key={test_key}")
        except RuntimeError:
            logger.exception("Provider execution failed")

    assert test_key not in caplog.text
    assert "AIzaSyZ" not in caplog.text
    assert "[REDACTED" in caplog.text
