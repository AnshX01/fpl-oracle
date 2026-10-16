"""
Structured Rotating File Logger for FPL Oracle.
Directs all application logs to logs/fpl_oracle.log with rotation,
and formats logs with timestamps, levels, module names, and line numbers.
Includes comprehensive secret and credential redaction across log records,
tracebacks, exceptions, and headers.
"""

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from typing import Any

from fpl_oracle.config import BASE_DIR, LOGS_DIR

_is_configured = False

# Capture environment secrets present at startup
_STARTUP_SECRETS: set[str] = set()


def _init_startup_secrets() -> None:
    """Populate secret cache from environment variables and local .env."""
    for key, val in os.environ.items():
        if not val or len(val.strip()) < 6:
            continue
        key_upper = key.upper()
        if any(term in key_upper for term in ("KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH", "CREDENTIAL", "API")):
            _STARTUP_SECRETS.add(val.strip())

    env_path = BASE_DIR / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    _, val = line.split("=", 1)
                    val = val.strip().strip("'\"")
                    if len(val) >= 6:
                        _STARTUP_SECRETS.add(val)
        except Exception:
            pass


_init_startup_secrets()


def redact_text(text: str) -> str:
    """Scrub sensitive credentials, API keys, Authorization headers, and URLs."""
    if not text:
        return text

    # 1. Exact value redaction of known environment secrets
    for secret in _STARTUP_SECRETS:
        if secret and secret in text:
            text = text.replace(secret, "[REDACTED]")

    # Check any dynamically added secrets in os.environ
    for key, val in os.environ.items():
        if not val:
            continue
        val_str = val.strip()
        if len(val_str) >= 6 and any(
            term in key.upper() for term in ("KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH", "CREDENTIAL", "API")
        ):
            if val_str in text:
                text = text.replace(val_str, "[REDACTED]")

    # 2. Query parameters (?key=..., &api_key=..., etc.)
    text = re.sub(
        r"(?i)([?&](?:key|api_key|apikey|access_token)=)[^&\s\"'\\]+",
        r"\g<1>[REDACTED]",
        text,
    )

    # 3. Authorization header & Bearer tokens
    text = re.sub(
        r"(?i)(Authorization:\s*(?:Bearer\s+)?)[^\r\n\s'\"]+",
        r"\g<1>[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(['\"]Authorization['\"]\s*:\s*['\"](?:Bearer\s+)?[^'\"]+['\"])",
        r"'Authorization': '[REDACTED]'",
        text,
    )
    text = re.sub(
        r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]+",
        r"Bearer [REDACTED]",
        text,
    )

    # 4. x-goog-api-key headers (plain and dict/json)
    text = re.sub(
        r"(?i)(x-goog-api-key:\s*)[^\r\n\s'\"]+",
        r"\g<1>[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(['\"]x-goog-api-key['\"]\s*:\s*['\"])[^'\"]+(['\"])",
        r"\g<1>[REDACTED]\g<2>",
        text,
    )

    # 5. Generic and prefix API key patterns
    text = re.sub(
        r"AIza[0-9A-Za-z-_]{35}",
        r"[REDACTED_GOOGLE_API_KEY]",
        text,
    )
    text = re.sub(
        r"AQ\.[A-Za-z0-9_-]{20,}",
        r"[REDACTED_GOOGLE_API_KEY]",
        text,
    )

    return text


class RedactingFormatter(logging.Formatter):
    """Formats log records and scrubs secrets from the entire rendered output, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        return redact_text(s)

    def formatException(self, ei: Any) -> str:
        s = super().formatException(ei)
        return redact_text(s)


class SecretRedactingFilter(logging.Filter):
    """
    Remove credentials and secrets from rendered messages and pre-formatted text.
    Preserves record.exc_info intact so exception tracebacks are available.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (redact_text(v) if isinstance(v, str) else v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_text(a) if isinstance(a, str) else a for a in record.args)
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        # Note: Do NOT clear record.exc_info! RedactingFormatter will format and redact the traceback.
        return True


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def setup_logging(
    log_file: str = "fpl_oracle.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    level: int = logging.INFO,
) -> logging.Logger:
    global _is_configured
    root_logger = logging.getLogger()

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    redacting_filter = SecretRedactingFilter()
    if not any(isinstance(f, SecretRedactingFilter) for f in root_logger.filters):
        root_logger.addFilter(redacting_filter)

    formatter = RedactingFormatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in root_logger.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(redacting_filter)
        handler.setFormatter(formatter)

    if _is_configured:
        return logging.getLogger("fpl_oracle")

    root_logger.setLevel(level)

    # 1. Console Handler (if not already present)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers
    )
    if not has_stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(redacting_filter)
        root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    log_path = LOGS_DIR / log_file
    file_handler = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redacting_filter)
    root_logger.addHandler(file_handler)

    _is_configured = True
    return logging.getLogger("fpl_oracle")
