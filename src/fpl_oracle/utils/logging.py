"""
Structured Rotating File Logger for FPL Oracle.
Directs all application logs to logs/fpl_oracle.log with rotation,
and formats logs with timestamps, levels, module names, and line numbers.
"""

import logging
from logging.handlers import RotatingFileHandler

from fpl_oracle.config import LOGS_DIR

_is_configured = False


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

    for handler in root_logger.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(redacting_filter)

    if _is_configured:
        return logging.getLogger("fpl_oracle")

    root_logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

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


class SecretRedactingFilter(logging.Filter):
    """Remove credentials and secrets from rendered messages, URLs, and exception tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        import os
        import re

        text = record.getMessage()
        if record.exc_text:
            text += "\n" + record.exc_text
            record.exc_text = None

        if record.exc_info:
            text += "\n" + logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
            record.exc_text = None

        # Redact query parameters (?key=..., &api_key=..., etc.)
        text = re.sub(
            r"(?i)([?&](?:key|api_key|apikey|access_token)=)[^&\s\"'\\]+",
            r"\g<1>[REDACTED]",
            text,
        )
        # Redact Authorization header & Bearer tokens
        text = re.sub(
            r"(?i)(Authorization:\s*(?:Bearer\s+)?)[^\r\n\s'\"]+",
            r"\g<1>[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]+",
            r"Bearer [REDACTED]",
            text,
        )
        # Redact Google API key pattern
        text = re.sub(
            r"AIza[0-9A-Za-z-_]{35}",
            r"[REDACTED_GOOGLE_API_KEY]",
            text,
        )
        for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            secret = os.getenv(name, "").strip()
            if secret:
                text = text.replace(secret, "[REDACTED]")

        record.msg, record.args = text, ()
        return True
