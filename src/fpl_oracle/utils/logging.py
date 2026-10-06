"""
Structured Rotating File Logger for FPL Oracle.
Directs all application logs to logs/fpl_oracle.log with rotation,
and formats logs with timestamps, levels, module names, and line numbers.
"""

import logging
from logging.handlers import RotatingFileHandler

from fpl_oracle.config import LOGS_DIR

_is_configured = False


def setup_logging(
    log_file: str = "fpl_oracle.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    level: int = logging.INFO,
) -> logging.Logger:
    global _is_configured
    root_logger = logging.getLogger()

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
        root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    log_path = LOGS_DIR / log_file
    file_handler = RotatingFileHandler(str(log_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _is_configured = True
    return logging.getLogger("fpl_oracle")
