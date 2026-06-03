"""Logging configuration for YOLO Demo."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging (ELK / Loki / Datadog)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


def _get_log_format() -> str:
    """Resolve log format from LOG_FORMAT env var.

    Set LOG_FORMAT=json for structured JSON output.
    Defaults to human-readable format.
    """
    fmt = os.environ.get("LOG_FORMAT", "").lower()
    if fmt == "json":
        return "json"
    return "text"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional path to log file. If None, logs to stderr only.
        format_string: Custom format string. Uses default if not provided.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    log_format = _get_log_format()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    if log_format == "json":
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(format_string))

    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        if log_format == "json":
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)

    logging.info("Logging initialized at level %s (format=%s)",
                 logging.getLevelName(level), log_format)
    if log_file:
        logging.info("Log file: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


# Default logger for this module
logger = get_logger(__name__)
