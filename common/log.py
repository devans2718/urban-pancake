"""Logging utilities for consistent logging across projects."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .paths import LOGS


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name: Logger name (usually __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_file: Custom log file path (uses date-based default if None)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = LOGS / f"{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Avoid double logging
    logger.propagate = False

    return logger


def setup_project_logger(
    project_name: str,
    level: int = logging.INFO,
    log_to_console: bool = True,
) -> logging.Logger:
    return setup_logger(
        name=project_name,
        level=level,
        log_to_console=log_to_console,
        log_file=LOGS / f"{project_name}.log",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a simple logger for quick use.

    Args:
        name: Logger name (uses 'common' if None)

    Returns:
        Configured logger instance
    """
    if name is None:
        name = 'common'
    return setup_logger(name)


# Usage examples:
# logger = setup_logger(__name__)
# logger.info("This is an info message")
# logger.debug("This is a debug message")
# logger.warning("This is a warning")
# logger.error("This is an error")
# logger.exception("This logs an exception with traceback")
