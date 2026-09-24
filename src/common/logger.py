"""Centralized structured logger for Athena AI."""

import logging
import sys
from pathlib import Path
from typing import Optional

from src.common.config import PATHS


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Get a structured logger that logs to console and optionally to a file.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger was already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = PATHS.logs_dir / "athena.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
