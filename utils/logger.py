"""
utils/logger.py
Centralised logging configuration for the deepfake detection project.
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to both console and a dated log file.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: Directory where log files are stored.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers when the logger is reused
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler – one file per day
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
