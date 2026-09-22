"""
utils/logger_v2.py - Logger with fallback log directory support.
"""
import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always works)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler with fallback to /tmp/logs
    for candidate_dir in [log_dir, "/tmp/deepfake_logs"]:
        try:
            os.makedirs(candidate_dir, exist_ok=True)
            log_file = os.path.join(candidate_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            break
        except (PermissionError, OSError):
            continue

    return logger
