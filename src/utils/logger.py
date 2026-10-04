"""
Structured logging utility.

Replaces scattered print() calls with a consistent logger that writes
to both stdout and a timestamped log file under logs/.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(name: str, log_dir: str = "logs", level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger that writes INFO+ to stdout and DEBUG+ to a file.
    Safe to call multiple times with the same name (returns cached logger).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File — DEBUG and above
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(Path(log_dir) / f"{ts}_{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
