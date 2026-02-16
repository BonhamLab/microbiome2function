import os
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def configure_logging(
    logs_dir: str,
    file_level: int = logging.DEBUG,
    console_level: int = logging.WARNING
):
    """
    Configure the root logger with a timed rotating file handler and console handler.

    Args:
        logs_dir: Directory where log files will be stored.
        file_level: Logging level for the file handler (default: DEBUG).
        console_level: Logging level for the console handler (default: WARNING).
    """
    root = logging.getLogger()
    if root.handlers:
        return

    os.makedirs(logs_dir, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    logfile = os.path.join(
        logs_dir,
        f"m2f{datetime.now():%Y-%m-%d_%H%M%S}.log"
    )
    file_h = TimedRotatingFileHandler(
        logfile,
        when="midnight",
        backupCount=7,
        encoding="utf-8"
    )
    file_h.setLevel(file_level)
    file_h.setFormatter(formatter)

    console_h = logging.StreamHandler()
    console_h.setLevel(console_level)
    console_h.setFormatter(formatter)

    # Ensure root level is low enough to handle both handlers
    root.setLevel(min(file_level, console_level))
    root.addHandler(file_h)
    root.addHandler(console_h)


__all__ = ["configure_logging"]
