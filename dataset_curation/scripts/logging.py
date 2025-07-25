import os, logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def configure_logging(logs_dir: str):

    root = logging.getLogger()

    if root.handlers: # don't want to be adding new handlers over and over
        return
    
    os.makedirs(logs_dir, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    logfile = os.path.join(logs_dir, f"data_mining_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log")
    file_h = TimedRotatingFileHandler(logfile, when="midnight", backupCount=7, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    console_h = logging.StreamHandler()
    console_h.setLevel(logging.WARNING)
    console_h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG) # need it to be .DEBUG too, so that the root doesn't drop file handler's msgs
    root.addHandler(file_h)
    root.addHandler(console_h)

__all__ = [
    "configure_logging"
]

if __name__ == "__main__":
    pass
