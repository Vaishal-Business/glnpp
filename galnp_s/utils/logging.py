from __future__ import annotations
import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, logfile: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:               # already configured
        return log
    log.setLevel(level)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    log.addHandler(sh)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter(_FMT))
        log.addHandler(fh)
    log.propagate = False
    return log