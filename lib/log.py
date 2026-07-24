"""Shared logging setup for the long-running CLIs.

Call ``setup_logging(logfile)`` at the top of any script that does slow work
(position generation, activation collection, BSP labels, SAE training/eval).
It configures the root logger with a timestamped format and writes to BOTH
stdout and, if given, a UTF-8 log file.

Writing to a FileHandler directly (not only via shell redirection) is
deliberate: it means progress is captured even when the process is detached
(the Windows nohup / console-detach problem), and the file is always UTF-8
regardless of the host console code page (cp1252) or the launching shell
(PowerShell 5.1 would otherwise write UTF-16 on `>`).

Usage:
    from lib.log import setup_logging
    log = setup_logging("logs/champYb_labels.log")  # or setup_logging() for stdout only
    log.info("started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
_DATEFMT = "%m-%d %H:%M:%S"


def setup_logging(logfile: str | Path | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger (timestamped) to stdout + optional UTF-8 file.

    Idempotent-safe via ``force=True`` (replaces any prior basicConfig, e.g. a
    library that called it first). Returns the root logger.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=_FORMAT, datefmt=_DATEFMT,
        handlers=handlers, force=True,
    )
    return logging.getLogger()
