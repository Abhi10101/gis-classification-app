# ============================================================
# logger.py
# ============================================================
# PURPOSE:
# - Centralized logging utility
# - Works on local, Streamlit, and HF Spaces
# - NO dependency on logging config files
#
# Python: 3.8 / 3.9 compatible
# ============================================================

from datetime import datetime
from pathlib import Path
import sys


_LOG_FILE = None
_LOG_FH = None
_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
_MIN_LEVEL = _LOG_LEVELS["INFO"]


def init_logger(log_dir=None):
    """
    Initialize file logging (optional).
    Call once from app.py if needed.
    """
    global _LOG_FILE, _LOG_FH

    if log_dir is None:
        return

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = log_dir / "run.log"

        # open once (performance)
        _LOG_FH = open(_LOG_FILE, "a", encoding="utf-8")

    except Exception:
        _LOG_FILE = None
        _LOG_FH = None


def log_event(message: str, level: str = "INFO"):
    """
    Log a message to console and optionally to file.
    """
    lvl = _LOG_LEVELS.get(level, _LOG_LEVELS["INFO"])
    if lvl < _MIN_LEVEL:
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {message}"

    # Console (Streamlit / HF safe)
    print(line, file=sys.stdout)

    # Optional file logging (no reopen per call)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass
