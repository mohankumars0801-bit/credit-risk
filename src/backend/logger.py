"""
Centralized logging module for the JACCS Credit Risk application.

Provides:
- Rotating file handler at ~/.jaccs_credit_app/logs/app.log (10 MB, 5 backups)
- Console handler for WARNING+ messages
- get_logger(name) factory function
- log_prediction() helper for prediction events
- log_model_event() helper for model lifecycle events
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _resolve_app_dir() -> str:
    """Return a writable base directory for the application."""
    custom = os.getenv("JACCS_APP_DIR")
    if custom:
        os.makedirs(custom, exist_ok=True)
        return custom
    candidate = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app")
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except OSError:
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "jaccs_credit_app")
        os.makedirs(fallback, exist_ok=True)
        return fallback

_APP_DIR = _resolve_app_dir()
_LOG_DIR = os.path.join(_APP_DIR, "logs")
_APP_LOG = os.path.join(_LOG_DIR, "app.log")
_PRED_LOG = os.path.join(_LOG_DIR, "predictions.log")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_log_dir() -> None:
    """Create the log directory if it does not already exist."""
    global _LOG_DIR, _APP_LOG, _PRED_LOG
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except OSError as exc:
        import tempfile, sys
        _LOG_DIR = os.path.join(tempfile.gettempdir(), "jaccs_credit_app", "logs")
        os.makedirs(_LOG_DIR, exist_ok=True)
        _APP_LOG = os.path.join(_LOG_DIR, "app.log")
        _PRED_LOG = os.path.join(_LOG_DIR, "predictions.log")


def _build_rotating_handler(path: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> RotatingFileHandler:
    """Return a configured RotatingFileHandler for *path*."""
    _ensure_log_dir()
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    return handler


def _build_console_handler() -> logging.StreamHandler:
    """Return a StreamHandler that only emits WARNING and above."""
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(levelname)-8s | %(name)s | %(message)s",
        )
    )
    return handler


# ---------------------------------------------------------------------------
# Root application logger (lazy-initialised once)
# ---------------------------------------------------------------------------

_root_app_logger: logging.Logger | None = None


def _get_root_app_logger() -> logging.Logger:
    """Return (and lazily initialise) the root application logger."""
    global _root_app_logger
    if _root_app_logger is not None:
        return _root_app_logger

    logger = logging.getLogger("jaccs_credit_app")
    if not logger.handlers:  # guard against double-init
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_build_rotating_handler(_APP_LOG))
        logger.addHandler(_build_console_handler())
        logger.propagate = False

    _root_app_logger = logger
    return logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the application root hierarchy.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger that writes to the rotating app log file and
        emits WARNING+ messages to the console.

    Example
    -------
    >>> from src.backend.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Module initialised")
    """
    _get_root_app_logger()  # ensure handlers are attached
    child = logging.getLogger(f"jaccs_credit_app.{name}")
    return child


# ---------------------------------------------------------------------------
# Prediction logger (separate file, lazily initialised)
# ---------------------------------------------------------------------------

_pred_logger: logging.Logger | None = None


def _get_pred_logger() -> logging.Logger:
    """Return (and lazily initialise) the predictions logger."""
    global _pred_logger
    if _pred_logger is not None:
        return _pred_logger

    logger = logging.getLogger("jaccs_credit_app.predictions")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_build_rotating_handler(_PRED_LOG))
        logger.propagate = False  # do not duplicate into app.log

    _pred_logger = logger
    return logger


def log_prediction(
    party_no: int | str,
    app_no: int | str,
    probability: float,
    credit_score: float | int,
    threshold: float,
) -> None:
    """
    Write a structured prediction record to predictions.log.

    Parameters
    ----------
    party_no:
        Customer party number.
    app_no:
        Loan application number.
    probability:
        Raw model output probability (0–1).
    credit_score:
        Derived credit score (e.g. 300–850 scale).
    threshold:
        Decision threshold used to classify the prediction.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "party_no": party_no,
        "app_no": app_no,
        "probability": round(float(probability), 6),
        "credit_score": credit_score,
        "threshold": threshold,
        "decision": "DECLINE" if probability >= threshold else "APPROVE",
    }
    _get_pred_logger().info(json.dumps(record))


def log_model_event(event_type: str, details_dict: dict[str, Any]) -> None:
    """
    Write a structured model lifecycle event to app.log.

    Parameters
    ----------
    event_type:
        Short label describing the event (e.g. ``"model_load"``,
        ``"model_reload"``, ``"prediction_error"``).
    details_dict:
        Arbitrary key/value metadata to include in the log record.

    Example
    -------
    >>> log_model_event("model_load", {"path": "/models/lgbm.joblib", "status": "ok"})
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **details_dict,
    }
    logging.getLogger("jaccs_credit_app.model").info(json.dumps(record))


# Ensure the root logger is initialised when the module is first imported.
_get_root_app_logger()
