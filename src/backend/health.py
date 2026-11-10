"""
Health check module for the JACCS Credit Scoring System.

Performs a lightweight system status check that is safe to call at startup
or from the Admin Dashboard without triggering any heavy operations like
model training.

Usage:
    from src.backend.health import health_check, HealthStatus

    status = health_check()
    if not status.is_healthy:
        print(status.summary())
"""
from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    """Resolve the project root directory for dev and frozen (PyInstaller) mode."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _project_root()


@dataclass
class HealthStatus:
    """
    Snapshot of the system's health at a given point in time.

    Attributes:
        model_cache_exists: Whether a cached `.joblib` model file is present.
        training_data_present: Whether all expected training Excel files exist.
        db_config_present: Whether a database configuration file exists.
        db_connected: Whether a live database connection was successful.
        db_error: Error message if the DB connection failed.
        model_version: Version integer from the cached model bundle.
        model_path: Absolute path to the found model cache file (or None).
        users_store_exists: Whether the user store (CSV or SQLite DB) exists.
        log_dir_writable: Whether the log directory is writable.
        timestamp: ISO 8601 timestamp of when the check was performed.
        python_version: Python version string.
        app_version: Application version string.
    """
    model_cache_exists: bool = False
    training_data_present: bool = False
    db_config_present: bool = False
    db_connected: bool = False
    db_error: Optional[str] = None
    model_version: Optional[int] = None
    model_path: Optional[str] = None
    users_store_exists: bool = False
    log_dir_writable: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    app_version: str = "2.0.0"

    @property
    def is_healthy(self) -> bool:
        """True if the minimum requirements for running are met."""
        return self.training_data_present and self.users_store_exists

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        lines = [
            f"Health Check — {self.timestamp}",
            f"  App version       : {self.app_version}",
            f"  Python            : {self.python_version}",
            f"  Training data     : {'✓' if self.training_data_present else '✗ MISSING'}",
            f"  Model cache       : {'✓' if self.model_cache_exists else '✗ not cached (will train)'}",
            f"  Model version     : {self.model_version if self.model_version else 'unknown'}",
            f"  Users store       : {'✓' if self.users_store_exists else '✗ MISSING'}",
            f"  DB config         : {'✓' if self.db_config_present else '✗ not configured'}",
            f"  DB connected      : {'✓' if self.db_connected else '✗ ' + (self.db_error or 'not tested')}",
            f"  Log dir writable  : {'✓' if self.log_dir_writable else '✗ WARNING'}",
            f"  Overall status    : {'HEALTHY ✓' if self.is_healthy else 'DEGRADED ✗'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "model_cache_exists": self.model_cache_exists,
            "training_data_present": self.training_data_present,
            "db_config_present": self.db_config_present,
            "db_connected": self.db_connected,
            "db_error": self.db_error,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "users_store_exists": self.users_store_exists,
            "log_dir_writable": self.log_dir_writable,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "app_version": self.app_version,
            "is_healthy": self.is_healthy,
        }


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

TRAINING_FILES = [
    "src/data/May 12 Status Contracts.xlsx",
    "src/data/Extracted New.xlsx",
]

MODELS_DIR = PROJECT_ROOT / "src" / "backend" / "models"


def _check_training_data() -> bool:
    """Return True if all expected training workbooks are present."""
    return all((PROJECT_ROOT / rel).exists() for rel in TRAINING_FILES)


def _check_model_cache() -> tuple[bool, Optional[str], Optional[int]]:
    """
    Look for any `.joblib` file in the models directory.

    Returns:
        (cache_exists, model_path_str, model_version_int)
    """
    if not MODELS_DIR.exists():
        return False, None, None
    joblobs = list(MODELS_DIR.glob("*.joblib"))
    if not joblobs:
        return False, None, None

    # Pick the most recently modified cache
    latest = max(joblobs, key=lambda p: p.stat().st_mtime)
    version = None
    try:
        import joblib  # type: ignore[import]
        bundle = joblib.load(latest)
        version = bundle.get("version") if isinstance(bundle, dict) else None
    except Exception:
        pass
    return True, str(latest), version


def _check_db_config() -> bool:
    """Return True if a non-empty db_config.json file exists."""
    cfg_path = PROJECT_ROOT / "src" / "data" / "database" / "db_config.json"
    if not cfg_path.exists():
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        # Consider it configured if at least server is non-empty
        return bool(cfg.get("server", "").strip())
    except Exception:
        return False


def _check_db_connection() -> tuple[bool, Optional[str]]:
    """
    Attempt a live database connection.

    Returns:
        (success, error_message_or_None)
    """
    try:
        from src.backend.db import get_connection  # type: ignore[import]
        conn = get_connection()
        conn.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def _check_users_store() -> bool:
    """Return True if either the SQLite user DB or CSV user store exists."""
    # Try SQLite DB first (new format)
    db_paths = [
        PROJECT_ROOT / "src" / "backend" / "users.db",
        Path(os.path.expanduser("~")) / ".jaccs_credit_app" / "users.db",
    ]
    for p in db_paths:
        if p.exists():
            return True
    # Fall back to CSV (legacy format)
    csv_paths = [
        PROJECT_ROOT / "src" / "backend" / "users.csv",
        Path(os.path.expanduser("~")) / ".jaccs_credit_app" / "users.csv",
    ]
    for p in csv_paths:
        if p.exists():
            return True
    return False


def _check_log_dir() -> bool:
    """Return True if the log directory is writable."""
    try:
        from src.backend.logger import _LOG_DIR
        test_file = Path(_LOG_DIR) / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def health_check(check_db: bool = False) -> HealthStatus:
    """
    Perform a full system health check.

    Args:
        check_db: If ``True``, attempt a live database connection (adds latency).
                  Defaults to ``False`` to keep startup fast.

    Returns:
        A :class:`HealthStatus` dataclass instance with all check results.
    """
    cache_exists, model_path, model_version = _check_model_cache()
    db_connected, db_error = (False, None)
    if check_db:
        db_connected, db_error = _check_db_connection()

    return HealthStatus(
        model_cache_exists=cache_exists,
        training_data_present=_check_training_data(),
        db_config_present=_check_db_config(),
        db_connected=db_connected,
        db_error=db_error,
        model_version=model_version,
        model_path=model_path,
        users_store_exists=_check_users_store(),
        log_dir_writable=_check_log_dir(),
    )


if __name__ == "__main__":  # pragma: no cover
    status = health_check(check_db=True)
    print(status.summary())
    sys.exit(0 if status.is_healthy else 1)

