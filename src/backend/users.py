"""
Production-hardened user management module for the JACCS Credit Risk application.

Storage backend: SQLite (thread-safe, concurrent-safe).
Password hashing: bcrypt when available, falls back to hashlib PBKDF2-HMAC-SHA256.
Features:
- Thread-safe operations via threading.Lock
- Rotating audit log at ~/.jaccs_credit_app/logs/audit.log
- Input validation (email regex, minimum password length)
- Rate-limiting: 3 failed login attempts → 30-second lockout per email
- Automatic migration from legacy users.csv if present
- Backward-compatible public API

Public API (unchanged from original):
    ensure_user_store()
    authenticate(email, password) -> Optional[str]
    add_user(email, password, role) -> None
    delete_user(email) -> None
    list_users() -> List[Dict]
    get_user(email) -> Optional[Dict]
    set_role(email, role) -> None
    update_password(email, new_password) -> None
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional bcrypt import (falls back to PBKDF2 if not installed)
# ---------------------------------------------------------------------------
try:
    import bcrypt as _bcrypt  # type: ignore[import]
    _BCRYPT_AVAILABLE = True
except ImportError:
    _bcrypt = None  # type: ignore[assignment]
    _BCRYPT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Audit logger (independent of the central logger to avoid circular imports)
# ---------------------------------------------------------------------------

def _resolve_app_dir() -> str:
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
_AUDIT_LOG = os.path.join(_LOG_DIR, "audit.log")


def _build_audit_logger() -> logging.Logger:
    """Create and return a dedicated rotating audit logger."""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except OSError:
        pass

    logger = logging.getLogger("jaccs_credit_app.audit")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        try:
            fh = RotatingFileHandler(
                _AUDIT_LOG, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fh.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%S%z")
            )
            logger.addHandler(fh)
        except OSError:
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
    return logger


_audit_log = _build_audit_logger()

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# Keep the legacy CSV path resolution for frozen / dev mode
def _get_users_csv_path() -> str:
    """Resolve the legacy users.csv path (frozen exe or dev mode)."""
    try:
        if getattr(sys, "frozen", False):
            bundled = os.path.join(sys._MEIPASS, "users.csv")  # type: ignore[attr-defined]
            if os.path.exists(bundled):
                return bundled
            app_dir = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app")
            os.makedirs(app_dir, exist_ok=True)
            return os.path.join(app_dir, "users.csv")
        return os.path.join(os.path.dirname(__file__), "users.csv")
    except Exception as exc:
        import tempfile
        print(f"[Users] Warning: Failed to resolve users.csv path ({exc}). Using temp location.")
        return os.path.join(tempfile.gettempdir(), "users.csv")


USERS_CSV = _get_users_csv_path()


def _get_db_path() -> str:
    """Return path to the SQLite users database."""
    try:
        if getattr(sys, "frozen", False):
            app_dir = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app")
            os.makedirs(app_dir, exist_ok=True)
            return os.path.join(app_dir, "users.db")
        return os.path.join(os.path.dirname(__file__), "users.db")
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "users.db")


_DB_PATH = _get_db_path()

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_MIN_PASSWORD_LEN = 8
_VALID_ROLES = {"admin", "user", "viewer"}

# ---------------------------------------------------------------------------
# Rate-limiting state (in-memory, per process)
# ---------------------------------------------------------------------------

# { email_lower: {"fails": int, "locked_until": float} }
_rate_limit: Dict[str, Dict] = {}
_RATE_LIMIT_MAX_FAILS = 3
_RATE_LIMIT_LOCKOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[str] = None):
    """
    Hash *password* using bcrypt (preferred) or PBKDF2-HMAC-SHA256 (fallback).

    Returns
    -------
    tuple[str, str]
        (salt_or_marker, password_hash) – both are strings suitable for
        storage in the database.
    """
    if _BCRYPT_AVAILABLE:
        hashed = _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt(rounds=12))
        # Use a fixed marker so we know the algo at verify time
        return "bcrypt", hashed.decode("utf-8")
    # Fallback: PBKDF2-HMAC-SHA256
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000)
    return salt, dk.hex()


def _verify_password(password: str, stored_salt: str, stored_hash: str) -> bool:
    """Return True if *password* matches the stored hash."""
    try:
        if stored_salt == "bcrypt":
            if not _BCRYPT_AVAILABLE:
                # Cannot verify bcrypt hash without the library
                return False
            return _bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        # PBKDF2 fallback
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), stored_salt.encode("utf-8"), 260_000)
        return dk.hex() == stored_hash
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_email(email: str) -> None:
    """Raise ValueError if *email* is not a valid email address."""
    if not email or not _EMAIL_RE.match(email.strip()):
        raise ValueError(f"Invalid email address: {email!r}")


def _validate_password(password: str) -> None:
    """Raise ValueError if *password* does not meet minimum requirements."""
    if len(password) < _MIN_PASSWORD_LEN:
        raise ValueError(
            f"Password must be at least {_MIN_PASSWORD_LEN} characters long."
        )


def _validate_role(role: str) -> None:
    """Raise ValueError if *role* is not recognised."""
    if role not in _VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}. Must be one of: {_VALID_ROLES}")

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode enabled for concurrency."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create the users table if it does not already exist."""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email         TEXT PRIMARY KEY COLLATE NOCASE,
                role          TEXT NOT NULL DEFAULT 'user',
                salt          TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()

# ---------------------------------------------------------------------------
# CSV migration
# ---------------------------------------------------------------------------

def _migrate_csv_if_needed() -> None:
    """
    If a legacy users.csv exists and the SQLite DB is empty, import all rows.
    Old SHA-256 hashes are kept as-is with their original salt; they will
    authenticate correctly via the legacy path until the user changes password.
    """
    if not os.path.exists(USERS_CSV):
        return
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count > 0:
            return  # DB already has data – skip migration

    try:
        with open(USERS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            return

        with _get_conn() as conn:
            for row in rows:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO users (email, role, salt, password_hash) VALUES (?,?,?,?)",
                        (row["email"].strip(), row.get("role", "user").strip(),
                         row["salt"].strip(), row["password_hash"].strip()),
                    )
                except Exception as exc:
                    _audit_log.warning("CSV migration: skipped row %s – %s", row.get("email"), exc)
            conn.commit()
        _audit_log.info("CSV migration complete: imported %d user(s) from %s", len(rows), USERS_CSV)
    except Exception as exc:
        _audit_log.error("CSV migration failed: %s", exc)


def _verify_legacy_sha256(password: str, salt: str, stored_hash: str) -> bool:
    """Verify a password against the original SHA-256 scheme used by the old module."""
    import hashlib as _hl
    test = _hl.sha256((salt + password).encode("utf-8")).hexdigest()
    return test == stored_hash

# ---------------------------------------------------------------------------
# Rate-limiting helpers
# ---------------------------------------------------------------------------

def _check_rate_limit(email_lower: str) -> None:
    """Raise RuntimeError if the account is currently locked out."""
    state = _rate_limit.get(email_lower)
    if state and state["locked_until"] > time.monotonic():
        remaining = int(state["locked_until"] - time.monotonic())
        raise RuntimeError(
            f"Account temporarily locked. Try again in {remaining} second(s)."
        )


def _record_failed_login(email_lower: str) -> None:
    """Increment fail counter and apply lockout if threshold exceeded."""
    state = _rate_limit.setdefault(email_lower, {"fails": 0, "locked_until": 0.0})
    state["fails"] += 1
    if state["fails"] >= _RATE_LIMIT_MAX_FAILS:
        state["locked_until"] = time.monotonic() + _RATE_LIMIT_LOCKOUT_SECONDS
        _audit_log.warning(
            "LOCKOUT email=%s fails=%d lockout=%ds",
            email_lower, state["fails"], _RATE_LIMIT_LOCKOUT_SECONDS,
        )


def _clear_rate_limit(email_lower: str) -> None:
    """Reset rate-limit counters on successful login."""
    _rate_limit.pop(email_lower, None)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_user_store() -> None:
    """
    Initialise the user store (SQLite database).

    Creates the database schema if needed, migrates from legacy CSV, and
    seeds a default admin account if no users exist.
    """
    _init_db()
    _migrate_csv_if_needed()

    with _LOCK, _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    if count == 0:
        initial_password = os.getenv("JACCS_DEFAULT_ADMIN_PASSWORD")
        if not initial_password:
            initial_password = secrets.token_urlsafe(16)
            print(
                "[Users] Generated a temporary admin password. "
                "Set JACCS_DEFAULT_ADMIN_PASSWORD to choose one explicitly."
            )
        add_user("admin@example.com", initial_password, role="admin")


def authenticate(email: str, password: str) -> Optional[str]:
    """
    Verify credentials and return the user's role on success, or None on failure.

    Parameters
    ----------
    email:
        User's email address.
    password:
        Plain-text password.

    Returns
    -------
    Optional[str]
        The user's role string (e.g. ``"admin"``, ``"user"``) on success,
        or ``None`` if authentication fails.

    Raises
    ------
    RuntimeError
        If the account is currently rate-limited / locked out.
    """
    email_lower = email.strip().lower()

    with _LOCK:
        _check_rate_limit(email_lower)

        with _get_conn() as conn:
            row = conn.execute(
                "SELECT role, salt, password_hash FROM users WHERE email = ?", (email_lower,)
            ).fetchone()

        if row is None:
            _audit_log.warning("AUTH_FAIL email=%s reason=unknown_user", email_lower)
            _record_failed_login(email_lower)
            return None

        role, salt, pw_hash = row["role"], row["salt"], row["password_hash"]

        # Try modern verifier first, then legacy SHA-256
        verified = _verify_password(password, salt, pw_hash)
        if not verified:
            # Attempt legacy SHA-256 (migrated accounts)
            verified = _verify_legacy_sha256(password, salt, pw_hash)
            if verified:
                # Upgrade hash to current scheme in-place
                try:
                    new_salt, new_hash = _hash_password(password)
                    with _get_conn() as conn:
                        conn.execute(
                            "UPDATE users SET salt=?, password_hash=? WHERE email=?",
                            (new_salt, new_hash, email_lower),
                        )
                        conn.commit()
                    _audit_log.info("HASH_UPGRADE email=%s", email_lower)
                except Exception as exc:
                    _audit_log.warning("HASH_UPGRADE_FAIL email=%s err=%s", email_lower, exc)

        if not verified:
            _audit_log.warning("AUTH_FAIL email=%s reason=bad_password", email_lower)
            _record_failed_login(email_lower)
            return None

        _clear_rate_limit(email_lower)
        _audit_log.info("AUTH_OK email=%s role=%s", email_lower, role)
        return role


def add_user(email: str, password: str, role: str = "user") -> None:
    """
    Create a new user account.

    Parameters
    ----------
    email:
        Must be a valid email address.
    password:
        Must be at least 8 characters long.
    role:
        One of ``"admin"``, ``"user"``, ``"viewer"``. Defaults to ``"user"``.

    Raises
    ------
    ValueError
        On invalid input or if the email already exists.
    """
    _validate_email(email)
    _validate_password(password)
    _validate_role(role)
    email_lower = email.strip().lower()

    with _LOCK:
        with _get_conn() as conn:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE email = ?", (email_lower,)
            ).fetchone()
            if existing:
                raise ValueError(f"User {email_lower!r} already exists.")
            salt, pw_hash = _hash_password(password)
            conn.execute(
                "INSERT INTO users (email, role, salt, password_hash) VALUES (?,?,?,?)",
                (email_lower, role, salt, pw_hash),
            )
            conn.commit()

    _audit_log.info("USER_ADD email=%s role=%s", email_lower, role)


def delete_user(email: str) -> None:
    """
    Remove a user account.

    Parameters
    ----------
    email:
        Email address of the account to delete.
    """
    email_lower = email.strip().lower()
    with _LOCK:
        with _get_conn() as conn:
            conn.execute("DELETE FROM users WHERE email = ?", (email_lower,))
            conn.commit()
    _audit_log.info("USER_DEL email=%s", email_lower)


def list_users() -> List[Dict[str, str]]:
    """
    Return a list of all users (email and role only – no credentials).

    Returns
    -------
    List[Dict[str, str]]
        Each dict has keys ``"email"`` and ``"role"``.
    """
    with _get_conn() as conn:
        rows = conn.execute("SELECT email, role FROM users ORDER BY email").fetchall()
    return [{"email": r["email"], "role": r["role"]} for r in rows]


def get_user(email: str) -> Optional[Dict[str, str]]:
    """
    Fetch a single user record (including salt and password_hash).

    Parameters
    ----------
    email:
        Email address to look up.

    Returns
    -------
    Optional[Dict[str, str]]
        Dict with keys ``email``, ``role``, ``salt``, ``password_hash``,
        or ``None`` if not found.
    """
    email_lower = email.strip().lower()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT email, role, salt, password_hash FROM users WHERE email = ?",
            (email_lower,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def set_role(email: str, role: str) -> None:
    """
    Change the role of an existing user.

    Parameters
    ----------
    email:
        Email address of the target user.
    role:
        New role. Must be one of ``"admin"``, ``"user"``, ``"viewer"``.

    Raises
    ------
    ValueError
        If the role is invalid or the user does not exist.
    """
    _validate_role(role)
    email_lower = email.strip().lower()
    with _LOCK:
        with _get_conn() as conn:
            result = conn.execute(
                "UPDATE users SET role = ? WHERE email = ?", (role, email_lower)
            )
            conn.commit()
        if result.rowcount == 0:
            raise ValueError(f"User {email_lower!r} not found.")
    _audit_log.info("ROLE_CHANGE email=%s new_role=%s", email_lower, role)


def update_password(email: str, new_password: str) -> None:
    """
    Set a new password for an existing user.

    Parameters
    ----------
    email:
        Email address of the target user.
    new_password:
        Must be at least 8 characters long.

    Raises
    ------
    ValueError
        If the password is too short or the user does not exist.
    """
    _validate_password(new_password)
    email_lower = email.strip().lower()
    salt, pw_hash = _hash_password(new_password)
    with _LOCK:
        with _get_conn() as conn:
            result = conn.execute(
                "UPDATE users SET salt = ?, password_hash = ? WHERE email = ?",
                (salt, pw_hash, email_lower),
            )
            conn.commit()
        if result.rowcount == 0:
            raise ValueError(f"User {email_lower!r} not found.")
    _audit_log.info("PASSWORD_CHANGE email=%s", email_lower)


# ---------------------------------------------------------------------------
# Dev / CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_user_store()
    print("Users:", list_users())
