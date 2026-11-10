"""
Production-hardened database access module for the JACCS Credit Risk application.

Features:
- Thread-local connection caching (connection pooling per thread)
- Configurable query timeout (default 30 s)
- Automatic connection retry with exponential back-off (up to 3 attempts)
- test_connection(cfg) -> (bool, str) health-check function
- Clear error differentiation: connection failure / no data / query failure
- XOR encryption for sensitive config fields (no new dependencies)
- Environment-variable fallback for server address

Public API (unchanged from original):
    load_config() -> dict
    save_config(cfg: dict) -> None
    get_connection(...) -> pyodbc.Connection
    get_connection_local(...) -> pyodbc.Connection
    fetch_ci_mock_crifv4(party_no, application_no, conn=None) -> Optional[dict]

New additions:
    test_connection(cfg=None) -> Tuple[bool, str]
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import threading
import time
from base64 import b64decode, b64encode
from typing import Any, Dict, Optional, Tuple

import pyodbc

# ---------------------------------------------------------------------------
# Logging (import after logger.py so it is always available)
# ---------------------------------------------------------------------------
try:
    from src.backend.logger import get_logger as _get_logger
    _log = _get_logger(__name__)
except Exception:
    import logging
    _log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive fields encrypted in the config file
# ---------------------------------------------------------------------------

SENSITIVE_FIELDS = {"password", "user"}

# ---------------------------------------------------------------------------
# Config path resolution (identical to original logic)
# ---------------------------------------------------------------------------

def _get_config_path() -> str:
    """
    Resolve the path to db_config.json.

    - Frozen (PyInstaller): prefer bundled read-only template, then fall back
      to the user's home directory.
    - Development: src/backend/data/database/db_config.json.
    """
    try:
        if getattr(sys, "frozen", False):
            bundled_config = os.path.join(
                sys._MEIPASS, "database", "db_config.json"  # type: ignore[attr-defined]
            )
            if os.path.exists(bundled_config):
                return bundled_config
            app_dir = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app")
            config_dir = os.path.join(app_dir, "config")
            os.makedirs(config_dir, exist_ok=True)
            return os.path.join(config_dir, "db_config.json")
        config_dir = os.path.join(os.path.dirname(__file__), "data", "database")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "db_config.json")
    except Exception as exc:
        _log.warning("Failed to resolve config path (%s). Using temp location.", exc)
        import tempfile
        return os.path.join(tempfile.gettempdir(), "db_config.json")


CONFIG_PATH = _get_config_path()

DEFAULT_CONFIG: Dict[str, str] = {
    "server": "",
    "database": "",
    "user": "",
    "password": "",
    "driver": "{ODBC Driver 17 for SQL Server}",
}

# ---------------------------------------------------------------------------
# XOR encryption helpers (no additional dependencies)
# ---------------------------------------------------------------------------

def _derive_encryption_key(salt: str) -> str:
    """Derive a 32-character encryption key from *salt* using SHA-256."""
    return hashlib.sha256((salt + "DB_CONFIG_SECRET").encode("utf-8")).hexdigest()[:32]


def _simple_encrypt(value: str, salt: str) -> str:
    """Encrypt *value* with XOR using a key derived from *salt*."""
    key = _derive_encryption_key(salt)
    encrypted = bytes([ord(c) ^ ord(key[i % len(key)]) for i, c in enumerate(value)])
    return b64encode(encrypted).decode("utf-8")


def _simple_decrypt(encrypted_value: str, salt: str) -> str:
    """Decrypt *encrypted_value* with XOR using a key derived from *salt*."""
    try:
        key = _derive_encryption_key(salt)
        encrypted = b64decode(encrypted_value.encode("utf-8"))
        decrypted = bytes([b ^ ord(key[i % len(key)]) for i, b in enumerate(encrypted)])
        return decrypted.decode("utf-8")
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Config load / save
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, str]:
    """
    Load and return the database configuration from disk.

    Sensitive fields (``user``, ``password``) are transparently decrypted.
    Returns a copy of ``DEFAULT_CONFIG`` if the file is missing or corrupt.

    Returns
    -------
    Dict[str, str]
        Configuration dictionary with plain-text values.
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg: Dict[str, Any] = json.load(f)
            salt = cfg.get("_salt", "")
            if salt:
                for field in SENSITIVE_FIELDS:
                    if field in cfg and str(cfg[field]).startswith("ENC:"):
                        cfg[field] = _simple_decrypt(cfg[field][4:], salt)
                cfg.pop("_salt", None)
            return cfg
        except Exception as exc:
            _log.error("Failed to load config from %s: %s", CONFIG_PATH, exc)
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, str]) -> None:
    """
    Persist *cfg* to disk with sensitive fields XOR-encrypted.

    Parameters
    ----------
    cfg:
        Configuration dictionary (plain-text values).

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    try:
        cfg_to_save = dict(cfg)
        salt = secrets.token_hex(16)
        cfg_to_save["_salt"] = salt
        for field in SENSITIVE_FIELDS:
            if field in cfg_to_save:
                cfg_to_save[field] = "ENC:" + _simple_encrypt(cfg_to_save[field], salt)
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg_to_save, f, indent=4)
        _log.info("Config saved to: %s", CONFIG_PATH)
        print(f"Config saved to: {CONFIG_PATH}")
    except Exception as exc:
        _log.error("Failed to save config: %s", exc)
        raise

# ---------------------------------------------------------------------------
# ODBC driver resolution
# ---------------------------------------------------------------------------

def _resolve_driver(driver_name: Optional[str] = None) -> str:
    """
    Return the best available ODBC driver name.

    Preference order: ODBC Driver 18, 17, Native Client 11, SQL Server.
    Falls back to the first installed driver, or a safe default string.
    """
    try:
        installed = pyodbc.drivers()
    except Exception:
        installed = []

    preferred = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server",
    ]
    if driver_name:
        return driver_name
    for name in preferred:
        if name in installed:
            return name
    return installed[0] if installed else "ODBC Driver 17 for SQL Server"


def _server_candidates(server: Optional[str] = None):
    """
    Return an ordered, deduplicated list of server addresses to try.

    Priority:
    1. Explicitly provided *server* argument
    2. ``SQL_SERVER`` environment variable
    3. Any remaining env-var fallbacks (no hardcoded names)
    """
    raw = [
        server,
        os.getenv("SQL_SERVER"),
        os.getenv("SQL_SERVER_FALLBACK"),
    ]
    seen: set = set()
    out = []
    for item in raw:
        if not item:
            continue
        norm = item.strip()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out

# ---------------------------------------------------------------------------
# Thread-local connection cache
# ---------------------------------------------------------------------------

_tls = threading.local()


def _get_cached_connection(conn_str: str, timeout: int) -> pyodbc.Connection:
    """
    Return a cached pyodbc connection for the current thread, or create one.

    Connections are cached per (thread, conn_str) pair.  Stale / closed
    connections are detected and replaced automatically.
    """
    cache: Dict[str, pyodbc.Connection] = getattr(_tls, "conn_cache", None)
    if cache is None:
        _tls.conn_cache = {}
        cache = _tls.conn_cache

    conn = cache.get(conn_str)
    if conn is not None:
        try:
            conn.cursor().execute("SELECT 1")  # lightweight liveness check
            return conn
        except Exception:
            _log.debug("Cached connection stale – reconnecting.")
            try:
                conn.close()
            except Exception:
                pass
            del cache[conn_str]

    conn = pyodbc.connect(conn_str, timeout=timeout)
    cache[conn_str] = conn
    return conn

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _connect_with_retry(
    conn_str: str,
    timeout: int = 30,
    max_attempts: int = 3,
    use_cache: bool = True,
) -> pyodbc.Connection:
    """
    Attempt to open (or retrieve from cache) a pyodbc connection, with
    exponential back-off on failure.

    Parameters
    ----------
    conn_str:
        Full ODBC connection string.
    timeout:
        Per-attempt connection timeout in seconds.
    max_attempts:
        Maximum number of attempts before raising.
    use_cache:
        Whether to use the thread-local connection cache.

    Raises
    ------
    RuntimeError
        After all attempts are exhausted, with a descriptive message.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            if use_cache:
                return _get_cached_connection(conn_str, timeout)
            return pyodbc.connect(conn_str, timeout=timeout)
        except pyodbc.OperationalError as exc:
            last_exc = exc
            _log.warning("Connection attempt %d/%d failed (OperationalError): %s", attempt, max_attempts, exc)
        except pyodbc.Error as exc:
            last_exc = exc
            _log.warning("Connection attempt %d/%d failed (pyodbc.Error): %s", attempt, max_attempts, exc)
        except Exception as exc:
            last_exc = exc
            _log.warning("Connection attempt %d/%d failed (unexpected): %s", attempt, max_attempts, exc)

        if attempt < max_attempts:
            backoff = 2 ** (attempt - 1)  # 1 s, 2 s
            _log.info("Retrying in %d second(s)…", backoff)
            time.sleep(backoff)

    raise RuntimeError(
        f"Could not establish a database connection after {max_attempts} attempt(s). "
        f"Last error: {last_exc}"
    )

# ---------------------------------------------------------------------------
# Public connection functions
# ---------------------------------------------------------------------------

def get_connection_local(
    server: Optional[str] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    driver: Optional[str] = None,
    encrypt: str = "yes",
    trustservercertificate: str = "yes",
    timeout: int = 30,
) -> pyodbc.Connection:
    """
    Open a local / developer SQL Server connection, trying available server addresses.

    Parameters
    ----------
    server, database, user, password, driver:
        Connection parameters; ``None`` values fall through to environment
        variables or sensible defaults.
    encrypt:
        ODBC ``Encrypt`` option (default ``"yes"``).
    trustservercertificate:
        ODBC ``TrustServerCertificate`` option (default ``"yes"``).
    timeout:
        Per-attempt connection timeout in seconds (default 30).

    Returns
    -------
    pyodbc.Connection

    Raises
    ------
    RuntimeError
        If no candidate server can be reached.
    """
    database = database or "ltid"
    driver = _resolve_driver(driver)
    last_error: Optional[Exception] = None

    for srv in _server_candidates(server):
        auth = f"UID={user};PWD={password};" if user and password else "Trusted_Connection=yes;"
        conn_str = (
            f"DRIVER={{{driver}}};SERVER={srv};DATABASE={database};{auth}"
            f"Encrypt={encrypt};TrustServerCertificate={trustservercertificate};"
        )
        try:
            conn = _connect_with_retry(conn_str, timeout=timeout, use_cache=False)
            _log.info("Local connection established: server=%s db=%s", srv, database)
            return conn
        except Exception as exc:
            last_error = exc
            _log.debug("Local connection attempt failed for %s: %s", srv, exc)
            print(f"Local connection attempt failed for {srv}: {exc}")

    raise RuntimeError(f"Could not connect locally. Last error: {last_error}")


def get_connection(
    server: Optional[str] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    driver: Optional[str] = None,
    timeout: int = 30,
) -> pyodbc.Connection:
    """
    Open a production SQL Server connection using stored configuration.

    Parameters missing from the call are sourced from the saved config file.

    Parameters
    ----------
    server, database, user, password, driver:
        Optional overrides for config-file values.
    timeout:
        Connection timeout in seconds (default 30).

    Returns
    -------
    pyodbc.Connection

    Raises
    ------
    RuntimeError
        If required configuration values are missing, or if all connection
        attempts fail.
    """
    cfg = load_config()
    server   = server   or cfg.get("server")
    database = database or cfg.get("database")
    user     = user     or cfg.get("user")
    password = password or cfg.get("password")
    driver   = driver   or cfg.get("driver")

    missing = [
        name for name, value in (
            ("server",   server),
            ("database", database),
            ("user",     user),
            ("password", password),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Database configuration is incomplete. Set the missing values in "
            f"{CONFIG_PATH}: {', '.join(missing)}"
        )

    driver = _resolve_driver(driver)
    conn_str = (
        f"DRIVER={driver};SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};"
    )
    return _connect_with_retry(conn_str, timeout=timeout, use_cache=True)

# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

def test_connection(cfg: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """
    Test whether a database connection can be established.

    Parameters
    ----------
    cfg:
        Optional configuration dictionary.  When ``None``, the saved config
        file is used.

    Returns
    -------
    Tuple[bool, str]
        ``(True, "Connection successful")`` on success, or
        ``(False, "<descriptive error message>")`` on failure.
    """
    if cfg is None:
        cfg = load_config()

    server   = cfg.get("server")
    database = cfg.get("database")
    user     = cfg.get("user")
    password = cfg.get("password")
    driver   = _resolve_driver(cfg.get("driver"))

    missing = [
        name for name, value in (
            ("server",   server),
            ("database", database),
            ("user",     user),
            ("password", password),
        )
        if not value
    ]
    if missing:
        return False, f"Configuration incomplete – missing: {', '.join(missing)}"

    conn_str = (
        f"DRIVER={driver};SERVER={server};DATABASE={database};"
        f"UID={user};PWD={password};"
    )
    try:
        conn = pyodbc.connect(conn_str, timeout=10)
        conn.cursor().execute("SELECT 1")
        conn.close()
        _log.info("test_connection OK server=%s db=%s", server, database)
        return True, "Connection successful"
    except pyodbc.OperationalError as exc:
        msg = f"Connection failed (server unreachable or credentials rejected): {exc}"
        _log.warning("test_connection FAIL: %s", msg)
        return False, msg
    except pyodbc.Error as exc:
        msg = f"Connection failed (ODBC error): {exc}"
        _log.warning("test_connection FAIL: %s", msg)
        return False, msg
    except Exception as exc:
        msg = f"Connection failed (unexpected error): {exc}"
        _log.error("test_connection ERROR: %s", msg)
        return False, msg

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ci_mock_crifv4(
    party_no: int,
    application_no: int,
    conn: Optional[pyodbc.Connection] = None,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Execute ``dbo.CIMockCrifv4`` and return the first result row as a dict.

    Parameters
    ----------
    party_no:
        Customer party number.
    application_no:
        Loan application number.
    conn:
        Optional existing connection.  When ``None``, a connection is opened
        and closed automatically.
    timeout:
        Query timeout in seconds (default 30).

    Returns
    -------
    Optional[Dict[str, Any]]
        A ``{column: value}`` dict for the first result row, or ``None`` if
        the stored procedure returned no rows.

    Raises
    ------
    RuntimeError
        On connection failure or query error.
    """
    close_conn = False
    if conn is None:
        try:
            conn = get_connection(timeout=timeout)
        except RuntimeError as exc:
            raise RuntimeError(f"fetch_ci_mock_crifv4: connection failed – {exc}") from exc
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.timeout = timeout
            try:
                cur.execute(
                    "EXEC dbo.CIMockCrifv4 @party = ?, @application = ?",
                    party_no,
                    application_no,
                )
            except pyodbc.Error as exc:
                raise RuntimeError(
                    f"fetch_ci_mock_crifv4: query failed for party={party_no} "
                    f"app={application_no} – {exc}"
                ) from exc

            columns = [c[0] for c in cur.description] if cur.description else []
            row = cur.fetchone()

            if not row:
                _log.info(
                    "fetch_ci_mock_crifv4: no data for party=%s application=%s",
                    party_no, application_no,
                )
                return None

            return dict(zip(columns, row))
    finally:
        if close_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass
