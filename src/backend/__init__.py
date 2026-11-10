"""
JACCS Credit Risk – backend package.

Exposes the most commonly used symbols from the backend sub-modules so that
consumers can import directly from the package rather than from individual
modules.

Example
-------
>>> from src.backend import authenticate, get_connection, get_logger, test_connection
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Logger (import first so other modules can use it during their own import)
# ---------------------------------------------------------------------------
from src.backend.logger import (  # noqa: F401
    get_logger,
    log_model_event,
    log_prediction,
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
from src.backend.db import (  # noqa: F401
    fetch_ci_mock_crifv4,
    get_connection,
    get_connection_local,
    load_config,
    save_config,
    test_connection,
)

# ---------------------------------------------------------------------------
# Users / auth
# ---------------------------------------------------------------------------
from src.backend.users import (  # noqa: F401
    add_user,
    authenticate,
    delete_user,
    ensure_user_store,
    get_user,
    list_users,
    set_role,
    update_password,
)

__all__ = [
    # logger
    "get_logger",
    "log_model_event",
    "log_prediction",
    # db
    "fetch_ci_mock_crifv4",
    "get_connection",
    "get_connection_local",
    "load_config",
    "save_config",
    "test_connection",
    # users
    "add_user",
    "authenticate",
    "delete_user",
    "ensure_user_store",
    "get_user",
    "list_users",
    "set_role",
    "update_password",
]
