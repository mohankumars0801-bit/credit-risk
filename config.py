"""
config.py – Centralized configuration for the JACCS Credit Scoring System.

Usage:
    from config import get_config
    cfg = get_config()
    print(cfg.app.version)
    print(cfg.model.training_data_paths)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """General application settings."""
    app_name: str = "JACCS Credit Scoring System"
    version: str = "2.0.0"
    window_title: str = "JACCS Credit Scoring System v2.0.0"
    window_size: str = "1280x800"
    min_window_size: tuple = (900, 600)
    icon_path: str = "ltid_logo(1).ico"
    debug: bool = False


@dataclass
class ModelConfig:
    """AI model and training data settings."""
    training_data_paths: List[str] = field(default_factory=lambda: [
        "src/data/May 12 Status Contracts.xlsx",
        "src/data/Extracted New.xlsx",
    ])
    model_version: int = 20
    model_cache_dir: str = "src/backend/models"
    # Decision threshold override (None = use model's own threshold)
    decision_threshold_override: Optional[float] = None
    # Minimum validation precision required during training
    min_validation_precision: float = 0.75


@dataclass
class DBConfig:
    """Database connection settings (populated from db_config.json)."""
    server: str = ""
    database: str = ""
    user: str = ""
    password: str = ""
    driver: str = "{ODBC Driver 17 for SQL Server}"

    @classmethod
    def from_dict(cls, data: dict) -> "DBConfig":
        return cls(
            server=data.get("server", ""),
            database=data.get("database", ""),
            user=data.get("user", ""),
            password=data.get("password", ""),
            driver=data.get("driver", "{ODBC Driver 17 for SQL Server}"),
        )

    def to_dict(self) -> dict:
        return {
            "server": self.server,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "driver": self.driver,
        }


@dataclass
class LogConfig:
    """Logging configuration."""
    log_dir: str = "logs"
    log_file: str = "jaccs_app.log"
    # Log level string; overridden by JACCS_LOG_LEVEL env var
    log_level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024   # 10 MB
    backup_count: int = 5
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class UIConfig:
    """Frontend / Tkinter styling settings."""
    # Colour palette
    primary_color: str = "#1a3c5e"       # dark navy
    secondary_color: str = "#2d7dd2"     # medium blue
    accent_color: str = "#f4a261"        # warm orange
    success_color: str = "#2ecc71"
    danger_color: str = "#e74c3c"
    warning_color: str = "#f39c12"
    background_color: str = "#f5f7fa"
    card_color: str = "#ffffff"
    text_primary: str = "#2c3e50"
    text_secondary: str = "#7f8c8d"
    text_on_primary: str = "#ffffff"

    # Fonts
    font_family: str = "Segoe UI"
    font_size_body: int = 10
    font_size_heading: int = 14
    font_size_title: int = 18
    font_size_small: int = 9

    # Widget geometry
    button_padding_x: int = 20
    button_padding_y: int = 8
    entry_width: int = 30
    card_padding: int = 16
    table_row_height: int = 28


# ---------------------------------------------------------------------------
# Root config container
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Singleton root configuration object."""
    app: AppConfig = field(default_factory=AppConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    db: DBConfig = field(default_factory=DBConfig)
    log: LogConfig = field(default_factory=LogConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root (works for dev and PyInstaller bundles)."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def _load_config_json(config_path: Path) -> dict:
    """Load an optional config.json override file, returning {} on any error."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not load %s: %s", config_path, exc
        )
        return {}


def _load_db_config_json() -> dict:
    """
    Load db_config.json from the standard backend location.
    Returns {} if the file is absent or unreadable.
    """
    root = _project_root()
    candidates = [
        root / "src" / "backend" / "data" / "database" / "db_config.json",
        Path(os.path.expanduser("~")) / ".jaccs_credit_app" / "config" / "db_config.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
    return {}


def _apply_env_overrides(cfg: Config) -> None:
    """
    Apply environment-variable overrides. Supported variables:

    App / general
    -------------
    JACCS_DEBUG               1 / true / yes  → app.debug = True

    Model
    -----
    JACCS_MODEL_CACHE_DIR     → model.model_cache_dir
    JACCS_MODEL_THRESHOLD     → model.decision_threshold_override (float)

    Database (read-only env fallbacks; db_config.json takes precedence via db.py)
    --------
    SQL_SERVER                → db.server  (if not already set from JSON)
    SQL_DATABASE              → db.database
    SQL_USER                  → db.user
    SQL_PASSWORD              → db.password
    SQL_DRIVER                → db.driver

    Logging
    -------
    JACCS_LOG_LEVEL           DEBUG / INFO / WARNING / ERROR / CRITICAL
    JACCS_LOG_DIR             → log.log_dir
    JACCS_LOG_FILE            → log.log_file
    """
    # --- App ---
    debug_env = os.getenv("JACCS_DEBUG", "").lower()
    if debug_env in ("1", "true", "yes"):
        cfg.app.debug = True

    # --- Model ---
    if val := os.getenv("JACCS_MODEL_CACHE_DIR"):
        cfg.model.model_cache_dir = val
    if val := os.getenv("JACCS_MODEL_THRESHOLD"):
        try:
            cfg.model.decision_threshold_override = float(val)
        except ValueError:
            pass

    # --- Database (env fallbacks, only if JSON didn't supply them) ---
    if not cfg.db.server and (val := os.getenv("SQL_SERVER")):
        cfg.db.server = val
    if not cfg.db.database and (val := os.getenv("SQL_DATABASE")):
        cfg.db.database = val
    if not cfg.db.user and (val := os.getenv("SQL_USER")):
        cfg.db.user = val
    if not cfg.db.password and (val := os.getenv("SQL_PASSWORD")):
        cfg.db.password = val
    if not cfg.db.driver and (val := os.getenv("SQL_DRIVER")):
        cfg.db.driver = val

    # --- Logging ---
    if val := os.getenv("JACCS_LOG_LEVEL"):
        cfg.log.log_level = val.upper()
    if val := os.getenv("JACCS_LOG_DIR"):
        cfg.log.log_dir = val
    if val := os.getenv("JACCS_LOG_FILE"):
        cfg.log.log_file = val


def _apply_json_overrides(cfg: Config, overrides: dict) -> None:
    """
    Apply nested dict from config.json into the Config object.
    Keys must match the dataclass section names (app, model, db, log, ui).
    """
    section_map = {
        "app": cfg.app,
        "model": cfg.model,
        "db": cfg.db,
        "log": cfg.log,
        "ui": cfg.ui,
    }
    for section_name, section_obj in section_map.items():
        section_data = overrides.get(section_name, {})
        for key, value in section_data.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Return the application-wide Config singleton.

    Load priority (each layer overrides the previous):
    1. Dataclass defaults (hard-coded sensible values)
    2. db_config.json  → db section
    3. config.json (optional, next to this file) → any section
    4. Environment variables

    Args:
        reload: Force a fresh load even if the singleton already exists.
                Useful in tests that modify environment variables.

    Returns:
        The populated Config singleton.
    """
    global _instance
    if _instance is not None and not reload:
        return _instance

    cfg = Config()

    # 2. Load db_config.json into cfg.db
    db_data = _load_db_config_json()
    if db_data:
        # Strip internal fields before mapping to DBConfig
        clean = {k: v for k, v in db_data.items() if not k.startswith("_")}
        cfg.db = DBConfig.from_dict(clean)

    # 3. Load optional config.json override
    root = _project_root()
    json_overrides = _load_config_json(root / "config.json")
    if json_overrides:
        _apply_json_overrides(cfg, json_overrides)

    # 4. Environment variables (highest priority)
    _apply_env_overrides(cfg)

    _instance = cfg
    return _instance


def configure_logging(cfg: Optional[Config] = None) -> None:
    """
    Configure the root logger using the LogConfig section.
    Call once at application start-up.
    """
    import logging.handlers

    if cfg is None:
        cfg = get_config()

    log_cfg = cfg.log
    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / log_cfg.log_file
    level = getattr(logging, log_cfg.log_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers on reload
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=log_cfg.log_format,
        datefmt=log_cfg.date_format,
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # Rotating file handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except OSError as exc:
        root_logger.warning("Could not open log file %s: %s", log_file, exc)
