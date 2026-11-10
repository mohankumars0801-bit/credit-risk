"""
tests/test_backend.py
=====================
Comprehensive unit tests for the JACCS Credit Scoring System backend modules:
  - src/backend/users.py
  - src/backend/db.py
  - src/backend/ai_model.py  (pure-function / static-method tests only)

Run with:
    pytest tests/test_backend.py -v
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup – make src importable without installing the package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ===========================================================================
# Helpers
# ===========================================================================

def _write_users_csv(path: Path, rows: list[dict]) -> None:
    """Write a users.csv with the standard schema."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["email", "role", "salt", "password_hash"])
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# TestUsers – tests for src/backend/users.py
# ===========================================================================

class TestUsers:
    """Unit tests for the user management module (users.py)."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(autouse=True)
    def _patch_users_csv(self, tmp_path):
        """Redirect the users DB and CSV paths to a temporary directory."""
        self.csv_path = tmp_path / "users.csv"
        self.db_path = tmp_path / "users.db"
        import src.backend.users as users_mod
        self._users_mod = users_mod
        users_mod._rate_limit.clear()
        with patch.object(users_mod, "USERS_CSV", str(self.csv_path)), \
             patch.object(users_mod, "_DB_PATH", str(self.db_path)):
            users_mod._init_db()
            yield

    # ------------------------------------------------------------------
    # Password hashing
    # ------------------------------------------------------------------

    def test_hash_password_produces_salt_and_hash(self):
        """_hash_password with no salt generates a valid salt and hash."""
        salt, hashed = self._users_mod._hash_password("SuperSecret123")
        assert isinstance(salt, str) and len(salt) > 0
        assert isinstance(hashed, str) and len(hashed) > 0

    def test_hash_password_deterministic_with_given_salt(self):
        """_verify_password successfully verifies the generated hash."""
        salt, hashed = self._users_mod._hash_password("SuperSecret123")
        assert self._users_mod._verify_password("SuperSecret123", salt, hashed)
        assert not self._users_mod._verify_password("WrongPassword123", salt, hashed)

    def test_hash_password_different_passwords_differ(self):
        """Different passwords produce different hashes."""
        _, h1 = self._users_mod._hash_password("Password_A_123")
        _, h2 = self._users_mod._hash_password("Password_B_123")
        assert h1 != h2

    # ------------------------------------------------------------------
    # add_user / get_user / list_users
    # ------------------------------------------------------------------

    def test_add_user_creates_entry(self):
        """add_user writes a new record that get_user can retrieve."""
        self._users_mod.add_user("alice@example.com", "hunter2Password", role="user")
        record = self._users_mod.get_user("alice@example.com")
        assert record is not None
        assert record["email"] == "alice@example.com"
        assert record["role"] == "user"

    def test_add_user_stores_hashed_password(self):
        """Stored password_hash is not the plaintext password."""
        self._users_mod.add_user("bob@example.com", "plain_text_123")
        record = self._users_mod.get_user("bob@example.com")
        assert record["password_hash"] != "plain_text_123"

    def test_add_user_duplicate_raises(self):
        """Adding a user with an already-registered email raises ValueError."""
        self._users_mod.add_user("dup@example.com", "password123")
        with pytest.raises(ValueError, match="already exists"):
            self._users_mod.add_user("dup@example.com", "password456")

    def test_add_user_case_insensitive_duplicate(self):
        """E-mail uniqueness check is case-insensitive."""
        self._users_mod.add_user("Test@Example.com", "password123")
        with pytest.raises(ValueError):
            self._users_mod.add_user("test@example.com", "password456")

    def test_list_users_excludes_credentials(self):
        """list_users must not expose salt or password_hash."""
        self._users_mod.add_user("carol@example.com", "SecretPass123", role="admin")
        users = self._users_mod.list_users()
        assert len(users) == 1
        assert "salt" not in users[0]
        assert "password_hash" not in users[0]
        assert users[0]["role"] == "admin"

    def test_get_user_returns_none_for_missing(self):
        """get_user returns None when no matching record exists."""
        result = self._users_mod.get_user("nobody@example.com")
        assert result is None

    # ------------------------------------------------------------------
    # authenticate
    # ------------------------------------------------------------------

    def test_authenticate_success_returns_role(self):
        """authenticate returns the user's role for valid credentials."""
        self._users_mod.add_user("dave@example.com", "correct_pass_123", role="admin")
        role = self._users_mod.authenticate("dave@example.com", "correct_pass_123")
        assert role == "admin"

    def test_authenticate_wrong_password_returns_none(self):
        """authenticate returns None for an incorrect password."""
        self._users_mod.add_user("eve@example.com", "right_pass_123")
        role = self._users_mod.authenticate("eve@example.com", "wrong_pass_123")
        assert role is None

    def test_authenticate_unknown_email_returns_none(self):
        """authenticate returns None for an email not in the store."""
        result = self._users_mod.authenticate("ghost@example.com", "any_password_123")
        assert result is None

    def test_authenticate_case_insensitive_email(self):
        """authenticate is case-insensitive on the email field."""
        self._users_mod.add_user("Frank@Example.com", "password123", role="user")
        role = self._users_mod.authenticate("frank@example.com", "password123")
        assert role == "user"

    # ------------------------------------------------------------------
    # delete_user
    # ------------------------------------------------------------------

    def test_delete_user_removes_entry(self):
        """delete_user removes the named user from the store."""
        self._users_mod.add_user("grace@example.com", "password123")
        self._users_mod.delete_user("grace@example.com")
        assert self._users_mod.get_user("grace@example.com") is None

    def test_delete_user_nonexistent_does_not_raise(self):
        """delete_user on a non-existent email silently succeeds."""
        self._users_mod.delete_user("nobody@example.com")  # Should not raise

    def test_delete_user_only_removes_target(self):
        """delete_user leaves other users intact."""
        self._users_mod.add_user("h@example.com", "password123")
        self._users_mod.add_user("i@example.com", "password456")
        self._users_mod.delete_user("h@example.com")
        assert self._users_mod.get_user("i@example.com") is not None

    # ------------------------------------------------------------------
    # update_password / set_role
    # ------------------------------------------------------------------

    def test_update_password_changes_credentials(self):
        """After update_password old credentials fail and new ones succeed."""
        self._users_mod.add_user("j@example.com", "old_password123")
        self._users_mod.update_password("j@example.com", "new_password123")
        assert self._users_mod.authenticate("j@example.com", "old_password123") is None
        assert self._users_mod.authenticate("j@example.com", "new_password123") is not None

    def test_set_role_updates_role(self):
        """set_role changes the role field for an existing user."""
        self._users_mod.add_user("k@example.com", "password123", role="user")
        self._users_mod.set_role("k@example.com", "admin")
        record = self._users_mod.get_user("k@example.com")
        assert record["role"] == "admin"

    def test_set_role_missing_user_raises(self):
        """set_role raises ValueError if the user does not exist."""
        with pytest.raises(ValueError, match="not found"):
            self._users_mod.set_role("nobody@example.com", "admin")

    # ------------------------------------------------------------------
    # ensure_user_store
    # ------------------------------------------------------------------

    def test_ensure_user_store_creates_admin(self):
        """ensure_user_store creates the default admin when missing."""
        with patch.dict(os.environ, {"JACCS_DEFAULT_ADMIN_PASSWORD": "admin_password_123"}):
            self._users_mod.ensure_user_store()
        users = self._users_mod.list_users()
        emails = [u["email"] for u in users]
        assert "admin@example.com" in emails

    def test_ensure_user_store_idempotent(self):
        """Calling ensure_user_store twice does not create a duplicate admin."""
        with patch.dict(os.environ, {"JACCS_DEFAULT_ADMIN_PASSWORD": "admin_password_123"}):
            self._users_mod.ensure_user_store()
            self._users_mod.ensure_user_store()
        users = self._users_mod.list_users()
        admin_entries = [u for u in users if u["email"] == "admin@example.com"]
        assert len(admin_entries) == 1


# ===========================================================================
# TestDB – tests for src/backend/db.py
# ===========================================================================

class TestDB:
    """Unit tests for the database configuration module (db.py)."""

    @pytest.fixture(autouse=True)
    def _patch_config_path(self, tmp_path):
        """Redirect CONFIG_PATH to a temp file so tests are side-effect-free."""
        self.config_file = tmp_path / "db_config.json"
        import src.backend.db as db_mod
        self._db_mod = db_mod
        with patch.object(db_mod, "CONFIG_PATH", str(self.config_file)):
            yield

    # ------------------------------------------------------------------
    # load_config
    # ------------------------------------------------------------------

    def test_load_config_returns_defaults_when_missing(self):
        """load_config returns DEFAULT_CONFIG when db_config.json is absent."""
        result = self._db_mod.load_config()
        assert result["server"] == ""
        assert result["database"] == ""

    def test_load_config_reads_saved_values(self):
        """load_config decrypts and returns values written by save_config."""
        cfg = {
            "server": "myserver",
            "database": "mydb",
            "user": "admin",
            "password": "s3cr3t",
            "driver": "{ODBC Driver 17 for SQL Server}",
        }
        self._db_mod.save_config(cfg)
        loaded = self._db_mod.load_config()
        assert loaded["server"] == "myserver"
        assert loaded["database"] == "mydb"
        assert loaded["user"] == "admin"
        assert loaded["password"] == "s3cr3t"

    def test_load_config_handles_corrupt_file(self):
        """load_config returns defaults when the JSON file is corrupt."""
        self.config_file.write_text("not valid json", encoding="utf-8")
        result = self._db_mod.load_config()
        assert result == self._db_mod.DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # save_config
    # ------------------------------------------------------------------

    def test_save_config_writes_file(self):
        """save_config creates a JSON file at CONFIG_PATH."""
        cfg = {"server": "srv", "database": "db", "user": "u", "password": "p", "driver": "drv"}
        self._db_mod.save_config(cfg)
        assert self.config_file.exists()

    def test_save_config_encrypts_sensitive_fields(self):
        """Sensitive fields in the JSON on disk should not be plaintext."""
        cfg = {"server": "s", "database": "d", "user": "myuser", "password": "mypassword", "driver": "drv"}
        self._db_mod.save_config(cfg)
        raw = json.loads(self.config_file.read_text(encoding="utf-8"))
        assert raw["password"] != "mypassword"
        assert raw["user"] != "myuser"
        assert raw["password"].startswith("ENC:")
        assert raw["user"].startswith("ENC:")

    def test_save_and_load_roundtrip(self):
        """Values survive a save → load roundtrip unchanged."""
        original = {
            "server": r"DESKTOP\SQLEXPRESS",
            "database": "credit_db",
            "user": "sa",
            "password": "P@$$w0rd!",
            "driver": "{ODBC Driver 17 for SQL Server}",
        }
        self._db_mod.save_config(original)
        loaded = self._db_mod.load_config()
        for key in ("server", "database", "user", "password", "driver"):
            assert loaded[key] == original[key], f"Mismatch on key '{key}'"

    # ------------------------------------------------------------------
    # test_connection (mocked pyodbc)
    # ------------------------------------------------------------------

    def test_get_connection_raises_when_config_incomplete(self):
        """get_connection raises RuntimeError when required fields are empty."""
        with pytest.raises(RuntimeError, match="incomplete"):
            self._db_mod.get_connection()

    def test_get_connection_calls_pyodbc_connect(self):
        """get_connection calls pyodbc.connect with a correctly-formed DSN."""
        cfg = {
            "server": "myhost",
            "database": "mydb",
            "user": "usr",
            "password": "pw",
            "driver": "{ODBC Driver 17 for SQL Server}",
        }
        self._db_mod.save_config(cfg)
        mock_conn = MagicMock()
        with patch("src.backend.db.pyodbc.connect", return_value=mock_conn) as mock_connect:
            conn = self._db_mod.get_connection()
            assert conn is mock_conn
            call_args = mock_connect.call_args[0][0]
            assert "myhost" in call_args
            assert "mydb" in call_args

    def test_derive_encryption_key_deterministic(self):
        """_derive_encryption_key returns the same key for identical inputs."""
        key1 = self._db_mod._derive_encryption_key("some_salt")
        key2 = self._db_mod._derive_encryption_key("some_salt")
        assert key1 == key2 and len(key1) == 32

    def test_encrypt_decrypt_roundtrip(self):
        """XOR encrypt/decrypt functions are symmetric."""
        salt = "cafecafe12345678"
        plaintext = "super_secret_value_123!"
        encrypted = self._db_mod._simple_encrypt(plaintext, salt)
        decrypted = self._db_mod._simple_decrypt(encrypted, salt)
        assert decrypted == plaintext

    def test_decrypt_bad_input_returns_empty(self):
        """_simple_decrypt returns '' for invalid base64 / corrupt data."""
        result = self._db_mod._simple_decrypt("!!!not_base64!!!", "some_salt")
        assert result == ""


# ===========================================================================
# TestAIModelUtils – pure-function tests for src/backend/ai_model.py
# ===========================================================================

class TestAIModelUtils:
    """Unit tests for the stateless helper functions in ai_model.py."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        import src.backend.ai_model as ai_mod
        self._ai = ai_mod

    # ------------------------------------------------------------------
    # resource_path
    # ------------------------------------------------------------------

    def test_resource_path_relative_resolves_to_absolute(self):
        """resource_path converts a relative path to an absolute path."""
        result = self._ai.resource_path("src/data/some_file.xlsx")
        assert os.path.isabs(result)

    def test_resource_path_absolute_passthrough(self):
        """resource_path returns absolute paths unchanged."""
        abs_path = "/tmp/test_file.txt"
        result = self._ai.resource_path(abs_path)
        assert result == abs_path

    def test_resource_path_empty_passthrough(self):
        """resource_path returns an empty string unchanged."""
        assert self._ai.resource_path("") == ""

    def test_resource_path_normalizes_backslashes(self):
        """resource_path converts Windows-style backslashes."""
        result = self._ai.resource_path("src\\data\\some_file.xlsx")
        # Result should be an absolute path, not a relative Windows path
        assert os.path.isabs(result)

    # ------------------------------------------------------------------
    # _clean_income
    # ------------------------------------------------------------------

    def test_clean_income_plain_number(self):
        """_clean_income parses a plain numeric string."""
        assert self._ai.AIModel._clean_income("50000") == pytest.approx(50000.0)

    def test_clean_income_with_commas(self):
        """_clean_income strips comma thousands-separators."""
        assert self._ai.AIModel._clean_income("1,200,000") == pytest.approx(1200000.0)

    def test_clean_income_k_suffix(self):
        """_clean_income expands 'k' suffix to thousands."""
        assert self._ai.AIModel._clean_income("30k") == pytest.approx(30000.0)

    def test_clean_income_annual_divides_by_12(self):
        """_clean_income divides by 12 when 'annual' appears in the text."""
        result = self._ai.AIModel._clean_income("120000 annual")
        assert result == pytest.approx(10000.0)

    def test_clean_income_range_averages_values(self):
        """_clean_income averages the two endpoints of a range."""
        result = self._ai.AIModel._clean_income("20000-40000")
        assert result == pytest.approx(30000.0)

    def test_clean_income_confidential_returns_nan(self):
        """_clean_income returns NaN for 'confidential' and similar strings."""
        assert np.isnan(self._ai.AIModel._clean_income("Confidential"))
        assert np.isnan(self._ai.AIModel._clean_income("not disclosed"))

    def test_clean_income_nan_input_returns_nan(self):
        """_clean_income returns NaN for a NaN input."""
        assert np.isnan(self._ai.AIModel._clean_income(np.nan))

    def test_clean_income_empty_string_returns_nan(self):
        """_clean_income returns NaN for an unreadable/empty string."""
        assert np.isnan(self._ai.AIModel._clean_income("no numbers here"))

    # ------------------------------------------------------------------
    # _clean_tenure
    # ------------------------------------------------------------------

    def test_clean_tenure_plain_integer(self):
        """_clean_tenure parses a plain integer string."""
        assert self._ai.AIModel._clean_tenure("24") == pytest.approx(24.0)

    def test_clean_tenure_plain_float(self):
        """_clean_tenure parses a decimal tenure."""
        assert self._ai.AIModel._clean_tenure("12.5") == pytest.approx(12.5)

    def test_clean_tenure_years_to_months(self):
        """_clean_tenure converts years to months."""
        assert self._ai.AIModel._clean_tenure("2 years") == pytest.approx(24.0)

    def test_clean_tenure_years_and_months(self):
        """_clean_tenure combines years and months components."""
        assert self._ai.AIModel._clean_tenure("1 year 6 months") == pytest.approx(18.0)

    def test_clean_tenure_months_only(self):
        """_clean_tenure handles explicit month specification."""
        assert self._ai.AIModel._clean_tenure("18 months") == pytest.approx(18.0)

    def test_clean_tenure_with_question_marks(self):
        """_clean_tenure strips question-marks before parsing."""
        assert self._ai.AIModel._clean_tenure("12?") == pytest.approx(12.0)

    def test_clean_tenure_nan_input_returns_nan(self):
        """_clean_tenure returns NaN for NaN input."""
        assert np.isnan(self._ai.AIModel._clean_tenure(np.nan))

    def test_clean_tenure_unreadable_returns_nan(self):
        """_clean_tenure returns NaN when no numbers are found."""
        assert np.isnan(self._ai.AIModel._clean_tenure("unknown"))

    # ------------------------------------------------------------------
    # _excel_sha1 / _combined_excel_sha1
    # ------------------------------------------------------------------

    def test_excel_sha1_stable(self, tmp_path):
        """_excel_sha1 produces the same hash for the same file content."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        h1 = self._ai._excel_sha1(str(f))
        h2 = self._ai._excel_sha1(str(f))
        assert h1 == h2 and len(h1) == 40  # SHA1 hex length

    def test_excel_sha1_differs_for_different_content(self, tmp_path):
        """_excel_sha1 produces different hashes for different file content."""
        f1, f2 = tmp_path / "a.bin", tmp_path / "b.bin"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        assert self._ai._excel_sha1(str(f1)) != self._ai._excel_sha1(str(f2))

    def test_combined_sha1_order_sensitive(self, tmp_path):
        """_combined_excel_sha1 differs when file order differs."""
        f1, f2 = tmp_path / "file1.bin", tmp_path / "file2.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        h_ab = self._ai._combined_excel_sha1([str(f1), str(f2)])
        h_ba = self._ai._combined_excel_sha1([str(f2), str(f1)])
        assert h_ab != h_ba


# ===========================================================================
# TestProcessData – tests for AIModel.process_data static pipeline
# ===========================================================================

class TestProcessData:
    """Tests for the data-processing pipeline in AIModel.process_data."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        import src.backend.ai_model as ai_mod
        self.AIModel = ai_mod.AIModel

    def _make_df(self, **extra):
        """Create a minimal DataFrame suitable for process_data."""
        base = {
            "Amount Finance": 300000,
            "Downpayment": 60000,
            "Downpayment %": 20,
            "Payment Terms": 36,
            "Asset": "Motorcycle",
            "Manufacturer": "Honda",
            "Type of Unit": "New",
            "Price Gap": -10000,
            "Loan History": "Fully paid Good Repeat Client",
            "BAP Type": "No BAP",
            "BAP Amount": np.nan,
            "Interest Rate": 0.18,
            "Claimed Income": 50000,
            "EV Income": np.nan,
            "BV Income": 48000,
            "Employment Type": "Locally Employed",
            "Claimed Tenure": 24,
            "EV Tenure": np.nan,
            "BV Tenure": 24,
            "DSCR": 1.2,
            "Age as of Contract": 30,
            "Home Ownership": "Owned",
            "Residence Tenure": 36,
            "Citizenship": "Filipino",
            "Dealer Name": "ABC Motors",
            "Submitted ID": "ID1",
            "Marital Status": "Single",
            "Loan Amort": 9500,
            "Product Type": "Auto Loan",
            "Mode of Payment": "ADA - BPI",
            "Co-Borrowers": "No",
            "Borrower Type": "Principal",
            "Contract Start Date": "01/15/2024",
            "Current DPD": 0,
        }
        base.update(extra)
        return pd.DataFrame([base])

    def test_process_data_adds_dti(self):
        """process_data computes a DTI column."""
        df = self._make_df()
        result = self.AIModel.process_data(df)
        assert "DTI" in result.columns

    def test_process_data_adds_ltv(self):
        """process_data computes an LTV column."""
        df = self._make_df()
        result = self.AIModel.process_data(df)
        assert "LTV" in result.columns

    def test_process_data_adds_target(self):
        """process_data creates the binary Target column from Current DPD."""
        df_no_dpd = self._make_df(**{"Current DPD": 0})
        df_dpd = self._make_df(**{"Current DPD": 10})
        result_no = self.AIModel.process_data(df_no_dpd)
        result_yes = self.AIModel.process_data(df_dpd)
        assert result_no["Target"].iloc[0] == 0
        assert result_yes["Target"].iloc[0] == 1

    def test_process_data_no_inf_values(self):
        """process_data produces no infinite values."""
        df = self._make_df()
        result = self.AIModel.process_data(df)
        numeric = result.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()

    def test_process_data_contract_date_is_dropped(self):
        """process_data drops Contract Start Date and replaces with year/month."""
        df = self._make_df()
        result = self.AIModel.process_data(df)
        assert "Contract Start Date" not in result.columns
        assert "Contract_Year" in result.columns
        assert "Contract_Month" in result.columns

    def test_process_data_inference_mode_keeps_all_marital_statuses(self):
        """In inference mode, unusual Marital Status values are retained."""
        df = self._make_df(**{"Marital Status": "Partner"})
        result = self.AIModel.process_data(df, inference=True)
        assert len(result) == 1

    def test_process_data_training_mode_filters_unusual_marital_status(self):
        """In training mode, 'Partner' and 'Unspecified' rows are dropped."""
        df = self._make_df(**{"Marital Status": "Partner"})
        result = self.AIModel.process_data(df, inference=False)
        assert len(result) == 0
