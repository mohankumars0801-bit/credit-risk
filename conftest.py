"""
Global pytest configuration and shared fixtures for the credit-risk project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator
import tempfile
import pytest

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_customer_row() -> Dict[str, Any]:
    """Return a representative dictionary matching a raw customer row."""
    return {
        "Party No": 201614,
        "Application No": 2,
        "Amount Finance": 450000.0,
        "Downpayment": 90000.0,
        "Downpayment %": 16.67,
        "Payment Terms": 48.0,
        "Asset": "Toyota",
        "Manufacturer": "Toyota Motor Corp",
        "Type of Unit": "Sedan",
        "Price Gap": 0.0,
        "Loan History": "Fully paid Good Repeat Client",
        "BAP Type": "Clean",
        "BAP Amount": 0.0,
        "Interest Rate": 9.5,
        "Claimed Income": 65000.0,
        "EV Income": 62000.0,
        "BV Income": 60000.0,
        "Employment Type": "Locally Employed",
        "Claimed Tenure": 48.0,
        "EV Tenure": 48.0,
        "BV Tenure": 48.0,
        "DSCR": 28.5,
        "Age as of Contract": 35.0,
        "Home Ownership": "Owned",
        "Residence Tenure": 60.0,
        "Citizenship": "Filipino",
        "Dealer Name": "Toyota Makati",
        "Submitted ID": "Passport",
        "Marital Status": "Married",
        "Loan Amort": 12500.0,
        "Product Type": "Auto Loan",
        "Mode of Payment": "ADA - BDO",
        "Co-Borrowers": "No",
        "Borrower Type": "Principal",
        "Contract Start Date": "2024-03-15",
    }


@pytest.fixture
def high_risk_customer_row() -> Dict[str, Any]:
    """Return a customer row with severe distress factors."""
    return {
        "Party No": 999999,
        "Application No": 1,
        "Amount Finance": 800000.0,
        "Downpayment": 0.0,
        "Downpayment %": 0.0,
        "Payment Terms": 60.0,
        "Asset": "Personal Loan",
        "Manufacturer": "Unknown",
        "Type of Unit": "None",
        "Price Gap": -100000.0,
        "Loan History": "With history of Pastdue",
        "BAP Type": "Adverse Loans",
        "BAP Amount": 350000.0,
        "Interest Rate": 24.0,
        "Claimed Income": 25000.0,
        "EV Income": None,
        "BV Income": None,
        "Employment Type": "Self-Employed",
        "Claimed Tenure": 6.0,
        "EV Tenure": None,
        "BV Tenure": None,
        "DSCR": 75.0,
        "Age as of Contract": 22.0,
        "Home Ownership": "Renting",
        "Residence Tenure": 3.0,
        "Citizenship": "Filipino",
        "Dealer Name": "Unknown",
        "Submitted ID": "Voter ID",
        "Marital Status": "Single/Not Married",
        "Loan Amort": 22000.0,
        "Product Type": "Personal Loan",
        "Mode of Payment": "OTC - Bank",
        "Co-Borrowers": "No",
        "Borrower Type": "Principal",
        "Contract Start Date": "2024-01-01",
    }


@pytest.fixture
def tmp_users_db(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary SQLite database for user tests."""
    db_path = tmp_path / "test_users.db"
    orig_env = os.environ.get("JACCS_USER_DB_PATH")
    os.environ["JACCS_USER_DB_PATH"] = str(db_path)
    yield db_path
    if orig_env is not None:
        os.environ["JACCS_USER_DB_PATH"] = orig_env
    else:
        os.environ.pop("JACCS_USER_DB_PATH", None)


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for configuration files."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    yield cfg_dir

