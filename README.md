# JACCS Credit Scoring System (v2.0.0 Production Release)

Enterprise desktop credit risk assessment and credit scoring platform built with Python, Tkinter, LightGBM, and SHAP.

---

## 🏛️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                            │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────────┐  │
│  │  Splash View │   │  Authentication │   │  Admin Dashboard View    │  │
│  │ (splash.py)  │   │   (main.py)     │   │   (admin_dashboard.py)   │  │
│  └──────┬───────┘   └────────┬────────┘   └────────────┬─────────────┘  │
│         │                    │                         │                │
│  ┌──────┴────────────────────┴─────────────────────────┴─────────────┐  │
│  │                   Credit Scoring Application                      │  │
│  │       • Single Assessment & Inline Feature Editor                 │  │
│  │       • Batch Inference Engine (CSV/Excel Upload)                 │  │
│  │       • Visual Risk Gauge (risk_gauge.py)                         │  │
│  │       • Multi-sheet Detailed Export (export_manager.py)           │  │
│  │       • Interactive SHAP Feature Contribution Explorer            │  │
│  └───────────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       │
┌──────────────────────────────────────┴──────────────────────────────────┐
│                         CORE BACKEND & ML ENGINE                        │
│  ┌────────────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │    AIModel (LightGBM)  │  │ User Security  │  │ Database Layer    │  │
│  │ • Two-Stage Delinquency│  │ • SQLite Store │  │ • Connection Pool │  │
│  │ • Policy Hazard Layer  │  │ • bcrypt/PBKDF2│  │ • Auto-Retry &    │  │
│  │ • Calibrated Classifier│  │ • Rate Limiting│  │   Timeout Control │  │
│  │ • SHAP TreeExplainer   │  │ • Audit Log    │  │ • SQL Server /    │  │
│  │   (ai_model.py)        │  │   (users.py)   │  │   Offline Fallback│  │
│  └────────────────────────┘  └────────────────┘  │     (db.py)       │  │
│                                                  └───────────────────┘  │
│  ┌────────────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ Health Check Monitor   │  │ Central Logger │  │ App Configuration │  │
│  │  (health.py)           │  │  (logger.py)   │  │   (config.py)     │  │
│  └────────────────────────┘  └────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features & Production Enhancements

1. **Robust Dual-Mode Prediction Engine**:
   - Single applicant scoring with real-time UI normalization and interactive inline editing.
   - High-throughput batch assessment supporting both Excel (`.xlsx`, `.xls`) and CSV files with tokenization and encoding self-healing.
   - Delinquency severity estimation & DPD classification alongside raw calibrated default probabilities.

2. **Hardened Multi-Tenant Security & Authentication**:
   - SQLite user credentials store with concurrent write-safety via WAL journal mode.
   - Modern adaptive password hashing (`bcrypt` with fallback to `PBKDF2-HMAC-SHA256`).
   - Anti-brute-force rate limiting (3 failed attempts triggers automatic 30s lockout per account).
   - Tamper-evident rotating audit trail (`audit.log`).

3. **Enterprise Connectivity & Resiliency**:
   - Resilient database connection caching and active connection health probing.
   - Exponential backoff retry logic (up to 3 attempts with timeout handling).
   - Seamless offline fallback to bundled local datasets (`src/data/Extracted New.csv`) when SQL Server is unreachable.

4. **Explainable AI (XAI)**:
   - Integrated TreeSHAP breakdown showing positive/negative impact factors per borrower.
   - Comprehensive multi-tab Excel reports featuring summary scorecards and granular SHAP driver tabs.

5. **Operational Health & Diagnostics**:
   - Centralized environment configuration via `config.py` and `.env`.
   - Comprehensive health diagnostic suite (`health_check()`) verifying dataset integrity, model cache validity, and database status.

---

## 📋 Prerequisites

- **Python**: Version 3.10+ (Recommended: Python 3.12 64-bit)
- **ODBC Driver**: Microsoft ODBC Driver 17 or 18 for SQL Server (required only for live database connectivity)
- **Operating System**: Windows 10/11 or Windows Server 2019+

---

## 🚀 Quick Start Guide

### 1. Installation

Clone or open the repository in PowerShell and execute:

```powershell
# Using the production runner script
.\run.ps1 setup
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` or set environment variables in your active shell:

```powershell
# Set initial admin credentials (optional - randomly generated if omitted)
$env:JACCS_DEFAULT_ADMIN_PASSWORD = "YourSecurePassword123!"

# Configure database server (optional)
$env:SQL_SERVER = "YOUR_SERVER\SQLEXPRESS"

# Set logging verbosity (DEBUG, INFO, WARNING, ERROR)
$env:JACCS_LOG_LEVEL = "INFO"
```

### 3. Running the Application

```powershell
# Quick launcher
.\run.ps1 run

# Or via Python directly
python src\frontend\main.py
```

### Default Credentials
Upon first launch with clean storage:
- **Email**: `admin@example.com`
- **Password**: Set via `JACCS_DEFAULT_ADMIN_PASSWORD` (or view console output if not set)

---

## 🧪 Automated Test Suite

The system includes 80+ unit and integration tests covering data transformations, underwriting logic, database resiliency, security, and ML pipelines.

```powershell
# Run the full test suite
.\run.ps1 test

# Or run directly with pytest
pytest tests/ -v
```

---

## 📦 Building Standalone Executable (.exe)

Package the application as a zero-dependency Windows desktop executable using PyInstaller:

```powershell
.\run.ps1 build
```

The compiled application will be generated in `dist/JACCSScore.exe`.

---

## 📁 System Logs & Storage Paths

All persistent application data and logs are isolated in a user-safe directory:

- **Logs**: `%USERPROFILE%\.jaccs_credit_app\logs\` (or fallback to `%TEMP%\jaccs_credit_app\logs\`)
  - `app.log`: Application and operational events (10 MB rotating, 5 backups)
  - `audit.log`: Security, authentication, and user administration events
  - `predictions.log`: Audit trail of all scored borrowers and calculated credit decisions
- **User Store**: `src/backend/users.db` / `%USERPROFILE%\.jaccs_credit_app\users.db`
- **Database Config**: `src/data/database/db_config.json`

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Could not connect to database"** | SQL Server instance offline or ODBC driver missing | The app automatically falls back to local data. To connect to live DB, ensure ODBC Driver 17/18 is installed and verify connection in `Settings (⚙)`. |
| **"User account temporarily locked"** | 3 consecutive failed login attempts | Wait 30 seconds for the rate-limiting lockout period to expire before retrying. |
| **First run takes 3-5 minutes** | AI model is training and compiling LightGBM bundles | This occurs only once when training data changes. Subsequent launches load the precompiled `.joblib` cache in < 2 seconds. |
| **XLSX export fails** | Missing `openpyxl` dependency | Verify installation with `pip install openpyxl>=3.1.0`. |
| **Permission Denied on AppData** | Restricted corporate security sandbox | Set `$env:JACCS_APP_DIR = "C:\YourWritablePath"` before running. |
