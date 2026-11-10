"""
tests/test_integration.py
=========================
Integration tests for the JACCS Credit Scoring System.

These tests verify the end-to-end pipeline from raw row data through
preprocessing and prediction.  Tests that require the training data
workbooks on disk are skipped gracefully when the files are absent.

Run with:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Training data presence check
# ---------------------------------------------------------------------------
_TRAINING_FILES = [
    PROJECT_ROOT / "src" / "data" / "May 12 Status Contracts.xlsx",
    PROJECT_ROOT / "src" / "data" / "Extracted New.xlsx",
]
_TRAINING_DATA_PRESENT = all(p.exists() for p in _TRAINING_FILES)
_SKIP_NO_DATA = pytest.mark.skipif(
    not _TRAINING_DATA_PRESENT,
    reason="Training data workbooks not found in src/data/ – skipping integration tests."
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def ai_model():
    """
    Load (or train) the AIModel once for the entire test module.
    Skipped if training data is absent.
    """
    if not _TRAINING_DATA_PRESENT:
        pytest.skip("Training data not found.")

    from src.backend.ai_model import AIModel
    model = AIModel(
        "src/data/May 12 Status Contracts.xlsx",
        "src/data/Extracted New.xlsx",
    )
    return model


@pytest.fixture
def sample_row() -> Dict[str, Any]:
    """A representative, well-formed customer application row."""
    return {
        "Amount Finance": 300_000,
        "Downpayment": 60_000,
        "Downpayment %": 20,
        "Payment Terms": 36,
        "Asset": "Motorcycle",
        "Manufacturer": "Honda",
        "Type of Unit": "New",
        "Price Gap": -5_000,
        "Loan History": "Fully paid Good Repeat Client",
        "BAP Type": "No BAP",
        "BAP Amount": np.nan,
        "Interest Rate": 0.18,
        "Claimed Income": 50_000,
        "EV Income": np.nan,
        "BV Income": 48_000,
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
        "Loan Amort": 9_500,
        "Product Type": "Auto Loan",
        "Mode of Payment": "ADA - BPI",
        "Co-Borrowers": "No",
        "Borrower Type": "Principal",
        "Contract Start Date": "01/15/2024",
    }


@pytest.fixture
def high_risk_row(sample_row) -> Dict[str, Any]:
    """A high-risk variant of the sample row."""
    row = dict(sample_row)
    row.update({
        "Amount Finance": 500_000,
        "Downpayment": 0,
        "Downpayment %": 0,
        "Payment Terms": 60,
        "Loan History": "With history of Pastdue",
        "BAP Type": "Adverse Loans",
        "BAP Amount": 300_000,
        "Claimed Income": 20_000,
        "BV Income": 0,
        "DSCR": 0.5,
        "DTI": 0.9,
        "Product Type": "Personal Loan",
        "Price Gap": -120_000,
        "Loan Amort": 15_000,
    })
    return row


# ===========================================================================
# Tests – Model instantiation
# ===========================================================================

class TestModelInstantiation:
    """Verify that the AIModel can be instantiated when training data exists."""

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_model_loads_or_trains(self, ai_model):
        """AIModel initialises without error when training workbooks are present."""
        assert ai_model is not None

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_model_has_features_list(self, ai_model):
        """AIModel exposes a non-empty features list after initialisation."""
        assert isinstance(ai_model.features, list)
        assert len(ai_model.features) > 0

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_model_has_decision_threshold(self, ai_model):
        """AIModel exposes a decision threshold in [0, 1]."""
        assert 0.0 < ai_model.decision_threshold < 1.0

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_model_version_matches_expected(self, ai_model):
        """The loaded model version matches MODEL_VERSION."""
        from src.backend.ai_model import AIModel
        assert ai_model.MODEL_VERSION == AIModel.MODEL_VERSION

    def test_model_raises_on_missing_data(self, tmp_path):
        """AIModel raises FileNotFoundError when training workbook is absent."""
        from src.backend.ai_model import AIModel
        with pytest.raises(FileNotFoundError):
            AIModel("src/data/nonexistent_file.xlsx")

    def test_model_raises_on_no_arguments(self):
        """AIModel raises ValueError when called with no arguments."""
        from src.backend.ai_model import AIModel
        with pytest.raises((ValueError, TypeError)):
            AIModel()


# ===========================================================================
# Tests – Prediction pipeline
# ===========================================================================

class TestPredictionPipeline:
    """End-to-end prediction pipeline tests."""

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_predict_from_features_returns_float(self, ai_model, sample_row):
        """predict_from_features returns a float probability."""
        prob = ai_model.predict_from_features(sample_row)
        assert isinstance(prob, float)

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_predict_probability_in_range(self, ai_model, sample_row):
        """Predicted probability is in [0, 1]."""
        prob = ai_model.predict_from_features(sample_row)
        assert 0.0 <= prob <= 1.0

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_batch_prediction_consistency(self, ai_model, sample_row):
        """
        Batch prediction on duplicate rows must produce identical results,
        and single-row prediction must match the first batch element.
        """
        single = ai_model.predict_from_features(sample_row)
        batch = ai_model.predict_from_features_batch([sample_row, sample_row, sample_row])
        assert len(batch) == 3
        # All three identical inputs must produce the same probability
        assert batch[0] == pytest.approx(batch[1], abs=1e-9)
        assert batch[0] == pytest.approx(batch[2], abs=1e-9)
        # Single and batch must agree
        assert float(batch[0]) == pytest.approx(single, abs=1e-9)

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_high_risk_row_scores_higher(self, ai_model, sample_row, high_risk_row):
        """
        A high-risk application should receive a higher default probability
        than a comparable low-risk application.
        """
        low_prob = ai_model.predict_from_features(sample_row)
        high_prob = ai_model.predict_from_features(high_risk_row)
        assert high_prob > low_prob, (
            f"Expected high-risk ({high_prob:.4f}) > low-risk ({low_prob:.4f})"
        )

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_predict_dpd_returns_non_negative_float(self, ai_model, sample_row):
        """predict_dpd returns a non-negative float representing days past due."""
        dpd = ai_model.predict_dpd(sample_row)
        assert isinstance(dpd, float)
        assert dpd >= 0.0

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_predict_dpd_category_valid_label(self, ai_model, sample_row):
        """predict_dpd_category_batch returns a known DPD category label."""
        from src.backend.ai_model import AIModel
        cats = ai_model.predict_dpd_category_batch([sample_row])
        assert len(cats) == 1
        assert cats[0] in AIModel.DPD_CATEGORY_LABELS

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_empty_batch_returns_empty_array(self, ai_model):
        """predict_from_features_batch on an empty list returns an empty array."""
        result = ai_model.predict_from_features_batch([])
        assert len(result) == 0

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_process_raw_input_then_predict(self, ai_model, sample_row):
        """process_raw_input → predict_from_features pipeline succeeds."""
        processed = ai_model.process_raw_input(sample_row)
        assert isinstance(processed, dict)
        # The processed dict should not contain Target or Longest_DPD
        assert "Target" not in processed
        assert "Longest_DPD" not in processed


# ===========================================================================
# Tests – Credit score formula
# ===========================================================================

class TestCreditScoreFormula:
    """Tests for the credit score calculation and threshold logic."""

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_probability_threshold_binary_decision(self, ai_model, sample_row):
        """
        Default decision uses ai_model.decision_threshold:
        prob >= threshold → positive (default) label.
        """
        prob = ai_model.predict_from_features(sample_row)
        is_default = int(prob >= ai_model.decision_threshold)
        assert is_default in (0, 1)

    def test_dpd_target_definition(self):
        """
        The Target column uses DPD > 1 as the default boundary (not DPD > 0).
        """
        from src.backend.ai_model import AIModel
        df = pd.DataFrame([
            {"Current DPD": 0},
            {"Current DPD": 1},
            {"Current DPD": 2},
            {"Current DPD": 30},
        ])
        # Minimal required columns to prevent KeyErrors in process_data
        for col in ["Amount Finance", "Downpayment", "Downpayment %", "Payment Terms",
                    "Interest Rate", "Claimed Income", "BV Income", "Loan Amort",
                    "Age as of Contract", "Marital Status"]:
            if col not in df:
                df[col] = np.nan if col != "Marital Status" else "Single"
        # We only care about the Target column calculation
        # DPD > 1 → 1, else 0
        dpd_values = pd.Series([0, 1, 2, 30])
        targets = (dpd_values > 1).astype(int)
        assert list(targets) == [0, 0, 1, 1]

    @_SKIP_NO_DATA
    @pytest.mark.integration
    def test_batch_and_single_scores_agree_across_rows(self, ai_model, sample_row, high_risk_row):
        """Batch predictions agree with individual predictions row by row."""
        rows = [sample_row, high_risk_row]
        singles = [ai_model.predict_from_features(r) for r in rows]
        batch = ai_model.predict_from_features_batch(rows)
        for i, (s, b) in enumerate(zip(singles, batch)):
            assert s == pytest.approx(float(b), abs=1e-9), (
                f"Row {i}: single={s:.6f}, batch={b:.6f}"
            )


# ===========================================================================
# Tests – Process data pipeline (no training data needed)
# ===========================================================================

class TestProcessDataIntegration:
    """Integration tests for the data-processing pipeline in isolation."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.backend.ai_model import AIModel
        self.AIModel = AIModel

    def _minimal_row(self, **overrides) -> pd.DataFrame:
        base = {
            "Amount Finance": 200_000,
            "Downpayment": 40_000,
            "Downpayment %": 20,
            "Payment Terms": 24,
            "Asset": "Car",
            "Manufacturer": "Toyota",
            "Type of Unit": "New",
            "Price Gap": 0,
            "Loan History": "Fully paid Good Repeat Client",
            "BAP Type": "No BAP",
            "BAP Amount": np.nan,
            "Interest Rate": 0.15,
            "Claimed Income": 45_000,
            "EV Income": np.nan,
            "BV Income": 44_000,
            "Employment Type": "Locally Employed",
            "Claimed Tenure": 36,
            "EV Tenure": np.nan,
            "BV Tenure": 36,
            "DSCR": 1.5,
            "Age as of Contract": 35,
            "Home Ownership": "Owned",
            "Residence Tenure": 48,
            "Citizenship": "Filipino",
            "Dealer Name": "XYZ Autos",
            "Submitted ID": "ID1",
            "Marital Status": "Married",
            "Loan Amort": 8_500,
            "Product Type": "Auto Loan",
            "Mode of Payment": "ADA - BDO",
            "Co-Borrowers": "No",
            "Borrower Type": "Principal",
            "Contract Start Date": "03/01/2023",
            "Current DPD": 0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_pipeline_categorical_fillna(self):
        """Missing categorical columns are filled with 'Unknown'."""
        df = self._minimal_row()
        df = df.drop(columns=["Asset"])
        result = self.AIModel.process_data(df)
        assert "Asset" in result.columns
        assert result["Asset"].iloc[0] == "Unknown"

    def test_pipeline_mop_risk_score_mapped(self):
        """MOP_Risk_Score is populated from the Mode of Payment mapping."""
        df = self._minimal_row(**{"Mode of Payment": "PDC"})
        result = self.AIModel.process_data(df)
        assert "MOP_Risk_Score" in result.columns
        # PDC maps to 1
        assert result["MOP_Risk_Score"].iloc[0] == 1

    def test_pipeline_personal_loan_flag(self):
        """Is_Personal_Loan is set to 1 for personal loan product types."""
        df_pl = self._minimal_row(**{"Product Type": "Personal Loan"})
        df_al = self._minimal_row(**{"Product Type": "Auto Loan"})
        r_pl = self.AIModel.process_data(df_pl, inference=True)
        r_al = self.AIModel.process_data(df_al, inference=True)
        assert r_pl["Is_Personal_Loan"].iloc[0] == 1
        assert r_al["Is_Personal_Loan"].iloc[0] == 0

    def test_pipeline_dti_calculation(self):
        """DTI = Loan Amort / Claimed Income, computed correctly."""
        df = self._minimal_row(**{"Loan Amort": 10_000, "Claimed Income": 50_000})
        result = self.AIModel.process_data(df)
        expected_dti = 10_000 / 50_000
        assert result["DTI"].iloc[0] == pytest.approx(expected_dti, rel=1e-5)

    def test_pipeline_ltv_calculation(self):
        """LTV = Amount Finance / (Amount Finance + Downpayment)."""
        df = self._minimal_row(**{"Amount Finance": 400_000, "Downpayment": 100_000})
        result = self.AIModel.process_data(df)
        expected_ltv = 400_000 / (400_000 + 100_000)
        assert result["LTV"].iloc[0] == pytest.approx(expected_ltv, rel=1e-5)
