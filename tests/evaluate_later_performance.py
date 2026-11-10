"""Evaluate the scoring model against later observed DPD outcomes.

Only rows with an observed label are evaluated. The observed DPD column is
used exclusively as the ground-truth label and is never passed to inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.ai_model import AIModel
from src.frontend.main import App


def evaluate(input_path: Path, label_column: str, threshold: float | None, dpd_cutoff: float = 10.0) -> dict[str, float | int | str]:
    """Return classification and ranking metrics for observed DPD outcomes."""
    df = pd.read_excel(input_path) if input_path.suffix.lower() == ".xlsx" else pd.read_csv(input_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column!r} was not found in {input_path}")

    observed_dpd = pd.to_numeric(df[label_column], errors="coerce")
    labeled = df.loc[observed_dpd.notna()].copy()
    if labeled.empty:
        raise ValueError(f"No numeric labels were found in {input_path}")

    y_true = (pd.to_numeric(labeled[label_column], errors="coerce") > dpd_cutoff).astype(int).to_numpy()
    if np.unique(y_true).size < 2:
        raise ValueError("ROC-AUC requires both default and non-default labels")

    model = AIModel("src/data/May 12 Status Contracts.xlsx", "src/data/Extracted New.xlsx")
    ui = App.__new__(App)
    feature_rows = [
        ui._row_to_ai_features(ui._normalize_input_row(row))
        for row in labeled.to_dict(orient="records")
    ]
    y_prob = np.asarray(model.predict_from_features_batch(feature_rows), dtype=float)
    decision_threshold = (
        float(getattr(model, "decision_threshold", 0.5))
        if threshold is None
        else threshold
    )
    y_pred = (y_prob >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "dataset": input_path.name,
        "default_rule": f"{label_column} > {dpd_cutoff}",
        "total_rows": int(len(df)),
        "labeled_rows": int(len(labeled)),
        "unlabeled_rows": int(len(df) - len(labeled)),
        "default_rows": int(y_true.sum()),
        "non_default_rows": int((y_true == 0).sum()),
        "default_rate": float(y_true.mean()),
        "threshold": decision_threshold,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "Raw Jan - Jun 2026.xlsx",
        help="Later-period workbook or CSV containing observed DPD outcomes.",
    )
    parser.add_argument(
        "--label",
        default="Current DPD",
        help="Observed DPD column used as the ground-truth label.",
    )
    parser.add_argument(
        "--dpd-cutoff",
        type=float,
        default=10.0,
        help="DPD cutoff threshold above which an account is defined as default (default: 10.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the model decision threshold.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for writing metrics as JSON.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.input.resolve(), args.label, args.threshold, args.dpd_cutoff)
    print(json.dumps(metrics, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {args.json}")


if __name__ == "__main__":
    main()
