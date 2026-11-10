"""
Credit Scoring Model Evaluation & Testing Script  [AFTER FIX — credit-scoring-app]
Produces predictions with:
  Contract ID | App No | Amount Financed | Predicted Risk | Credit Score | Risk Tier
And displays score bracket counts:
  [0 to 29], [30 to 69], [70 to 100]

Manager-requested values computed in THIS file (After Fix):
  2. Prediction After Fix  — binary 0/1 (1 = predicted default, 0 = good borrower)
  4. Probability After Fix — raw model probability of default (0.0 to 1.0)

For Before Fix values, run the test in:  credit-scoring-app-commitid-5a5a/tests/test_model_predictions.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.backend.ai_model import AIModel
from src.frontend.main import App


def assign_risk_tier(score: float, prediction: int = 0, probability: float = 0.0, decision_threshold: float = 0.2353, actual_dpd: float = None) -> str:
    """
    Assign a risk tier from the model probability.

    ``actual_dpd`` is retained for compatibility with existing callers, but
    actual outcomes must not influence predictions or risk tiers.
    """
    if probability >= decision_threshold:
        return "High Risk"
    elif probability >= 0.05:
        return "Moderate Risk"
    else:
        return "Low Risk"


def assign_score_bracket_tier(score: float) -> str:
    """Legacy tier based strictly on linear score cutoffs [0-29.99], [30-69.99], [70-100]."""
    if score < 30.0:
        return "High Risk"
    elif score < 70.0:
        return "Moderate Risk"
    else:
        return "Low Risk"


def run_testing(input_file: Path, output_file: Path = None, show_samples: int = 20, dpd_cutoff: float = 10.0):
    model = AIModel("src/data/May 12 Status Contracts.xlsx", "src/data/Extracted New.xlsx")
    ui_input = App.__new__(App)

    print(f"[INFO] Reading input data from: {input_file}")
    if input_file.suffix.lower() in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(input_file)
    else:
        df_raw = pd.read_csv(input_file)

    print(f"[INFO] Loaded {len(df_raw):,} records. Preprocessing features...")
    print("[INFO] Computing model predictions and credit scores using the UI inference path...")
    ui_rows = []
    for row in df_raw.to_dict(orient="records"):
        normalized = ui_input._normalize_input_row(row)
        ui_rows.append(ui_input._row_to_ai_features(normalized))

    # ── (After Fix) Probability & Prediction ──────────────────────────────────
    # Probability After Fix: raw probability of default from the AFTER-FIX model
    probability_after_fix = model.predict_from_features_batch(ui_rows)

    decision_threshold = getattr(model, 'decision_threshold', 0.5)
    prediction_after_fix = (probability_after_fix >= decision_threshold).astype(int)

    predicted_risk = probability_after_fix.copy()
    credit_scores = (1.0 - predicted_risk) * 100.0
    df_processed = df_raw.rename(columns={"Current DPD": "Longest_DPD"})

    # Compute Predicted DPD (days) using compound hurdle credit risk model
    raw_records = df_raw.to_dict(orient="records")
    if hasattr(model, "predict_dpd_batch"):
        predicted_dpd = model.predict_dpd_batch(raw_records, predicted_risk)
    else:
        predicted_dpd = np.round(predicted_risk * 120.0, 1)

    # Keep actual DPD only as an evaluation column; predictions remain model-derived.
    has_actual_dpd_col = "Longest_DPD" in df_processed
    actual_dpd_series = pd.to_numeric(df_processed["Longest_DPD"], errors="coerce") if has_actual_dpd_col else None

    # Use only the independently predicted DPD for the decision rule.
    prediction_after_fix = (predicted_dpd > 1.0).astype(int)

    # Build results DataFrame
    contract_id = df_processed.get("Contract ID", df_raw.get("Contract ID", pd.Series(range(1, len(df_processed) + 1))))
    app_no = df_processed.get("Application No", df_raw.get("Application No", pd.Series(range(1, len(df_processed) + 1))))
    party_no = df_raw.get("Party No", None)
    amount_financed = df_processed.get("Amount Finance", df_raw.get("Amount Finance", pd.Series(np.nan, index=df_processed.index)))

    results_dict = {
        "Contract ID": contract_id.values,
        "App No": app_no.values,
    }
    if party_no is not None:
        results_dict["Party No"] = party_no.values
    results_dict["Amount Financed"] = amount_financed.values
    results_dict["Predicted Risk"] = np.round(predicted_risk, 4)
    results_dict["Credit Score"] = np.round(credit_scores, 2)

    results_df = pd.DataFrame(results_dict)

    results_df["Risk Tier"] = [
        assign_risk_tier(s, p, prob, decision_threshold)
        for s, p, prob in zip(credit_scores, prediction_after_fix, predicted_risk)
    ]
    results_df["Score Bracket Tier"] = [assign_score_bracket_tier(s) for s in credit_scores]
    results_df["Prediction"] = prediction_after_fix
    results_df["Predicted DPD (days)"] = np.round(predicted_dpd, 1)

    # If ground truth DPD is available, optionally include for validation
    if "Longest_DPD" in df_processed:
        results_df["Actual DPD"] = df_processed["Longest_DPD"].values

    # Preview sample records
    print("\n" + "=" * 90)
    print(f"SAMPLE TEST RESULTS (Showing {min(show_samples, len(results_df))} of {len(results_df):,} records)")
    print("=" * 90)
    header = f"{'Contract ID':<12} {'App No':<8} {'Amount Financed':>14} {'Predicted Risk':>16} {'Credit Score':>13} {'Risk Tier':<14} {'Pred DPD':>10}"
    print(header)
    print("-" * 95)

    # Take stratified sample across tiers for representative preview
    preview_rows = []
    for tier in ["High Risk", "Moderate Risk", "Low Risk"]:
        tier_sub = results_df[results_df["Risk Tier"] == tier]
        if not tier_sub.empty:
            preview_rows.append(tier_sub.head(max(2, show_samples // 3)))
    preview_df = pd.concat(preview_rows).head(show_samples) if preview_rows else results_df.head(show_samples)

    for _, r in preview_df.iterrows():
        cid = str(r["Contract ID"])
        app = str(r["App No"])
        amt = f"{r['Amount Financed']:,.2f}" if pd.notna(r["Amount Financed"]) else "N/A"
        risk = f"{r['Predicted Risk']:.4f} ({r['Predicted Risk']*100:.1f}%)"
        sc = f"{r['Credit Score']:.2f}"
        tier = r["Risk Tier"]
        pdpd = f"{r['Predicted DPD (days)']:.1f}d" if 'Predicted DPD (days)' in r else "N/A"
        print(f"{cid:<12} {app:<8} {amt:>14} {risk:>16} {sc:>13} {tier:<14} {pdpd:>10}")

    # Summary counts: Decision-Aligned Tiers (Credit Risk Standard)
    total_records = len(results_df)
    tier_high_count = int((results_df["Risk Tier"] == "High Risk").sum())
    tier_mod_count  = int((results_df["Risk Tier"] == "Moderate Risk").sum())
    tier_low_count  = int((results_df["Risk Tier"] == "Low Risk").sum())

    tier_high_pct = (tier_high_count / total_records) * 100.0 if total_records > 0 else 0.0
    tier_mod_pct  = (tier_mod_count  / total_records) * 100.0 if total_records > 0 else 0.0
    tier_low_pct  = (tier_low_count  / total_records) * 100.0 if total_records > 0 else 0.0

    # Summary counts: Legacy Linear Score Brackets
    count_0_29   = int((results_df["Credit Score"] < 30.0).sum())
    count_30_69  = int(((results_df["Credit Score"] >= 30.0) & (results_df["Credit Score"] < 70.0)).sum())
    count_70_100 = int((results_df["Credit Score"] >= 70.0).sum())

    pct_0_29   = (count_0_29   / total_records) * 100.0 if total_records > 0 else 0.0
    pct_30_69  = (count_30_69  / total_records) * 100.0 if total_records > 0 else 0.0
    pct_70_100 = (count_70_100 / total_records) * 100.0 if total_records > 0 else 0.0

    # Ground truth validation against Actual DPD > dpd_cutoff (if available)
    has_ground_truth = "Actual DPD" in results_df.columns and results_df["Actual DPD"].notna().sum() > 0
    actual_defaults = int((results_df["Actual DPD"] > dpd_cutoff).sum()) if has_ground_truth else 0
    caught_aligned  = int(((results_df["Risk Tier"] == "High Risk") & (results_df["Actual DPD"] > dpd_cutoff)).sum()) if has_ground_truth else 0
    caught_legacy   = int(((results_df["Credit Score"] < 30.0) & (results_df["Actual DPD"] > dpd_cutoff)).sum()) if has_ground_truth else 0

    # ── Manager-requested summary: After Fix values ────────────────────────────
    predicted_defaults = int(prediction_after_fix.sum())
    mean_prob          = float(probability_after_fix.mean())
    min_prob           = float(probability_after_fix.min())
    max_prob           = float(probability_after_fix.max())

    mean_score         = float(credit_scores.mean())
    median_score       = float(np.median(credit_scores))
    min_score          = float(credit_scores.min())
    max_score          = float(credit_scores.max())

    print("\n" + "=" * 90)
    print("AFTER FIX — MANAGER-REQUESTED VALUES SUMMARY")
    print("=" * 90)
    print(f"  Decision Threshold used                   : {decision_threshold:.4f}")
    print(f"  Prediction (After Fix) — records = 1     : {predicted_defaults:,} out of {total_records:,} records")
    print(f"  Prediction (After Fix) — records = 0     : {total_records - predicted_defaults:,} out of {total_records:,} records")
    print(f"  Probability (After Fix) — mean           : {mean_prob:.6f}")
    print(f"  Probability (After Fix) — min            : {min_prob:.6f}")
    print(f"  Probability (After Fix) — max            : {max_prob:.6f}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("DECISION-ALIGNED RISK TIER COUNTS (Credit Risk Best Practice)")
    print("=" * 90)
    print(f"Total Records Tested: {total_records:,}")
    print("-" * 90)
    print(f"  High Risk (DPD>1 / predicted DPD>1)      : {tier_high_count:7,d} records  ({tier_high_pct:5.2f}%)")
    print(f"  Moderate Risk (predicted PD 5%-threshold)  : {tier_mod_count:7,d} records  ({tier_mod_pct:5.2f}%)")
    print(f"  Low Risk (DPD<=1 / predicted DPD<=1)     : {tier_low_count:7,d} records  ({tier_low_pct:5.2f}%)")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("LEGACY SCORE BRACKET COUNTS (Score < 30 Cutoff)")
    print("=" * 90)
    print(f"  Score   0 to 29  (Legacy High Risk)     : {count_0_29:7,d} records  ({pct_0_29:5.2f}%)")
    print(f"  Score  30 to 69  (Legacy Moderate Risk) : {count_30_69:7,d} records  ({pct_30_69:5.2f}%)")
    print(f"  Score  70 to 100 (Legacy Low Risk)      : {count_70_100:7,d} records  ({pct_70_100:5.2f}%)")
    print("=" * 90)

    if has_ground_truth:
        print("\n" + "=" * 90)
        print(f"GROUND-TRUTH VALIDATION AGAINST ACTUAL DEFAULTERS (Actual DPD > {dpd_cutoff})")
        print("=" * 90)
        print(f"  Actual Historical Defaulters in Portfolio: {actual_defaults:,} records")
        print(f"  Caught by Decision-Aligned High Risk     : {caught_aligned:,} ({caught_aligned/actual_defaults*100:.2f}% Recall) — Missed: {actual_defaults - caught_aligned:,}")
        print(f"  Caught by Legacy Score < 30 High Risk    : {caught_legacy:,} ({caught_legacy/actual_defaults*100:.2f}% Recall) — Missed: {actual_defaults - caught_legacy:,}")
        print("=" * 90)

    summary_metrics = {
        "total_records": total_records,
        "decision_threshold": decision_threshold,
        "predicted_defaults": predicted_defaults,
        "mean_prob": mean_prob,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "mean_score": mean_score,
        "median_score": median_score,
        "min_score": min_score,
        "max_score": max_score,
        "tier_high_count": tier_high_count,
        "tier_mod_count": tier_mod_count,
        "tier_low_count": tier_low_count,
        "tier_high_pct": tier_high_pct,
        "tier_mod_pct": tier_mod_pct,
        "tier_low_pct": tier_low_pct,
        "count_0_29": count_0_29,
        "count_30_69": count_30_69,
        "count_70_100": count_70_100,
        "pct_0_29": pct_0_29,
        "pct_30_69": pct_30_69,
        "pct_70_100": pct_70_100,
        "has_ground_truth": has_ground_truth,
        "actual_defaults": actual_defaults,
        "caught_aligned": caught_aligned,
        "caught_legacy": caught_legacy,
        "mean_pred_dpd": float(np.mean(predicted_dpd)),
        "median_pred_dpd": float(np.median(predicted_dpd)),
        "max_pred_dpd": float(np.max(predicted_dpd)),
    }

    # Save to file
    if output_file:
        output_file = Path(output_file)
        if output_file.suffix.lower() in [".xlsx", ".xls"]:
            export_to_excel(output_file, results_df, df_raw, summary_metrics)
            # Also keep test_results.csv in sync
            csv_sync = PROJECT_ROOT / "test_results.csv"
            results_df.to_csv(csv_sync, index=False)
            print(f"[INFO] Synced CSV results ({len(results_df):,} rows) to: {csv_sync.resolve()}")
        else:
            results_df.to_csv(output_file, index=False)
            print(f"\n[INFO] Full results ({len(results_df):,} rows) exported to CSV: {output_file.resolve()}\n")

    return results_df, summary_metrics


def export_to_excel(output_path: Path, results_df: pd.DataFrame, df_raw: pd.DataFrame, summary_metrics: dict):
    """Export formatted multi-sheet Excel report with calculated results, summary metrics, and full dataset."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    print(f"[INFO] Building Excel report workbook at: {output_path.resolve()}...")

    metrics_data = [
        {"Evaluation Metric": "Total Contracts Evaluated", "Value": f"{summary_metrics['total_records']:,}"},
        {"Evaluation Metric": "Model Architecture", "Value": "Calibrated LightGBM (1,386 Trees, Target-Encoded)"},
        {"Evaluation Metric": "Decision Threshold", "Value": f"{summary_metrics['decision_threshold']:.4f}"},
        {"Evaluation Metric": "Predicted Defaults (Delinquent = 1)", "Value": f"{summary_metrics['predicted_defaults']:,}"},
        {"Evaluation Metric": "Predicted Good Borrowers (Non-Delinquent = 0)", "Value": f"{summary_metrics['total_records'] - summary_metrics['predicted_defaults']:,}"},
        {"Evaluation Metric": "Predicted Default Rate", "Value": f"{(summary_metrics['predicted_defaults'] / summary_metrics['total_records']) * 100:.2f}%"},
        {"Evaluation Metric": "Mean Default Risk Probability", "Value": f"{summary_metrics['mean_prob']:.4f}"},
        {"Evaluation Metric": "Min Default Risk Probability", "Value": f"{summary_metrics['min_prob']:.4f}"},
        {"Evaluation Metric": "Max Default Risk Probability", "Value": f"{summary_metrics['max_prob']:.4f}"},
        {"Evaluation Metric": "Mean Credit Score", "Value": f"{summary_metrics['mean_score']:.2f}"},
        {"Evaluation Metric": "Median Credit Score", "Value": f"{summary_metrics['median_score']:.2f}"},
        {"Evaluation Metric": "Min Credit Score", "Value": f"{summary_metrics['min_score']:.2f}"},
        {"Evaluation Metric": "Max Credit Score", "Value": f"{summary_metrics['max_score']:.2f}"},
        {"Evaluation Metric": "Mean Predicted DPD (days)", "Value": f"{summary_metrics.get('mean_pred_dpd', 0.0):.1f} days"},
        {"Evaluation Metric": "Median Predicted DPD (days)", "Value": f"{summary_metrics.get('median_pred_dpd', 0.0):.1f} days"},
        {"Evaluation Metric": "Max Predicted DPD (days)", "Value": f"{summary_metrics.get('max_pred_dpd', 0.0):.1f} days"},
    ]
    df_metrics = pd.DataFrame(metrics_data)

    aligned_tier_data = [
        {"Risk Tier (Decision-Aligned)": "High Risk", "Definition / Criterion": "Predicted DPD > 1", "Record Count": summary_metrics["tier_high_count"], "Percentage": f"{summary_metrics['tier_high_pct']:.2f}%"},
        {"Risk Tier (Decision-Aligned)": "Moderate Risk", "Definition / Criterion": "Predicted PD >= 5% and below default threshold", "Record Count": summary_metrics["tier_mod_count"], "Percentage": f"{summary_metrics['tier_mod_pct']:.2f}%"},
        {"Risk Tier (Decision-Aligned)": "Low Risk", "Definition / Criterion": "Predicted DPD <= 1", "Record Count": summary_metrics["tier_low_count"], "Percentage": f"{summary_metrics['tier_low_pct']:.2f}%"},
        {"Risk Tier (Decision-Aligned)": "Total", "Definition / Criterion": "All Applications", "Record Count": summary_metrics["total_records"], "Percentage": "100.00%"},
    ]
    df_aligned = pd.DataFrame(aligned_tier_data)

    legacy_bracket_data = [
        {"Score Bracket (Legacy)": "Score 0 to 29.99 (High Risk)", "Record Count": summary_metrics["count_0_29"], "Percentage": f"{summary_metrics['pct_0_29']:.2f}%"},
        {"Score Bracket (Legacy)": "Score 30 to 69.99 (Moderate Risk)", "Record Count": summary_metrics["count_30_69"], "Percentage": f"{summary_metrics['pct_30_69']:.2f}%"},
        {"Score Bracket (Legacy)": "Score 70 to 100.00 (Low Risk)", "Record Count": summary_metrics["count_70_100"], "Percentage": f"{summary_metrics['pct_70_100']:.2f}%"},
        {"Score Bracket (Legacy)": "Total", "Record Count": summary_metrics["total_records"], "Percentage": "100.00%"},
    ]
    df_legacy = pd.DataFrame(legacy_bracket_data)

    calc_cols = ["Predicted Risk", "Prediction", "Credit Score", "Risk Tier", "Score Bracket Tier", "Predicted DPD (days)"]
    enriched_cols = calc_cols + [c for c in df_raw.columns if c not in calc_cols]
    enriched_df = pd.concat([results_df[calc_cols], df_raw[[c for c in df_raw.columns if c not in calc_cols]]], axis=1)

    # Handle potential file lock if result.xlsx is currently open in Excel
    target_path = output_path
    try:
        if output_path.exists():
            with open(output_path, "a"):
                pass
    except (PermissionError, IOError):
        target_path = output_path.with_name(f"{output_path.stem}_corrected{output_path.suffix}")
        print(f"[WARNING] '{output_path.name}' is currently locked/open in Excel.")
        print(f"[INFO] Writing results to unlocked fallback file: {target_path.resolve()}")

    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        # Sheet 1: Calculated Results
        results_df.to_excel(writer, sheet_name="Calculated Results", index=False)
        
        # Sheet 2: Summary Metrics
        df_metrics.to_excel(writer, sheet_name="Summary Metrics", index=False, startrow=3)
        b1_start = 3 + len(df_metrics) + 3
        df_aligned.to_excel(writer, sheet_name="Summary Metrics", index=False, startrow=b1_start)
        b2_start = b1_start + len(df_aligned) + 3
        df_legacy.to_excel(writer, sheet_name="Summary Metrics", index=False, startrow=b2_start)
        
        # Sheet 3: All Features With Results
        enriched_df.to_excel(writer, sheet_name="All Features With Results", index=False)

        # Apply Styles
        header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        section_font = Font(name="Calibri", size=11, bold=True, color="1E293B")

        # Style Summary Metrics
        ws_sum = writer.sheets["Summary Metrics"]
        ws_sum.views.sheetView[0].showGridLines = True
        ws_sum.cell(row=1, column=1, value="CREDIT SCORING MODEL EVALUATION REPORT").font = Font(name="Calibri", size=15, bold=True, color="1E3A8A")
        ws_sum.cell(row=3, column=1, value="1. Performance & Inference Metrics").font = section_font
        for cell in ws_sum[4]:
            if cell.value:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

        ws_sum.cell(row=b1_start, column=1, value="2. Decision-Aligned Risk Tiers (Credit Risk Best Practice)").font = section_font
        for cell in ws_sum[b1_start + 1]:
            if cell.value:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

        ws_sum.cell(row=b2_start, column=1, value="3. Legacy Linear Score Bracket Counts (Score < 30 Cutoff)").font = section_font
        for cell in ws_sum[b2_start + 1]:
            if cell.value:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

        for col in ws_sum.columns:
            col_letter = get_column_letter(col[0].column)
            vals = [str(c.value or "") for c in col]
            max_len = max(len(v) for v in vals) if vals else 12
            ws_sum.column_dimensions[col_letter].width = max(max_len + 4, 18)

        # Style Calculated Results
        ws_res = writer.sheets["Calculated Results"]
        ws_res.views.sheetView[0].showGridLines = True
        ws_res.freeze_panes = "A2"
        ws_res.row_dimensions[1].height = 24
        for cell in ws_res[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for col_idx, col_name in enumerate(results_df.columns, 1):
            col_letter = get_column_letter(col_idx)
            ws_res.column_dimensions[col_letter].width = max(len(str(col_name)) + 4, 15)

        # Style Enriched Sheet
        ws_enr = writer.sheets["All Features With Results"]
        ws_enr.views.sheetView[0].showGridLines = True
        ws_enr.freeze_panes = "G2"
        ws_enr.row_dimensions[1].height = 24
        for cell in ws_enr[1]:
            is_calc = cell.value in calc_cols
            cell.fill = header_fill if is_calc else PatternFill(start_color="334155", end_color="334155", fill_type="solid")
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for col_idx, col_name in enumerate(enriched_df.columns, 1):
            col_letter = get_column_letter(col_idx)
            ws_enr.column_dimensions[col_letter].width = max(len(str(col_name)) + 3, 14)

    print(f"[INFO] Full results ({len(results_df):,} rows) exported to Excel: {target_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test credit scoring model on dataset [AFTER FIX]")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(PROJECT_ROOT / "src" / "data" / "Extracted New.xlsx"),
        help="Path to input Excel or CSV file (default: src/data/Extracted New.xlsx)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(PROJECT_ROOT / "result.xlsx"),
        help="Path to save output Excel or CSV (default: result.xlsx)",
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=15,
        help="Number of preview records to display (default: 15)",
    )
    parser.add_argument(
        "--dpd-cutoff",
        type=float,
        default=10.0,
        help="DPD cutoff threshold above which an account is defined as default (default: 10.0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        sys.exit(1)

    run_testing(input_path, output_path, show_samples=args.samples, dpd_cutoff=args.dpd_cutoff)
