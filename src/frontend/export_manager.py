"""
export_manager.py – Standalone export logic extracted from main.py.

Provides :class:`ExportManager` with three export methods:
  * ``export_to_xlsx``    – multi-sheet workbook: metadata + summary + per-record detail
  * ``export_to_csv``     – per-record CSV files in a folder
  * ``export_single_result`` – single XLSX or CSV for one scored record

All methods return ``(success: bool, message: str)`` so callers can decide
how to surface feedback (messagebox, logger, etc.) without coupling to Tk.

Dependencies
------------
  * openpyxl  (bundled with the project)
  * csv       (stdlib)
  * datetime  (stdlib)
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Result = Dict[str, Any]
BreakdownFn = Callable[[Dict[str, Any]], Dict[str, Any]]


# ---------------------------------------------------------------------------
# ExportManager
# ---------------------------------------------------------------------------
class ExportManager:
    """Handles all data-export operations for the Credit Scoring application.

    Parameters
    ----------
    threshold : float
        Decision threshold stored in metadata rows (default 0.5).
    model_name : str
        Human-readable model identifier written to the metadata sheet.
    """

    # Styling constants for XLSX output
    _HEADER_BG = "2E7D32"      # green header fill
    _HEADER_FG = "FFFFFF"      # white header text
    _META_BG = "F2F4F8"        # light grey metadata cells
    _RISK_HIGH_BG = "FFEBEE"   # light red
    _RISK_MOD_BG = "FFF9C4"    # light yellow
    _RISK_LOW_BG = "E8F5E9"    # light green

    # Column definitions for the Summary sheet
    _SUMMARY_COLS = [
        "Application No",
        "Party No",
        "AI Predicted Probability (%)",
        "Predicted DPD (Any)",
        "Credit Score",
        "Risk Category",
        "Base Value",
        "Reconstructed",
        "Status",
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = "JACCS Credit Scoring Model",
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def export_to_xlsx(
        self,
        batch_results: List[Result],
        file_path: str,
        breakdown_fn: Optional[BreakdownFn] = None,
    ) -> Tuple[bool, str]:
        """Export *batch_results* to a single XLSX file.

        Sheets produced:
          1. **Metadata** – run timestamp, record count, threshold used.
          2. **Summary**  – one row per record (formatted probabilities, scores).
          3. **Detail_<app>_<party>** – per-record SHAP breakdown (when *breakdown_fn* given).

        Parameters
        ----------
        batch_results : list[dict]
            Output from the batch scoring pipeline.
        file_path : str
            Destination ``.xlsx`` path.
        breakdown_fn : callable | None
            ``breakdown_fn(source_row_dict) -> breakdown_dict``  
            When supplied a detail sheet is created for each successfully
            scored record.  Pass *None* to skip detail sheets.

        Returns
        -------
        (bool, str)
            ``(True, "Exported to …")`` on success, ``(False, error_message)`` on failure.
        """
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        except ImportError:
            return False, (
                "openpyxl is not installed.\n"
                "Run:  pip install openpyxl"
            )

        try:
            wb = Workbook()
            wb.remove(wb.active)   # remove default blank sheet

            header_fill = PatternFill(start_color=self._HEADER_BG, end_color=self._HEADER_BG, fill_type="solid")
            header_font = Font(bold=True, color=self._HEADER_FG)
            meta_fill = PatternFill(start_color=self._META_BG, end_color=self._META_BG, fill_type="solid")
            center = Alignment(horizontal="center", vertical="center")
            left = Alignment(horizontal="left", vertical="center")

            valid_records = [r for r in batch_results if r.get("Predicted DPD (Any)") is not None]

            # ── Sheet 1: Metadata ─────────────────────────────────────────
            meta_ws = wb.create_sheet("Metadata", 0)
            meta_rows = [
                ("Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                ("Model", self.model_name),
                ("Total Records Processed", len(batch_results)),
                ("Successfully Scored", len(valid_records)),
                ("Failed / Skipped", len(batch_results) - len(valid_records)),
                ("Decision Threshold", f"{self.threshold:.4f}"),
            ]
            for r_idx, (key, val) in enumerate(meta_rows, 1):
                key_cell = meta_ws.cell(row=r_idx, column=1, value=key)
                val_cell = meta_ws.cell(row=r_idx, column=2, value=val)
                key_cell.fill = meta_fill
                key_cell.font = Font(bold=True)
                val_cell.alignment = left

            meta_ws.column_dimensions["A"].width = 30
            meta_ws.column_dimensions["B"].width = 36

            # ── Sheet 2: Summary ──────────────────────────────────────────
            sum_ws = wb.create_sheet("Summary", 1)

            for col_idx, col_name in enumerate(self._SUMMARY_COLS, 1):
                cell = sum_ws.cell(row=1, column=col_idx, value=col_name)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = center

            row_num = 2
            for rec in valid_records:
                score = rec.get("Credit Score")
                prob = rec.get("AI Predicted Probability")
                risk_cat = self._score_to_category(score)

                values = [
                    rec.get("Application No"),
                    rec.get("Party No"),
                    self._fmt_pct(prob),
                    rec.get("Predicted DPD (Any)"),
                    self._fmt_score(score),
                    risk_cat,
                    self._fmt_score(rec.get("Base Value")),
                    self._fmt_score(rec.get("Reconstructed")),
                    rec.get("Status", "OK"),
                ]

                # Row background based on risk
                risk_bg = {
                    "Low Risk": self._RISK_LOW_BG,
                    "Moderate Risk": self._RISK_MOD_BG,
                    "High Risk": self._RISK_HIGH_BG,
                }.get(risk_cat, None)

                row_fill = PatternFill(start_color=risk_bg, end_color=risk_bg, fill_type="solid") if risk_bg else None

                for col_idx, val in enumerate(values, 1):
                    cell = sum_ws.cell(row=row_num, column=col_idx, value=val)
                    cell.alignment = center
                    if row_fill:
                        cell.fill = row_fill

                row_num += 1

            # Auto-size summary columns
            self._auto_size_ws(sum_ws)

            # ── Sheets 3+: Per-record detail ──────────────────────────────
            if breakdown_fn is not None:
                for rec in valid_records:
                    app_no = rec.get("Application No", "?")
                    party_no = rec.get("Party No", "?")
                    sheet_name = f"Detail_{app_no}_{party_no}"[:31]

                    det_ws = wb.create_sheet(sheet_name)

                    # Record summary at the top
                    meta_pairs = [
                        ("Application No", app_no),
                        ("Party No", party_no),
                        ("Predicted DPD (Any)", rec.get("Predicted DPD (Any)")),
                        ("Credit Score", self._fmt_score(rec.get("Credit Score"))),
                        ("Risk Category", self._score_to_category(rec.get("Credit Score"))),
                        ("AI Prob (%)", self._fmt_pct(rec.get("AI Predicted Probability"))),
                        ("Base Value", self._fmt_score(rec.get("Base Value"))),
                        ("Reconstructed", self._fmt_score(rec.get("Reconstructed"))),
                        ("Status", rec.get("Status", "OK")),
                    ]
                    for r_idx, (key, val) in enumerate(meta_pairs, 1):
                        k_cell = det_ws.cell(row=r_idx, column=1, value=key)
                        k_cell.fill = meta_fill
                        k_cell.font = Font(bold=True)
                        det_ws.cell(row=r_idx, column=2, value=val).alignment = left

                    # SHAP breakdown table
                    breakdown_header_row = len(meta_pairs) + 2
                    bd_cols = ["Feature", "Value", "Impact (SHAP)", "|Impact|"]
                    for c_idx, hdr in enumerate(bd_cols, 1):
                        cell = det_ws.cell(row=breakdown_header_row, column=c_idx, value=hdr)
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = center

                    try:
                        src = rec.get("source_row") or {}
                        breakdown = breakdown_fn(src)
                        for bd_row_idx, bd_row in enumerate(breakdown.get("rows", []), breakdown_header_row + 1):
                            det_ws.cell(row=bd_row_idx, column=1, value=bd_row.get("feature"))
                            det_ws.cell(row=bd_row_idx, column=2, value=bd_row.get("value"))
                            det_ws.cell(row=bd_row_idx, column=3, value=round(bd_row.get("shap", 0), 6))
                            det_ws.cell(row=bd_row_idx, column=4, value=round(bd_row.get("abs_shap", 0), 6))
                    except Exception as exc:
                        det_ws.cell(
                            row=breakdown_header_row + 1, column=1,
                            value=f"Breakdown unavailable: {exc}",
                        )

                    # Column widths
                    det_ws.column_dimensions["A"].width = 38
                    det_ws.column_dimensions["B"].width = 22
                    det_ws.column_dimensions["C"].width = 16
                    det_ws.column_dimensions["D"].width = 16

            wb.save(file_path)
            return True, f"Exported {len(valid_records)} record(s) to:\n{file_path}"

        except Exception as exc:
            return False, f"XLSX export failed:\n{exc}"

    # ------------------------------------------------------------------

    def export_to_csv(
        self,
        batch_results: List[Result],
        folder_path: str,
        breakdown_fn: Optional[BreakdownFn] = None,
    ) -> Tuple[bool, str]:
        """Export each scored record to its own CSV file inside *folder_path*.

        File names follow the pattern: ``<ApplicationNo>_<PartyNo>.csv``

        Parameters
        ----------
        batch_results : list[dict]
            Output from the batch scoring pipeline.
        folder_path : str
            Destination directory (created if it does not exist).
        breakdown_fn : callable | None
            When provided the SHAP breakdown rows are appended to each CSV.

        Returns
        -------
        (bool, str)
            ``(True, "Saved N files …")`` on success, ``(False, msg)`` on
            partial failure, ``(False, msg)`` on total failure.
        """
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as exc:
            return False, f"Cannot create output folder:\n{exc}"

        saved: List[str] = []
        failed: int = 0

        for rec in batch_results:
            if rec.get("Predicted DPD (Any)") is None:
                continue

            app_no = rec.get("Application No", "unknown")
            party_no = rec.get("Party No", "unknown")
            fname = f"{app_no}_{party_no}.csv"
            out_path = os.path.join(folder_path, fname)

            try:
                score = rec.get("Credit Score")
                prob = rec.get("AI Predicted Probability")

                meta_rows = [
                    ["Application No", app_no],
                    ["Party No", party_no],
                    ["Predicted DPD (Any)", rec.get("Predicted DPD (Any)")],
                    ["Credit Score", self._fmt_score(score)],
                    ["Risk Category", self._score_to_category(score)],
                    ["AI Predicted Probability (%)", self._fmt_pct(prob)],
                    ["Base Value", self._fmt_score(rec.get("Base Value"))],
                    ["Reconstructed", self._fmt_score(rec.get("Reconstructed"))],
                    ["Status", rec.get("Status", "OK")],
                    ["Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    [],
                ]

                detail_rows: List[Dict[str, Any]] = []
                if breakdown_fn is not None:
                    try:
                        src = rec.get("source_row") or {}
                        breakdown = breakdown_fn(src)
                        for bd_row in breakdown.get("rows", []):
                            detail_rows.append({
                                "Feature": bd_row.get("feature"),
                                "Value": bd_row.get("value"),
                                "Impact (SHAP)": round(bd_row.get("shap", 0), 6),
                                "|Impact|": round(bd_row.get("abs_shap", 0), 6),
                            })
                    except Exception:
                        detail_rows = []

                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    for row in meta_rows:
                        writer.writerow(row)
                    if detail_rows:
                        writer.writerow(["Feature", "Value", "Impact (SHAP)", "|Impact|"])
                        for d in detail_rows:
                            writer.writerow([d["Feature"], d["Value"], d["Impact (SHAP)"], d["|Impact|"]])

                saved.append(out_path)

            except Exception:
                failed += 1

        if not saved and failed == 0:
            return False, "No successfully scored records to export."
        if failed > 0 and not saved:
            return False, f"All {failed} export(s) failed."
        msg = f"Saved {len(saved)} file(s) to:\n{folder_path}"
        if failed:
            msg += f"\n({failed} file(s) failed)"
        return True, msg

    # ------------------------------------------------------------------

    def export_single_result(
        self,
        result: Result,
        breakdown: Optional[Dict[str, Any]],
        file_path: str,
    ) -> Tuple[bool, str]:
        """Export a single scored record to XLSX (or CSV based on extension).

        Parameters
        ----------
        result : dict
            Single record result dict (same structure as batch results).
        breakdown : dict | None
            Pre-computed SHAP breakdown dict (``{"rows": [...], "base_value": …}``).
            Pass *None* to skip the SHAP section.
        file_path : str
            Destination path.  Extension decides format: ``.xlsx`` → Excel,
            anything else → CSV.

        Returns
        -------
        (bool, str)
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".xlsx":
            return self._export_single_xlsx(result, breakdown, file_path)
        return self._export_single_csv(result, breakdown, file_path)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _export_single_xlsx(
        self,
        result: Result,
        breakdown: Optional[Dict[str, Any]],
        file_path: str,
    ) -> Tuple[bool, str]:
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError:
            return False, "openpyxl is not installed.  Run: pip install openpyxl"

        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Result"  # type: ignore[union-attr]

            header_fill = PatternFill(start_color=self._HEADER_BG, end_color=self._HEADER_BG, fill_type="solid")
            header_font = Font(bold=True, color=self._HEADER_FG)
            meta_fill = PatternFill(start_color=self._META_BG, end_color=self._META_BG, fill_type="solid")
            center = Alignment(horizontal="center", vertical="center")
            left = Alignment(horizontal="left", vertical="center")

            score = result.get("Credit Score")
            prob = result.get("AI Predicted Probability")
            meta_pairs = [
                ("Application No", result.get("Application No", "N/A")),
                ("Party No", result.get("Party No", "N/A")),
                ("Predicted DPD (Any)", result.get("Predicted DPD (Any)")),
                ("Credit Score", self._fmt_score(score)),
                ("Risk Category", self._score_to_category(score)),
                ("AI Predicted Probability (%)", self._fmt_pct(prob)),
                ("Base Value", self._fmt_score(result.get("Base Value"))),
                ("Reconstructed", self._fmt_score(result.get("Reconstructed"))),
                ("Status", result.get("Status", "OK")),
                ("Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ]

            for r_idx, (key, val) in enumerate(meta_pairs, 1):
                k_cell = ws.cell(row=r_idx, column=1, value=key)  # type: ignore[union-attr]
                k_cell.fill = meta_fill
                k_cell.font = Font(bold=True)
                ws.cell(row=r_idx, column=2, value=val).alignment = left  # type: ignore[union-attr]

            if breakdown:
                hdr_row = len(meta_pairs) + 2
                for c_idx, hdr in enumerate(["Feature", "Value", "Impact (SHAP)", "|Impact|"], 1):
                    cell = ws.cell(row=hdr_row, column=c_idx, value=hdr)  # type: ignore[union-attr]
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = center
                for bd_row_idx, bd_row in enumerate(breakdown.get("rows", []), hdr_row + 1):
                    ws.cell(row=bd_row_idx, column=1, value=bd_row.get("feature"))  # type: ignore[union-attr]
                    ws.cell(row=bd_row_idx, column=2, value=bd_row.get("value"))  # type: ignore[union-attr]
                    ws.cell(row=bd_row_idx, column=3, value=round(bd_row.get("shap", 0), 6))  # type: ignore[union-attr]
                    ws.cell(row=bd_row_idx, column=4, value=round(bd_row.get("abs_shap", 0), 6))  # type: ignore[union-attr]

            ws.column_dimensions["A"].width = 38  # type: ignore[union-attr]
            ws.column_dimensions["B"].width = 22  # type: ignore[union-attr]
            ws.column_dimensions["C"].width = 16  # type: ignore[union-attr]
            ws.column_dimensions["D"].width = 16  # type: ignore[union-attr]

            wb.save(file_path)
            return True, f"Result saved to:\n{file_path}"
        except Exception as exc:
            return False, f"XLSX export failed:\n{exc}"

    def _export_single_csv(
        self,
        result: Result,
        breakdown: Optional[Dict[str, Any]],
        file_path: str,
    ) -> Tuple[bool, str]:
        try:
            score = result.get("Credit Score")
            prob = result.get("AI Predicted Probability")
            rows = [
                ["Application No", result.get("Application No", "N/A")],
                ["Party No", result.get("Party No", "N/A")],
                ["Predicted DPD (Any)", result.get("Predicted DPD (Any)")],
                ["Credit Score", self._fmt_score(score)],
                ["Risk Category", self._score_to_category(score)],
                ["AI Predicted Probability (%)", self._fmt_pct(prob)],
                ["Base Value", self._fmt_score(result.get("Base Value"))],
                ["Reconstructed", self._fmt_score(result.get("Reconstructed"))],
                ["Status", result.get("Status", "OK")],
                ["Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                [],
            ]
            with open(file_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                for row in rows:
                    writer.writerow(row)
                if breakdown:
                    writer.writerow(["Feature", "Value", "Impact (SHAP)", "|Impact|"])
                    for bd_row in breakdown.get("rows", []):
                        writer.writerow([
                            bd_row.get("feature"),
                            bd_row.get("value"),
                            round(bd_row.get("shap", 0), 6),
                            round(bd_row.get("abs_shap", 0), 6),
                        ])
            return True, f"Result saved to:\n{file_path}"
        except Exception as exc:
            return False, f"CSV export failed:\n{exc}"

    # ── formatting helpers ─────────────────────────────────────────────────

    @staticmethod
    def _fmt_pct(value: Any) -> str:
        """Format a 0-1 probability as a percentage string (e.g. ``"34.56 %"``)."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value) * 100:.2f} %"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_score(value: Any) -> str:
        """Format a score/float to 2 decimal places."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _score_to_category(score: Any) -> str:
        """Return a risk category label for a credit score."""
        if score is None:
            return "Unknown"
        try:
            s = float(score)
        except (TypeError, ValueError):
            return "Unknown"
        if s >= 70:
            return "Low Risk"
        if s >= 30:
            return "Moderate Risk"
        return "High Risk"

    @staticmethod
    def _auto_size_ws(ws, max_width: int = 50) -> None:
        """Auto-fit column widths in an openpyxl worksheet."""
        for col in ws.columns:
            max_len = 0
            for cell in col:
                try:
                    cell_len = len(str(cell.value)) if cell.value is not None else 0
                    max_len = max(max_len, cell_len)
                except Exception:
                    pass
            col_letter = col[0].column_letter
            ws.column_dimensions[col_letter].width = min(max_len + 4, max_width)
