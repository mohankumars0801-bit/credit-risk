"""
progress_dialog.py – Reusable thread-safe progress dialog for the Credit Scoring frontend.

Usage
-----
    dialog = ProgressDialog(parent, title="Processing…", message="Please wait…", cancellable=True)

    # From a worker thread – fully thread-safe:
    dialog.update_message("Step 2 of 5…")
    dialog.update_progress(40)   # 0-100 for determinate mode, ignored for indeterminate
    dialog.close()               # always safe to call from any thread

    # Check whether the user pressed Cancel:
    if dialog.cancelled:
        return
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional


class ProgressDialog(tk.Toplevel):
    """Modal progress dialog with optional cancel button.

    Parameters
    ----------
    parent : tk.Widget
        The parent window.
    title : str
        Window title.
    message : str
        Initial status message shown below the progress bar.
    cancellable : bool
        When *True* a **Cancel** button is shown; `self.cancelled` becomes
        *True* if the user presses it.
    mode : str
        ``"indeterminate"`` (default) or ``"determinate"``.
    width : int
        Dialog width in pixels (default 420).
    """

    # Styling constants – match the application's green theme
    _GREEN = "#2E7D32"
    _BG = "white"
    _FG = "#1C1C1C"
    _SUB = "#6A6F76"

    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Processing…",
        message: str = "Please wait…",
        cancellable: bool = False,
        mode: str = "indeterminate",
        width: int = 420,
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self.cancelled: bool = False
        self._closed: bool = False
        self._mode = mode

        # ── window chrome ──────────────────────────────────────────────────
        self.title(title)
        self.configure(bg=self._BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Prevent the user from closing via the ✕ button (use Cancel instead)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel if cancellable else lambda: None)

        # ── size & centre ──────────────────────────────────────────────────
        h = 180 if cancellable else 150
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        px = getattr(parent, "winfo_rootx", lambda: (sw - width) // 2)()
        py = getattr(parent, "winfo_rooty", lambda: (sh - h) // 2)()
        pw = getattr(parent, "winfo_width", lambda: width)()
        ph = getattr(parent, "winfo_height", lambda: h)()
        x = px + (pw - width) // 2
        y = py + (ph - h) // 2
        self.geometry(f"{width}x{h}+{x}+{y}")

        # ── content ────────────────────────────────────────────────────────
        pad = tk.Frame(self, bg=self._BG, padx=24, pady=20)
        pad.pack(fill="both", expand=True)

        # Title label (bold)
        self._title_lbl = tk.Label(
            pad, text=title, bg=self._BG, fg=self._GREEN,
            font=("Segoe UI", 12, "bold"), anchor="w",
        )
        self._title_lbl.pack(fill="x", pady=(0, 10))

        # Progress bar
        style = ttk.Style(pad)
        style.configure(
            "Progress.Horizontal.TProgressbar",
            troughcolor="#F2F4F8", bordercolor="#F2F4F8",
            background=self._GREEN, lightcolor=self._GREEN, darkcolor=self._GREEN,
        )
        self._bar = ttk.Progressbar(
            pad, mode=mode, style="Progress.Horizontal.TProgressbar", length=width - 48,
        )
        self._bar.pack(fill="x", pady=(0, 10))
        if mode == "indeterminate":
            self._bar.start(12)

        # Status message label
        self._msg_var = tk.StringVar(value=message)
        self._msg_lbl = tk.Label(
            pad, textvariable=self._msg_var, bg=self._BG, fg=self._SUB,
            font=("Segoe UI", 10), wraplength=width - 56, justify="left", anchor="w",
        )
        self._msg_lbl.pack(fill="x")

        # Optional Cancel button
        if cancellable:
            self._cancel_btn = tk.Button(
                pad, text="Cancel", command=self._on_cancel,
                bg="#EEEEEE", fg="#000000", cursor="hand2",
                font=("Segoe UI", 10), bd=0, relief="flat",
                highlightthickness=1, highlightbackground="#CCCCCC",
                activebackground="#DDDDDD",
            )
            self._cancel_btn.pack(pady=(14, 0), ipadx=16, ipady=4)

        self.lift()
        self.update()

    # ── public API (all thread-safe) ───────────────────────────────────────

    def update_message(self, msg: str) -> None:
        """Set the status label text.  Safe to call from any thread."""
        self._schedule(lambda: self._msg_var.set(msg))

    def update_progress(self, value: float) -> None:
        """Set progress bar value (0-100). Only meaningful in *determinate* mode."""
        if self._mode == "determinate":
            self._schedule(lambda: self._bar.configure(value=float(value)))

    def set_title(self, title: str) -> None:
        """Update the bold title label (not the window title)."""
        self._schedule(lambda: self._title_lbl.configure(text=title))

    def close(self) -> None:
        """Destroy the dialog.  Safe to call from any thread."""
        self._schedule(self._do_close)

    @property
    def is_cancelled(self) -> bool:
        """Return *True* if the user pressed Cancel."""
        return self.cancelled

    # ── private helpers ────────────────────────────────────────────────────

    def _schedule(self, fn) -> None:
        """Schedule *fn* on the Tk main thread via after(0, …)."""
        try:
            if not self._closed:
                self.after(0, fn)
        except tk.TclError:
            # Widget already destroyed – nothing to do
            pass

    def _on_cancel(self) -> None:
        self.cancelled = True
        self._do_close()

    def _do_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bar.stop()
            self.grab_release()
            self.destroy()
        except tk.TclError:
            pass
