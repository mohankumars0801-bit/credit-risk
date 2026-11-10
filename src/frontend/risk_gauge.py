"""
risk_gauge.py – Canvas-based animated semicircle credit-risk gauge widget.

The gauge maps a credit score (0 – 100) to three coloured zones:
  ● Green   70 – 100   Low Risk
  ● Yellow  30 – 69    Moderate Risk
  ● Red      0 – 29    High Risk

Usage
-----
    gauge = RiskGauge(parent, width=320, height=180)
    gauge.pack()
    gauge.set_score(72.5)          # animated needle
    gauge.set_score(23.0, animate=False)  # instant
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Optional


class RiskGauge(tk.Canvas):
    """Animated semicircle gauge that visualises a credit score (0-100).

    Parameters
    ----------
    parent : tk.Widget
        Parent widget.
    width : int
        Canvas width (default 320).
    height : int
        Canvas height (default 190) – should be roughly *width / 2 + padding*.
    **kwargs
        Forwarded to :class:`tk.Canvas`.
    """

    # ── colour palette ─────────────────────────────────────────────────────
    _BG = "white"
    _TRACK_BG = "#E8E8E8"       # arc track background
    _GREEN = "#2E7D32"
    _YELLOW = "#F9A825"
    _RED = "#C62828"
    _NEEDLE = "#1A1A2E"
    _TEXT_SCORE = "#1C1C1C"
    _TEXT_LABEL = "#6A6F76"

    # ── layout constants ───────────────────────────────────────────────────
    _ARC_THICKNESS = 22          # px thickness of the coloured arc
    _NEEDLE_LEN_RATIO = 0.78     # fraction of gauge radius
    _ANIMATION_STEPS = 30        # frames for smooth animation
    _ANIMATION_DELAY_MS = 16     # ~60 fps

    def __init__(
        self,
        parent: tk.Widget,
        width: int = 320,
        height: int = 190,
        **kwargs,
    ) -> None:
        kwargs.setdefault("bg", self._BG)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, width=width, height=height, **kwargs)

        self._w = width
        self._h = height
        self._score: float = 0.0          # current displayed score
        self._target_score: float = 0.0   # score we are animating toward
        self._anim_id: Optional[str] = None

        # Derived geometry – recalculated on resize
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._r: float = 0.0

        self.bind("<Configure>", self._on_resize)
        self._compute_geometry(width, height)
        self._draw_all(self._score)

    # ── public API ─────────────────────────────────────────────────────────

    def set_score(self, score: float, animate: bool = True) -> None:
        """Move the needle to *score* (0-100).

        Parameters
        ----------
        score : float
            Credit score value, clamped to [0, 100].
        animate : bool
            When *True* (default) the needle moves smoothly over ~0.5 s.
        """
        score = max(0.0, min(100.0, float(score)))
        self._target_score = score

        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None

        if animate:
            self._animate()
        else:
            self._score = score
            self._draw_all(score)

    # ── geometry ───────────────────────────────────────────────────────────

    def _on_resize(self, event: tk.Event) -> None:  # type: ignore[override]
        self._compute_geometry(event.width, event.height)
        self._draw_all(self._score)

    def _compute_geometry(self, w: int, h: int) -> None:
        self._w = w
        self._h = h
        margin = max(16, int(min(w, h) * 0.08))
        # Centre x at mid-width, centre y at ~65 % down (semicircle opens upward)
        self._cx = w / 2.0
        self._cy = h * 0.70
        self._r = min(w / 2.0 - margin, self._cy - margin)

    # ── drawing ────────────────────────────────────────────────────────────

    def _draw_all(self, score: float) -> None:
        self.delete("all")
        self._draw_track()
        self._draw_zones()
        self._draw_tick_marks()
        self._draw_needle(score)
        self._draw_labels(score)

    def _draw_track(self) -> None:
        """Draw grey background arc (180°)."""
        x0, y0 = self._cx - self._r, self._cy - self._r
        x1, y1 = self._cx + self._r, self._cy + self._r
        t = self._ARC_THICKNESS
        # outer ring bg
        self.create_arc(
            x0, y0, x1, y1,
            start=0, extent=180, style="arc",
            outline=self._TRACK_BG, width=t,
        )

    def _draw_zones(self) -> None:
        """Draw the three risk-zone coloured arcs."""
        x0, y0 = self._cx - self._r, self._cy - self._r
        x1, y1 = self._cx + self._r, self._cy + self._r
        t = self._ARC_THICKNESS

        # Mapping: score 0-100 → angle 180°→0° (left → right on the semicircle)
        # score 0  → 180° (left)   score 100 → 0° (right)
        # Red   zone: score 0-29   → 180°→124.2°  (extent=55.8°)
        # Yellow zone: score 30-69 → 124.2°→48.6° (extent=75.6°)
        # Green  zone: score 70-100 → 48.6°→0°    (extent=48.6°)

        def s2a(s: float) -> float:
            """Score (0-100) → start angle (180→0 degrees)."""
            return 180.0 - s * 1.8

        zones = [
            (0, 29, self._RED),
            (30, 69, self._YELLOW),
            (70, 100, self._GREEN),
        ]
        for lo, hi, colour in zones:
            start_ang = s2a(hi)
            extent = (hi - lo) * 1.8
            self.create_arc(
                x0, y0, x1, y1,
                start=start_ang, extent=extent, style="arc",
                outline=colour, width=t,
            )

    def _draw_tick_marks(self) -> None:
        """Draw subtle tick marks at the zone boundaries."""
        for score in (0, 30, 70, 100):
            angle_deg = 180.0 - score * 1.8
            angle_rad = math.radians(angle_deg)
            inner = self._r - self._ARC_THICKNESS / 2 - 2
            outer = self._r + self._ARC_THICKNESS / 2 + 2
            x1 = self._cx + inner * math.cos(angle_rad)
            y1 = self._cy - inner * math.sin(angle_rad)
            x2 = self._cx + outer * math.cos(angle_rad)
            y2 = self._cy - outer * math.sin(angle_rad)
            self.create_line(x1, y1, x2, y2, fill="#BDBDBD", width=1)

    def _draw_needle(self, score: float) -> None:
        """Draw needle pointing to *score*."""
        angle_deg = 180.0 - score * 1.8
        angle_rad = math.radians(angle_deg)
        needle_len = self._r * self._NEEDLE_LEN_RATIO

        tip_x = self._cx + needle_len * math.cos(angle_rad)
        tip_y = self._cy - needle_len * math.sin(angle_rad)

        self.create_line(
            self._cx, self._cy, tip_x, tip_y,
            fill=self._NEEDLE, width=3, capstyle="round",
        )
        # Pivot circle
        pivot_r = max(5, int(self._r * 0.07))
        self.create_oval(
            self._cx - pivot_r, self._cy - pivot_r,
            self._cx + pivot_r, self._cy + pivot_r,
            fill=self._NEEDLE, outline=self._NEEDLE,
        )

    def _draw_labels(self, score: float) -> None:
        """Draw score number and risk label."""
        score_font_size = max(16, int(self._r * 0.22))
        label_font_size = max(10, int(self._r * 0.13))
        minor_font_size = max(8, int(self._r * 0.10))

        # Zone boundary labels
        for s, label, anchor_x_ratio in ((0, "0", -1.15), (30, "30", -0.75), (70, "70", 0.55), (100, "100", 1.1)):
            angle_deg = 180.0 - s * 1.8
            angle_rad = math.radians(angle_deg)
            lx = self._cx + (self._r + self._ARC_THICKNESS + 10) * math.cos(angle_rad)
            ly = self._cy - (self._r + self._ARC_THICKNESS + 10) * math.sin(angle_rad)
            self.create_text(
                lx, ly, text=str(s),
                font=("Segoe UI", minor_font_size),
                fill=self._TEXT_LABEL, anchor="center",
            )

        # Numeric score – centred inside the arc
        self.create_text(
            self._cx, self._cy - self._r * 0.30,
            text=f"{score:.1f}",
            font=("Segoe UI", score_font_size, "bold"),
            fill=self._TEXT_SCORE, anchor="center",
        )

        # Risk category label
        risk_label, label_colour = self._score_to_label(score)
        self.create_text(
            self._cx, self._cy - self._r * 0.30 + score_font_size + 6,
            text=risk_label,
            font=("Segoe UI", label_font_size, "bold"),
            fill=label_colour, anchor="center",
        )

    # ── animation ──────────────────────────────────────────────────────────

    def _animate(self) -> None:
        diff = self._target_score - self._score
        if abs(diff) < 0.5:
            self._score = self._target_score
            self._draw_all(self._score)
            self._anim_id = None
            return

        step = diff / self._ANIMATION_STEPS
        self._score += step
        self._draw_all(self._score)
        self._anim_id = self.after(self._ANIMATION_DELAY_MS, self._animate)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_label(score: float) -> tuple[str, str]:
        """Return (label_text, hex_colour) for a given score."""
        if score >= 70:
            return "Low Risk", "#2E7D32"
        if score >= 30:
            return "Moderate Risk", "#F9A825"
        return "High Risk", "#C62828"
