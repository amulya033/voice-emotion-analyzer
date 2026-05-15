"""Reusable Tkinter widgets: BarRow, WaveformCanvas, TimelineCanvas."""

import tkinter as tk
import numpy as np
from collections import deque

from .theme import (
    SURFACE, SURFACE2, BORDER, TEXT, SUBTEXT, ACCENT,
    BAR_W, BAR_H,
    WAVE_H, TIMELINE_H, CANVAS_PAD,
    FONT_BODY, FONT_SMALL,
)
from core.config import EMOTIONS, EMOTION_COLORS, LABEL_EXPAND, TIMELINE_SECS


# ═══════════════════════════════════════════════════════════════════════ #
#  BarRow
# ═══════════════════════════════════════════════════════════════════════ #

class BarRow(tk.Frame):
    """Single labeled progress bar for a score in [0, 1]."""

    def __init__(self, parent, label: str, color: str, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self._color = color

        tk.Label(self, text=label, width=16, anchor="w",
                 bg=SURFACE, fg=TEXT, font=FONT_BODY).pack(side="left")

        self._track = tk.Frame(self, width=BAR_W, height=BAR_H, bg=BORDER)
        self._track.pack(side="left", padx=(0, 8))
        self._track.pack_propagate(False)

        self._fill = tk.Frame(self._track, width=0, height=BAR_H, bg=color)
        self._fill.place(x=0, y=0, height=BAR_H)

        self._pct = tk.Label(self, text=" 0.0%", width=6, anchor="e",
                             bg=SURFACE, fg=TEXT, font=FONT_BODY)
        self._pct.pack(side="left")

    def set(self, score: float):
        w = int(np.clip(score, 0.0, 1.0) * BAR_W)
        self._fill.place(x=0, y=0, width=w, height=BAR_H)
        self._pct.config(text=f"{score * 100:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════ #
#  WaveformCanvas
# ═══════════════════════════════════════════════════════════════════════ #

class WaveformCanvas(tk.Canvas):
    """Real-time oscilloscope-style waveform drawn on a Canvas."""

    def __init__(self, parent, width: int, **kw):
        super().__init__(
            parent,
            width=width, height=WAVE_H,
            bg=SURFACE2, highlightthickness=0,
            **kw,
        )
        self._w = width
        self._mid = WAVE_H // 2
        self._line_id = None
        self._draw_baseline()

    def _draw_baseline(self):
        self.create_line(
            CANVAS_PAD, self._mid, self._w - CANVAS_PAD, self._mid,
            fill=BORDER, width=1, dash=(4, 4),
        )

    def update_waveform(self, samples: np.ndarray):
        """Draw `samples` (float32 [-1,1]) as a centered oscilloscope line."""
        n = len(samples)
        if n < 2:
            return

        x_step = (self._w - 2 * CANVAS_PAD) / (n - 1)
        amplitude = (WAVE_H // 2) - 4

        coords = []
        for i, s in enumerate(samples):
            x = CANVAS_PAD + i * x_step
            y = self._mid - np.clip(s, -1.0, 1.0) * amplitude
            coords.extend([x, y])

        if self._line_id:
            self.coords(self._line_id, *coords)
        else:
            self._line_id = self.create_line(*coords, fill=ACCENT, width=1, smooth=True)


# ═══════════════════════════════════════════════════════════════════════ #
#  TimelineCanvas
# ═══════════════════════════════════════════════════════════════════════ #

class TimelineCanvas(tk.Canvas):
    """
    Scrolling line chart showing emotion scores over the last TIMELINE_SECS seconds.
    Each emotion is a colored polyline; the chart scrolls left with each new reading.
    """

    _LABEL_W = 56   # pixels reserved on left for y-axis labels
    _AXIS_H  = 14   # pixels reserved at bottom for axis

    def __init__(self, parent, width: int, **kw):
        super().__init__(
            parent,
            width=width, height=TIMELINE_H,
            bg=SURFACE2, highlightthickness=0,
            **kw,
        )
        self._w   = width
        self._cw  = width - self._LABEL_W - CANVAS_PAD   # chart area width
        self._ch  = TIMELINE_H - self._AXIS_H - 4        # chart area height
        self._ox  = self._LABEL_W                        # chart origin x
        self._oy  = 2                                    # chart origin y

        # Rolling buffer: one list of scores per emotion, newest on right
        self._history: dict[str, deque] = {
            raw: deque(maxlen=TIMELINE_SECS) for raw in EMOTIONS
        }
        self._line_ids: dict[str, int | None] = {raw: None for raw in EMOTIONS}

        self._draw_static()

    # ------------------------------------------------------------------ #

    def _draw_static(self):
        """Draw static grid, y-axis labels, and legend."""
        # Horizontal grid lines at 25 / 50 / 75 / 100 %
        for pct in (25, 50, 75, 100):
            y = self._oy + self._ch - int(pct / 100 * self._ch)
            self.create_line(
                self._ox, y, self._ox + self._cw, y,
                fill=BORDER, width=1,
            )
            self.create_text(
                self._ox - 4, y, text=f"{pct}%",
                anchor="e", fill=SUBTEXT, font=FONT_SMALL,
            )

        # Legend (bottom-left)
        lx = self._LABEL_W
        ly = TIMELINE_H - self._AXIS_H + 2
        for raw in EMOTIONS:
            lbl   = LABEL_EXPAND.get(raw, raw)
            color = EMOTION_COLORS.get(raw, "#ffffff")
            self.create_rectangle(lx, ly, lx + 8, ly + 8, fill=color, outline="")
            self.create_text(lx + 11, ly + 4, text=lbl, anchor="w",
                             fill=SUBTEXT, font=FONT_SMALL)
            lx += 72

    # ------------------------------------------------------------------ #

    def push(self, emotions: list[dict]):
        """Append one reading and redraw all lines."""
        score_map = {e["label"]: e["score"] for e in emotions}
        for raw in EMOTIONS:
            score = score_map.get(raw, score_map.get(LABEL_EXPAND.get(raw, ""), 0.0))
            self._history[raw].append(score)

        self._redraw_lines()

    # ------------------------------------------------------------------ #

    def _redraw_lines(self):
        for raw in EMOTIONS:
            history = list(self._history[raw])
            n = len(history)
            if n < 2:
                continue

            color = EMOTION_COLORS.get(raw, "#ffffff")
            x_step = self._cw / (TIMELINE_SECS - 1)

            # Pad shorter histories so lines always end at the right edge
            start_x = self._ox + self._cw - (n - 1) * x_step

            coords = []
            for i, score in enumerate(history):
                x = start_x + i * x_step
                y = self._oy + self._ch - score * self._ch
                coords.extend([x, y])

            if len(coords) < 4:
                continue

            if self._line_ids[raw]:
                self.coords(self._line_ids[raw], *coords)
                self.itemconfig(self._line_ids[raw], fill=color)
            else:
                self._line_ids[raw] = self.create_line(
                    *coords, fill=color, width=2, smooth=True
                )
