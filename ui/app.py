"""Main application window."""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .theme import (
    ACCENT, ACCENT2, BG, BORDER, FONT_BODY, FONT_HEADER,
    FONT_SMALL, FONT_TITLE, FONT_VERDICT, SUBTEXT, SURFACE, SURFACE2,
    TEXT, WIN_W, WAVE_H, TIMELINE_H,
)
from .widgets import BarRow, TimelineCanvas, WaveformCanvas
from ..core.audio import AudioStream, analyze_file, is_silence, list_input_devices
from ..core.config import (
    CALIBRATION_N, EMOTIONS, EMOTION_COLORS, HOP_SEC,
    LABEL_EXPAND, SAMPLE_RATE, STRESS_COLORS, STRESS_KEYS,
    STRESS_LABELS, WINDOW_SEC,
)
from ..core.model import classify, confidence_level, load_model, smooth_emotions
from ..core.session import SessionRecorder
from ..core.stress import StressAnalyzer


# ═══════════════════════════════════════════════════════════════════════ #
#  Helpers
# ═══════════════════════════════════════════════════════════════════════ #

def _btn(parent, text, command, bg="#374151", width=None):
    kw = dict(
        text=text, command=command,
        bg=bg, fg=TEXT,
        activebackground="#4b5563", activeforeground=TEXT,
        relief="flat", font=FONT_BODY,
        padx=10, pady=5, cursor="hand2",
    )
    if width:
        kw["width"] = width
    return tk.Button(parent, **kw)


def _section_frame(parent, title: str) -> tk.Frame:
    outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
    outer.pack(fill="x", padx=14, pady=(0, 10))
    inner = tk.Frame(outer, bg=SURFACE, padx=10, pady=8)
    inner.pack(fill="x")
    tk.Label(inner, text=title, bg=SURFACE, fg=ACCENT,
             font=FONT_HEADER).pack(anchor="w", pady=(0, 6))
    return inner


# ═══════════════════════════════════════════════════════════════════════ #
#  Main App
# ═══════════════════════════════════════════════════════════════════════ #

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Voice Emotion Analyzer")
        self.configure(bg=BG)
        self.resizable(False, False)

        # State
        self._running    = False
        self._thread: threading.Thread | None = None
        self._results: dict = {}
        self._lock       = threading.Lock()
        self._prev_emo   = None           # for EMA smoothing
        self._session    = SessionRecorder()
        self._stream: AudioStream | None = None
        self._clf        = None           # shared model reference

        # Settings vars (created before _build_ui so sliders can bind to them)
        self._window_sec_var  = tk.IntVar(value=WINDOW_SEC)
        self._sensitivity_var = tk.IntVar(value=5)   # 1 = very sensitive, 10 = less

        # Build UI then start
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_analysis()

    # ================================================================== #
    # UI Construction
    # ================================================================== #

    def _build_ui(self):
        self._build_header()
        self._notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self._notebook.pack(fill="both", expand=True, padx=0, pady=0)
        self._style_notebook()

        self._tab_live     = tk.Frame(self._notebook, bg=BG)
        self._tab_history  = tk.Frame(self._notebook, bg=BG)
        self._tab_settings = tk.Frame(self._notebook, bg=BG)

        self._notebook.add(self._tab_live,     text="  LIVE  ")
        self._notebook.add(self._tab_history,  text="  HISTORY  ")
        self._notebook.add(self._tab_settings, text="  SETTINGS  ")

        self._build_live_tab()
        self._build_history_tab()
        self._build_settings_tab()

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG, padx=14, pady=10)
        hdr.pack(fill="x")

        tk.Label(hdr, text="Voice Emotion Analyzer",
                 bg=BG, fg=TEXT, font=FONT_TITLE).pack(side="left")

        right = tk.Frame(hdr, bg=BG)
        right.pack(side="right")

        self._status_dot = tk.Label(right, text="●", bg=BG, fg="#374151",
                                    font=("Consolas", 12))
        self._status_dot.pack(side="left", padx=(0, 6))

        self._clock_lbl = tk.Label(right, text="", bg=BG, fg=SUBTEXT,
                                   font=FONT_BODY)
        self._clock_lbl.pack(side="left")
        self._tick_clock()

        # Status bar
        sf = tk.Frame(self, bg=SURFACE2, padx=14, pady=4)
        sf.pack(fill="x")
        self._status_lbl = tk.Label(
            sf, text="Initializing…", bg=SURFACE2, fg=SUBTEXT,
            font=FONT_SMALL, anchor="w",
        )
        self._status_lbl.pack(fill="x")

    # ------------------------------------------------------------------ #
    # Live tab
    # ------------------------------------------------------------------ #

    def _build_live_tab(self):
        tab = self._tab_live

        # ── top row: emotion | stress ──────────────────────────────── #
        top_row = tk.Frame(tab, bg=BG)
        top_row.pack(fill="x", padx=14, pady=(10, 0))

        # Emotion panel
        emo_outer = tk.Frame(top_row, bg=BORDER, padx=1, pady=1)
        emo_outer.pack(side="left", fill="both", expand=True, padx=(0, 6))
        emo_inner = tk.Frame(emo_outer, bg=SURFACE, padx=10, pady=8)
        emo_inner.pack(fill="both", expand=True)
        tk.Label(emo_inner, text="EMOTION ANALYSIS", bg=SURFACE, fg=ACCENT,
                 font=FONT_HEADER).pack(anchor="w", pady=(0, 6))

        self._emo_bars: dict[str, BarRow] = {}
        for raw in EMOTIONS:
            lbl   = LABEL_EXPAND[raw]
            color = EMOTION_COLORS[raw]
            row   = BarRow(emo_inner, lbl, color)
            row.pack(fill="x", pady=2)
            self._emo_bars[raw] = row

        tk.Frame(emo_inner, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))

        conf_row = tk.Frame(emo_inner, bg=SURFACE)
        conf_row.pack(fill="x")
        self._emo_verdict = tk.Label(conf_row, text="Overall: —",
                                     bg=SURFACE, fg=TEXT, font=FONT_VERDICT)
        self._emo_verdict.pack(side="left")
        self._conf_lbl = tk.Label(conf_row, text="", bg=SURFACE,
                                  fg=SUBTEXT, font=FONT_SMALL)
        self._conf_lbl.pack(side="right", padx=(0, 2))

        # Stress panel
        stress_outer = tk.Frame(top_row, bg=BORDER, padx=1, pady=1)
        stress_outer.pack(side="left", fill="both", expand=True, padx=(6, 0))
        stress_inner = tk.Frame(stress_outer, bg=SURFACE, padx=10, pady=8)
        stress_inner.pack(fill="both", expand=True)
        tk.Label(stress_inner, text="VOICE STRESS", bg=SURFACE, fg=ACCENT,
                 font=FONT_HEADER).pack(anchor="w", pady=(0, 6))

        self._cal_label = tk.Label(
            stress_inner,
            text=f"Speak normally for ~{CALIBRATION_N}s to calibrate…",
            bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL,
        )
        self._cal_label.pack(anchor="w", pady=(0, 6))

        self._cal_bar = BarRow(stress_inner, "Calibration", ACCENT)
        self._cal_bar.pack(fill="x", pady=2)

        self._stress_frame = tk.Frame(stress_inner, bg=SURFACE)
        self._stress_bars: dict[str, BarRow] = {}
        for key in STRESS_KEYS:
            lbl   = STRESS_LABELS[key]
            color = STRESS_COLORS[key]
            row   = BarRow(self._stress_frame, lbl, color)
            row.pack(fill="x", pady=2)
            self._stress_bars[key] = row

        tk.Frame(stress_inner, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))
        self._stress_verdict = tk.Label(stress_inner, text="Stress level: —",
                                        bg=SURFACE, fg=TEXT, font=FONT_VERDICT)
        self._stress_verdict.pack(anchor="w")
        tk.Label(stress_inner, text="* deception score is just for fun :)",
                 bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w", pady=(2, 0))

        # ── waveform ───────────────────────────────────────────────── #
        wave_sec = _section_frame(tab, "LIVE WAVEFORM")
        self._waveform_canvas = WaveformCanvas(wave_sec, width=WIN_W - 56)
        self._waveform_canvas.pack(fill="x")

        # ── timeline ──────────────────────────────────────────────── #
        tl_sec = _section_frame(tab, f"EMOTION TIMELINE  (last {60}s)")
        self._timeline = TimelineCanvas(tl_sec, width=WIN_W - 56)
        self._timeline.pack(fill="x")

        # ── footer controls ───────────────────────────────────────── #
        foot = tk.Frame(tab, bg=BG, padx=14, pady=10)
        foot.pack(fill="x")

        self._toggle_btn = _btn(foot, "  Stop  ", self._toggle)
        self._toggle_btn.pack(side="left", padx=(0, 8))

        _btn(foot, "  Save Session  ", self._save_session).pack(side="left", padx=(0, 8))
        _btn(foot, "  Analyze File…  ", self._analyze_file).pack(side="left")

        # session counter
        self._session_lbl = tk.Label(
            foot, text="0 samples recorded", bg=BG, fg=SUBTEXT, font=FONT_SMALL,
        )
        self._session_lbl.pack(side="right")

    # ------------------------------------------------------------------ #
    # History tab
    # ------------------------------------------------------------------ #

    def _build_history_tab(self):
        tab = self._tab_history
        tk.Frame(tab, bg=BG, height=10).pack()

        hdr = _section_frame(tab, "SAVED SESSIONS")
        tk.Label(hdr, text="Sessions are saved to the sessions/ folder.",
                 bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w")
        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x", pady=8)

        list_frame = tk.Frame(hdr, bg=SURFACE)
        list_frame.pack(fill="x")

        scrollbar = tk.Scrollbar(list_frame, bg=SURFACE)
        scrollbar.pack(side="right", fill="y")

        self._history_list = tk.Listbox(
            list_frame,
            bg=SURFACE2, fg=TEXT, selectbackground=ACCENT,
            font=FONT_BODY, relief="flat", height=10,
            yscrollcommand=scrollbar.set,
        )
        self._history_list.pack(side="left", fill="x", expand=True)
        scrollbar.config(command=self._history_list.yview)

        btn_row = tk.Frame(hdr, bg=SURFACE)
        btn_row.pack(anchor="w", pady=(8, 0))
        _btn(btn_row, "  Refresh  ", self._refresh_history).pack(side="left", padx=(0, 6))
        _btn(btn_row, "  Open Folder  ", self._open_sessions_folder).pack(side="left")

        self._refresh_history()

    # ------------------------------------------------------------------ #
    # Settings tab
    # ------------------------------------------------------------------ #

    def _build_settings_tab(self):
        tab = self._tab_settings
        tk.Frame(tab, bg=BG, height=10).pack()

        # Microphone device
        mic_sec = _section_frame(tab, "INPUT DEVICE")
        tk.Label(mic_sec, text="Microphone:", bg=SURFACE, fg=SUBTEXT,
                 font=FONT_SMALL).pack(anchor="w")

        devices = list_input_devices()
        device_names = [f"{idx}: {name}" for idx, name in devices]
        self._device_var = tk.StringVar(value=device_names[0] if device_names else "")
        device_menu = tk.OptionMenu(mic_sec, self._device_var, *device_names)
        device_menu.config(bg=SURFACE2, fg=TEXT, activebackground=ACCENT,
                           activeforeground=TEXT, relief="flat", font=FONT_BODY,
                           highlightthickness=0)
        device_menu["menu"].config(bg=SURFACE2, fg=TEXT, font=FONT_BODY)
        device_menu.pack(anchor="w", pady=(4, 0))

        tk.Frame(mic_sec, bg=BORDER, height=1).pack(fill="x", pady=8)
        tk.Label(
            mic_sec,
            text="Restart analysis (Stop → Start) after changing the device.",
            bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL,
        ).pack(anchor="w")

        # ── Analysis parameters ──────────────────────────────────── #
        params_sec = _section_frame(tab, "ANALYSIS PARAMETERS")

        # Window size
        def _slider_row(parent, label, description):
            tk.Label(parent, text=label, bg=SURFACE, fg=TEXT,
                     font=FONT_BODY).pack(anchor="w")
            tk.Label(parent, text=description, bg=SURFACE, fg=SUBTEXT,
                     font=FONT_SMALL).pack(anchor="w")
            row = tk.Frame(parent, bg=SURFACE)
            row.pack(fill="x", pady=(4, 10))
            return row

        win_row = _slider_row(
            params_sec,
            "Analysis window",
            "How many seconds of audio to analyze per hop. "
            "Longer = more accurate but slower to react.",
        )
        self._win_val_lbl = tk.Label(win_row, text=f"{WINDOW_SEC}s",
                                     width=4, anchor="e", bg=SURFACE, fg=ACCENT,
                                     font=FONT_BODY)
        self._win_val_lbl.pack(side="right")
        tk.Scale(
            win_row, from_=2, to=6, orient="horizontal",
            variable=self._window_sec_var,
            bg=SURFACE, fg=TEXT, troughcolor=BORDER,
            activebackground=ACCENT, highlightthickness=0, showvalue=False,
            command=lambda v: self._win_val_lbl.config(text=f"{int(float(v))}s"),
        ).pack(side="left", fill="x", expand=True)

        sens_row = _slider_row(
            params_sec,
            "Mic sensitivity",
            "How sensitive the silence detector is. "
            "Raise if background noise triggers false readings.",
        )
        _SENS_LABELS = ["", "Very High", "High", "High", "Medium", "Medium",
                        "Low", "Low", "Very Low", "Very Low", "Very Low"]
        self._sens_val_lbl = tk.Label(sens_row, text=_SENS_LABELS[5],
                                      width=10, anchor="e", bg=SURFACE, fg=ACCENT,
                                      font=FONT_BODY)
        self._sens_val_lbl.pack(side="right")
        tk.Scale(
            sens_row, from_=1, to=10, orient="horizontal",
            variable=self._sensitivity_var,
            bg=SURFACE, fg=TEXT, troughcolor=BORDER,
            activebackground=ACCENT, highlightthickness=0, showvalue=False,
            command=lambda v: self._sens_val_lbl.config(
                text=_SENS_LABELS[int(float(v))]
            ),
        ).pack(side="left", fill="x", expand=True)

        tk.Frame(params_sec, bg=BORDER, height=1).pack(fill="x", pady=(0, 8))
        _btn(params_sec, "  Apply & Restart  ", self._apply_settings,
             bg=ACCENT).pack(anchor="w")
        tk.Label(params_sec,
                 text="Stops and restarts analysis with the new settings.",
                 bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w", pady=(4, 0))

        # Model info
        model_sec = _section_frame(tab, "MODEL INFO")
        info_lines = [
            ("Model",   "superb/wav2vec2-base-superb-er"),
            ("Dataset", "IEMOCAP (Interactive Emotional Dyadic Motion Capture)"),
            ("Labels",  "Happy · Neutral · Sad · Angry"),
            ("Window",  f"{WINDOW_SEC}s analysis window, {HOP_SEC}s hop"),
            ("SR",      f"{SAMPLE_RATE // 1000} kHz"),
        ]
        for key, val in info_lines:
            row = tk.Frame(model_sec, bg=SURFACE)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{key}:", width=10, anchor="w",
                     bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL).pack(side="left")
            tk.Label(row, text=val, anchor="w",
                     bg=SURFACE, fg=TEXT, font=FONT_SMALL).pack(side="left")

        # About
        about_sec = _section_frame(tab, "ABOUT")
        about_text = (
            "Voice Emotion Analyzer — CS 488 Independent Study\n"
            "Amulya Prasad · NJIT · Advisor: Dr. Pramod Abhichandani\n\n"
            "Real-time acoustic emotion classification using wav2vec2\n"
            "combined with prosodic stress feature analysis."
        )
        tk.Label(about_sec, text=about_text, bg=SURFACE, fg=SUBTEXT,
                 font=FONT_SMALL, justify="left").pack(anchor="w")

    # ================================================================== #
    # Styling
    # ================================================================== #

    def _style_notebook(self):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(
            "Dark.TNotebook",
            background=BG, borderwidth=0,
        )
        style.configure(
            "Dark.TNotebook.Tab",
            background=SURFACE, foreground=SUBTEXT,
            font=FONT_HEADER, padding=[14, 6],
            borderwidth=0,
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", SURFACE2), ("active", SURFACE2)],
            foreground=[("selected", TEXT),    ("active", ACCENT2)],
        )

    # ================================================================== #
    # Clock
    # ================================================================== #

    def _tick_clock(self):
        self._clock_lbl.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ================================================================== #
    # Analysis thread
    # ================================================================== #

    def _start_analysis(self):
        self._running = True
        self._toggle_btn.config(text="  Stop  ")
        self._status_dot.config(fg="#10b981")   # green = running
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.after(200, self._poll_ui)

    def _toggle(self):
        if self._running:
            self._running = False
            self._toggle_btn.config(text="  Start  ")
            self._status_dot.config(fg="#374151")
            self._set_status("Stopped. Press Start to resume.")
            if self._stream:
                self._stream.stop()
        else:
            self._prev_emo = None
            self._start_analysis()

    def _run_loop(self):
        self._set_status("Loading model…")

        # Read current settings before starting stream
        selected    = self._device_var.get()
        dev_id      = int(selected.split(":")[0]) if selected else None
        window_sec  = self._window_sec_var.get()
        sensitivity = self._sensitivity_var.get()
        # Map 1–10 → 0.0001–0.001 (higher = less sensitive = louder threshold)
        silence_rms = 0.0001 + (sensitivity - 1) * (0.0009 / 9)

        self._stream = AudioStream(device=dev_id, window_sec=window_sec)

        if self._clf is None:
            self._clf = load_model()
        analyzer = StressAnalyzer()

        self._set_status(
            "Speak normally for ~6 seconds to calibrate your stress baseline."
        )
        self._stream.start()

        while self._running:
            time.sleep(HOP_SEC)

            audio, ratio = self._stream.get_audio()
            if audio is None:
                self._set_status(f"Buffering… {ratio * 100:.0f}%")
                continue

            if is_silence(audio, threshold=silence_rms):
                rms = float(np.sqrt(np.mean(audio ** 2)))
                self._set_status(f"Waiting for speech…  RMS {rms:.5f}")
                continue

            emotions = classify(clf, audio)
            emotions = smooth_emotions(self._prev_emo, emotions)
            self._prev_emo = {"raw": emotions}

            stress   = analyzer.analyze(audio)
            cal_prog = analyzer.calibration_progress
            waveform = self._stream.get_waveform()

            with self._lock:
                self._results = {
                    "emotions":   emotions,
                    "stress":     stress,
                    "cal_prog":   cal_prog,
                    "calibrated": analyzer.calibrated,
                    "waveform":   waveform,
                    "confidence": confidence_level(emotions),
                }

            self._session.record(emotions, stress)
            self._set_status("Listening…")

    # ================================================================== #
    # UI Poll (runs on main thread every 200 ms)
    # ================================================================== #

    def _poll_ui(self):
        with self._lock:
            data = dict(self._results)

        if data:
            self._update_emotions(data["emotions"], data["confidence"])
            self._update_stress(data["stress"], data["cal_prog"], data["calibrated"])
            self._waveform_canvas.update_waveform(data["waveform"])
            self._timeline.push(data["emotions"])
            self._session_lbl.config(
                text=f"{self._session.count} samples  ·  "
                     f"{int(self._session.duration_seconds)}s"
            )

        if self._running:
            self.after(200, self._poll_ui)

    # ================================================================== #
    # Display updates
    # ================================================================== #

    def _update_emotions(self, emotions: list[dict], confidence: str):
        for e in emotions:
            raw = e["label"]
            if raw in self._emo_bars:
                self._emo_bars[raw].set(e["score"])

        top      = emotions[0]
        top_lbl  = LABEL_EXPAND.get(top["label"], top["label"]).upper()
        color    = EMOTION_COLORS.get(top["label"], TEXT)
        self._emo_verdict.config(
            text=f"{top_lbl}  {top['score'] * 100:.1f}%",
            fg=color,
        )
        conf_color = {"HIGH": "#10b981", "MODERATE": "#f59e0b", "LOW": "#ef4444"}
        self._conf_lbl.config(
            text=f"confidence: {confidence}",
            fg=conf_color.get(confidence, SUBTEXT),
        )

    def _update_stress(self, stress, cal_prog: int, calibrated: bool):
        if not calibrated:
            prog = cal_prog / CALIBRATION_N
            self._cal_bar.set(prog)
            self._cal_label.config(
                text=f"Calibrating… ({cal_prog}/{CALIBRATION_N} samples)"
            )
            self._stress_frame.pack_forget()
        else:
            self._cal_label.config(text="Baseline calibrated.")
            self._cal_bar.set(1.0)
            self._stress_frame.pack(fill="x")

            if stress:
                for key, bar in self._stress_bars.items():
                    bar.set(stress.get(key, 0.0))
                overall       = stress["overall"]
                label, color  = StressAnalyzer.verdict(overall)
                self._stress_verdict.config(
                    text=f"Stress level: {label}  ({overall * 100:.1f}%)",
                    fg=color,
                )

    # ================================================================== #
    # Actions
    # ================================================================== #

    def _save_session(self):
        if self._session.count == 0:
            messagebox.showinfo("Nothing to save", "No data has been recorded yet.")
            return
        json_path = self._session.save_json()
        csv_path  = self._session.save_csv()
        self._refresh_history()
        messagebox.showinfo(
            "Session saved",
            f"Saved {self._session.count} samples.\n\n"
            f"JSON → {json_path}\nCSV  → {csv_path}",
        )

    def _analyze_file(self):
        if self._clf is None:
            messagebox.showwarning(
                "Model not ready",
                "The model is still loading. Wait a moment and try again."
            )
            return

        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
        )
        if not path:
            return

        self._set_status(f"Analyzing file: {os.path.basename(path)}…")

        def _run():
            try:
                audio    = analyze_file(path)
                clf      = self._clf   # reuse already-loaded model
                analyzer = StressAnalyzer()

                hop   = SAMPLE_RATE * 3   # 3-second windows
                n     = len(audio)
                segments = max(1, n // hop)
                all_emotions: list[dict[str, float]] = []
                prev = None

                for i in range(segments):
                    chunk = audio[i * hop: (i + 1) * hop]
                    if len(chunk) < hop:
                        chunk = np.pad(chunk, (0, hop - len(chunk)))
                    emos = classify(clf, chunk)
                    emos = smooth_emotions(prev, emos)
                    prev = {"raw": emos}
                    analyzer.analyze(chunk)
                    all_emotions.append({e["label"]: e["score"] for e in emos})

                # Average across segments
                avg_scores: dict[str, float] = {}
                for label in all_emotions[0]:
                    avg_scores[label] = float(np.mean([s.get(label, 0) for s in all_emotions]))

                top_label = max(avg_scores, key=avg_scores.get)
                top_score = avg_scores[top_label]
                expanded  = LABEL_EXPAND.get(top_label, top_label).upper()

                self.after(0, lambda: messagebox.showinfo(
                    "File Analysis Complete",
                    f"File: {os.path.basename(path)}\n"
                    f"Duration: {n / SAMPLE_RATE:.1f}s  ({segments} segments)\n\n"
                    f"Dominant emotion: {expanded} ({top_score * 100:.1f}%)\n\n"
                    + "\n".join(
                        f"  {LABEL_EXPAND.get(k, k):<9}: {v * 100:.1f}%"
                        for k, v in sorted(avg_scores.items(), key=lambda x: -x[1])
                    )
                ))
                self._set_status("Listening…" if self._running else "Stopped.")
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Error", str(exc)))
                self._set_status("Error during file analysis.")

        threading.Thread(target=_run, daemon=True).start()

    def _refresh_history(self):
        self._history_list.delete(0, "end")
        sessions_dir = "sessions"
        if not os.path.isdir(sessions_dir):
            self._history_list.insert("end", "  (no sessions saved yet)")
            return
        files = sorted(
            [f for f in os.listdir(sessions_dir) if f.endswith((".json", ".csv"))],
            reverse=True,
        )
        if not files:
            self._history_list.insert("end", "  (no sessions saved yet)")
        for f in files:
            path = os.path.join(sessions_dir, f)
            size = os.path.getsize(path)
            self._history_list.insert("end", f"  {f}  ({size // 1024} KB)")

    def _apply_settings(self):
        """Stop and restart analysis with the current slider values."""
        if self._running:
            self._running = False
            self._status_dot.config(fg="#374151")
            if self._stream:
                self._stream.stop()
        self._prev_emo = None
        self._start_analysis()

    def _on_close(self):
        """Prompt to save unsaved session data before quitting."""
        if self._session.count > 0:
            result = messagebox.askyesnocancel(
                "Save session?",
                f"You recorded {self._session.count} samples "
                f"({int(self._session.duration_seconds)}s).\n\n"
                "Save session to JSON + CSV before closing?",
            )
            if result is None:    # Cancel — don't close
                return
            if result:            # Yes — save then close
                self._save_session()
        if self._running:
            self._running = False
            if self._stream:
                self._stream.stop()
        self.destroy()

    def _open_sessions_folder(self):
        import subprocess
        folder = os.path.abspath("sessions")
        os.makedirs(folder, exist_ok=True)
        subprocess.Popen(f'explorer "{folder}"')

    # ================================================================== #
    # Utilities
    # ================================================================== #

    def _set_status(self, msg: str):
        self.after(0, lambda: self._status_lbl.config(text=msg))
