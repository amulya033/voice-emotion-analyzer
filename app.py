import tkinter as tk
import threading
import time
import numpy as np
import torch
import librosa
import sounddevice as sd
from transformers import pipeline
from collections import deque

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID      = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE   = 16_000
WINDOW_SEC    = 3
HOP_SEC       = 1
SILENCE_RMS   = 0.0002
CALIBRATION_N = 6

LABEL_EXPAND = {
    "ang": "Angry", "hap": "Happy", "neu": "Neutral", "sad": "Sad",
    "angry": "Angry", "happy": "Happy", "neutral": "Neutral",
}
EMOTION_COLORS = {
    "hap": "#10b981", "happy": "#10b981",
    "sad": "#60a5fa",
    "ang": "#ef4444", "angry": "#ef4444",
    "neu": "#8b5cf6", "neutral": "#8b5cf6",
}
STRESS_COLORS = {
    "Pitch Elevation":   "#f59e0b",
    "Voice Tremor":      "#f97316",
    "Amplitude Tremor":  "#fb923c",
    "Hesitation":        "#fbbf24",
}

# ── Theme ──────────────────────────────────────────────────────────────────
BG      = "#111827"
SURFACE = "#1f2937"
BORDER  = "#374151"
TEXT    = "#f9fafb"
SUBTEXT = "#9ca3af"
ACCENT  = "#6366f1"
BAR_W   = 320
BAR_H   = 20


# ── Stress Analyzer ────────────────────────────────────────────────────────
class StressAnalyzer:
    WEIGHTS = {"pitch_elevation": 0.30, "voice_tremor": 0.30,
                "amplitude_tremor": 0.20, "speech_hesitation": 0.20}

    def __init__(self):
        self._cal_buffer = []
        self.baseline    = None

    @property
    def calibrated(self):
        return self.baseline is not None

    @property
    def calibration_progress(self):
        return min(len(self._cal_buffer), CALIBRATION_N)

    def _extract(self, audio):
        f0     = librosa.yin(audio, fmin=60, fmax=500, sr=SAMPLE_RATE)
        voiced = f0[f0 > 60]
        if len(voiced) < 20:
            return None
        mean_pitch = float(np.mean(voiced))
        jitter     = float(np.mean(np.abs(np.diff(voiced))) / mean_pitch) if mean_pitch else 0.0
        rms        = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_v      = rms[rms > 0]
        shimmer    = float(np.mean(np.abs(np.diff(rms_v))) / np.mean(rms_v)) if len(rms_v) > 1 else 0.0
        pause_ratio = float(np.sum(rms < SILENCE_RMS) / len(rms))
        return {"mean_pitch": mean_pitch, "jitter": jitter,
                "shimmer": shimmer, "pause_ratio": pause_ratio}

    def _calibrate(self, feats):
        self._cal_buffer.append(feats)
        if len(self._cal_buffer) >= CALIBRATION_N:
            self.baseline = {k: float(np.mean([c[k] for c in self._cal_buffer]))
                             for k in feats}

    def _ratio_score(self, cur, base, scale=2.0):
        if base == 0: return 0.0
        return float(np.clip((cur / base - 1.0) / scale, 0.0, 1.0))

    def analyze(self, audio):
        feats = self._extract(audio)
        if feats is None:
            return None
        if not self.calibrated:
            self._calibrate(feats)
            return None
        b = self.baseline
        scores = {
            "pitch_elevation":   self._ratio_score(feats["mean_pitch"], b["mean_pitch"], 0.5),
            "voice_tremor":      self._ratio_score(feats["jitter"],     b["jitter"],     2.0),
            "amplitude_tremor":  self._ratio_score(feats["shimmer"],    b["shimmer"],    2.0),
            "speech_hesitation": float(np.clip(abs(feats["pause_ratio"] - b["pause_ratio"]) * 4, 0, 1)),
        }
        scores["overall"] = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return scores

    @staticmethod
    def verdict(score):
        if score < 0.25: return "LOW",       "#10b981"
        if score < 0.50: return "MODERATE",  "#f59e0b"
        if score < 0.75: return "HIGH",       "#f97316"
        return               "VERY HIGH",    "#ef4444"


# ── Custom bar widget ──────────────────────────────────────────────────────
class BarRow(tk.Frame):
    def __init__(self, parent, label, color, **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self._color = color

        tk.Label(self, text=label, width=16, anchor="w",
                 bg=SURFACE, fg=TEXT, font=("Consolas", 10)).pack(side="left")

        self._track = tk.Frame(self, width=BAR_W, height=BAR_H, bg=BORDER)
        self._track.pack(side="left", padx=(0, 8))
        self._track.pack_propagate(False)

        self._fill = tk.Frame(self._track, width=0, height=BAR_H, bg=color)
        self._fill.place(x=0, y=0, height=BAR_H)

        self._pct = tk.Label(self, text=" 0.0%", width=6, anchor="e",
                             bg=SURFACE, fg=TEXT, font=("Consolas", 10))
        self._pct.pack(side="left")

    def set(self, score: float):
        w = int(np.clip(score, 0, 1) * BAR_W)
        self._fill.place(x=0, y=0, width=w, height=BAR_H)
        self._pct.config(text=f"{score*100:5.1f}%")


# ── Main App ───────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Emotion Analyzer")
        self.configure(bg=BG)
        self.resizable(False, False)

        self._results = {}
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

        self._build_ui()
        self._start()

    # ── UI build ────────────────────────────────────────────────────────
    def _section(self, parent, title):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=16, pady=(0, 12))
        inner = tk.Frame(outer, bg=SURFACE, padx=12, pady=10)
        inner.pack(fill="x")
        tk.Label(inner, text=title, bg=SURFACE, fg=ACCENT,
                 font=("Consolas", 11, "bold")).pack(anchor="w", pady=(0, 8))
        return inner

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, padx=16, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Voice Emotion Analyzer", bg=BG, fg=TEXT,
                 font=("Consolas", 16, "bold")).pack(side="left")
        self._clock = tk.Label(hdr, text="", bg=BG, fg=SUBTEXT,
                               font=("Consolas", 10))
        self._clock.pack(side="right", pady=4)

        # Status bar
        sf = tk.Frame(self, bg=SURFACE, padx=16, pady=6)
        sf.pack(fill="x")
        self._status = tk.Label(sf, text="Loading model...", bg=SURFACE,
                                fg=SUBTEXT, font=("Consolas", 10), anchor="w")
        self._status.pack(fill="x")

        tk.Frame(self, bg=BG, height=10).pack()

        # ── Emotion section
        emo_sec = self._section(self, "EMOTION ANALYSIS")
        self._emo_bars = {}
        for raw in ["hap", "neu", "sad", "ang"]:
            lbl   = LABEL_EXPAND[raw]
            color = EMOTION_COLORS[raw]
            row   = BarRow(emo_sec, lbl, color)
            row.pack(fill="x", pady=2)
            self._emo_bars[raw] = row

        tk.Frame(emo_sec, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._emo_verdict = tk.Label(emo_sec, text="Overall: —",
                                     bg=SURFACE, fg=TEXT,
                                     font=("Consolas", 12, "bold"))
        self._emo_verdict.pack(anchor="w")

        # ── Stress section
        stress_sec = self._section(self, "VOICE STRESS INDICATORS")
        self._cal_label = tk.Label(stress_sec,
                                   text="Speak normally for ~6 seconds to calibrate...",
                                   bg=SURFACE, fg=SUBTEXT, font=("Consolas", 10))
        self._cal_label.pack(anchor="w", pady=(0, 8))
        self._cal_bar = BarRow(stress_sec, "Calibration", ACCENT)
        self._cal_bar.pack(fill="x", pady=2)

        self._stress_frame = tk.Frame(stress_sec, bg=SURFACE)

        stress_keys = [
            ("pitch_elevation",   "Pitch Elevation"),
            ("voice_tremor",      "Voice Tremor"),
            ("amplitude_tremor",  "Amplitude Tremor"),
            ("speech_hesitation", "Hesitation"),
        ]
        self._stress_bars = {}
        for key, lbl in stress_keys:
            color = STRESS_COLORS.get(lbl, "#f59e0b")
            row   = BarRow(self._stress_frame, lbl, color)
            row.pack(fill="x", pady=2)
            self._stress_bars[key] = row

        tk.Frame(stress_sec, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._stress_verdict = tk.Label(stress_sec, text="Deception likelihood: —",
                                        bg=SURFACE, fg=TEXT,
                                        font=("Consolas", 12, "bold"))
        self._stress_verdict.pack(anchor="w")
        tk.Label(stress_sec, text="* not accurate, just a fun feature :)",
                 bg=SURFACE, fg=SUBTEXT, font=("Consolas", 9)).pack(anchor="w", pady=(4, 0))

        # Footer
        tk.Frame(self, bg=BG, height=10).pack()
        foot = tk.Frame(self, bg=BG, padx=16, pady=10)
        foot.pack()
        self._btn = tk.Button(foot, text="  Stop  ", command=self._toggle,
                              bg="#374151", fg=TEXT, activebackground="#4b5563",
                              activeforeground=TEXT, relief="flat",
                              font=("Consolas", 11), padx=12, pady=6, cursor="hand2")
        self._btn.pack()

        self._tick_clock()

    # ── Clock ────────────────────────────────────────────────────────────
    def _tick_clock(self):
        self._clock.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ── Analysis thread ──────────────────────────────────────────────────
    def _start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.after(200, self._poll)

    def _toggle(self):
        if self._running:
            self._running = False
            self._btn.config(text="  Start  ")
            self._status.config(text="Stopped.")
        else:
            self._btn.config(text="  Stop  ")
            self._start()

    def _run(self):
        self._set_status("Loading model...")
        device = 0 if torch.cuda.is_available() else -1
        clf    = pipeline("audio-classification", model=MODEL_ID, device=device)
        analyzer = StressAnalyzer()

        window_size = SAMPLE_RATE * WINDOW_SEC
        buf  = deque(maxlen=window_size)
        lock = threading.Lock()

        def callback(indata, frames, t, status):
            with lock:
                buf.extend(indata[:, 0].tolist())

        self._set_status("Listening... speak normally to calibrate stress baseline.")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", callback=callback):
            while self._running:
                time.sleep(HOP_SEC)
                with lock:
                    if len(buf) < window_size:
                        prog = len(buf) / window_size
                        self._set_status(f"Buffering... {prog*100:.0f}%")
                        continue
                    audio = np.array(buf, dtype=np.float32)

                rms = np.sqrt(np.mean(audio ** 2))
                if rms < SILENCE_RMS:
                    self._set_status(f"Waiting for speech...  (RMS {rms:.5f})")
                    continue

                emotions = clf(audio, top_k=None)
                emotions.sort(key=lambda x: x["score"], reverse=True)
                stress   = analyzer.analyze(audio)
                cal_prog = analyzer.calibration_progress

                with self._lock:
                    self._results = {
                        "emotions": emotions,
                        "stress":   stress,
                        "cal_prog": cal_prog,
                        "calibrated": analyzer.calibrated,
                    }

                self._set_status("Listening...")

    def _set_status(self, msg):
        self.after(0, lambda: self._status.config(text=msg))

    # ── UI polling ───────────────────────────────────────────────────────
    def _poll(self):
        with self._lock:
            data = dict(self._results)

        if data:
            self._update_emotions(data["emotions"])
            self._update_stress(data["stress"], data["cal_prog"], data["calibrated"])

        if self._running:
            self.after(200, self._poll)

    def _update_emotions(self, emotions):
        for e in emotions:
            raw   = e["label"]
            score = e["score"]
            if raw in self._emo_bars:
                self._emo_bars[raw].set(score)

        top     = emotions[0]
        top_lbl = LABEL_EXPAND.get(top["label"], top["label"]).upper()
        color   = EMOTION_COLORS.get(top["label"], TEXT)
        self._emo_verdict.config(
            text=f"Overall: {top_lbl}  ({top['score']*100:.1f}%)",
            fg=color)

    def _update_stress(self, stress, cal_prog, calibrated):
        if not calibrated:
            prog = cal_prog / CALIBRATION_N
            self._cal_bar.set(prog)
            self._cal_label.config(
                text=f"Calibrating... ({cal_prog}/{CALIBRATION_N} samples)")
            self._stress_frame.pack_forget()
        else:
            self._cal_label.config(text="Baseline calibrated.")
            self._cal_bar.set(1.0)
            self._stress_frame.pack(fill="x")

            if stress:
                for key, bar in self._stress_bars.items():
                    bar.set(stress.get(key, 0.0))
                overall        = stress["overall"]
                label, color   = StressAnalyzer.verdict(overall)
                self._stress_verdict.config(
                    text=f"Deception likelihood: {label}  ({overall*100:.1f}%)",
                    fg=color)


if __name__ == "__main__":
    app = App()
    app.mainloop()
