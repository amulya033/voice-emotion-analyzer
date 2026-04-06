import time
import threading
from collections import deque

import numpy as np
import torch
import librosa
import sounddevice as sd
from transformers import pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID        = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE     = 16_000
WINDOW_SEC      = 3
HOP_SEC         = 1
SILENCE_RMS     = 0.0002
CALIBRATION_N   = 6    # chunks needed before stress scores activate (~6s)

EMOTION_EMOJI = {
    "ang": "😠", "hap": "😊", "neu": "😐", "sad": "😢",
    "angry": "😠", "happy": "😊", "neutral": "😐",
}

LABEL_EXPAND = {
    "ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad",
}

# ---------------------------------------------------------------------------
# Emotion model
# ---------------------------------------------------------------------------
def load_model():
    print("Loading emotion model (first run downloads ~1 GB)...")
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("audio-classification", model=MODEL_ID, device=device)
    label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"Model ready on {label}.")
    print(f"This model is using the IEMOCAP dataset (Interactive Emotional Dyadic Motion Capture).\n")
    return clf

def classify_emotions(clf, audio: np.ndarray) -> list[dict]:
    results = clf(audio, top_k=None)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Stress / deception indicator (acoustic feature analysis)
# ---------------------------------------------------------------------------
class StressAnalyzer:
    """
    Computes voice stress indicators from acoustic features.
    Requires a short personal calibration period before scoring.

    Features tracked:
      - Pitch elevation  : mean F0 higher than your own baseline → tension
      - Voice tremor     : jitter (F0 micro-variation) → nervousness
      - Amplitude tremor : shimmer (amplitude micro-variation) → stress
      - Speech hesitation: change in pause ratio → cognitive load
    """

    WEIGHTS = {
        "pitch_elevation":   0.30,
        "voice_tremor":      0.30,
        "amplitude_tremor":  0.20,
        "speech_hesitation": 0.20,
    }

    def __init__(self):
        self._cal_buffer: list[dict] = []
        self.baseline: dict | None = None

    # ---- feature extraction ------------------------------------------------
    def _extract(self, audio: np.ndarray) -> dict | None:
        # Pitch via YIN
        f0 = librosa.yin(audio, fmin=60, fmax=500, sr=SAMPLE_RATE)
        voiced = f0[f0 > 60]  # filter unvoiced frames
        if len(voiced) < 20:
            return None  # not enough speech

        mean_pitch = float(np.mean(voiced))
        jitter     = float(np.mean(np.abs(np.diff(voiced))) / mean_pitch) if mean_pitch else 0.0

        # Shimmer via RMS frames
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_v = rms[rms > 0]
        shimmer = float(np.mean(np.abs(np.diff(rms_v))) / np.mean(rms_v)) if len(rms_v) > 1 else 0.0

        # Pause ratio: fraction of frames below silence threshold
        pause_ratio = float(np.sum(rms < SILENCE_RMS) / len(rms))

        return {
            "mean_pitch":  mean_pitch,
            "jitter":      jitter,
            "shimmer":     shimmer,
            "pause_ratio": pause_ratio,
        }

    # ---- calibration -------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    @property
    def calibration_progress(self) -> int:
        return min(len(self._cal_buffer), CALIBRATION_N)

    def _calibrate(self, feats: dict):
        self._cal_buffer.append(feats)
        if len(self._cal_buffer) >= CALIBRATION_N:
            self.baseline = {
                k: float(np.mean([c[k] for c in self._cal_buffer]))
                for k in feats
            }

    # ---- scoring -----------------------------------------------------------
    def _ratio_score(self, current, baseline, scale=2.0) -> float:
        if baseline == 0:
            return 0.0
        return float(np.clip((current / baseline - 1.0) / scale, 0.0, 1.0))

    def analyze(self, audio: np.ndarray) -> dict | None:
        feats = self._extract(audio)
        if feats is None:
            return None

        if not self.calibrated:
            self._calibrate(feats)
            return None

        b = self.baseline
        scores = {
            "pitch_elevation":   self._ratio_score(feats["mean_pitch"],  b["mean_pitch"],  scale=0.5),
            "voice_tremor":      self._ratio_score(feats["jitter"],      b["jitter"],      scale=2.0),
            "amplitude_tremor":  self._ratio_score(feats["shimmer"],     b["shimmer"],     scale=2.0),
            "speech_hesitation": float(np.clip(abs(feats["pause_ratio"] - b["pause_ratio"]) * 4.0, 0.0, 1.0)),
        }
        overall = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        scores["overall"] = float(overall)
        return scores

    def verdict(self, overall: float) -> str:
        if overall < 0.25: return "LOW"
        if overall < 0.50: return "MODERATE"
        if overall < 0.75: return "HIGH"
        return "VERY HIGH"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
BAR = 26

def _bar(score: float) -> str:
    filled = int(score * BAR)
    return "█" * filled + "░" * (BAR - filled)

def _pct(score: float) -> str:
    return f"{score * 100:5.1f}%"

_last_lines = 0

def _clear_prev():
    global _last_lines
    for _ in range(_last_lines):
        print("\033[A\033[K", end="")
    _last_lines = 0

def _print_block(lines: list[str]):
    global _last_lines
    text = "\n".join(lines)
    print(text)
    _last_lines = len(lines)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(emotions: list[dict], stress: dict | None, cal_progress: int, ts: str):
    W = 52
    lines = []

    # ── Emotion block ────────────────────────────────────────────
    lines.append(f"┌{'─'*W}┐")
    lines.append(f"│  Emotion Analysis  {ts:<32}│")
    lines.append(f"├{'─'*W}┤")
    for e in emotions:
        raw   = e["label"]
        lbl   = LABEL_EXPAND.get(raw, raw)
        score = e["score"]
        lines.append(f"│  {lbl:<9s} [{_bar(score)}] {_pct(score)} │")
    top     = emotions[0]
    top_lbl = LABEL_EXPAND.get(top["label"], top["label"]).upper()
    lines.append(f"├{'─'*W}┤")
    lines.append(f"│  Emotion → {top_lbl:<15s} ({_pct(top['score'])})           │")
    lines.append(f"└{'─'*W}┘")

    # ── Stress block ─────────────────────────────────────────────
    lines.append("")
    lines.append(f"┌{'─'*W}┐")
    lines.append(f"│  Voice Stress Indicators{' '*27}│")
    lines.append(f"├{'─'*W}┤")

    if stress is None:
        prog = f"{cal_progress}/{CALIBRATION_N}"
        lines.append(f"│  Calibrating personal baseline... {prog:<17}│")
        lines.append(f"│  (keep speaking normally for a few seconds){' '*8}│")
    else:
        labels = {
            "pitch_elevation":   "Pitch elevation ",
            "voice_tremor":      "Voice tremor    ",
            "amplitude_tremor":  "Amplitude tremor",
            "speech_hesitation": "Hesitation      ",
        }
        for key, lbl in labels.items():
            s = stress[key]
            lines.append(f"│  {lbl} [{_bar(s)}] {_pct(s)} │")

        overall  = stress["overall"]
        verdict  = StressAnalyzer().verdict(overall)   # stateless call
        lines.append(f"├{'─'*W}┤")
        lines.append(f"│  Deception likelihood → {verdict:<10s} ({_pct(overall)})     │")
        lines.append(f"├{'─'*W}┤")
        lines.append(f"│  * not accurate, just a fun feature :){' '*13}│")

    lines.append(f"└{'─'*W}┘")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    clf      = load_model()
    analyzer = StressAnalyzer()

    window_size = SAMPLE_RATE * WINDOW_SEC
    buffer: deque = deque(maxlen=window_size)
    lock   = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        with lock:
            buffer.extend(indata[:, 0].tolist())

    print(f"Listening on microphone  (window={WINDOW_SEC}s, hop={HOP_SEC}s)")
    print("Speak normally for ~6 seconds first to calibrate your stress baseline.")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=audio_callback):
        while True:
            time.sleep(HOP_SEC)

            with lock:
                if len(buffer) < window_size:
                    remaining = (window_size - len(buffer)) / SAMPLE_RATE
                    print(f"\rBuffering... {remaining:.1f}s ", end="", flush=True)
                    continue
                audio = np.array(buffer, dtype=np.float32)

            rms = np.sqrt(np.mean(audio ** 2))
            if rms < SILENCE_RMS:
                print(f"\r[silence — mic RMS: {rms:.5f}, threshold: {SILENCE_RMS}]  ", end="", flush=True)
                continue

            # Run both analyses
            emotions = classify_emotions(clf, audio)
            stress   = analyzer.analyze(audio)

            ts    = time.strftime("%H:%M:%S")
            lines = render(emotions, stress, analyzer.calibration_progress, ts)

            _clear_prev()
            _print_block(lines)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
