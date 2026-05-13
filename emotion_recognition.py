"""
CLI mode — prints a live ASCII dashboard to the terminal.

Usage:
    python emotion_recognition.py
    python emotion_recognition.py --device 1
    python emotion_recognition.py --file path/to/audio.wav
"""

import argparse
import time

import numpy as np

from core.audio import AudioStream, analyze_file, is_silence
from core.config import CALIBRATION_N, LABEL_EXPAND, SAMPLE_RATE, WINDOW_SEC, HOP_SEC
from core.model import classify, load_model, smooth_emotions
from core.stress import StressAnalyzer


# ── ASCII rendering ──────────────────────────────────────────────────── #

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
    print("\n".join(lines))
    _last_lines = len(lines)


def render(emotions: list[dict], stress: dict | None, cal_progress: int, ts: str) -> list[str]:
    W = 54
    lines = []
    lines.append(f"┌{'─' * W}┐")
    lines.append(f"│  Emotion Analysis  {ts:<34}│")
    lines.append(f"├{'─' * W}┤")
    for e in emotions:
        raw   = e["label"]
        lbl   = LABEL_EXPAND.get(raw, raw)
        score = e["score"]
        lines.append(f"│  {lbl:<9s} [{_bar(score)}] {_pct(score)} │")
    top     = emotions[0]
    top_lbl = LABEL_EXPAND.get(top["label"], top["label"]).upper()
    lines.append(f"├{'─' * W}┤")
    lines.append(f"│  Dominant → {top_lbl:<10s} ({_pct(top['score'])})             │")
    lines.append(f"└{'─' * W}┘")

    lines.append("")
    lines.append(f"┌{'─' * W}┐")
    lines.append(f"│  Voice Stress Indicators{' ' * 29}│")
    lines.append(f"├{'─' * W}┤")
    if stress is None:
        prog = f"{cal_progress}/{CALIBRATION_N}"
        lines.append(f"│  Calibrating… {prog:<10} (keep speaking normally){' ' * 6}│")
    else:
        labels = {
            "pitch_elevation":   "Pitch elevation ",
            "voice_tremor":      "Voice tremor    ",
            "amplitude_tremor":  "Amplitude tremor",
            "speech_hesitation": "Hesitation      ",
        }
        for key, lbl in labels.items():
            s = stress.get(key, 0.0)
            lines.append(f"│  {lbl} [{_bar(s)}] {_pct(s)} │")
        overall = stress["overall"]
        label, _ = StressAnalyzer.verdict(overall)
        lines.append(f"├{'─' * W}┤")
        lines.append(f"│  Stress level → {label:<10s} ({_pct(overall)})           │")
        lines.append(f"├{'─' * W}┤")
        lines.append(f"│  * deception score is just for fun :){' ' * 17}│")
    lines.append(f"└{'─' * W}┘")
    return lines


# ── File analysis ────────────────────────────────────────────────────── #

def run_file(path: str):
    print("Loading model…")
    clf = load_model()
    analyzer = StressAnalyzer()

    audio = analyze_file(path)
    hop   = SAMPLE_RATE * 3
    n     = len(audio)
    segs  = max(1, n // hop)

    print(f"Analyzing {path}  ({n / SAMPLE_RATE:.1f}s, {segs} segments)\n")

    all_emotions: list[dict] = []
    prev = None
    for i in range(segs):
        chunk = audio[i * hop: (i + 1) * hop]
        if len(chunk) < hop:
            chunk = np.pad(chunk, (0, hop - len(chunk)))
        emos = classify(clf, chunk)
        emos = smooth_emotions(prev, emos)
        prev = {"raw": emos}
        analyzer.analyze(chunk)
        all_emotions.append({e["label"]: e["score"] for e in emos})
        ts   = f"{i * 3:.0f}s–{(i + 1) * 3:.0f}s"
        top  = emos[0]
        print(f"  [{ts:>8}]  {LABEL_EXPAND.get(top['label'], top['label']):<8} {top['score'] * 100:.1f}%")

    avg: dict[str, float] = {}
    for label in all_emotions[0]:
        avg[label] = float(np.mean([s.get(label, 0) for s in all_emotions]))

    top_lbl = max(avg, key=avg.get)
    print(f"\nDominant emotion: {LABEL_EXPAND.get(top_lbl, top_lbl).upper()}  ({avg[top_lbl] * 100:.1f}%)")
    for k, v in sorted(avg.items(), key=lambda x: -x[1]):
        print(f"  {LABEL_EXPAND.get(k, k):<9}: {v * 100:.1f}%")


# ── Live mode ────────────────────────────────────────────────────────── #

def run_live(device: int | None):
    print("Loading model…")
    clf      = load_model()
    analyzer = StressAnalyzer()
    stream   = AudioStream(device=device)
    prev     = None

    print(f"Listening (window={WINDOW_SEC}s, hop={HOP_SEC}s)")
    print("Speak normally for ~6 seconds to calibrate your stress baseline.")
    print("Press Ctrl+C to stop.\n")

    stream.start()
    try:
        while True:
            time.sleep(HOP_SEC)
            audio, ratio = stream.get_audio()
            if audio is None:
                print(f"\rBuffering… {ratio * 100:.0f}% ", end="", flush=True)
                continue
            if is_silence(audio):
                rms = float(np.sqrt(np.mean(audio ** 2)))
                print(f"\r[silence — RMS {rms:.5f}]  ", end="", flush=True)
                continue

            emotions = classify(clf, audio)
            emotions = smooth_emotions(prev, emotions)
            prev     = {"raw": emotions}
            stress   = analyzer.analyze(audio)
            ts       = time.strftime("%H:%M:%S")

            _clear_prev()
            _print_block(render(emotions, stress, analyzer.calibration_progress, ts))
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop()


# ── Entry point ─────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Emotion Analyzer — CLI")
    parser.add_argument("--file",   "-f", type=str, help="Path to audio file to analyze")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Mic device index (see sounddevice.query_devices())")
    args = parser.parse_args()

    if args.file:
        run_file(args.file)
    else:
        run_live(args.device)
