"""
Data collection script for report tables 6, 7, and 8.
Run this for 3-4 minutes, speaking in different emotional styles as prompted.
It will print a summary of real measured data at the end.
"""
import time
import threading
import numpy as np
import torch
import librosa
import sounddevice as sd
from transformers import pipeline
from collections import deque

MODEL_ID      = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE   = 16_000
WINDOW_SEC    = 3
HOP_SEC       = 1
SILENCE_RMS   = 0.0002
CALIBRATION_N = 6

LABEL_EXPAND = {
    "ang": "Angry", "hap": "Happy", "neu": "Neutral", "sad": "Sad",
}

SCENARIOS = [
    ("NEUTRAL",    "Speak calmly and normally — describe your day.",         20),
    ("HAPPY",      "Speak enthusiastically — talk about something exciting.", 20),
    ("ANGRY",      "Speak in a raised, tense voice — complain about something.", 20),
    ("SAD",        "Speak slowly and quietly — describe something unfortunate.", 20),
    ("HESITANT",   "Speak with lots of pauses and filler words — um, uh...", 20),
]

class StressAnalyzer:
    WEIGHTS = {"pitch_elevation": 0.30, "voice_tremor": 0.30,
               "amplitude_tremor": 0.20, "speech_hesitation": 0.20}

    def __init__(self):
        self._cal_buffer = []
        self.baseline    = None

    @property
    def calibrated(self):
        return self.baseline is not None

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

    def calibrate(self, feats):
        self._cal_buffer.append(feats)
        if len(self._cal_buffer) >= CALIBRATION_N:
            self.baseline = {k: float(np.mean([c[k] for c in self._cal_buffer]))
                             for k in feats}

    def _ratio_score(self, cur, base, scale=2.0):
        if base == 0: return 0.0
        return float(np.clip((cur / base - 1.0) / scale, 0.0, 1.0))

    def score(self, feats):
        b = self.baseline
        return {
            "pitch_elevation":   self._ratio_score(feats["mean_pitch"], b["mean_pitch"], 0.5),
            "voice_tremor":      self._ratio_score(feats["jitter"],     b["jitter"],     2.0),
            "amplitude_tremor":  self._ratio_score(feats["shimmer"],    b["shimmer"],    2.0),
            "speech_hesitation": float(np.clip(abs(feats["pause_ratio"] - b["pause_ratio"]) * 4, 0, 1)),
        }


def run():
    print("Loading model...")
    device = 0 if torch.cuda.is_available() else -1
    clf    = pipeline("audio-classification", model=MODEL_ID, device=device)
    analyzer = StressAnalyzer()

    window_size = SAMPLE_RATE * WINDOW_SEC
    buf  = deque(maxlen=window_size)
    lock = threading.Lock()

    def callback(indata, frames, t, status):
        with lock:
            buf.extend(indata[:, 0].tolist())

    # ── Storage ──────────────────────────────────────────────────────────
    inference_times  = []
    scenario_results = {}  # scenario_name -> list of (emotions_dict, stress_dict)

    # ── Calibration phase ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CALIBRATION PHASE (~6 seconds)")
    print("Speak normally — just chat about anything.")
    print("="*60)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", callback=callback):

        # calibrate
        while not analyzer.calibrated:
            time.sleep(HOP_SEC)
            with lock:
                if len(buf) < window_size:
                    continue
                audio = np.array(buf, dtype=np.float32)
            if np.sqrt(np.mean(audio**2)) < SILENCE_RMS:
                continue
            feats = analyzer._extract(audio)
            if feats:
                analyzer.calibrate(feats)
                print(f"  Calibration: {len(analyzer._cal_buffer)}/{CALIBRATION_N}")

        print("  Baseline calibrated.\n")

        # ── Scenario collection ───────────────────────────────────────────
        for scenario_name, prompt, duration_sec in SCENARIOS:
            print("="*60)
            print(f"SCENARIO: {scenario_name}")
            print(f"  {prompt}")
            print(f"  Speak for {duration_sec} seconds. Starting in 3...")
            time.sleep(3)
            print("  GO!")

            scenario_results[scenario_name] = []
            end_time = time.time() + duration_sec

            while time.time() < end_time:
                time.sleep(HOP_SEC)
                with lock:
                    if len(buf) < window_size:
                        continue
                    audio = np.array(buf, dtype=np.float32)

                if np.sqrt(np.mean(audio**2)) < SILENCE_RMS:
                    continue

                t0       = time.perf_counter()
                emotions = clf(audio, top_k=None)
                feats    = analyzer._extract(audio)
                elapsed  = time.perf_counter() - t0
                inference_times.append(elapsed)

                emotions.sort(key=lambda x: x["score"], reverse=True)
                emo_dict = {LABEL_EXPAND.get(e["label"], e["label"]): round(e["score"]*100, 1)
                            for e in emotions}

                stress_dict = None
                if feats:
                    stress_dict = analyzer.score(feats)
                    stress_dict = {k: round(v*100, 1) for k, v in stress_dict.items()}

                scenario_results[scenario_name].append((emo_dict, stress_dict))
                remaining = int(end_time - time.time())
                print(f"  [{remaining:2d}s left] {emo_dict}  |  stress: {stress_dict}")

            print()

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY — COPY THIS INTO YOUR REPORT")
    print("="*60)

    # Table 8: performance
    if inference_times:
        print("\n--- TABLE 8: System Performance ---")
        print(f"  Samples collected : {len(inference_times)}")
        print(f"  Inference time    : {np.mean(inference_times):.3f}s avg "
              f"(min {np.min(inference_times):.3f}s, max {np.max(inference_times):.3f}s)")
        print(f"  GPU used          : {'Yes' if device >= 0 else 'No (CPU)'}")

    # Table 6: emotion classification per scenario
    print("\n--- TABLE 6: Emotion Classification by Scenario ---")
    for scenario_name, records in scenario_results.items():
        emo_records = [r[0] for r in records if r[0]]
        if not emo_records:
            continue
        # average each emotion across records
        all_labels = list(emo_records[0].keys())
        avg = {lbl: round(np.mean([r.get(lbl, 0) for r in emo_records]), 1)
               for lbl in all_labels}
        # dominant emotion
        dominant = max(avg, key=avg.get)
        print(f"  {scenario_name:<12}: dominant={dominant:<8} | {avg}")

    # Table 7: stress indicators per scenario
    print("\n--- TABLE 7: Stress Indicators by Scenario ---")
    for scenario_name, records in scenario_results.items():
        stress_records = [r[1] for r in records if r[1]]
        if not stress_records:
            continue
        keys = list(stress_records[0].keys())
        avg  = {k: round(np.mean([r[k] for r in stress_records]), 1) for k in keys}
        overall = round(sum(avg[k] * w for k, w in [
            ("pitch_elevation", 0.30), ("voice_tremor", 0.30),
            ("amplitude_tremor", 0.20), ("speech_hesitation", 0.20)
        ]), 1)
        print(f"  {scenario_name:<12}: {avg}  overall={overall}%")

    print("\n" + "="*60)
    print("Done. Paste the output above to Amulya for report tables.")
    print("="*60)


if __name__ == "__main__":
    run()
