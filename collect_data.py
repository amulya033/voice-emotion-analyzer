"""
Data collection script — generates tables for the research report.
Run for ~3 minutes, speaking in different emotional styles as prompted.

Usage:
    python collect_data.py
    python collect_data.py --device 1
"""

import argparse
import time

import numpy as np

from core.audio import AudioStream, is_silence
from core.config import CALIBRATION_N, LABEL_EXPAND, SAMPLE_RATE, HOP_SEC
from core.model import classify, load_model
from core.stress import StressAnalyzer


SCENARIOS = [
    ("NEUTRAL",  "Speak calmly and normally — describe your day.",              20),
    ("HAPPY",    "Speak enthusiastically — talk about something exciting.",      20),
    ("ANGRY",    "Speak in a raised, tense voice — complain about something.",   20),
    ("SAD",      "Speak slowly and quietly — describe something unfortunate.",   20),
    ("HESITANT", "Speak with lots of pauses and filler words — um, uh…",        20),
]


def run(device: int | None = None):
    print("Loading model…")
    clf      = load_model()
    analyzer = StressAnalyzer()
    stream   = AudioStream(device=device)

    inference_times: list[float]  = []
    scenario_results: dict        = {}

    # ── calibration ──────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("CALIBRATION PHASE (~6 seconds)")
    print("Speak normally — just chat about anything.")
    print("=" * 60)

    stream.start()
    try:
        while not analyzer.calibrated:
            time.sleep(HOP_SEC)
            audio, _ = stream.get_audio()
            if audio is None or is_silence(audio):
                continue
            feats = analyzer.extract(audio)
            if feats:
                analyzer._calibrate(feats)
                print(f"  Calibration: {analyzer.calibration_progress}/{CALIBRATION_N}")

        print("  Baseline calibrated.\n")

        # ── scenarios ────────────────────────────────────────────────── #
        for scenario_name, prompt, duration_sec in SCENARIOS:
            print("=" * 60)
            print(f"SCENARIO: {scenario_name}")
            print(f"  {prompt}")
            print(f"  Speak for {duration_sec} seconds. Starting in 3…")
            time.sleep(3)
            print("  GO!")

            scenario_results[scenario_name] = []
            end_time = time.time() + duration_sec

            while time.time() < end_time:
                time.sleep(HOP_SEC)
                audio, _ = stream.get_audio()
                if audio is None or is_silence(audio):
                    continue

                t0 = time.perf_counter()
                emotions = classify(clf, audio)
                feats    = analyzer.extract(audio)
                elapsed  = time.perf_counter() - t0
                inference_times.append(elapsed)

                emo_dict = {
                    LABEL_EXPAND.get(e["label"], e["label"]): round(e["score"] * 100, 1)
                    for e in emotions
                }
                stress_dict = None
                if feats and analyzer.calibrated:
                    raw_stress  = analyzer.analyze(audio)
                    stress_dict = {k: round(v * 100, 1) for k, v in raw_stress.items()} if raw_stress else None

                scenario_results[scenario_name].append((emo_dict, stress_dict))
                remaining = int(end_time - time.time())
                print(f"  [{remaining:2d}s left] {emo_dict}  |  stress: {stress_dict}")

            print()
    finally:
        stream.stop()

    # ── summary ──────────────────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if inference_times:
        print("\n--- Performance ---")
        print(f"  Samples     : {len(inference_times)}")
        print(f"  Inference   : {np.mean(inference_times):.3f}s avg "
              f"(min {np.min(inference_times):.3f}s, max {np.max(inference_times):.3f}s)")

    print("\n--- Emotion Classification by Scenario ---")
    for name, records in scenario_results.items():
        emo_recs = [r[0] for r in records if r[0]]
        if not emo_recs:
            continue
        labels = list(emo_recs[0].keys())
        avg    = {lbl: round(np.mean([r.get(lbl, 0) for r in emo_recs]), 1) for lbl in labels}
        dominant = max(avg, key=avg.get)
        print(f"  {name:<12}: dominant={dominant:<8} | {avg}")

    print("\n--- Stress Indicators by Scenario ---")
    for name, records in scenario_results.items():
        stress_recs = [r[1] for r in records if r[1]]
        if not stress_recs:
            continue
        keys = list(stress_recs[0].keys())
        avg  = {k: round(np.mean([r[k] for r in stress_recs]), 1) for k in keys}
        print(f"  {name:<12}: {avg}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection for research report")
    parser.add_argument("--device", "-d", type=int, default=None)
    args = parser.parse_args()
    run(device=args.device)
