import numpy as np
import librosa
from .config import (
    SAMPLE_RATE, SILENCE_RMS, CALIBRATION_N,
    STRESS_WEIGHTS,
)


class StressAnalyzer:
    """
    Acoustic stress indicator derived from four prosodic features
    measured against a personal baseline calibrated at session start.

    Features:
        pitch_elevation   - mean F0 deviation from your own baseline
        voice_tremor      - jitter (F0 cycle-to-cycle variation)
        amplitude_tremor  - shimmer (RMS amplitude variation)
        speech_hesitation - change in pause-to-speech ratio
    """

    def __init__(self):
        self._cal_buffer: list[dict] = []
        self.baseline: dict | None = None

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    @property
    def calibration_progress(self) -> int:
        return min(len(self._cal_buffer), CALIBRATION_N)

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #

    def extract(self, audio: np.ndarray) -> dict | None:
        f0 = librosa.yin(audio, fmin=60, fmax=500, sr=SAMPLE_RATE)
        voiced = f0[f0 > 60]
        if len(voiced) < 20:
            return None

        mean_pitch = float(np.mean(voiced))
        jitter = (
            float(np.mean(np.abs(np.diff(voiced))) / mean_pitch)
            if mean_pitch else 0.0
        )

        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_v = rms[rms > 0]
        shimmer = (
            float(np.mean(np.abs(np.diff(rms_v))) / np.mean(rms_v))
            if len(rms_v) > 1 else 0.0
        )

        pause_ratio = float(np.sum(rms < SILENCE_RMS) / len(rms))

        # Spectral centroid: higher value → more energetic/harsh speech
        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
        mean_centroid = float(np.mean(centroid))

        return {
            "mean_pitch":    mean_pitch,
            "jitter":        jitter,
            "shimmer":       shimmer,
            "pause_ratio":   pause_ratio,
            "mean_centroid": mean_centroid,
        }

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def _calibrate(self, feats: dict):
        self._cal_buffer.append(feats)
        if len(self._cal_buffer) >= CALIBRATION_N:
            self.baseline = {
                k: float(np.mean([c[k] for c in self._cal_buffer]))
                for k in feats
            }

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ratio_score(current: float, baseline: float, scale: float = 2.0) -> float:
        if baseline == 0:
            return 0.0
        return float(np.clip((current / baseline - 1.0) / scale, 0.0, 1.0))

    def analyze(self, audio: np.ndarray) -> dict | None:
        feats = self.extract(audio)
        if feats is None:
            return None
        if not self.calibrated:
            self._calibrate(feats)
            return None

        b = self.baseline
        scores: dict[str, float] = {
            "pitch_elevation":   self._ratio_score(feats["mean_pitch"],    b["mean_pitch"],    scale=0.5),
            "voice_tremor":      self._ratio_score(feats["jitter"],        b["jitter"],        scale=2.0),
            "amplitude_tremor":  self._ratio_score(feats["shimmer"],       b["shimmer"],       scale=2.0),
            "speech_hesitation": float(np.clip(
                abs(feats["pause_ratio"] - b["pause_ratio"]) * 4.0, 0.0, 1.0
            )),
            "spectral_shift":    self._ratio_score(feats["mean_centroid"], b["mean_centroid"], scale=0.3),
        }
        scores["overall"] = float(
            sum(scores[k] * v for k, v in STRESS_WEIGHTS.items())
        )
        return scores

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #

    @staticmethod
    def verdict(overall: float) -> tuple[str, str]:
        if overall < 0.25:
            return "LOW",       "#10b981"
        if overall < 0.50:
            return "MODERATE",  "#f59e0b"
        if overall < 0.75:
            return "HIGH",      "#f97316"
        return "VERY HIGH",     "#ef4444"
