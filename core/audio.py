import threading
from collections import deque

import numpy as np
import sounddevice as sd
import librosa

from .config import SAMPLE_RATE, WINDOW_SEC, SILENCE_RMS

_WAVEFORM_SAMPLES = 800  # display resolution for live waveform


class AudioStream:
    """Continuous microphone capture with thread-safe audio buffer."""

    def __init__(self, device: int | None = None, window_sec: int = WINDOW_SEC):
        self.device = device
        self._window = SAMPLE_RATE * window_sec
        self._buf: deque = deque(maxlen=self._window)
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._waveform = np.zeros(_WAVEFORM_SAMPLES, dtype=np.float32)

    # ------------------------------------------------------------------ #

    def start(self):
        def _callback(indata, frames, t, status):
            chunk = indata[:, 0]
            with self._lock:
                self._buf.extend(chunk.tolist())
                # Downsample latest chunk for waveform display
                n = len(chunk)
                if n >= _WAVEFORM_SAMPLES:
                    step = n // _WAVEFORM_SAMPLES
                    self._waveform = chunk[::step][:_WAVEFORM_SAMPLES].copy()
                else:
                    self._waveform = np.roll(self._waveform, -n)
                    self._waveform[-n:] = chunk

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------ #

    def get_audio(self) -> tuple[np.ndarray | None, float]:
        """Return (audio_array, fill_ratio). audio is None if not full yet."""
        with self._lock:
            ratio = len(self._buf) / self._window
            if len(self._buf) < self._window:
                return None, ratio
            return np.array(self._buf, dtype=np.float32), 1.0

    def get_waveform(self) -> np.ndarray:
        with self._lock:
            return self._waveform.copy()

    @property
    def buffered_ratio(self) -> float:
        with self._lock:
            return len(self._buf) / self._window


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def list_input_devices() -> list[tuple[int, str]]:
    devices = sd.query_devices()
    return [
        (i, d["name"])
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def analyze_file(path: str) -> np.ndarray:
    """Load an audio file resampled to SAMPLE_RATE, mono."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def is_silence(audio: np.ndarray, threshold: float = SILENCE_RMS) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) < threshold
