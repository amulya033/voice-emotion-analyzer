MODEL_ID      = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE   = 16_000
WINDOW_SEC    = 3
HOP_SEC       = 1
SILENCE_RMS   = 0.0002
CALIBRATION_N = 6
TIMELINE_SECS = 60   # seconds of history shown in the timeline
EMA_ALPHA     = 0.35  # exponential moving average smoothing for emotion scores

# Canonical label order for display
EMOTIONS = ["hap", "neu", "sad", "ang"]

LABEL_EXPAND = {
    "ang": "Angry",   "angry":   "Angry",
    "hap": "Happy",   "happy":   "Happy",
    "neu": "Neutral", "neutral": "Neutral",
    "sad": "Sad",
}

EMOTION_COLORS = {
    "hap": "#10b981", "happy":   "#10b981",
    "sad": "#60a5fa",
    "ang": "#ef4444", "angry":   "#ef4444",
    "neu": "#8b5cf6", "neutral": "#8b5cf6",
}

STRESS_KEYS = [
    "pitch_elevation",
    "voice_tremor",
    "amplitude_tremor",
    "speech_hesitation",
]

STRESS_LABELS = {
    "pitch_elevation":   "Pitch Elevation",
    "voice_tremor":      "Voice Tremor",
    "amplitude_tremor":  "Amplitude Tremor",
    "speech_hesitation": "Hesitation",
}

STRESS_COLORS = {
    "pitch_elevation":   "#f59e0b",
    "voice_tremor":      "#f97316",
    "amplitude_tremor":  "#fb923c",
    "speech_hesitation": "#fbbf24",
}

STRESS_WEIGHTS = {
    "pitch_elevation":   0.30,
    "voice_tremor":      0.30,
    "amplitude_tremor":  0.20,
    "speech_hesitation": 0.20,
}
