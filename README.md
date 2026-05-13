# Voice Emotion Analyzer

A real-time speech emotion recognition and voice stress analysis desktop application built with Python and PyTorch.

**Developed by:** Amulya Prasad  
**Advisor:** Dr. Pramod Abhichandani  
**Course:** Independent Study (CS 488) — NJIT

---

## Features

- **Live emotion classification** — Happy, Sad, Angry, Neutral with per-class confidence bars
- **EMA-smoothed scores** — exponential moving average removes frame-to-frame jitter
- **Confidence indicator** — flags when the model is uncertain (top score < 40%)
- **Real-time waveform** — live oscilloscope display of your microphone signal
- **60-second emotion timeline** — scrolling line chart showing how emotion shifts over time
- **Voice stress analysis** — pitch elevation, voice tremor, amplitude tremor, hesitation (calibrated to your own baseline)
- **Spectral shift tracking** — additional stress feature based on spectral centroid deviation
- **Session recording** — every hop is logged and exportable to JSON and CSV
- **Audio file analysis** — analyze any `.wav / .mp3 / .flac / .ogg / .m4a` file from the GUI
- **Tabbed interface** — Live · History · Settings panels
- **Mic device selector** — choose your input device in the Settings tab
- **CLI mode** — headless terminal dashboard with the same analysis pipeline

---

## Architecture

```
voice-emotion-analyzer/
├── core/                   # Business logic — no UI dependency
│   ├── config.py           # All constants and label maps
│   ├── audio.py            # AudioStream (live capture), analyze_file, list_input_devices
│   ├── model.py            # load_model, classify, EMA smooth_emotions, confidence_level
│   ├── stress.py           # StressAnalyzer — feature extraction + calibration + scoring
│   └── session.py          # SessionRecorder — JSON / CSV export
│
├── ui/                     # Tkinter presentation layer
│   ├── theme.py            # Palette, fonts, dimensions
│   ├── widgets.py          # BarRow, WaveformCanvas, TimelineCanvas
│   └── app.py              # App(tk.Tk) — Live / History / Settings tabs
│
├── app.py                  # GUI entry point
├── emotion_recognition.py  # CLI entry point  (--file, --device flags)
└── collect_data.py         # Research data collection script
```

---

## Setup (Windows)

**Requirements:** Python 3.10 or later

```
git clone https://github.com/amulya033/voice-emotion-analyzer.git
cd voice-emotion-analyzer
setup.bat        # installs all dependencies
run.bat          # launches the GUI
```

> First run downloads the model (~360 MB). After that it loads from cache instantly.

### Manual install

```bash
pip install -r requirements.txt
python app.py
```

---

## CLI Usage

```bash
# Live microphone analysis
python emotion_recognition.py

# Specify a microphone device
python emotion_recognition.py --device 1

# Analyze an audio file
python emotion_recognition.py --file recording.wav
```

---

## Model

| Property | Value |
|---|---|
| Model | `superb/wav2vec2-base-superb-er` |
| Dataset | IEMOCAP (Interactive Emotional Dyadic Motion Capture) |
| Labels | Happy · Neutral · Sad · Angry |
| Window | 3 seconds, 1 second hop |
| Sample rate | 16 kHz |
| Backend | GPU (CUDA) if available, otherwise CPU |

---

## Dependencies

| Package | Purpose |
|---|---|
| torch | PyTorch inference |
| transformers | HuggingFace model loading |
| sounddevice | Real-time microphone capture |
| librosa | Acoustic feature extraction (pitch, jitter, shimmer) |
| numpy | Audio array operations |
| scipy | Signal processing utilities |

---

## Notes

- Speak normally for ~6 seconds at startup to calibrate your personal stress baseline
- The "deception likelihood" score is a voice stress indicator only and has no forensic validity
- Session exports are saved to `sessions/` (gitignored)
