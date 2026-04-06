# Voice Emotion Analyzer

A real-time speech emotion recognition and voice stress analysis desktop application built with Python and PyTorch.

**Developed by:** Amulya Prasad  
**Advisor:** Dr. Pramod Abhichandani — New Jersey Institute of Technology  
**Course:** Independent Study (CS 488)

---

## What it does

- Listens to your microphone in real time
- Classifies your voice into emotions (Happy, Sad, Angry, Neutral) with percentage scores
- Analyzes voice stress indicators (pitch elevation, voice tremor, amplitude tremor, hesitation)
- Updates live every second

The emotion model uses **wav2vec2-base-superb-er** fine-tuned on the **IEMOCAP** dataset.

---

## Setup (Windows)

**Requirements:** Python 3.9 or later

1. Clone the repo
2. Double-click `setup.bat` — installs all dependencies automatically
3. Double-click `run.bat` to launch the app

> The first run downloads the model (~360 MB). After that it loads instantly from cache.

---

## Dependencies

| Package | Purpose |
|---|---|
| torch | PyTorch inference backend |
| transformers | HuggingFace model loading |
| sounddevice | Real-time microphone capture |
| librosa | Acoustic feature extraction |
| numpy | Audio array operations |

---

## Notes

- Speak normally for ~6 seconds at startup to calibrate your personal stress baseline
- GPU is used automatically if available (CUDA), otherwise runs on CPU
- The deception likelihood score is a voice stress indicator only — not a validated lie detector
