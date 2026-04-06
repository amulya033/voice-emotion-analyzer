# Real-Time Speech Emotion Recognition and Voice Stress Analysis System

**Student:** Amulya Prasad  
**Student ID:** 31593937  
**Advisor:** Dr. Pramod Abhichandani  
**Institution:** New Jersey Institute of Technology  
**Course:** Independent Study (CS 488)  
**Date:** April 2026  

---

## Abstract

This report presents the design, development, and evaluation of a real-time speech emotion recognition and voice stress analysis system. The system leverages a pre-trained transformer-based model — wav2vec2-base-superb-er, fine-tuned on the IEMOCAP dataset — to classify emotional states from continuous microphone input. In parallel, a custom acoustic feature extraction module analyzes fundamental frequency, jitter, shimmer, and pause ratio to generate a voice stress profile calibrated to the individual speaker. The system produces per-emotion probability distributions and a deception likelihood score updated every second over a rolling three-second audio window, presented through a purpose-built desktop graphical interface. Development spanned ten weeks and encompassed literature review, model evaluation, real-time pipeline engineering, and iterative refinement. The result is a fully functional, low-latency desktop application written in Python using PyTorch, the HuggingFace Transformers library, and tkinter, capable of running on consumer hardware without GPU acceleration.

---

## 1. Introduction

Human communication is rich with affective information. Beyond the literal content of speech, vocal characteristics such as pitch, tempo, energy, and micro-variations in amplitude encode emotional states that listeners intuitively interpret in real time. Automating this process — enabling a machine to classify emotion from raw audio — is a long-standing challenge in affective computing with practical applications in mental health monitoring, human-computer interaction, security screening, call center analytics, and accessibility tools.

This independent study project set out to build a real-time speech emotion recognition (SER) system that operates on live microphone input, produces probability scores across multiple emotional categories, and augments that analysis with a voice stress module inspired by acoustic correlates of cognitive load and deception. The project was motivated by an interest in the intersection of deep learning and human behavioral analysis, and by the recent accessibility of high-quality pre-trained speech models produced by self-supervised learning research.

### 1.1 Problem Statement

Despite rapid advances in SER research, most systems remain confined to academic benchmarks, operating on pre-recorded audio clips under controlled conditions. Deploying such a system in real time introduces a distinct set of engineering challenges: low-latency audio capture, continuous windowed inference, speaker-specific normalization, and responsive user feedback. This project addresses all four, delivering a deployable desktop application that operates at inference cadences of one second or better.

### 1.2 Objectives

The primary objectives of this project were:

1. Survey the landscape of SER models, datasets, and acoustic feature methodologies.
2. Select and evaluate pre-trained models for real-time applicability on consumer hardware.
3. Engineer a low-latency audio pipeline capable of processing continuous microphone input.
4. Implement a secondary voice stress analysis module using handcrafted acoustic features with personal baseline calibration.
5. Design an informative, user-facing graphical interface that presents results clearly and updates in real time.

The final system fulfills all five objectives and is distributed as an open-source repository with automated setup scripts for Windows.

---

## 2. Background and Literature Review

### 2.1 Speech Emotion Recognition

Speech emotion recognition (SER) is the task of automatically inferring a speaker's emotional state from audio signals. The field has evolved through three broad generations of methodology.

**First generation (1990s–2000s):** Early approaches relied entirely on handcrafted acoustic features fed into classical classifiers. Mel-frequency cepstral coefficients (MFCCs), zero-crossing rate, spectral centroid, spectral rolloff, and pitch contours were computed per frame and summarized as utterance-level statistics (mean, variance, range). These features were fed to support vector machines (SVMs), Gaussian mixture models (GMMs), or hidden Markov models (HMMs). While foundational, these methods suffered from limited generalization across speakers and recording conditions.

**Second generation (2010s):** Deep learning architectures replaced manual feature engineering. Convolutional neural networks (CNNs) applied directly to mel spectrograms learned spatial representations of frequency-time patterns. Recurrent architectures — long short-term memory networks (LSTMs) and gated recurrent units (GRUs) — captured temporal dynamics across frames. Attention mechanisms allowed models to weight the most emotionally informative segments of an utterance. Performance on standard benchmarks improved substantially.

**Third generation (2019–present):** Self-supervised pre-training on massive unlabeled audio corpora produced general-purpose speech representations that transfer effectively to SER with minimal fine-tuning data. This approach — exemplified by wav2vec 2.0, HuBERT, and WavLM — represents the current state of the art. The model used in this project belongs to this generation.

### 2.2 Self-Supervised Learning for Speech: wav2vec 2.0

wav2vec 2.0, introduced by Baevski et al. (2020) at Meta AI Research, is a self-supervised framework that learns speech representations directly from raw audio waveforms without labeled data. Its architecture consists of three components:

**Feature encoder:** A stack of temporal convolutional layers transforms the raw audio waveform into a sequence of latent feature vectors at a rate of approximately one vector per 20 milliseconds.

**Transformer encoder:** A 12- or 24-layer transformer processes the full sequence of latent vectors, building contextualized representations that incorporate information from the entire utterance.

**Quantization module:** During pre-training, target representations are discretized into a finite codebook. The model is trained to identify the correct quantized target for masked time steps — a contrastive objective that forces the model to learn linguistically and acoustically meaningful representations.

```
┌─────────────────────────────────────────────────────────────┐
│                    wav2vec 2.0 Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Raw Audio (16 kHz waveform)                               │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │  Feature    │  7-layer temporal CNN                     │
│   │  Encoder    │  stride: 5,2,2,2,2,2,2                   │
│   └──────┬──────┘  output: 512-dim @ 50 Hz                  │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ Transformer │  12 layers, 768 hidden dim                │
│   │  Encoder    │  8 attention heads                        │
│   └──────┬──────┘  output: contextualized representations   │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ Classifier  │  fine-tuned on IEMOCAP                   │
│   │    Head     │  output: 4-class softmax                  │
│   └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

        Figure 1: wav2vec 2.0 model architecture as used
        in this project (base configuration).
```

After pre-training on 960 hours of unlabeled LibriSpeech audio, the model is fine-tuned for downstream tasks by appending a task-specific head and training on labeled data. For emotion recognition, a linear classification layer produces a probability distribution over emotion classes.

### 2.3 The IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset, introduced by Busso et al. (2008) at the University of Southern California, is one of the most widely used benchmarks in SER research. It contains approximately twelve hours of audiovisual data recorded from ten actors (five male, five female) performing scripted and improvised emotional dialogues in dyadic (two-person) sessions.

Audio recordings are segmented into utterances and annotated by multiple human evaluators, with both categorical labels (angry, happy, neutral, sad, excited, frustrated, disgust, surprised, fear, other) and dimensional scores on valence, activation, and dominance scales. The categorical labels are typically reduced to four primary classes — angry, happy, neutral, and sad — for benchmark evaluation, with excited utterances merged into happy.

IEMOCAP's dyadic, conversational structure distinguishes it from isolated utterance databases and makes it particularly relevant for the naturalistic, continuous speech scenarios targeted by this project.

| Property | Value |
|---|---|
| Duration | ~12 hours |
| Sessions | 5 dyadic sessions |
| Actors | 10 (5M, 5F) |
| Utterances | ~10,000 |
| Emotion classes (used) | Angry, Happy, Neutral, Sad |
| Annotation | Multi-annotator categorical + dimensional |
| Modalities | Audio, video, motion capture |

*Table 1: IEMOCAP dataset properties.*

### 2.4 The RAVDESS Dataset

During the model evaluation phase, the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), introduced by Livingstone and Russo (2018), was also examined. RAVDESS contains 24 professional actors vocalizing two lexically matched statements across eight emotional categories: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Its broad emotion vocabulary was initially appealing. However, as documented in Section 4.2, the primary model trained on RAVDESS exhibited classifier head architecture incompatibilities with the current HuggingFace Transformers library version, leading to unreliable predictions.

| Property | Value |
|---|---|
| Duration | ~3.6 hours |
| Actors | 24 (12M, 12F) |
| Utterances | 7,356 |
| Emotion classes | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| Setting | Professional studio, controlled |
| Modalities | Audio, video |

*Table 2: RAVDESS dataset properties.*

### 2.5 The SUPERB Benchmark

The Speech processing Universal PERformance Benchmark (SUPERB), introduced by Yang et al. (2021), provides a standardized evaluation framework for pre-trained speech models across a suite of tasks including automatic speech recognition, speaker verification, keyword spotting, intent classification, and emotion recognition. The emotion recognition task in SUPERB uses IEMOCAP and evaluates models on unweighted average recall (UAR) across the four primary emotion classes.

The model used in this project — superb/wav2vec2-base-superb-er — is the official SUPERB emotion recognition checkpoint, achieving competitive UAR on the IEMOCAP test set.

### 2.6 Voice Stress Analysis

Voice stress analysis (VSA) refers to the extraction of acoustic features hypothesized to correlate with psychological stress, cognitive load, and deceptive intent. The theoretical basis draws from psychophysiology: stress activates the autonomic nervous system, affecting laryngeal muscle tension, respiratory patterns, and vocal fold vibration. Observable acoustic consequences include elevated fundamental frequency (F0), increased cycle-to-cycle instability in pitch (jitter), increased cycle-to-cycle variation in amplitude (shimmer), and changes in speaking rate and pause patterns.

Key acoustic features used in VSA and their theoretical basis are summarized below:

| Feature | Definition | Stress Association |
|---|---|---|
| F0 (Fundamental Frequency) | Rate of vocal fold vibration (pitch) | Elevated under stress due to increased laryngeal tension |
| Jitter | Cycle-to-cycle variation in F0 period | Increased by irregular vocal fold vibration under stress |
| Shimmer | Cycle-to-cycle variation in amplitude | Increased by irregular subglottal pressure under stress |
| Pause Ratio | Fraction of speech window with silence | Increases with hesitation and cognitive load |
| Speaking Rate | Syllables or words per second | May increase (anxiety) or decrease (deliberation) under stress |

*Table 3: Acoustic features used in the voice stress analysis module.*

VSA tools have been commercially marketed since the 1970s (Psychological Stress Evaluator, CVSA). However, the scientific literature on their reliability for deception detection is deeply skeptical. A landmark 2003 report by the National Research Council concluded there is no credible scientific evidence supporting voice-based lie detection. Subsequent meta-analyses have confirmed this assessment. This project implements VSA as an educational demonstration of acoustic feature analysis, explicitly framed in the interface as non-validated.

---

## 3. System Architecture

The system is organized into four main components that operate in a coordinated real-time loop.

```
┌──────────────────────────────────────────────────────────────────┐
│                        System Architecture                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                                               │
│   │  Microphone  │                                               │
│   └──────┬───────┘                                               │
│          │  raw PCM (16 kHz, float32, mono)                      │
│          ▼                                                       │
│   ┌──────────────────────────────────────────┐                   │
│   │           Audio Capture Pipeline          │                   │
│   │   sounddevice InputStream (callback)      │                   │
│   │   → Thread-safe deque (3s rolling buffer) │                   │
│   │   → RMS silence gate                      │                   │
│   └──────┬──────────────────────────┬─────────┘                   │
│          │                          │                             │
│          ▼                          ▼                             │
│   ┌──────────────┐         ┌────────────────────┐                │
│   │   Emotion    │         │  Voice Stress       │                │
│   │ Classifier   │         │  Analyzer           │                │
│   │              │         │                     │                │
│   │ wav2vec2     │         │  librosa:           │                │
│   │ HuggingFace  │         │  - YIN (F0/pitch)   │                │
│   │ pipeline     │         │  - Jitter           │                │
│   │              │         │  - Shimmer          │                │
│   │ → softmax    │         │  - Pause ratio      │                │
│   │   over 4     │         │  → Personal         │                │
│   │   emotions   │         │    baseline norm    │                │
│   └──────┬───────┘         └────────┬────────────┘                │
│          │                          │                             │
│          └──────────┬───────────────┘                             │
│                     ▼                                             │
│   ┌──────────────────────────────────────────┐                   │
│   │              Display Layer                │                   │
│   │   tkinter GUI (polling every 200ms)       │                   │
│   │   - Emotion bars + overall verdict        │                   │
│   │   - Stress bars + deception score         │                   │
│   └──────────────────────────────────────────┘                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

        Figure 2: High-level system architecture showing
        data flow from microphone to GUI output.
```

### 3.1 Audio Capture Pipeline

Audio is captured from the system's default microphone using the sounddevice library, which wraps the PortAudio cross-platform audio I/O library. A non-blocking InputStream is opened with a callback function that appends incoming samples to a thread-safe rolling buffer implemented as a Python deque with a maximum length of 48,000 samples (three seconds at 16 kHz). The callback executes in a dedicated audio thread managed by PortAudio.

Every second, the main inference thread acquires a snapshot of the buffer contents under a threading lock, computes the root-mean-square (RMS) energy to gate silence, and passes the audio array to both analysis modules. The one-second hop with three-second window means consecutive analysis windows overlap by two seconds, providing temporal smoothing without sacrificing update frequency.

```
  Time →
  ├────────────────────────────────────────────────────────────┤
  │  Window 1: [0s ──────────────── 3s]                        │
  │  Window 2:     [1s ──────────────── 4s]                    │
  │  Window 3:         [2s ──────────────── 5s]                │
  │  Window 4:             [3s ──────────────── 6s]            │
  │         ↑                                                  │
  │    1s hop (inference trigger)                              │
  └────────────────────────────────────────────────────────────┘

        Figure 3: Rolling window strategy. Each window is
        3 seconds wide with a 1-second hop between inferences.
        Windows overlap by 2 seconds for temporal smoothing.
```

### 3.2 Emotion Classification Module

The emotion classification module wraps the pre-trained model in a HuggingFace `pipeline` with task `audio-classification`. On each inference cycle, the 48,000-sample float32 numpy array is passed to the pipeline, which applies the model's internal feature extractor (16 kHz normalization), runs the transformer forward pass, and returns softmax probability scores for all four emotion classes. GPU inference is used automatically when a CUDA-capable device is detected; otherwise inference runs on CPU.

### 3.3 Voice Stress Analysis Module

The `StressAnalyzer` class computes four acoustic features per window using librosa, then scores each relative to a personal baseline established during the first six windows of voiced speech.

**Pitch Extraction (YIN Algorithm):** The YIN algorithm (de Cheveigné and Kawahara, 2002) estimates the fundamental frequency frame by frame by minimizing a difference function derived from the autocorrelation of the signal. It was selected over simpler autocorrelation methods for its robustness to harmonics and background noise. Frames with F0 below 60 Hz are classified as unvoiced and excluded.

**Jitter Calculation:** Jitter is computed as the mean absolute difference between consecutive voiced F0 values, normalized by the mean F0:

```
           1/(N-1) × Σ|F0[i] - F0[i-1]|
  Jitter = ─────────────────────────────
                    mean(F0)
```

**Shimmer Calculation:** Shimmer is computed from short-time RMS energy frames (512-sample window, 256-sample hop). The mean absolute difference between consecutive non-zero RMS frames is normalized by the mean RMS:

```
             1/(M-1) × Σ|RMS[j] - RMS[j-1]|
  Shimmer = ──────────────────────────────────
                       mean(RMS)
```

**Pause Ratio:** The fraction of RMS frames falling below the silence energy threshold quantifies the proportion of the window containing silence or near-silence.

**Personal Baseline Normalization:** Each feature is scored as a ratio relative to the speaker's own calibration-phase baseline, clipped to [0, 1]:

```
               (current / baseline − 1.0)
  score = clip(──────────────────────────, 0, 1)
                        scale
```

The scale parameter differs per feature to reflect the typical dynamic range of stress-induced changes. The four feature scores are combined as a weighted sum to produce the overall stress score.

### 3.4 Graphical User Interface

The GUI is built with Python's standard tkinter library using a dark theme (background: `#111827`) designed for readability in varied lighting conditions. Two primary panels display emotion and stress results using custom bar widgets built from tk.Frame elements with dynamically resized fill regions — providing precise color control without the constraints of ttk.Progressbar's native styling.

```
  ┌─────────────────────────────────────────────────────┐
  │  Voice Emotion Analyzer                  21:23:36   │
  ├─────────────────────────────────────────────────────┤
  │  Status: Listening...                               │
  ├─────────────────────────────────────────────────────┤
  │  EMOTION ANALYSIS                                   │
  │  ┌──────────────────────────────────────────────┐  │
  │  │ Happy    [████████████████████░░░░░░]  78.4% │  │
  │  │ Neutral  [████░░░░░░░░░░░░░░░░░░░░░]  16.2% │  │
  │  │ Sad      [█░░░░░░░░░░░░░░░░░░░░░░░░]   3.9% │  │
  │  │ Angry    [░░░░░░░░░░░░░░░░░░░░░░░░░]   1.5% │  │
  │  │                                              │  │
  │  │ Overall: HAPPY  (78.4%)                      │  │
  │  └──────────────────────────────────────────────┘  │
  │                                                     │
  │  VOICE STRESS INDICATORS                            │
  │  ┌──────────────────────────────────────────────┐  │
  │  │ Pitch Elevation  [████████░░░░░░░░░]  29.7%  │  │
  │  │ Voice Tremor     [██████░░░░░░░░░░░]  23.0%  │  │
  │  │ Amplitude Tremor [████░░░░░░░░░░░░░]  12.4%  │  │
  │  │ Hesitation       [█████████████████]  74.5%  │  │
  │  │                                              │  │
  │  │ Deception likelihood: MODERATE  (33.2%)      │  │
  │  │ * not accurate, just a fun feature :)        │  │
  │  └──────────────────────────────────────────────┘  │
  │                    [ Stop ]                         │
  └─────────────────────────────────────────────────────┘

        Figure 4: GUI layout showing the emotion analysis
        and voice stress indicator panels during active inference.
```

The GUI polls a shared results dictionary every 200 milliseconds via tkinter's `after()` scheduling mechanism, updating bar widths and label text without blocking the main event loop. This poll-based design ensures thread safety without requiring tkinter to be called from background threads, which is unsupported on most platforms.

---

## 4. Implementation

### 4.1 Development Environment

The project was developed on Windows 11 using Python 3.14. All dependencies are installable via pip and documented in `requirements.txt`. A `setup.bat` script automates virtual environment creation and dependency installation for end users.

| Dependency | Version | Purpose |
|---|---|---|
| torch | ≥ 2.0 | PyTorch inference backend |
| transformers | ≥ 4.35 | HuggingFace model loading and pipeline API |
| sounddevice | ≥ 0.4.6 | Cross-platform real-time audio I/O |
| librosa | ≥ 0.10 | Acoustic feature extraction |
| numpy | ≥ 1.24 | Numerical array operations |

*Table 4: Project dependencies.*

### 4.2 Model Selection and Evaluation

Three candidate models were evaluated for the emotion classification task:

| Model | Dataset | Emotions | Architecture | Loads Cleanly | Selected |
|---|---|---|---|---|---|
| ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition | RAVDESS | 8 | wav2vec2-large-xlsr | No (head mismatch) | ✗ |
| superb/wav2vec2-base-superb-er | IEMOCAP | 4 | wav2vec2-base | Yes | ✓ |
| speechbrain/emotion-recognition-wav2vec2-IEMOCAP | IEMOCAP | 4 | wav2vec2-base | Requires SpeechBrain | ✗ |

*Table 5: Model evaluation summary.*

The ehcalabres model was eliminated due to a classifier head architecture mismatch with the current HuggingFace `Wav2Vec2ForSequenceClassification` implementation. The model checkpoint saved weights under `classifier.dense` and `classifier.output` keys, while the current library expects `projector` and `classifier` layer names. The HuggingFace LOAD REPORT flagged these as UNEXPECTED and MISSING respectively, indicating the classification head was randomly initialized rather than loaded from the checkpoint. This produced unreliable, effectively random emotion predictions despite the base wav2vec2 layers loading correctly.

The SpeechBrain model required installation of the SpeechBrain library as an additional dependency, introducing complexity for end-user setup. The SUPERB model (superb/wav2vec2-base-superb-er) loaded without warnings, integrated cleanly with the standard HuggingFace pipeline API, and required no additional dependencies beyond `transformers` and `torch`.

### 4.3 Threading Architecture

The application uses two threads:

**Main thread (tkinter event loop):** Manages the GUI, handles user interactions, and polls the shared results dictionary every 200 ms via `after()`.

**Analysis thread (daemon):** Manages audio capture and inference. Runs the sounddevice InputStream context, processes audio windows, and writes results to the shared dictionary under a threading lock.

```
  Main Thread                    Analysis Thread
  ────────────                   ───────────────
  tkinter mainloop()             while running:
       │                              │
       ├── after(200ms, poll)         ├── sleep(1s)
       │        │                     │
       │        ▼                     ├── acquire buffer lock
       │   read results dict ◄────────┼── write results dict
       │        │                     │
       │   update bars/labels         ├── classify emotions
       │                              ├── analyze stress
       │                              └── (repeat)
       └── (repeat)

        Figure 5: Threading model showing interaction between
        the main GUI thread and the background analysis thread.
```

### 4.4 Silence Gating and Microphone Calibration

A critical practical challenge was silence detection. The initial RMS threshold of 0.01 — derived from general guidance — was too high for the test hardware, which operated at a peak RMS of approximately 0.00048 during normal speech. This caused the system to classify all audio as silence and never proceed to inference.

Diagnosis was accomplished by modifying the silence message to display the live RMS value:

```
[silence — mic RMS: 0.00048, threshold: 0.001]
```

This revealed that the microphone's output was two orders of magnitude below the initial threshold. The threshold was lowered to 0.0002, resolving the issue. This experience motivated the design of a diagnostic utility (`mic_test.py`) that records three seconds of audio and reports RMS and peak levels with a recommended threshold value, included in the repository for future users.

---

## 5. Project Timeline

The project ran from January 31 to April 5, 2026 — a period of ten weeks structured into five phases.

```
Phase                        Jan  Feb  Feb  Mar  Mar  Apr
                             31   14   28   14   28   5
─────────────────────────────┼────┼────┼────┼────┼────┤
1. Literature Review         ████████                  │
2. Model Evaluation                  ████████          │
3. Core Implementation                    ████████████ │
4. Stress Analysis Module                         ████ │
5. Testing & Refinement                               ██

        Figure 6: Project timeline (Gantt chart).
        Each block represents approximately one week.
```

**Phase 1 — Literature Review and Scoping (January 31 – February 13)**

The first two weeks were dedicated to surveying the academic and technical landscape of speech emotion recognition. This included reading foundational papers on wav2vec 2.0, HuBERT, and WavLM; reviewing the IEMOCAP and RAVDESS dataset documentation; examining the SUPERB benchmark leaderboard; and exploring existing open-source SER implementations on HuggingFace and GitHub. The acoustic psychophysiology of stress was also reviewed to understand the theoretical basis for jitter, shimmer, and F0 as stress correlates. By the end of this phase, the project scope was defined: a real-time, single-speaker system targeting consumer hardware without requiring model training, with a secondary voice stress analysis module as a distinguishing feature.

**Phase 2 — Model Evaluation and Architecture Design (February 14 – February 28)**

Three pre-trained models were evaluated against the criteria described in Section 4.2. The RAVDESS-trained model was tested first due to its broader emotion vocabulary, but was eliminated after the architecture mismatch issue was identified and confirmed. The SUPERB model was selected as the primary classifier. Concurrently, the overall system architecture was designed: rolling buffer strategy, threading model, dual-module inference loop, and GUI layout. The decision to use tkinter rather than a web-based frontend (e.g., Flask + HTML) was made to keep the distribution simple — a single Python file with no server process.

**Phase 3 — Core Implementation (March 1 – March 21)**

The three-week core implementation phase produced the real-time audio pipeline and emotion classification system. Implementation proceeded in layers: sounddevice InputStream and callback, thread-safe deque buffer, silence energy gate, HuggingFace pipeline integration, and the tkinter GUI framework. The custom bar widget (`BarRow`) was developed iteratively — initial attempts using `ttk.Progressbar` were abandoned due to the inability to set custom bar colors through the ttk styling API on Windows. The final implementation uses dynamically resized `tk.Frame` elements as fill bars, providing full color control. The ANSI-based terminal interface from the initial prototype was preserved as `emotion_recognition.py` for command-line use.

**Phase 4 — Voice Stress Analysis Module (March 22 – April 1)**

The `StressAnalyzer` class was designed and implemented as a parallel analysis stream alongside the emotion classifier. This phase required in-depth engagement with the librosa library: specifically, the `yin()` pitch estimator, `feature.rms()` for short-time energy frames, and the design of the normalization and scoring functions. The personal baseline calibration mechanism was the most conceptually significant design decision of this phase: without speaker normalization, absolute feature values would be meaningless for comparison across individuals. The calibration UI — a progress bar in the stress panel that fills during the first six voiced windows — was designed to communicate the calibration status clearly to the user.

**Phase 5 — Testing, Refinement, and Documentation (April 2 – April 5)**

The final phase addressed system-level issues discovered through sustained use: the microphone RMS threshold problem (Section 4.4), abbreviated model output labels (hap, neu, ang expanded to full words), emoji removal for terminal compatibility, display layout adjustments, and the addition of a Start/Stop button to the GUI. The setup and run batch scripts were authored and tested on a clean Python installation. The report and repository were finalized.

---

## 6. Results and Evaluation

### 6.1 Emotion Classification Performance

The superb/wav2vec2-base-superb-er model was evaluated across five controlled vocal scenarios. For each scenario, the speaker sustained a distinct vocal style for 20 seconds while the system recorded per-window emotion probability distributions. Table 6 reports the average scores across all windows collected per scenario (78 total samples across all scenarios).

| Scenario | Dominant Class | Happy | Neutral | Sad | Angry |
|---|---|---|---|---|---|
| Neutral (calm, normal speech) | Happy | 64.9% | 17.9% | 16.7% | 0.5% |
| Happy (enthusiastic, excited) | Neutral | 33.8% | 38.9% | 23.0% | 4.3% |
| Angry (raised, tense voice) | Angry | 34.1% | 15.3% | 5.1% | 45.5% |
| Sad (slow, quiet speech) | Happy | 35.2% | 34.7% | 29.7% | 0.3% |
| Hesitant (pauses, filler words) | Happy | 60.8% | 26.2% | 12.7% | 0.2% |

*Table 6: Measured emotion classification results across five vocal scenarios (78 samples, CPU inference, no GPU).*

These results reveal important characteristics of the model's behavior. The Angry scenario was the only one where the expected class achieved clear dominance (45.5%), suggesting the model is most discriminative for high-energy negative affect. The Neutral and Hesitant scenarios were both dominated by Happy, likely because the speaker's natural conversational tone shares prosodic features — steady pace, moderate pitch — with the IEMOCAP "happy" class. The Happy scenario counterintuitively returned Neutral as dominant (38.9%), with Happy second (33.8%), suggesting the model was trained on acted emotional speech with more exaggerated affect than natural conversation. The Sad scenario produced near-even distribution across Happy, Neutral, and Sad, consistent with the known difficulty of distinguishing low-arousal states. These results highlight that the model performs best on high-arousal, distinctly different states (angry vs. neutral) and struggles with subtle naturalistic emotion, a known limitation in the SER literature.

### 6.2 Voice Stress Analysis Behavior

The stress module was evaluated over the same five scenarios used for Table 6. Table 7 reports the average value of each stress indicator per scenario, measured after personal baseline calibration.

| Scenario | Pitch Elevation | Voice Tremor | Amplitude Tremor | Hesitation | Overall |
|---|---|---|---|---|---|
| Neutral | 0.9% | 33.2% | 1.9% | 64.1% | 23.4% |
| Happy | 7.6% | 34.6% | 8.7% | 28.3% | 20.1% |
| Angry | 19.6% | 34.7% | 13.9% | 24.9% | 24.0% |
| Sad | 11.9% | 36.9% | 12.0% | 43.0% | 25.6% |
| Hesitant | 8.1% | 45.1% | 3.7% | 42.9% | 25.3% |

*Table 7: Measured voice stress indicator values across five vocal scenarios (averages over 20-second recording windows, speaker-normalized to personal baseline).*

Several patterns emerge from the measured data. Voice tremor was consistently elevated across all scenarios (33–45%), suggesting this feature may have been overfit during calibration or is sensitive to normal variation in conversational speech rather than stress specifically. Pitch elevation behaved as expected — highest during the Angry scenario (19.6%) and near-zero during Neutral (0.9%). Hesitation showed the largest scenario-dependent variation: 64.1% during the deliberately slow Neutral reading (likely because natural pausing was classified as hesitation relative to the faster calibration baseline) versus 24.9% during the Angry scenario where speech was more continuous. The overall stress scores were tightly clustered between 20.1% and 25.6% across all scenarios, which reflects the limitations of VSA discussed in Section 2.6 — the acoustic features extracted do not reliably differentiate emotional scenarios at the overall score level when speaker-normalized.

### 6.3 System Performance

System performance was measured across 78 inference cycles collected during the five-scenario evaluation session, running on CPU with no GPU acceleration.

| Metric | Measured Value |
|---|---|
| Total samples collected | 78 |
| Average inference time (CPU) | 0.325 s |
| Minimum inference time | 0.258 s |
| Maximum inference time | 0.509 s |
| Acoustic feature extraction time | ~0.05 – 0.10 s |
| Total cycle time (inference + features) | ~0.375 – 0.610 s |
| GUI poll interval | 200 ms |
| Effective update rate | 1 per second (1 s hop) |
| GPU used | No (CPU only) |
| Model download size | ~360 MB |

*Table 8: Measured system performance metrics (78 samples, CPU-only, Windows 11).*

The average inference time of 0.325 seconds comfortably fits within the one-second hop interval, leaving approximately 0.675 seconds of margin per cycle for feature extraction, GUI updates, and audio buffering. The maximum observed inference time of 0.509 seconds — likely caused by CPU scheduling variance or thermal throttling during the session — also remained within budget. This confirms that the system runs reliably at real-time cadence on consumer hardware without GPU acceleration. With GPU acceleration, inference time would be expected to drop below 100 ms, enabling a hop interval reduction to 0.5 seconds for more responsive updates.

---

## 7. Challenges and Limitations

### 7.1 Microphone Sensitivity and Threshold Calibration

The most immediately practical challenge was microphone gain calibration. The initial silence threshold was derived from general guidance in the sounddevice documentation and proved two orders of magnitude too high for the specific hardware under test. The system rejected all audio as silence, appearing non-functional. The debugging process — adding a live RMS readout to the silence gate message — was straightforward once the issue was framed correctly, but the experience highlighted a broader design gap: the threshold should ideally be determined automatically from a brief ambient noise measurement at startup rather than hardcoded. Automatic gain calibration is identified as a priority for future work.

### 7.2 Model Architecture Compatibility

The initial model selection failed silently due to a weight key mismatch between the saved checkpoint and the current HuggingFace `Wav2Vec2ForSequenceClassification` class. The LOAD REPORT printed by the transformers library flagged this with UNEXPECTED and MISSING keys, but the model still loaded and ran without raising an exception — producing random, uncorrelated predictions. This class of failure is particularly insidious in machine learning deployments because the system appears to function normally while producing meaningless output. It underscores the importance of verifying model loading completeness at integration time, not just checking that the pipeline runs without error.

### 7.3 Emotion Vocabulary

The four-class IEMOCAP vocabulary (angry, happy, neutral, sad) does not cover the full spectrum of emotionally significant states. Nervousness, anxiousness, excitement, surprise, disgust, and fear — all states with distinct vocal correlates — are not represented as discrete classes. Nervous or anxious speech in practice tends to be classified as neutral or angry depending on energy level, which is a meaningful limitation for applications that specifically target those states.

### 7.4 Voice Stress Validity

As established in the literature review, voice stress analysis lacks scientific validation as a deception detection tool. The stress module is designed and framed as an educational feature demonstrating acoustic analysis, not a reliable behavioral inference tool. Its value in this project is in illustrating how acoustic features can be extracted, normalized, and combined into a composite score — a methodology applicable to validated tasks such as clinical tremor monitoring or fatigue detection.

### 7.5 Single-Speaker Assumption

The system assumes a single speaker throughout a session. The calibration baseline is set once at startup and is not updated or extended. In multi-speaker scenarios, a speaker diarization step would be required before emotion or stress analysis could be applied meaningfully per speaker.

---

## 8. Conclusion and Future Work

### 8.1 Conclusion

This project successfully produced a real-time speech emotion recognition and voice stress analysis system built on modern self-supervised speech representations. The system processes continuous microphone input, classifies emotional states with associated probability scores across four classes, and provides a speaker-normalized stress profile across four acoustic dimensions — all updated at one-second intervals through a custom desktop GUI, with no GPU required.

The project provided substantial hands-on experience across the full stack of a machine learning application: model evaluation and selection, real-time audio engineering, acoustic feature extraction, threading and concurrency, GUI development, and open-source distribution. The debugging arc — from model architecture mismatches to microphone threshold calibration — reflected challenges typical of real-world ML systems deployment and reinforced the importance of end-to-end testing beyond benchmark evaluation.

### 8.2 Future Work

Several directions could extend this work meaningfully:

**Broader emotion vocabulary:** Fine-tuning wav2vec2 or a HuBERT model on a dataset with more emotion categories — such as CREMA-D (6 classes) or MSP-IMPROV (4 + dimensional) — would increase expressive range. The RAVDESS-trained model with 8 classes remains a candidate if its architecture compatibility issue is resolved by using an older transformers version.

**Dimensional emotion modeling:** Replacing categorical classification with a valence-arousal-dominance (VAD) regression model (e.g., audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) would enable continuous, nuanced emotional state tracking and would naturally accommodate states like nervousness and anxiousness as combinations of high arousal and low valence.

**Automatic gain calibration:** A brief ambient noise measurement at startup could set the silence threshold automatically, eliminating the manual tuning step that caused significant friction during development.

**Speaker diarization:** Integrating a speaker embedding model (e.g., ECAPA-TDNN) would enable per-speaker emotion and stress tracking in multi-speaker conversations — significantly expanding the range of applicable scenarios.

**Longitudinal emotion tracking:** Visualizing emotion and stress trajectories over time (e.g., with a scrolling matplotlib canvas) would allow users to observe patterns across a conversation rather than only point-in-time snapshots.

**Mobile or web deployment:** Porting the inference pipeline to ONNX or TensorFlow Lite would enable deployment on mobile devices or in the browser via WebAssembly, greatly expanding accessibility.

---

## References

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems*, 33, 12449–12460.

Busso, C., Bulut, M., Lee, C. C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., & Narayanan, S. S. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4), 335–359.

Cao, H., Cooper, D. G., Kuchinsky, M. K., Ghosh, S., Ma, C., & Bhatt, R. (2014). CREMA-D: Crowd-sourced emotional multimodal actors dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.

de Cheveigné, A., & Kawahara, H. (2002). YIN, a fundamental frequency estimator for speech and music. *Journal of the Acoustical Society of America*, 111(4), 1917–1930.

Hsu, W. N., Bolte, B., Tsai, Y. H. H., Lakhotia, K., Salakhutdinov, R., & Mohamed, A. (2021). HuBERT: Self-supervised speech representation learning by masked prediction of hidden units. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29, 3451–3460.

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLOS ONE*, 13(5), e0196391.

McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25.

National Research Council. (2003). *The polygraph and lie detection*. National Academies Press.

Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A review of affective computing: From unimodal analysis to multimodal fusion. *Information Fusion*, 37, 98–125.

Yang, S., Chi, P. H., Chuang, Y. S., Lai, C. I. J., Lakhotia, K., Lin, Y. Y., Liu, A. T., Shi, J., Chang, X., Lin, G. T., Huang, T. H., Tseng, H. J., & Lee, H. Y. (2021). SUPERB: Speech processing universal performance benchmark. *Proceedings of Interspeech 2021*, 1194–1198.
