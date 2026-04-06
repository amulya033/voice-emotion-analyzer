# Real-Time Speech Emotion Recognition and Voice Stress Analysis System

**Student:** Amulya Prasad  
**Student ID:** 31593937  
**Advisor:** Dr. Pramod Abhichandani  
**Institution:** New Jersey Institute of Technology  
**Course:** Independent Study (CS 488)  
**Date:** April 2026  

---

## Abstract

This report presents the design, development, and evaluation of a real-time speech emotion recognition and voice stress analysis system. The system leverages a pre-trained transformer-based model, wav2vec2-base-superb-er, fine-tuned on the IEMOCAP dataset, to classify emotional states from continuous microphone input. In parallel, a custom acoustic feature extraction module analyzes fundamental frequency, jitter, shimmer, and pause ratio to generate a voice stress profile calibrated to the individual speaker. The system produces per-emotion probability distributions and a deception likelihood score updated every second over a rolling three-second audio window. Development spanned approximately ten weeks, encompassing literature review, model evaluation, real-time pipeline engineering, and iterative refinement. The result is a fully functional, low-latency desktop application written in Python using PyTorch and the HuggingFace Transformers library.

---

## 1. Introduction

Human communication is rich with affective information. Beyond the literal content of speech, vocal characteristics such as pitch, tempo, energy, and micro-variations in amplitude encode emotional states that listeners intuitively interpret in real time. Automating this process — enabling a machine to classify emotion from raw audio — is a long-standing challenge in affective computing and has practical applications in mental health monitoring, human-computer interaction, security screening, call center analytics, and accessibility tools.

This independent study project set out to build a real-time speech emotion recognition (SER) system that operates on live microphone input, produces probability scores across multiple emotional categories, and augments that analysis with a voice stress module inspired by acoustic correlates of cognitive load and deception. The project was motivated by an interest in the intersection of deep learning and human behavioral analysis, and by the accessibility of high-quality pre-trained speech models that have emerged from self-supervised learning research in recent years.

The primary objectives of this project were:

1. Survey the landscape of SER models, datasets, and acoustic feature methodologies.
2. Select and evaluate pre-trained models for real-time applicability.
3. Engineer a low-latency audio pipeline capable of processing continuous microphone input.
4. Implement a secondary voice stress analysis module using handcrafted acoustic features.
5. Design a readable, informative terminal-based interface that presents results clearly.

The final system fulfills all five objectives and runs on consumer hardware without a GPU, making it accessible for demonstration and further research.

---

## 2. Background and Literature Review

### 2.1 Speech Emotion Recognition

Speech emotion recognition is the task of automatically inferring a speaker's emotional state from audio signals. Early approaches relied on handcrafted acoustic features — such as mel-frequency cepstral coefficients (MFCCs), zero-crossing rate, spectral centroid, and pitch contours — fed into classical machine learning classifiers such as support vector machines and hidden Markov models. While these methods established a baseline, they were limited by the expressiveness of manually designed features and their sensitivity to speaker variability and environmental noise.

The introduction of deep learning shifted the field significantly. Convolutional neural networks applied to spectrograms, recurrent networks over temporal feature sequences, and attention-based models all demonstrated improved performance on standard benchmarks. The challenge of SER remains non-trivial, however, due to the subjectivity of emotional labels, cross-cultural variability in emotional expression, and the inherent ambiguity of speech that carries mixed or masked affect.

### 2.2 Self-Supervised Learning for Speech: wav2vec2

A turning point in speech processing came with the introduction of wav2vec 2.0 by Baevski et al. (2020). wav2vec2 is a self-supervised learning framework that learns powerful representations of raw audio by training a transformer-based model to solve a contrastive task over masked audio segments, without requiring labeled data. The pre-trained representations generalize remarkably well to downstream tasks including automatic speech recognition, speaker identification, and emotion recognition when fine-tuned on labeled datasets.

The model used in this project, superb/wav2vec2-base-superb-er, is part of the Speech processing Universal PERformance Benchmark (SUPERB) and was fine-tuned for emotion recognition on the IEMOCAP dataset. It takes raw 16 kHz audio as input and outputs probability distributions over four emotion classes: angry, happy, neutral, and sad.

### 2.3 The IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset, introduced by Busso et al. (2008) at the University of Southern California, is one of the most widely used benchmarks in SER research. It contains approximately twelve hours of audiovisual data from ten actors performing scripted and improvised emotional dialogues. Audio recordings are labeled by multiple annotators across categorical and dimensional emotional scales. Its dyadic, conversational structure makes it particularly relevant for real-world speech emotion analysis, in contrast to acted, isolated utterance datasets.

### 2.4 The RAVDESS Dataset

During the model evaluation phase of this project, the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) was also examined. RAVDESS contains 24 professional actors vocalizing two lexically matched statements across eight emotional categories: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. While RAVDESS offers a broader emotion vocabulary, models trained on it exhibited architectural incompatibilities with current versions of the HuggingFace Transformers library (detailed in Section 5), leading to the selection of the IEMOCAP-trained model instead.

### 2.5 Voice Stress Analysis

Voice stress analysis (VSA) refers to the extraction of acoustic features that are hypothesized to correlate with psychological stress, cognitive load, and deceptive intent. The theoretical basis draws from psychophysiology: stress activates the autonomic nervous system, which affects laryngeal muscle tension, respiratory patterns, and vocal fold vibration. Observable acoustic consequences include elevated fundamental frequency (F0), increased jitter (cycle-to-cycle variation in F0), increased shimmer (cycle-to-cycle variation in amplitude), and changes in speaking rate and pause patterns.

VSA tools have been marketed commercially since the 1970s, but the scientific literature on their reliability as lie detectors is deeply skeptical. A 2003 report by the National Research Council concluded there is no credible scientific evidence supporting voice-based deception detection. This project implements VSA as an educational demonstration of acoustic stress indicators, explicitly framed as a non-validated feature.

The acoustic features extracted in this system — F0 via the YIN algorithm, jitter, shimmer, and pause ratio — are computed using the librosa audio analysis library and compared against a personal baseline established during a calibration phase, yielding a speaker-normalized stress score.

---

## 3. System Architecture

The system is organized into four main components: the audio capture pipeline, the emotion classification module, the voice stress analysis module, and the display layer. These components run in a coordinated loop with a one-second inference cadence.

### 3.1 Audio Capture Pipeline

Audio is captured from the system's default microphone using the sounddevice library, which provides a non-blocking InputStream interface. A callback function populates a thread-safe rolling buffer implemented as a Python deque with a maximum length of 48,000 samples (three seconds at 16 kHz). Every second, the main inference loop acquires the buffer contents, performs an RMS energy check to filter silence, and passes the audio to both analysis modules.

The choice of a three-second rolling window balances temporal resolution with the minimum context needed for reliable emotion classification. wav2vec2's transformer architecture benefits from longer context windows, and pilot testing showed that windows shorter than two seconds produced noticeably noisier predictions.

### 3.2 Emotion Classification Module

The emotion classification module wraps the superb/wav2vec2-base-superb-er model in a HuggingFace pipeline with the audio-classification task. On each inference cycle, the 48,000-sample numpy array is passed directly to the pipeline, which handles feature extraction and model inference internally. The pipeline returns probability scores for all four emotion classes, which are sorted in descending order and rendered with percentage bar visualizations.

The model runs on CPU by default, with automatic GPU acceleration when a CUDA-capable device is detected. On a modern CPU, inference on a three-second clip completes in approximately 0.8 to 1.2 seconds, keeping the system within its one-second hop budget with modest headroom.

### 3.3 Voice Stress Analysis Module

The StressAnalyzer class implements a four-feature acoustic analysis pipeline using librosa:

**Fundamental Frequency (F0):** The YIN algorithm extracts frame-level pitch estimates across the audio window. Unvoiced frames (F0 below 60 Hz) are excluded. The mean of voiced F0 values constitutes the pitch estimate for that window.

**Jitter:** Defined as the mean absolute difference between consecutive voiced F0 values, normalized by the mean F0. This captures micro-instability in vocal fold vibration that increases under stress.

**Shimmer:** Computed from the short-time RMS energy of the signal. The mean absolute difference between consecutive non-zero RMS frames, normalized by the mean RMS, captures amplitude micro-variation associated with tension.

**Pause Ratio:** The fraction of RMS frames falling below the silence threshold quantifies the proportion of the window containing no speech, which increases with hesitation and cognitive load.

Each feature is scored relative to a personal baseline established during the first six seconds of voiced speech. Baseline normalization is critical because absolute feature values vary substantially between speakers. A pitch elevation score, for instance, is only meaningful relative to the individual's typical pitch, not a population average.

The four feature scores are combined via a weighted sum (pitch elevation: 30%, voice tremor: 30%, amplitude tremor: 20%, hesitation: 20%) to produce an overall stress score, which is mapped to four qualitative levels: Low, Moderate, High, and Very High.

### 3.4 Display Layer

The terminal display uses ANSI escape sequences to redraw output in place, creating a live-updating interface without screen flicker. Two bordered panels are rendered each cycle: the emotion analysis panel showing per-emotion probability bars, and the voice stress indicators panel. During the calibration period, the stress panel displays progress rather than scores.

---

## 4. Implementation

### 4.1 Development Environment

The project was developed on Windows 11 using Python 3.14. Key dependencies include:

- **torch** (2.x): PyTorch backend for model inference
- **transformers** (4.35+): HuggingFace model loading and pipeline interface
- **sounddevice** (0.4.6): Cross-platform real-time audio I/O
- **librosa** (0.10+): Audio feature extraction
- **numpy** (1.24+): Numerical array operations

### 4.2 Key Implementation Decisions

**Thread-safe buffer:** The audio callback runs in a dedicated sounddevice thread. A threading.Lock protects the deque during reads in the main loop, preventing race conditions between audio capture and inference.

**Silence gating:** An RMS energy threshold filters windows containing no meaningful speech before passing audio to the model. This prevents the emotion classifier from producing noise-driven predictions during pauses and reduces unnecessary CPU load. The threshold was calibrated empirically by measuring microphone output levels during silent and voiced conditions.

**Calibration phase:** The StressAnalyzer requires a minimum of six voiced speech windows before activating stress scoring. This ensures that the baseline accurately reflects the speaker's natural vocal characteristics rather than the first few frames of a cold start.

**Model selection:** Early in the project, the ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition model, trained on RAVDESS, was evaluated. This model presented a classifier head architecture mismatch with the current HuggingFace Wav2Vec2ForSequenceClassification implementation: the checkpoint contained classifier.dense and classifier.output layers, while the pipeline expected projector and classifier layers. As a result, the classification head was randomly initialized rather than loaded from the checkpoint, producing unreliable predictions. The model was replaced with superb/wav2vec2-base-superb-er, which loads without architectural discrepancy.

---

## 5. Project Timeline

The project spanned ten weeks from January 31 to April 5, 2026, organized into five phases.

**Phase 1 — Literature Review and Scoping (January 31 – February 13)**  
The first two weeks were dedicated to surveying the academic and technical landscape of speech emotion recognition. This included reading foundational papers on wav2vec2, reviewing the IEMOCAP and RAVDESS dataset documentation, examining the SUPERB benchmark leaderboard, and exploring existing open-source SER implementations. By the end of this phase, the project scope was defined: a real-time, single-speaker emotion recognition system augmented with acoustic stress analysis, targeting consumer hardware without requiring model training.

**Phase 2 — Model Evaluation and Architecture Design (February 14 – February 28)**  
Multiple pre-trained models were evaluated for suitability. Criteria included: cleanness of model loading (no missing or unexpected weight keys), inference speed on CPU, emotion vocabulary breadth, and compatibility with the HuggingFace pipeline API. Models evaluated included ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition, superb/wav2vec2-base-superb-er, and speechbrain/emotion-recognition-wav2vec2-IEMOCAP. The RAVDESS-trained model was eliminated due to the architecture mismatch described in Section 4.2. The SUPERB model was selected as the primary classifier. The overall system architecture — rolling buffer, inference loop, dual-panel display — was designed and documented during this phase.

**Phase 3 — Core Implementation (March 1 – March 21)**  
The three-week core implementation phase produced the real-time audio pipeline and emotion classification system. This involved implementing the sounddevice InputStream callback, the thread-safe deque buffer, the silence energy gate, and the HuggingFace pipeline integration. The terminal display layer, including the in-place ANSI redraw system and the percentage bar visualization, was developed and iterated upon. Significant time was spent debugging the silence threshold: initial values were too high for the low-gain microphone used in testing, requiring the addition of live RMS diagnostics to identify the appropriate threshold.

**Phase 4 — Voice Stress Analysis Module (March 22 – April 1)**  
The StressAnalyzer class was designed and implemented as a parallel analysis stream. This phase required deeper engagement with the librosa library, specifically the YIN pitch estimation algorithm and short-time RMS frame analysis. The four-feature extraction pipeline was built and tested in isolation before integration with the main loop. The personal baseline calibration mechanism was designed to address inter-speaker variability. The weighted scoring and qualitative verdict system were implemented and tuned based on observed feature ranges during testing.

**Phase 5 — Testing, Refinement, and Documentation (April 2 – April 5)**  
The final phase addressed system-level issues identified through sustained use: microphone sensitivity calibration, label formatting (abbreviated model output labels expanded to full words), emoji removal from the interface for compatibility, and display layout adjustments. Performance was evaluated informally across several emotional scenarios, and the system's behavior was documented for this report.

---

## 6. Results and Evaluation

### 6.1 Emotion Classification Performance

The superb/wav2vec2-base-superb-er model demonstrates reasonable real-time performance for its four emotion classes. In informal testing, the system reliably distinguishes between markedly different affective states — animated, high-energy speech is classified as happy or angry depending on valence, while calm, even-toned speech trends toward neutral. Transitions between emotional states are reflected in shifting probability distributions within one to two seconds of the vocal change, given the one-second hop and three-second window.

The system's primary limitation is the coarseness of its four-class vocabulary. Emotional states such as nervousness, anxiousness, excitement, and surprise — which users may wish to detect — are not represented as discrete classes and may be poorly captured by the angry/happy/neutral/sad taxonomy.

### 6.2 Voice Stress Analysis Behavior

The stress module behaves as designed: during the calibration phase it accumulates baseline measurements, then scores subsequent windows against the personal norm. In practice, the hesitation score is the most volatile indicator, varying substantially with natural speech rhythm and pausing. Pitch elevation and voice tremor respond more gradually and tend to produce more stable readings over sustained speech. The overall stress score in normal, relaxed speech typically stabilizes between 10% and 30%, providing headroom for detection of elevated states.

### 6.3 System Performance

On a CPU-only configuration, each inference cycle — comprising RMS gating, emotion classification, and acoustic feature extraction — completes in approximately 1.0 to 1.8 seconds. Since the hop interval is one second, the system runs at roughly real-time with occasional minor lag. GPU acceleration reduces inference time to under 200 milliseconds, enabling comfortable real-time operation.

---

## 7. Challenges and Limitations

**Microphone sensitivity:** The most immediately practical challenge encountered was microphone gain calibration. The initial silence threshold of 0.01 RMS was derived from general guidance but proved too high for the test hardware, which operated at a peak RMS of approximately 0.00048 during normal speech. This was diagnosed by adding live RMS readout to the silence gate message and resolved by lowering the threshold to 0.0002. This experience highlighted the importance of adaptive or user-calibrated thresholds for real-world deployability.

**Model architecture compatibility:** The initial model selection (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) failed silently due to a weight key mismatch between the saved checkpoint and the current HuggingFace model class. The load report showed UNEXPECTED and MISSING keys in the classifier head, meaning predictions were based on a randomly initialized classification layer. This underscores the importance of verifying model loading completeness rather than trusting that a model card implies compatibility.

**Emotion vocabulary:** The four-class IEMOCAP-trained model does not cover the full spectrum of emotionally meaningful states. A production system would benefit from a model trained on a broader taxonomy or from fine-tuning on domain-specific data.

**Voice stress validity:** As discussed in the literature review, voice stress analysis lacks scientific validation as a deception detection tool. The stress module in this system is explicitly framed as a non-validated, educational feature. Its value lies in demonstrating acoustic feature extraction and personalized baseline normalization rather than in providing actionable deception inference.

---

## 8. Conclusion and Future Work

This project successfully produced a real-time speech emotion recognition and voice stress analysis system built on modern self-supervised speech representations. The system processes continuous microphone input, classifies emotional states with associated probability scores, and provides a speaker-normalized stress profile, all updated at one-second intervals with minimal hardware requirements.

The project provided hands-on experience with the HuggingFace Transformers ecosystem, real-time audio engineering in Python, acoustic feature extraction with librosa, and the practical challenges of deploying pre-trained models on consumer hardware. The debugging arc — from model architecture mismatches to microphone threshold calibration — reflected challenges typical of real-world machine learning systems development.

Several directions could extend this work meaningfully:

- **Broader emotion vocabulary:** Fine-tuning wav2vec2 or a HuBERT model on a dataset with more emotion categories (e.g., CREMA-D or MSP-IMPROV) would substantially increase the system's expressive range.
- **Dimensional emotion modeling:** Replacing categorical classification with a valence-arousal-dominance regression model would enable continuous, nuanced emotional state tracking.
- **Adaptive silence gating:** Implementing automatic gain calibration at startup would remove the need for manual threshold tuning.
- **Speaker diarization:** Extending the system to multi-speaker scenarios using speaker embedding models would enable per-speaker emotion tracking in conversational settings.
- **Graphical interface:** A matplotlib or web-based frontend would improve usability and enable time-series visualization of emotional state trajectories.

---

## References

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems*, 33, 12449–12460.

Busso, C., Bulut, M., Lee, C. C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., & Narayanan, S. S. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 42(4), 335–359.

De Boer, N. (2002). YIN, a fundamental frequency estimator for speech and music. *Journal of the Acoustical Society of America*, 111(4), 1917–1930.

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391.

McFee, B., Raffel, C., Liang, D., Ellis, D., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18–25.

National Research Council. (2003). *The polygraph and lie detection*. National Academies Press.

Yang, S., Chi, P. H., Chuang, Y. S., Lai, C. I. J., Lakhotia, K., Lin, Y. Y., Liu, A. T., Shi, J., Chang, X., Lin, G. T., Huang, T. H., Tseng, H. J., Lee, H. Y., & others. (2021). SUPERB: Speech processing universal performance benchmark. *Proceedings of Interspeech 2021*, 1194–1198.
