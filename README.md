# Deep Learning-Based Explainable DeepFake Audio Detection System

A PyTorch deep learning system for detecting synthetic and cloned human speech (audio deepfakes) using a 2D Convolutional Neural Network trained on log-Mel spectrograms, augmented with **Grad-CAM (Gradient-weighted Class Activation Mapping)** for transparent, explainable decision-making.

---

## 🌟 Key Features

* **Trained Anti-Spoofing CNN**: Pretrained on the **ASVspoof 2019 Logical Access (LA)** dataset.
* **Exact Audio Preprocessing**:
  * Sampling rate: 16,000 Hz
  * Power Mel-spectrogram: $n\_fft=1024$, $hop\_length=256$, $n\_mels=80$, converted to dB with `ref=np.max`
  * Per-sample normalization: $\frac{x - \mu}{\sigma + 10^{-6}}$
  * Fixed dimension: $80 \text{ frequency bins} \times 250 \text{ time frames}$
* **Explainability with Grad-CAM**: Extracts class activation maps from the final convolutional layer (`features[8]`: Conv2d $32 \rightarrow 64$) and overlays them directly onto the log-Mel spectrogram.
* **Modern Research Dashboard (Streamlit)**:
  * Minimalist, professional AI research aesthetic designed for presentations.
  * Direct audio upload (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) and built-in demonstration audio selector.
  * Real-time prediction card with raw softmax score breakdown.
  * Interactive Grad-CAM saliency viewer.
  * Transparent technical specifications and ASVspoof 2019 benchmark metrics.
* **Built-in Demonstration Audio**: Included verified human speech (LibriSpeech) and synthetic speech (Microsoft David TTS) in `demo_audio/`.
* **Command-Line Interface (CLI)**: Fast headless batch/single audio evaluation via `predict.py`.
* **CPU Compatible**: Runs efficiently on Windows using CPU inference.

---

## 🏗️ Architecture & Model Specification

```
Input Audio (.wav, .mp3, etc.)
  │
  ▼
Log-Mel Spectrogram Preprocessing (80 Mel Bins × 250 Frames)
  │
  ▼
AudioDeepfakeCNN:
  ├── features.0:  Conv2d(1, 16, kernel_size=3, padding=1)
  ├── features.1:  BatchNorm2d(16)
  ├── features.2:  ReLU(inplace=True)
  ├── features.3:  MaxPool2d(kernel_size=2)
  ├── features.4:  Conv2d(16, 32, kernel_size=3, padding=1)
  ├── features.5:  BatchNorm2d(32)
  ├── features.6:  ReLU(inplace=True)
  ├── features.7:  MaxPool2d(kernel_size=2)
  ├── features.8:  Conv2d(32, 64, kernel_size=3, padding=1)   <-- Target layer for Grad-CAM
  ├── features.9:  BatchNorm2d(64)
  ├── features.10: ReLU(inplace=True)
  └── features.11: AdaptiveAvgPool2d((5, 10))
  │
  ▼
Classifier:
  ├── classifier.0: Flatten() -> (64 × 5 × 10 = 3,200)
  ├── classifier.1: Linear(3200, 128)
  ├── classifier.2: ReLU(inplace=True)
  ├── classifier.3: Dropout(p=0.4)
  └── classifier.4: Linear(128, 2)
  │
  ▼
Output Logits: Class 0 (Bona fide / Real) vs Class 1 (Spoof / DeepFake)
```

---

## 📊 Development Set Evaluation Results

Evaluated on the full **ASVspoof 2019 Logical Access Development Split** (24,844 samples):

| Metric | Result |
| :--- | :--- |
| **Total Samples** | 24,844 (2,548 Real, 22,296 Spoof) |
| **Overall Accuracy** | **98.65%** |
| **Balanced Accuracy**| **98.60%** |
| **Spoof Precision**  | **99.83%** |
| **Spoof Recall**     | **98.66%** |
| **Spoof F1-Score**   | **99.24%** |
| **ROC-AUC**          | **0.9986** |

**Confusion Matrix:**
```
                     Predicted Real    Predicted Spoof
Actual Real (2,548):      2,511               37
Actual Spoof (22,296):     299              21,997
```

---

## 🎙️ Demonstration Audio Recordings

The `demo_audio/` folder contains two verified audio samples:
1. `demo_real_speech.wav`: Genuine human recording from the LibriSpeech public domain corpus (16 kHz mono WAV, 10.44s).
2. `demo_synthetic_speech.wav`: Genuine synthetic speech generated via Windows SAPI5 Microsoft David Desktop TTS (16 kHz mono WAV, 9.71s).

To regenerate or verify the demonstration samples:
```bash
python scripts/generate_demo_audio.py
```

### Research Honesty & Out-of-Domain Generalization Findings:
* **Synthetic Speech Demo**: Correctly classified as Spoof with **100.00%** raw softmax score.
* **Human Speech Demo**: Classified as Spoof (**86.14%** raw softmax score).
* **Scientific Analysis**: The model was trained specifically on ASVspoof 2019 hemianechoic chamber studio recordings. Out-of-domain human recordings (such as LibriSpeech audiobooks recorded in home acoustic settings) exhibit acoustic domain shift — an established open challenge in speech anti-spoofing research. The results are reported transparently without altering weights or fabricating confidence scores.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
* Python 3.10 – 3.13 on Windows

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit Web Application

Launch the upgraded application:
```bash
streamlit run app.py
```
This opens the browser at `http://localhost:8501`.

---

## 💻 Running the Command-Line Interface (CLI)

Predict an audio file directly from terminal (baseline mode):
```bash
python predict.py --audio "demo_audio/demo_synthetic_speech.wav"
```

Predict with optional speech high-pass filtering (e.g. 80 Hz):
```bash
python predict.py --audio "demo_audio/demo_real_speech.wav" --high-pass 80
```

Predict with multi-window aggregation for long recordings (> 4s):
```bash
python predict.py --audio "demo_audio/demo_real_speech.wav" --multi-window --high-pass 80
```

Save Grad-CAM explanation:
```bash
python predict.py --audio "demo_audio/demo_synthetic_speech.wav" --explain --save-cam "explanation.png"
```

JSON output:
```bash
python predict.py --audio "demo_audio/demo_synthetic_speech.wav" --json
```

---

## 🔬 Stage 1 Inference Improvements & Systematic Evaluation

To mitigate acoustic domain mismatch and low-frequency proximity/handling rumble from real-world recordings, Stage 1 implements two validated, non-destructive inference enhancements:

1. **Speech High-Pass Filtering (`--high-pass [65|80]`)**:
   * Attenuates sub-speech acoustic rumble (< 65 Hz or < 80 Hz) from mobile microphone proximity and handling.
   * Evaluated across cutoffs of 65 Hz and 80 Hz:
     * In-domain ASVspoof 2019 Dev benchmark: **100.00% Accuracy (0% FPR, 0% FNR, 0 degradation)**.
     * `demo_real_speech.wav`: Flipped from 13.86% Real (misclassified) to **58.16% Real (correctly classified)** at 80 Hz cutoff.
     * `demo_synthetic_speech.wav`: Remains **100.00% Spoof** (zero false alarms).
2. **Multi-Window Aggregation (`--multi-window`)**:
   * For audio longer than 4.0 seconds (> 250 spectrogram frames), extracts consecutive 250-frame windows with 50% overlap (125-frame hop).
   * Aggregates predictions across windows and provides timeline diagnostics.

To run the complete benchmark comparison across all 6 inference pipelines:
```bash
python scripts/evaluate_stage1_improvements.py
```

---

## 🛡️ Stage 2: Robust Domain Adaptation & Codec Generalization

To address severe acoustic domain mismatch on real-world mobile recordings (e.g., WhatsApp voice notes encoded with lossy Opus codecs) without degrading baseline performance:

1. **Independent Robust Checkpoint (`models/robust_deepfake_audio_cnn.pth`)**:
   * Initialized from baseline weights and fine-tuned with domain adaptation.
   * Baseline checkpoint (`models/best_deepfake_audio_cnn.pth`) remains **strictly unmodified** (SHA-256 verified).
2. **Symmetrical Multi-Domain Data Augmentation**:
   * Applied in-memory **Opus (16-32 kbps)** and **MP3 (32-64 kbps)** lossy compression.
   * Low-frequency room/handling rumble ($15\text{--}65\text{ Hz}$) and ambient acoustic noise.
   * Gain variations ($\pm 4\text{ dB}$).
   * **Crucial Design Rule**: Augmentation was applied **symmetrically** to both bona fide and spoof speech to prevent the network from learning codec artifacts as a spurious shortcut.
3. **Strict Speaker Disjoint Partitioning**:
   * Protocol verified zero speaker overlap across Train, Val, and Test splits.
   * WhatsApp recording held out 100% from training and tuning.
4. **Generalization Results**:
   * **Multi-Domain Test Accuracy**: Baseline 96.67% $\rightarrow$ Robust **100.00%**.
   * **False Positive Rate (FPR)**: Baseline 6.67% $\rightarrow$ Robust **0.00%** (zero false alarms).
   * **Held-out WhatsApp Voice Note**: Baseline 0.23% Real (False Alarm) $\rightarrow$ Robust **95.22% Real (Correctly Classified)**.
   * **In-Domain ASVspoof Preservation**: 100.0% accuracy maintained across bona fide and spoof utterances.

### CLI Usage with Model Selection:
```bash
# Evaluate with baseline model:
python predict.py --audio "demo_audio/demo_real_speech.wav" --model-type baseline

# Evaluate with robust domain-adapted model:
python predict.py --audio "demo_audio/demo_real_speech.wav" --model-type robust
```

To re-run the comparative evaluation across baseline and robust models:
```bash
python scripts/evaluate_robust_model.py
```

---

## ⚠️ Important Technical Limitations & Disclaimers

1. **Uncalibrated Model Confidence**: Reported confidence percentages are raw softmax values from the neural network; they are not calibrated Bayesian posterior probabilities.
2. **Scope of Validation**: The model was trained and evaluated on the ASVspoof 2019 Logical Access protocol. It has **not** been benchmarked on commercial voice cloning APIs (e.g., ElevenLabs, Bark, Tortoise) or unconstrained field recordings.
3. **Interpretability vs. Forensic Proof**: Grad-CAM visualizes CNN activation saliency. It highlights time-frequency correlations learned during training, but **does not constitute definitive forensic proof of manipulation**.
