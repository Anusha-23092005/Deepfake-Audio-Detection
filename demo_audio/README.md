# Demonstration Audio Samples Documentation

This directory contains two audio recordings prepared for live demonstrations and testing of the **Explainable DeepFake Audio Detection System**:

1. [`demo_real_speech.wav`](demo_real_speech.wav) — Genuine Human Speech Recording
2. [`demo_synthetic_speech.wav`](demo_synthetic_speech.wav) — Genuine Text-to-Speech (TTS) Recording

---

## 📄 Audio Metadata & Provenance

### 1. `demo_real_speech.wav` (Bona fide Human Speech)
* **Audio Source**: Canonical **LibriSpeech** corpus (clean speech partition, extracted from public domain audiobooks).
* **Provenance & Verification**: Read by a human volunteer for the LibriVox project; distributed under Public Domain (CC0).
* **Speaker**: Human male reader.
* **Spoken Transcript**:
  > *"He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flower-fatten sauce."*
* **Duration**: 10.44 seconds
* **Sample Rate**: 16,000 Hz (standard mono PCM 16-bit)
* **Channels**: 1 (Mono)
* **File Size**: ~326.1 KB

### 2. `demo_synthetic_speech.wav` (Synthetic / Cloned Speech)
* **Audio Source**: Genuine local speech synthesis engine (**Windows SAPI5 `System.Speech.Synthesis.SpeechSynthesizer`**).
* **Voice Model**: Microsoft David Desktop (English US, standard concatenative/formant synthesis voice).
* **Provenance & Verification**: Generated programmatically using `scripts/generate_demo_audio.py`. No artificial noise, pitch warping, or post-processing alterations were applied.
* **Spoken Transcript**:
  > *"He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fattened sauce."*
* **Duration**: 9.71 seconds
* **Sample Rate**: 16,000 Hz (standard mono PCM 16-bit)
* **Channels**: 1 (Mono)
* **File Size**: ~303.6 KB

---

## 🔬 Actual Model Inference Results (Research Honesty)

These recordings were evaluated using the unchanged trained checkpoint [`models/best_deepfake_audio_cnn.pth`](../models/best_deepfake_audio_cnn.pth) via the exact training preprocessing pipeline:

| Audio Sample | Ground Truth | Model Prediction | Raw Softmax: Real | Raw Softmax: Spoof |
| :--- | :--- | :--- | :--- | :--- |
| **`demo_real_speech.wav`** | **Human Speech** | **Spoof (Misclassified)** | 13.86% | **86.14%** |
| **`demo_synthetic_speech.wav`** | **Synthetic TTS** | **Spoof (Correct)** | 0.00% | **100.00%** |

### Scientific Analysis of Out-of-Domain Generalization:
* **Synthetic Speech Result**: The model correctly classified the synthetic speech sample as spoof with a 100.00% raw softmax score, successfully detecting electronic vocoder and synthesis artifacts.
* **Human Speech Result**: The human speech sample was misclassified as spoof (86.14% raw softmax score).
  * **Why?** The CNN model was trained and validated on the **ASVspoof 2019 Logical Access** database (where bona fide speech was recorded in a hemianechoic chamber on high-end studio condenser microphones from the VCTK corpus). LibriSpeech audiobooks, although genuine human speech, were recorded in home/non-studio acoustic environments with different reverberation, microphone frequency responses, and compression histories.
  * **Research Significance**: This illustrates the fundamental open research challenge in speech anti-spoofing known as **acoustic domain shift / cross-dataset generalization**.
  * **Academic Integrity**: Model weights, thresholds, and outputs were **not altered** to artificially force a "correct" result on the demo recording.

---

## ⚠️ Important Note
These two samples are intended strictly for **functional demonstration** of the pipeline, spectrogram generation, and Grad-CAM explainability, and do **not** constitute a statistically valid evaluation benchmark. Formal evaluation metrics are documented in [`models/dev_evaluation_results.json`](../models/dev_evaluation_results.json).

