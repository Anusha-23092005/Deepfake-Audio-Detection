#!/usr/bin/env python3
"""
Evaluation Script for Stage 1 Inference Improvements:
- Speech High-Pass Filter (65 Hz, 80 Hz)
- Overlapping-Window Aggregation for long recordings (> 4s)

Compares 6 inference pipelines across:
1. Balanced ASVspoof 2019 Dev Benchmark (12 bona fide, 12 spoof)
2. Demonstration recordings (demo_real_speech.wav, demo_synthetic_speech.wav)
3. Genuine WhatsApp Voice Note recording (Opus compressed, mobile)

Calculates:
- Confusion Matrix (TP, TN, FP, FN)
- Accuracy, False Positive Rate (FPR), False Negative Rate (FNR)
- Mean softmax probabilities per class
"""

import os
import sys
import json
import hashlib
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model import load_model, CLASS_MAPPING, CLASS_DESCRIPTIONS
from src.audio_processor import process_audio, SAMPLE_RATE

EXPECTED_CHECKPOINT_SHA256 = "6ec42917f8fd0e87aa6a5a29970fde4a4f7b91e87b90b0ee2c42a0e7ffc8afd1"

def verify_checkpoint(checkpoint_path: str) -> bool:
    with open(checkpoint_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest().lower()
    return sha256 == EXPECTED_CHECKPOINT_SHA256

def run_model_inference(
    model: torch.nn.Module,
    audio_path: str,
    cutoff: Any = None,
    use_windows: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    tensor_input, mel_db, meta = process_audio(
        audio_path,
        high_pass_cutoff=cutoff,
        use_windows=use_windows
    )
    tensor_input = tensor_input.to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        all_probs = torch.softmax(logits, dim=1).cpu().numpy()

    if use_windows and all_probs.shape[0] > 1:
        probs = np.mean(all_probs, axis=0)
    else:
        probs = all_probs[0]

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    return {
        "pred_idx": pred_idx,
        "label": CLASS_MAPPING[pred_idx],
        "confidence": conf,
        "probs": [float(probs[0]), float(probs[1])],
        "num_windows": len(all_probs),
        "per_window_probs": [[float(p[0]), float(p[1])] for p in all_probs],
        "meta": meta
    }

def main():
    checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_deepfake_audio_cnn.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if not verify_checkpoint(checkpoint_path):
        print("Error: Checkpoint SHA-256 integrity check failed!", file=sys.stderr)
        sys.exit(1)

    model, _ = load_model(checkpoint_path, device="cpu")
    model.eval()

    # Identify ASVspoof dev samples in scratch directory or fallback
    scratch_dir = os.path.expanduser("~/.gemini/antigravity/brain/df622718-b659-4ac1-817a-e157640df4a5/scratch")
    if not os.path.exists(scratch_dir):
        scratch_dir = os.path.join(PROJECT_ROOT, "scratch")

    flac_files = [
        f for f in os.listdir(scratch_dir)
        if f.startswith("LA_D_") and f.endswith(".flac")
    ] if os.path.exists(scratch_dir) else []

    # ASVspoof ground truth mapping (Class 0: bona fide, Class 1: spoof)
    asv_ground_truth = {
        # 12 Bona Fide samples
        "LA_D_1047731": 0,
        "LA_D_1105538": 0,
        "LA_D_1125976": 0,
        "LA_D_1293230": 0,
        "LA_D_1340209": 0,
        "LA_D_1376638": 0,
        "LA_D_1377420": 0,
        "LA_D_1403371": 0,
        "LA_D_1590530": 0,
        "LA_D_1595036": 0,
        "LA_D_1618135": 0,
        "LA_D_1624385": 0,
        # 12 Spoof samples (A01 - A06)
        "LA_D_1008730": 1,  # A01
        "LA_D_1034049": 1,  # A01
        "LA_D_1017803": 1,  # A02
        "LA_D_1042992": 1,  # A02
        "LA_D_1050796": 1,  # A03
        "LA_D_1111959": 1,  # A03
        "LA_D_1006800": 1,  # A04
        "LA_D_1012014": 1,  # A04
        "LA_D_1002910": 1,  # A05
        "LA_D_1006756": 1,  # A05
        "LA_D_1029617": 1,  # A06
        "LA_D_1059142": 1,  # A06
    }

    # WhatsApp Recording paths
    wa_candidates = [
        os.path.expanduser(r"C:\Users\Sreenithi\Downloads\WhatsApp Ptt 2026-10-17 at 5.14.35 PM.ogg"),
        os.path.expanduser(r"C:\Users\Sreenithi\Downloads\WhatsApp_Ptt_16kHz_mono.wav"),
        os.path.join(PROJECT_ROOT, "demo_audio", "whatsapp_sample.wav")
    ]
    wa_path = next((p for p in wa_candidates if os.path.exists(p)), None)

    demo_real_path = os.path.join(PROJECT_ROOT, "demo_audio", "demo_real_speech.wav")
    demo_synth_path = os.path.join(PROJECT_ROOT, "demo_audio", "demo_synthetic_speech.wav")

    configs = [
        ("Pipeline 1: Baseline (No filter, Single crop)", None, False),
        ("Pipeline 2: Speech HP 65 Hz (Single crop)", 65, False),
        ("Pipeline 3: Speech HP 80 Hz (Single crop)", 80, False),
        ("Pipeline 4: Multi-Window (No filter, 50% overlap)", None, True),
        ("Pipeline 5: Speech HP 65 Hz + Multi-Window", 65, True),
        ("Pipeline 6: Speech HP 80 Hz + Multi-Window", 80, True),
    ]

    print("=" * 90)
    print("STAGE 1 INFERENCE IMPROVEMENTS: SYSTEMATIC BENCHMARK EVALUATION")
    print("=" * 90)
    print(f"Checkpoint Verified: models/best_deepfake_audio_cnn.pth (SHA-256: {EXPECTED_CHECKPOINT_SHA256[:12]}...)")
    print(f"Available ASVspoof Dev Samples: {len(flac_files)} in {scratch_dir}")
    print(f"WhatsApp Audio Available: {wa_path}")
    print(f"Demo Real Available: {os.path.exists(demo_real_path)}")
    print(f"Demo Synth Available: {os.path.exists(demo_synth_path)}")

    results = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "asvspoof_dev_benchmark": {},
        "target_speech_samples": {}
    }

    # 1. ASVspoof 2019 Dev Benchmark
    print("\n" + "-" * 90)
    print("PART 1: IN-DOMAIN ASVspoof 2019 DEV BENCHMARK (12 BONA FIDE, 12 SPOOF)")
    print("-" * 90)
    print(f"{'Pipeline Name':<42} | {'Accuracy':<8} | {'TP':<3} {'TN':<3} {'FP':<3} {'FN':<3} | {'FPR':<6} {'FNR':<6} | {'Mean BonaFide':<13} | {'Mean Spoof':<13}")
    print("-" * 90)

    for pipe_name, cutoff, win in configs:
        tp, tn, fp, fn = 0, 0, 0, 0
        bonafide_probs = []
        spoof_probs = []

        for fid, gt in asv_ground_truth.items():
            flac_path = os.path.join(scratch_dir, f"{fid}.flac")
            if not os.path.exists(flac_path):
                continue

            res = run_model_inference(model, flac_path, cutoff=cutoff, use_windows=win)
            pred = res["pred_idx"]
            p_real = res["probs"][0]
            p_spoof = res["probs"][1]

            if gt == 1:  # Spoof (Positive)
                spoof_probs.append(p_spoof)
                if pred == 1:
                    tp += 1
                else:
                    fn += 1
            else:  # Bona fide (Negative)
                bonafide_probs.append(p_real)
                if pred == 0:
                    tn += 1
                else:
                    fp += 1

        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        mean_b = np.mean(bonafide_probs) * 100 if bonafide_probs else 0.0
        mean_s = np.mean(spoof_probs) * 100 if spoof_probs else 0.0

        print(f"{pipe_name:<42} | {acc*100:6.2f}% | {tp:2d}  {tn:2d}  {fp:2d}  {fn:2d} | {fpr*100:5.1f}% {fnr*100:5.1f}% | {mean_b:12.2f}% | {mean_s:12.2f}%")

        results["asvspoof_dev_benchmark"][pipe_name] = {
            "total_samples": total,
            "accuracy": round(float(acc), 4),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "mean_bonafide_real_prob": round(float(mean_b), 2),
            "mean_spoof_fake_prob": round(float(mean_s), 2)
        }

    # 2. Key Target Recordings
    print("\n" + "-" * 90)
    print("PART 2: OUT-OF-DOMAIN & IN-THE-WILD AUDIO SAMPLES")
    print("-" * 90)

    target_audios = []
    if demo_real_path and os.path.exists(demo_real_path):
        target_audios.append(("demo_real_speech.wav", demo_real_path, "Human (LibriSpeech clean)"))
    if demo_synth_path and os.path.exists(demo_synth_path):
        target_audios.append(("demo_synthetic_speech.wav", demo_synth_path, "Synthetic TTS (SAPI5)"))
    if wa_path and os.path.exists(wa_path):
        target_audios.append(("WhatsApp Voice Note", wa_path, "Human (WhatsApp Opus mobile)"))

    for label, audio_file, desc in target_audios:
        print(f"\nRecording: {label} ({desc})")
        print(f"{'Pipeline':<42} | {'Prediction':<10} | {'Real Prob':<10} | {'Fake Prob':<10} | {'Windows':<8} | {'Window Real% Breakdown'}")
        print("-" * 90)

        results["target_speech_samples"][label] = {"description": desc, "pipelines": {}}

        for pipe_name, cutoff, win in configs:
            res = run_model_inference(model, audio_file, cutoff=cutoff, use_windows=win)
            p_real = res["probs"][0] * 100
            p_fake = res["probs"][1] * 100
            n_win = res["num_windows"]
            win_real_strs = [f"{w[0]*100:.1f}%" for w in res["per_window_probs"]]

            print(f"{pipe_name:<42} | {res['label']:<10} | {p_real:9.2f}% | {p_fake:9.2f}% | {n_win:<8} | {win_real_strs}")

            results["target_speech_samples"][label]["pipelines"][pipe_name] = {
                "predicted_label": res["label"],
                "real_probability": round(float(res["probs"][0]), 4),
                "spoof_probability": round(float(res["probs"][1]), 4),
                "num_windows": n_win,
                "window_real_probabilities": [round(float(w[0]), 4) for w in res["per_window_probs"]]
            }

    # Save to report JSON
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    report_out = os.path.join(PROJECT_ROOT, "reports", "stage1_evaluation_results.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Evaluation report successfully saved to: {report_out}")
    print("=" * 90)

if __name__ == "__main__":
    main()

