#!/usr/bin/env python3
"""
Systematic Comparative Evaluation: Baseline Model vs. Robust Model (Stage 2 Pilot).

Evaluates both models on identical, untouched evaluation data:
1. Held-Out Test Split (30 samples: 15 bona fide, 15 spoof with disjoint speakers).
2. Out-of-Domain Human Speech (demo_real_speech.wav).
3. Modern Synthetic Speech (demo_synthetic_speech.wav).
4. Real-World Mobile Recording (WhatsApp voice note, completely untouched by training).

Computes:
- Confusion Matrix (TP, TN, FP, FN)
- Accuracy, Balanced Accuracy, Precision, Recall, F1
- False Positive Rate (FPR), False Negative Rate (FNR)
- ROC-AUC
- Per-domain performance breakdown
"""

import os
import sys
import json
import csv
import numpy as np
import soundfile as sf
import librosa
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model import load_model, CLASS_MAPPING
from src.audio_processor import process_audio

BASELINE_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "best_deepfake_audio_cnn.pth")
ROBUST_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "robust_deepfake_audio_cnn.pth")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "pilot", "metadata", "test_split.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def evaluate_model_on_dataset(model: torch.nn.Module, test_samples: list, device: str = "cpu"):
    y_true = []
    y_pred = []
    y_prob_spoof = []
    per_domain_results = {}

    for item in test_samples:
        file_path = item["file_path"]
        gt = int(item["class_label"])
        domain = item.get("domain", "Unknown")

        if domain not in per_domain_results:
            per_domain_results[domain] = {"y_true": [], "y_pred": [], "probs": []}

        try:
            tensor_input, _, _ = process_audio(file_path)
            tensor_input = tensor_input.to(device)
            with torch.no_grad():
                logits = model(tensor_input)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_class = int(np.argmax(probs))
            spoof_prob = float(probs[1])

            y_true.append(gt)
            y_pred.append(pred_class)
            y_prob_spoof.append(spoof_prob)

            per_domain_results[domain]["y_true"].append(gt)
            per_domain_results[domain]["y_pred"].append(pred_class)
            per_domain_results[domain]["probs"].append(spoof_prob)
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")

    # Compute overall metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob_spoof)
    except Exception:
        auc = float("nan")

    # Per domain metrics
    domain_summary = {}
    for d, data in per_domain_results.items():
        d_true = data["y_true"]
        d_pred = data["y_pred"]
        d_acc = accuracy_score(d_true, d_pred)
        d_cm = confusion_matrix(d_true, d_pred, labels=[0, 1])
        d_tn, d_fp, d_fn, d_tp = d_cm.ravel()
        mean_p_spoof = float(np.mean(data["probs"]))
        domain_summary[d] = {
            "samples": len(d_true),
            "accuracy": round(float(d_acc), 4),
            "confusion_matrix": {"TP": int(d_tp), "TN": int(d_tn), "FP": int(d_fp), "FN": int(d_fn)},
            "mean_spoof_prob": round(mean_p_spoof, 4)
        }

    return {
        "overall": {
            "samples": len(y_true),
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(b_acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
        },
        "per_domain": domain_summary
    }

def evaluate_single_file(model: torch.nn.Module, audio_path: str, device: str = "cpu"):
    if not os.path.exists(audio_path):
        return None
    tensor_input, _, meta = process_audio(audio_path)
    tensor_input = tensor_input.to(device)
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "pred_idx": pred_idx,
        "label": CLASS_MAPPING[pred_idx],
        "bonafide_prob": round(float(probs[0]), 4),
        "spoof_prob": round(float(probs[1]), 4)
    }

def main():
    print("=" * 90)
    print("SYSTEMATIC COMPARATIVE EVALUATION: BASELINE vs. ROBUST (PILOT)")
    print("=" * 90)

    if not os.path.exists(BASELINE_CHECKPOINT):
        print(f"Error: Baseline checkpoint missing: {BASELINE_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(ROBUST_CHECKPOINT):
        print(f"Error: Robust checkpoint missing: {ROBUST_CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    base_model, base_meta = load_model(BASELINE_CHECKPOINT, device="cpu")
    robust_model, robust_meta = load_model(ROBUST_CHECKPOINT, device="cpu")
    base_model.eval()
    robust_model.eval()

    # Load test split
    test_samples = []
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_samples.append(row)

    print(f"Loaded {len(test_samples)} untouched test samples from {TEST_CSV}")

    # Evaluate both models on test set
    base_results = evaluate_model_on_dataset(base_model, test_samples)
    robust_results = evaluate_model_on_dataset(robust_model, test_samples)

    # Completely held-out target files
    wa_path = r"C:\Users\Sreenithi\Downloads\WhatsApp Ptt 2026-10-17 at 5.14.35 PM.ogg"
    demo_real = os.path.join(PROJECT_ROOT, "demo_audio", "demo_real_speech.wav")
    demo_synth = os.path.join(PROJECT_ROOT, "demo_audio", "demo_synthetic_speech.wav")

    wa_base = evaluate_single_file(base_model, wa_path)
    wa_robust = evaluate_single_file(robust_model, wa_path)

    real_base = evaluate_single_file(base_model, demo_real)
    real_robust = evaluate_single_file(robust_model, demo_real)

    synth_base = evaluate_single_file(base_model, demo_synth)
    synth_robust = evaluate_single_file(robust_model, demo_synth)

    # Print Comparative Report
    print("\n" + "-" * 90)
    print("PART 1: HELD-OUT TEST BENCHMARK METRICS (30 DISJOINT SAMPLES)")
    print("-" * 90)
    print(f"{'Metric':<25} | {'Baseline Model':<15} | {'Robust Model (Stage 2)':<25} | {'Delta':<10}")
    print("-" * 90)

    b_ov = base_results["overall"]
    r_ov = robust_results["overall"]

    metrics = [
        ("Accuracy", b_ov["accuracy"], r_ov["accuracy"]),
        ("Balanced Accuracy", b_ov["balanced_accuracy"], r_ov["balanced_accuracy"]),
        ("Precision (Spoof)", b_ov["precision"], r_ov["precision"]),
        ("Recall (Spoof)", b_ov["recall"], r_ov["recall"]),
        ("F1-Score (Spoof)", b_ov["f1"], r_ov["f1"]),
        ("False Positive Rate", b_ov["fpr"], r_ov["fpr"]),
        ("False Negative Rate", b_ov["fnr"], r_ov["fnr"]),
        ("ROC-AUC", b_ov["roc_auc"], r_ov["roc_auc"])
    ]

    for m_name, b_val, r_val in metrics:
        if b_val is not None and r_val is not None:
            delta = r_val - b_val
            sign = "+" if delta >= 0 else ""
            print(f"{m_name:<25} | {b_val*100:13.2f}% | {r_val*100:23.2f}% | {sign}{delta*100:6.2f}%")
        else:
            print(f"{m_name:<25} | {str(b_val):<15} | {str(r_val):<25} | N/A")

    print("\nConfusion Matrices (Held-Out Test Split):")
    print(f"  Baseline Model : TP={b_ov['confusion_matrix']['TP']:2d}, TN={b_ov['confusion_matrix']['TN']:2d}, FP={b_ov['confusion_matrix']['FP']:2d}, FN={b_ov['confusion_matrix']['FN']:2d}")
    print(f"  Robust Model   : TP={r_ov['confusion_matrix']['TP']:2d}, TN={r_ov['confusion_matrix']['TN']:2d}, FP={r_ov['confusion_matrix']['FP']:2d}, FN={r_ov['confusion_matrix']['FN']:2d}")

    print("\n" + "-" * 90)
    print("PART 2: PER-DOMAIN PERFORMANCE BREAKDOWN")
    print("-" * 90)
    for domain in sorted(base_results["per_domain"].keys()):
        b_dom = base_results["per_domain"][domain]
        r_dom = robust_results["per_domain"].get(domain, {})
        print(f"Domain: {domain:<30} | Baseline Acc: {b_dom['accuracy']*100:5.1f}% | Robust Acc: {r_dom.get('accuracy', 0)*100:5.1f}%")

    print("\n" + "-" * 90)
    print("PART 3: HELD-OUT REAL-WORLD & DEMO AUDIO RECORDINGS")
    print("-" * 90)
    print(f"{'Recording':<28} | {'Ground Truth':<12} | {'Baseline Prediction':<22} | {'Robust Prediction':<22}")
    print("-" * 90)

    if real_base and real_robust:
        print(f"{'demo_real_speech.wav':<28} | {'Human':<12} | {real_base['label']:<15} ({real_base['bonafide_prob']*100:4.1f}%) | {real_robust['label']:<15} ({real_robust['bonafide_prob']*100:4.1f}%)")

    if synth_base and synth_robust:
        print(f"{'demo_synthetic_speech.wav':<28} | {'Synthetic':<12} | {synth_base['label']:<15} ({synth_base['spoof_prob']*100:4.1f}%) | {synth_robust['label']:<15} ({synth_robust['spoof_prob']*100:4.1f}%)")

    if wa_base and wa_robust:
        print(f"{'WhatsApp Voice Note':<28} | {'Human':<12} | {wa_base['label']:<15} ({wa_base['bonafide_prob']*100:4.1f}%) | {wa_robust['label']:<15} ({wa_robust['bonafide_prob']*100:4.1f}%)")

    # Save output report
    report_data = {
        "baseline_model": {
            "checkpoint": os.path.basename(BASELINE_CHECKPOINT),
            "results": base_results,
            "demo_real": real_base,
            "demo_synth": synth_base,
            "whatsapp": wa_base
        },
        "robust_model": {
            "checkpoint": os.path.basename(ROBUST_CHECKPOINT),
            "results": robust_results,
            "demo_real": real_robust,
            "demo_synth": synth_robust,
            "whatsapp": wa_robust
        }
    }

    report_path = os.path.join(REPORTS_DIR, "pilot_robust_evaluation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 90)
    print(f"Full comparative evaluation saved to: {report_path}")
    print("=" * 90)

if __name__ == "__main__":
    main()

