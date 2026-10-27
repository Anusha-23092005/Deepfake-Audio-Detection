#!/usr/bin/env python3
"""
CLI Predictor for Deep Learning-Based Explainable DeepFake Audio Detection.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model import load_model, CLASS_MAPPING, CLASS_DESCRIPTIONS, MODEL_REGISTRY, BASELINE_MODEL_PATH, ROBUST_MODEL_PATH
from src.audio_processor import process_audio, AudioProcessingError
from src.explainability import GradCAM, plot_explainability

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether an audio file is Real (Bona fide) or Fake (Spoof) with Grad-CAM explainability."
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        required=True,
        help="Path to the input audio file (.wav, .mp3, .flac, .ogg, .m4a)."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to the model checkpoint .pth file."
    )
    parser.add_argument(
        "--model-type", "-t",
        type=str,
        default="baseline",
        choices=["baseline", "robust"],
        help="Model variant: 'baseline' (ASVspoof 2019 CNN) or 'robust' (Stage 2 Domain-Adapted CNN)."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)."
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=None,
        help="Optional speech high-pass filter cutoff in Hz (e.g. 65 or 80) to attenuate low-frequency rumble."
    )
    parser.add_argument(
        "--multi-window",
        action="store_true",
        help="Enable overlapping-window inference (250-frame windows with 50%% overlap) for audio longer than 4s."
    )
    parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Compute Grad-CAM explainability map."
    )
    parser.add_argument(
        "--save-cam",
        type=str,
        default=None,
        help="Path to save the generated Grad-CAM explanation figure (e.g. cam_result.png)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the prediction results in JSON format."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Resolve model checkpoint
    if args.model:
        checkpoint_path = args.model
    elif args.model_type == "robust":
        checkpoint_path = ROBUST_MODEL_PATH
    else:
        checkpoint_path = BASELINE_MODEL_PATH

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load Model
    try:
        model, metadata = load_model(checkpoint_path, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Process Audio
    try:
        tensor_input, mel_db, meta = process_audio(
            args.audio,
            high_pass_cutoff=args.high_pass,
            use_windows=args.multi_window
        )
        tensor_input = tensor_input.to(args.device)
    except AudioProcessingError as ape:
        print(f"Audio Processing Error: {ape}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error processing audio: {e}", file=sys.stderr)
        sys.exit(2)

    # 3. Model Inference
    with torch.no_grad():
        logits = model(tensor_input)
        all_probs = torch.softmax(logits, dim=1).cpu().numpy()

    if args.multi_window and all_probs.shape[0] > 1:
        probs = np.mean(all_probs, axis=0)
    else:
        probs = all_probs[0]

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_MAPPING[pred_idx]
    pred_desc = CLASS_DESCRIPTIONS[pred_idx]
    confidence_score = float(probs[pred_idx])

    result = {
        "audio_file": os.path.abspath(args.audio),
        "model_variant": args.model_type,
        "model_name": MODEL_REGISTRY.get(args.model_type, {}).get("name", args.model_type),
        "model_checkpoint": os.path.basename(checkpoint_path),
        "prediction_class_index": pred_idx,
        "prediction_label": pred_label,
        "prediction_description": pred_desc,
        "model_confidence_score": round(confidence_score, 4),
        "softmax_probabilities": {
            "bonafide_real": round(float(probs[0]), 4),
            "spoof_fake": round(float(probs[1]), 4)
        },
        "audio_metadata": {
            "duration_seconds": round(meta["duration_sec"], 2),
            "sample_rate": meta["sample_rate"],
            "original_frames": meta["original_frames"],
            "target_frames": meta["target_frames"],
            "n_mels": meta["n_mels"],
            "high_pass_cutoff_hz": args.high_pass,
            "is_windowed": meta.get("is_windowed", False),
            "num_windows": meta.get("num_windows", 1)
        },
        "disclaimer": (
            "Model confidence score represents the raw uncalibrated softmax distribution from the CNN. "
            "Trained and evaluated on the ASVspoof 2019 Logical Access protocol. "
            "Performance on unseen synthesis techniques or in the wild may vary."
        )
    }

    if args.multi_window and all_probs.shape[0] > 1:
        result["window_analysis"] = {
            "total_windows": int(all_probs.shape[0]),
            "windows": [
                {
                    "window_index": i,
                    "bonafide_real": round(float(all_probs[i, 0]), 4),
                    "spoof_fake": round(float(all_probs[i, 1]), 4)
                }
                for i in range(all_probs.shape[0])
            ]
        }

    # 4. Optional Grad-CAM Explainability
    if args.explain or args.save_cam:
        try:
            cam_input = tensor_input[:1]
            grad_cam = GradCAM(model)
            cam_map, _, _ = grad_cam.generate_cam(cam_input, target_class=pred_idx)
            grad_cam.remove_hooks()

            save_path = args.save_cam or "gradcam_explanation.png"
            fig = plot_explainability(
                mel_db=mel_db[:, :250] if mel_db.shape[1] > 250 else mel_db,
                cam_map=cam_map,
                predicted_class_name=pred_label,
                confidence_score=confidence_score,
                sample_rate=meta["sample_rate"],
                hop_length=meta["hop_length"]
            )
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            result["grad_cam_saved_to"] = os.path.abspath(save_path)
        except Exception as e:
            result["grad_cam_error"] = str(e)

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print("EXPLAINABLE DEEPFAKE AUDIO DETECTION — PREDICTION RESULT")
        print("=" * 65)
        print(f"Audio File        : {result['audio_file']}")
        print(f"Duration          : {result['audio_metadata']['duration_seconds']}s (Sample Rate: {result['audio_metadata']['sample_rate']} Hz)")
        print(f"Verdict           : {result['prediction_description'].upper()}")
        print(f"Classification    : {result['prediction_label']}")
        print(f"Model Confidence  : {result['model_confidence_score'] * 100:.2f}% (Raw Softmax Score)")
        print(f"  - Real (Bona fide) : {result['softmax_probabilities']['bonafide_real'] * 100:.2f}%")
        print(f"  - Fake (Spoof)     : {result['softmax_probabilities']['spoof_fake'] * 100:.2f}%")
        if "grad_cam_saved_to" in result:
            print(f"Grad-CAM Heatmap  : Saved to {result['grad_cam_saved_to']}")
        print("-" * 65)
        print(f"Note: {result['disclaimer']}")
        print("=" * 65)

if __name__ == "__main__":
    main()
