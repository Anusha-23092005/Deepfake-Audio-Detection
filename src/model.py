import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

CLASS_MAPPING = {
    0: "bonafide (real)",
    1: "spoof (fake)"
}

CLASS_DESCRIPTIONS = {
    0: "Bona fide Human Speech (Real)",
    1: "Synthetic / Cloned Speech (Spoof)"
}

class AudioDeepfakeCNN(nn.Module):
    """
    Exact CNN Architecture for DeepFake Audio Detection.

    Structure:
    - 3 Convolutional Blocks:
        Block 1: Conv2d(1, 16, 3, padding=1) -> BatchNorm2d(16) -> ReLU -> MaxPool2d(2)
        Block 2: Conv2d(16, 32, 3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2)
        Block 3: Conv2d(32, 64, 3, padding=1) -> BatchNorm2d(64) -> ReLU (No 3rd MaxPool2d)
    - AdaptiveAvgPool2d((5, 10)) -> maps 64 channels to (64, 5, 10)
    - Classification Head:
        Flatten() -> Linear(3200, 128) -> ReLU -> Dropout(0.4) -> Linear(128, 2)
    """
    def __init__(self):
        super(AudioDeepfakeCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Adaptive Pooling to (5, 10)
            nn.AdaptiveAvgPool2d((5, 10))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[AudioDeepfakeCNN, Dict[str, Any]]:
    """
    Loads the trained model weights from the specified checkpoint.

    Args:
        checkpoint_path: Path to best_deepfake_audio_cnn.pth
        device: Device to map model weights ('cpu' or 'cuda')

    Returns:
        Tuple of (model, metadata_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Expected 'model_state_dict' in checkpoint keys: {list(checkpoint.keys())}")

    model = AudioDeepfakeCNN()
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
        raise RuntimeError(
            f"State dict mismatch! Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    metadata = {
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "history": checkpoint.get("history", {}),
        "model_type": checkpoint.get("model_type", "baseline"),
        "training_metadata": checkpoint.get("training_metadata", {})
    }

    return model, metadata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_deepfake_audio_cnn.pth")
ROBUST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "robust_deepfake_audio_cnn.pth")

MODEL_REGISTRY = {
    "baseline": {
        "id": "baseline",
        "name": "Baseline Model (ASVspoof 2019 CNN)",
        "path": BASELINE_MODEL_PATH,
        "description": "Trained exclusively on ASVspoof 2019 LA studio recordings (best for hemianechoic studio conditions).",
        "checkpoint_file": "best_deepfake_audio_cnn.pth"
    },
    "robust": {
        "id": "robust",
        "name": "Robust Model (Stage 2 Domain-Adapted CNN)",
        "path": ROBUST_MODEL_PATH,
        "description": "Adapted with Opus/MP3 codec and acoustic noise augmentation (recommended for mobile recordings and in-the-wild audio).",
        "checkpoint_file": "robust_deepfake_audio_cnn.pth"
    }
}
