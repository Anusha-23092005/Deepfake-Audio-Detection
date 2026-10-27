#!/usr/bin/env python3
"""
Reproducible Training Pipeline for Robust DeepFake Audio Detection (Stage 2 Pilot).

Features:
- Initializes from baseline checkpoint: models/best_deepfake_audio_cnn.pth
- Preserves baseline checkpoint strictly unmodified.
- Saves robust model to: models/robust_deepfake_audio_cnn.pth
- Dynamic on-the-fly codec (Opus, MP3) and acoustic augmentation on TRAIN set only.
- Validation and Test sets are NEVER augmented.
- Symmetric augmentation across both classes (no shortcut bias).
- CPU-efficient batching and data loading.
"""

import os
import sys
import io
import time
import json
import random
import csv
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model import AudioDeepfakeCNN, load_model
from src.audio_processor import extract_mel_spectrogram, SAMPLE_RATE, TARGET_FRAMES

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pilot")
META_DIR = os.path.join(DATA_DIR, "metadata")
BASELINE_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "best_deepfake_audio_cnn.pth")
ROBUST_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "robust_deepfake_audio_cnn.pth")

class RobustAudioDataset(Dataset):
    """
    PyTorch Dataset with on-the-fly codec and acoustic augmentations for training data.
    """
    def __init__(self, csv_path: str, is_train: bool = True, seed: int = 42):
        self.is_train = is_train
        self.rng = np.random.RandomState(seed)
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def apply_augmentations(self, y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """
        Applies randomized codec compression and acoustic perturbations symmetrically
        to both bona fide and spoof audio to avoid class-specific shortcuts.
        """
        # 1. Gain variation (50% chance, +/- 4 dB)
        if self.rng.rand() < 0.50:
            gain_db = self.rng.uniform(-4.0, 4.0)
            y = y * (10.0 ** (gain_db / 20.0))

        # 2. Opus codec compression (40% chance)
        if self.rng.rand() < 0.40:
            try:
                buf = io.BytesIO()
                sf.write(buf, y, sr, format='OGG', subtype='OPUS')
                buf.seek(0)
                y_dec, _ = sf.read(buf, dtype='float32')
                if len(y_dec) > 0 and not np.any(np.isnan(y_dec)):
                    y = y_dec
            except Exception:
                pass

        # 3. MP3 compression (25% chance)
        if self.rng.rand() < 0.25:
            try:
                buf = io.BytesIO()
                sf.write(buf, y, sr, format='MP3')
                buf.seek(0)
                y_dec, _ = sf.read(buf, dtype='float32')
                if len(y_dec) > 0 and not np.any(np.isnan(y_dec)):
                    y = y_dec
            except Exception:
                pass

        # 4. Low-frequency handling rumble + mild ambient noise (35% chance)
        if self.rng.rand() < 0.35:
            t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
            rumble_freq = self.rng.uniform(20.0, 60.0)
            rumble_amp = self.rng.uniform(0.005, 0.03) * np.max(np.abs(y))
            rumble = rumble_amp * np.sin(2 * np.pi * rumble_freq * t)
            noise_amp = self.rng.uniform(0.001, 0.008) * np.std(y)
            ambient = noise_amp * self.rng.randn(len(y)).astype(np.float32)
            y = y + rumble + ambient

        # Clip to prevent digital clipping
        max_val = np.max(np.abs(y))
        if max_val > 1.0:
            y = y / max_val

        return y.astype(np.float32)

    def __getitem__(self, idx):
        item = self.samples[idx]
        file_path = item["file_path"]
        label = int(item["class_label"])

        try:
            y, sr = sf.read(file_path, dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception as e:
            # Fallback to librosa if soundfile fails
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        if len(y) == 0:
            y = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

        # Apply augmentation strictly to training data
        if self.is_train:
            y = self.apply_augmentations(y, sr=SAMPLE_RATE)

        # Extract normalized log-Mel spectrogram matching Kaggle settings
        tensor_spec, _, _ = extract_mel_spectrogram(y, sr=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        # tensor_spec is (1, 1, 80, 250), squeeze outer batch dim -> (1, 80, 250)
        return tensor_spec.squeeze(0), torch.tensor(label, dtype=torch.long)

def train_robust_model(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    device: str = "cpu"
):
    print("=" * 80)
    print("STAGE 2 ROBUST MODEL TRAINING (PILOT)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr}")
    print(f"Base Checkpoint: {BASELINE_CHECKPOINT}")
    print(f"Target Checkpoint: {ROBUST_CHECKPOINT}")

    train_csv = os.path.join(META_DIR, "train_split.csv")
    val_csv = os.path.join(META_DIR, "val_split.csv")

    assert os.path.exists(train_csv), f"Train CSV missing: {train_csv}"
    assert os.path.exists(val_csv), f"Val CSV missing: {val_csv}"

    train_ds = RobustAudioDataset(train_csv, is_train=True)
    val_ds = RobustAudioDataset(val_csv, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Dataset: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val Dataset:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # 1. Initialize from baseline checkpoint (preserving baseline untouched)
    model, base_meta = load_model(BASELINE_CHECKPOINT, device=device)
    print(f"Loaded baseline weights (Validation Acc in baseline: {base_meta.get('best_val_accuracy', 0)*100:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": []
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state_dict = None

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_b).sum().item()
            total_train += len(y_b)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        train_loss /= total_train
        train_acc = train_correct / total_train

        # VALIDATION (unaugmented)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                logits = model(x_v)
                loss = criterion(logits, y_v)

                val_loss += loss.item() * len(y_v)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_v).sum().item()
                total_val += len(y_v)

        val_loss /= total_val
        val_acc = val_correct / total_val

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(curr_lr)

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}% | LR: {curr_lr:.2e}")

        # Checkpoint selection (prefer higher accuracy, then lower loss)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    t_total = time.time() - t_start
    print(f"\nTraining completed in {t_total:.2f} seconds ({t_total/epochs:.2f}s per epoch on CPU)!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%, Loss: {best_val_loss:.4f}")

    # Save to separate robust checkpoint
    checkpoint_payload = {
        "model_state_dict": best_state_dict,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "history": history,
        "model_type": "robust_domain_adapted",
        "training_metadata": {
            "trained_epochs": epochs,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "base_checkpoint": os.path.basename(BASELINE_CHECKPOINT),
            "augmentations": ["opus_codec", "mp3_codec", "rumble_noise", "gain_variation"],
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    torch.save(checkpoint_payload, ROBUST_CHECKPOINT)
    print(f"Robust checkpoint successfully saved to: {ROBUST_CHECKPOINT}")
    print(f"Baseline checkpoint remains untouched:  {BASELINE_CHECKPOINT}")
    print("=" * 80)

if __name__ == "__main__":
    train_robust_model(epochs=10, batch_size=16, lr=1e-4, device="cpu")

