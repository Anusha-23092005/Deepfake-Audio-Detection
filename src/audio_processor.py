import os
import io
from typing import Tuple, Dict, Any, Union, Optional, List
import numpy as np
import soundfile as sf
import librosa
import torch
import scipy.signal as signal

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
TARGET_FRAMES = 250

class AudioProcessingError(Exception):
    """Custom exception raised for invalid, corrupt, or unsupported audio."""
    pass

def apply_high_pass_filter(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    cutoff_hz: Optional[float] = None,
    order: int = 4
) -> np.ndarray:
    """
    Applies an optional Butterworth high-pass filter to attenuate low-frequency
    sub-speech rumble (e.g. < 65 Hz or < 80 Hz) from microphone proximity and handling.
    Uses Second-Order Sections (SOS) for numerical stability.

    Args:
        y: Audio waveform array (1D float).
        sr: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz. If None or <= 0, audio is returned unmodified.
        order: Filter order (default: 4th order Butterworth).

    Returns:
        Filtered waveform as float32 numpy array.
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return y.astype(np.float32)

    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        raise AudioProcessingError(
            f"High-pass cutoff ({cutoff_hz} Hz) must be less than the Nyquist frequency ({nyquist} Hz)."
        )

    try:
        sos = signal.butter(order, cutoff_hz, btype='highpass', fs=sr, output='sos')
        y_filtered = signal.sosfilt(sos, y)
        return y_filtered.astype(np.float32)
    except Exception as e:
        raise AudioProcessingError(f"Failed to apply high-pass filter: {e}")

def load_audio_signal(
    audio_source: Union[str, bytes, io.BytesIO],
    sr: int = SAMPLE_RATE,
    high_pass_cutoff: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Safely loads audio from a file path, byte string, or BytesIO buffer,
    with optional speech high-pass filtering.

    Args:
        audio_source: Path to audio file or byte buffer.
        sr: Target sample rate (16000 Hz).
        high_pass_cutoff: Optional cutoff in Hz for speech high-pass filtering.

    Returns:
        Tuple of (audio_waveform, duration_seconds)
    """
    try:
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise AudioProcessingError(f"Audio file does not exist: {audio_source}")
            if os.path.getsize(audio_source) == 0:
                raise AudioProcessingError(f"Audio file is empty (0 bytes): {audio_source}")
            y, loaded_sr = librosa.load(audio_source, sr=sr, mono=True)
        elif isinstance(audio_source, (bytes, io.BytesIO)):
            buffer = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
            buffer.seek(0)
            data, loaded_sr = sf.read(buffer, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1) # downmix to mono
            if loaded_sr != sr:
                data = librosa.resample(data, orig_sr=loaded_sr, target_sr=sr)
            y = data
        else:
            raise AudioProcessingError(f"Unsupported audio source type: {type(audio_source)}")
    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Could not decode audio file: {e}")

    if len(y) == 0:
        raise AudioProcessingError("Audio signal has 0 samples.")

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise AudioProcessingError("Audio signal contains NaN or Infinite values.")

    # Check for silent audio (signal amplitude threshold)
    max_amplitude = np.max(np.abs(y))
    rms_energy = np.sqrt(np.mean(y ** 2))
    if max_amplitude < 1e-4 or rms_energy < 1e-5:
        raise AudioProcessingError(
            f"Audio signal is completely silent or too faint (max amplitude: {max_amplitude:.2e}, RMS: {rms_energy:.2e}). "
            "Please provide an audio recording with audible speech."
        )

    # Optional high-pass filtering
    if high_pass_cutoff is not None and high_pass_cutoff > 0:
        y = apply_high_pass_filter(y, sr=sr, cutoff_hz=high_pass_cutoff)

    duration = len(y) / sr
    return y, duration

def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_frames: int = TARGET_FRAMES
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    Extracts log-Mel spectrogram matching the exact Kaggle training settings:
    - Sample rate: 16000 Hz
    - n_fft: 1024
    - hop_length: 256
    - n_mels: 80
    - Feature type: power Mel-spectrogram converted to dB using librosa.power_to_db(ref=np.max)
    - Per-sample normalization: (x - mean) / (std + 1e-6)
    - Fixed dimensions: 80 frequency bins x 250 time frames (crop or pad)

    Returns:
        tensor_input: PyTorch tensor (1, 1, 80, 250)
        mel_db_fixed: numpy array (80, 250) in decibels (before standardization, for visualization)
        meta: dictionary containing processing metadata
    """
    # 1. Compute power Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    raw_frames = mel_spec.shape[1]

    # 2. Convert to decibels (log-Mel) using ref=np.max
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Crop or pad along time axis to exactly target_frames (250)
    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        # In dB scale (ref=np.max), 0.0 dB is peak loudness and np.min(mel_db) (~ -80 dB) is silence.
        # Padding must use the silence floor to avoid injecting artificial full-scale bursts.
        silence_floor_db = float(np.min(mel_db))
        mel_db_fixed = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=silence_floor_db)
    else:
        mel_db_fixed = mel_db[:, :target_frames]

    # 4. Per-sample normalization: (x - mean) / (std + 1e-6)
    mean = np.mean(mel_db_fixed)
    std = np.std(mel_db_fixed)
    norm_mel = (mel_db_fixed - mean) / (std + 1e-6)

    # 5. Format tensor: (batch_size=1, channels=1, n_mels=80, time_frames=250)
    tensor_input = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    meta = {
        "sample_rate": sr,
        "duration_sec": len(y) / sr,
        "original_frames": raw_frames,
        "target_frames": target_frames,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
    }

    return tensor_input, mel_db_fixed, meta

def extract_mel_spectrogram_windows(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_frames: int = TARGET_FRAMES,
    hop_frames: int = 125
) -> Tuple[torch.Tensor, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts multi-window normalized log-Mel spectrograms for audio longer than target_frames (4.0s).
    If audio duration is <= target_frames, a single window is returned (padded if needed).
    If audio duration is > target_frames, overlapping windows (default 50% overlap) are extracted.

    Returns:
        tensor_batch: PyTorch tensor of shape (num_windows, 1, 80, 250)
        mel_db_full: Full log-Mel spectrogram in decibels (80, total_frames)
        windows_info: List of dicts with window boundaries and timestamps
        meta: Dictionary containing processing metadata
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    raw_frames = mel_spec.shape[1]
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    windows_tensors = []
    windows_info = []

    if raw_frames <= target_frames:
        if raw_frames < target_frames:
            pad_width = target_frames - raw_frames
            silence_floor_db = float(np.min(mel_db))
            mel_chunk = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=silence_floor_db)
        else:
            mel_chunk = mel_db[:, :target_frames]

        mean = np.mean(mel_chunk)
        std = np.std(mel_chunk)
        norm_chunk = (mel_chunk - mean) / (std + 1e-6)
        windows_tensors.append(norm_chunk)
        windows_info.append({
            "window_idx": 0,
            "start_frame": 0,
            "end_frame": raw_frames,
            "start_sec": 0.0,
            "end_sec": round(len(y) / sr, 3)
        })
    else:
        starts = list(range(0, raw_frames - target_frames + 1, hop_frames))
        if starts[-1] + target_frames < raw_frames:
            starts.append(raw_frames - target_frames)
        starts = sorted(list(set(starts)))

        for idx, start in enumerate(starts):
            end = start + target_frames
            chunk = mel_db[:, start:end]
            mean = np.mean(chunk)
            std = np.std(chunk)
            norm_chunk = (chunk - mean) / (std + 1e-6)
            windows_tensors.append(norm_chunk)
            windows_info.append({
                "window_idx": idx,
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start * HOP_LENGTH / sr, 3),
                "end_sec": round(end * HOP_LENGTH / sr, 3)
            })

    batch_np = np.stack(windows_tensors, axis=0)
    tensor_batch = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1)

    meta = {
        "sample_rate": sr,
        "duration_sec": len(y) / sr,
        "original_frames": raw_frames,
        "target_frames": target_frames,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "hop_frames": hop_frames,
        "num_windows": len(windows_info),
        "is_windowed": len(windows_info) > 1
    }

    return tensor_batch, mel_db, windows_info, meta

def process_audio(
    audio_source: Union[str, bytes, io.BytesIO],
    sr: int = SAMPLE_RATE,
    target_frames: int = TARGET_FRAMES,
    high_pass_cutoff: Optional[float] = None,
    use_windows: bool = False,
    hop_frames: int = 125
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    High-level pipeline: loads audio signal and extracts normalized log-Mel spectrogram(s).

    Args:
        audio_source: Path to audio file or byte buffer.
        sr: Target sample rate (16000 Hz).
        target_frames: Number of time frames per window (250 = ~4.0s).
        high_pass_cutoff: Optional speech high-pass filter cutoff in Hz (e.g. 65.0 or 80.0).
        use_windows: Whether to extract overlapping windows for audio > 4s.
        hop_frames: Frame step between consecutive windows (125 = 50% overlap).

    Returns:
        tensor_input: PyTorch tensor (1, 1, 80, 250) or (num_windows, 1, 80, 250)
        mel_db_viz: numpy array in decibels for visualization
        meta: dictionary containing processing metadata
    """
    y, duration = load_audio_signal(audio_source, sr=sr, high_pass_cutoff=high_pass_cutoff)

    if use_windows:
        tensor_batch, mel_db_full, windows_info, meta = extract_mel_spectrogram_windows(
            y, sr=sr, target_frames=target_frames, hop_frames=hop_frames
        )
        meta["duration_sec"] = duration
        meta["high_pass_cutoff"] = high_pass_cutoff
        meta["windows_info"] = windows_info
        return tensor_batch, mel_db_full, meta
    else:
        tensor_input, mel_db_fixed, meta = extract_mel_spectrogram(
            y, sr=sr, target_frames=target_frames
        )
        meta["duration_sec"] = duration
        meta["high_pass_cutoff"] = high_pass_cutoff
        meta["is_windowed"] = False
        meta["num_windows"] = 1
        return tensor_input, mel_db_fixed, meta

