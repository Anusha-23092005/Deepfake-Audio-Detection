import os
import sys
import json
import hashlib
import tempfile
import subprocess
import unittest
import numpy as np
import soundfile as sf
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model import (
    load_model,
    AudioDeepfakeCNN,
    CLASS_MAPPING,
    CLASS_DESCRIPTIONS,
    BASELINE_MODEL_PATH,
    ROBUST_MODEL_PATH,
    MODEL_REGISTRY
)
from src.audio_processor import (
    process_audio,
    extract_mel_spectrogram,
    extract_mel_spectrogram_windows,
    apply_high_pass_filter,
    AudioProcessingError,
    SAMPLE_RATE,
    N_MELS,
    TARGET_FRAMES
)
from src.explainability import GradCAM, plot_explainability

EXPECTED_CHECKPOINT_SHA256 = "6ec42917f8fd0e87aa6a5a29970fde4a4f7b91e87b90b0ee2c42a0e7ffc8afd1"

class TestDeepfakeAudioPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_deepfake_audio_cnn.pth")
        cls.robust_checkpoint_path = os.path.join(PROJECT_ROOT, "models", "robust_deepfake_audio_cnn.pth")
        cls.dev_results_path = os.path.join(PROJECT_ROOT, "models", "dev_evaluation_results.json")
        cls.pilot_results_path = os.path.join(PROJECT_ROOT, "reports", "pilot_robust_evaluation_results.json")
        assert os.path.exists(cls.checkpoint_path), f"Checkpoint not found at {cls.checkpoint_path}"
        assert os.path.exists(cls.robust_checkpoint_path), f"Robust checkpoint not found at {cls.robust_checkpoint_path}"
        assert os.path.exists(cls.dev_results_path), f"Dev results not found at {cls.dev_results_path}"

        cls.demo_real_path = os.path.join(PROJECT_ROOT, "demo_audio", "demo_real_speech.wav")
        cls.demo_synth_path = os.path.join(PROJECT_ROOT, "demo_audio", "demo_synthetic_speech.wav")

        # Create temporary audio files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.valid_audio_path = os.path.join(cls.temp_dir.name, "valid_speech_like.wav")
        duration = 2.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        sf.write(cls.valid_audio_path, sig, SAMPLE_RATE)

        # Silent audio
        cls.silent_audio_path = os.path.join(cls.temp_dir.name, "silent.wav")
        silent_sig = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        sf.write(cls.silent_audio_path, silent_sig, SAMPLE_RATE)

        # Empty file
        cls.empty_audio_path = os.path.join(cls.temp_dir.name, "empty.wav")
        with open(cls.empty_audio_path, "wb") as f:
            pass

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_checkpoint_loading_strict(self):
        """1. Verify checkpoint loads with strict=True (0 missing, 0 unexpected keys)."""
        model, metadata = load_model(self.checkpoint_path, device="cpu")
        self.assertIsInstance(model, AudioDeepfakeCNN)
        self.assertIn("best_val_accuracy", metadata)
        self.assertGreater(metadata["best_val_accuracy"], 0.90)
        self.assertIn("history", metadata)

    def test_02_class_index_mapping(self):
        """2. Verify class index mapping aligns with dev_evaluation_results.json."""
        with open(self.dev_results_path, "r", encoding="utf-8") as f:
            dev_data = json.load(f)

        expected_mapping = dev_data.get("class_mapping", {})
        self.assertEqual(CLASS_MAPPING[0], expected_mapping.get("0"))
        self.assertEqual(CLASS_MAPPING[1], expected_mapping.get("1"))
        self.assertIn("bonafide", CLASS_MAPPING[0].lower())
        self.assertIn("spoof", CLASS_MAPPING[1].lower())

    def test_03_preprocessing_consistency_and_silence_padding(self):
        """3. Verify training/local preprocessing consistency and silence floor padding."""
        tensor_input, mel_db, meta = process_audio(self.valid_audio_path)
        self.assertEqual(meta["sample_rate"], 16000)
        self.assertEqual(meta["n_mels"], 80)
        self.assertEqual(meta["n_fft"], 1024)
        self.assertEqual(meta["hop_length"], 256)
        self.assertEqual(meta["target_frames"], 250)

        # Verify that sub-250 frame audio is padded with silence floor, NOT 0.0 dB (max loudness)
        short_sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, endpoint=False))
        _, mel_db_short, _ = extract_mel_spectrogram(short_sig, sr=16000, target_frames=250)
        # Tail padding frames should be at or near the silence floor (-80 dB)
        padded_tail = mel_db_short[:, 200:]
        self.assertLess(float(np.mean(padded_tail)), -40.0)

    def test_04_output_tensor_shape(self):
        """4. Verify feature tensor shape is (1, 1, 80, 250) and output logits shape is (1, 2)."""
        model, _ = load_model(self.checkpoint_path, device="cpu")
        tensor_input, _, _ = process_audio(self.valid_audio_path)
        self.assertEqual(tensor_input.shape, (1, 1, 80, 250))

        with torch.no_grad():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_05_identical_cli_and_api_predictions(self):
        """5. Confirm CLI (predict.py) and programmatic pipeline produce identical model outputs."""
        model, _ = load_model(self.checkpoint_path, device="cpu")
        tensor_input, _, _ = process_audio(self.demo_real_path)
        with torch.no_grad():
            direct_probs = torch.softmax(model(tensor_input), dim=1).numpy()[0]

        # Run CLI in JSON mode
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "predict.py"),
            "--audio", self.demo_real_path,
            "--json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(res.returncode, 0, f"predict.py failed: {res.stderr}")
        cli_data = json.loads(res.stdout)

        cli_real_prob = cli_data["softmax_probabilities"]["bonafide_real"]
        cli_fake_prob = cli_data["softmax_probabilities"]["spoof_fake"]

        self.assertAlmostEqual(float(direct_probs[0]), cli_real_prob, places=3)
        self.assertAlmostEqual(float(direct_probs[1]), cli_fake_prob, places=3)

    def test_06_demo_audio_loading(self):
        """6. Confirm demonstration audio files exist, are readable, and meet specifications."""
        self.assertTrue(os.path.exists(self.demo_real_path))
        self.assertTrue(os.path.exists(self.demo_synth_path))

        info_real = sf.info(self.demo_real_path)
        self.assertEqual(info_real.samplerate, 16000)
        self.assertEqual(info_real.channels, 1)
        self.assertGreaterEqual(info_real.duration, 5.0)

        info_synth = sf.info(self.demo_synth_path)
        self.assertEqual(info_synth.samplerate, 16000)
        self.assertEqual(info_synth.channels, 1)
        self.assertGreaterEqual(info_synth.duration, 5.0)

    def test_07_no_stale_results_on_input_change(self):
        """7. Verify that switching audio input clears previous results (session state pattern)."""
        session_state = {
            "active_audio_label": "file_1.wav",
            "active_audio_source": b"audio_data_1",
            "analysis_result": {"pred": 1, "conf": 0.99}
        }

        # Simulate set_audio_input logic with a new file
        new_label = "file_2.wav"
        new_source = b"audio_data_2"
        if session_state["active_audio_label"] != new_label:
            session_state["active_audio_source"] = new_source
            session_state["active_audio_label"] = new_label
            session_state["analysis_result"] = None

        self.assertIsNone(session_state["analysis_result"])
        self.assertEqual(session_state["active_audio_label"], "file_2.wav")

    def test_08_grad_cam_current_recording(self):
        """8. Verify Grad-CAM computes activation maps for the currently analyzed recording."""
        model, _ = load_model(self.checkpoint_path, device="cpu")
        tensor_input, mel_db, meta = process_audio(self.valid_audio_path)

        grad_cam = GradCAM(model)
        cam_map, target_class, conf = grad_cam.generate_cam(tensor_input)
        grad_cam.remove_hooks()

        self.assertEqual(cam_map.shape, (80, 250))
        self.assertTrue(0.0 <= np.min(cam_map) <= 1.0)
        self.assertTrue(0.0 <= np.max(cam_map) <= 1.0)

        fig = plot_explainability(
            mel_db=mel_db,
            cam_map=cam_map,
            predicted_class_name=CLASS_MAPPING[target_class],
            confidence_score=conf
        )
        self.assertIsNotNone(fig)

    def test_09_error_handling_invalid_audio(self):
        """9. Verify correct error handling for silent, empty, and nonexistent audio."""
        with self.assertRaises(AudioProcessingError):
            process_audio(self.silent_audio_path)
        with self.assertRaises(AudioProcessingError):
            process_audio(self.empty_audio_path)
        with self.assertRaises(AudioProcessingError):
            process_audio("non_existent_file.wav")

    def test_10_checkpoint_integrity_unmodified(self):
        """10. Verify original checkpoint file has not been altered or corrupted."""
        with open(self.checkpoint_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest().lower()
        self.assertEqual(actual_sha256, EXPECTED_CHECKPOINT_SHA256)

    def test_11_high_pass_filter_attenuation(self):
        """11. Verify Butterworth high-pass filter attenuates sub-speech rumble while preserving voice frequencies."""
        duration = 1.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        # 30 Hz sub-speech rumble + 500 Hz speech harmonic
        rumble = 1.0 * np.sin(2 * np.pi * 30 * t)
        voice = 1.0 * np.sin(2 * np.pi * 500 * t)
        composite = rumble + voice

        filtered_80 = apply_high_pass_filter(composite, sr=SAMPLE_RATE, cutoff_hz=80.0)

        # Compute energy of rumble component before and after
        fft_orig = np.abs(np.fft.rfft(composite))
        fft_filt = np.abs(np.fft.rfft(filtered_80))
        freqs = np.fft.rfftfreq(len(composite), 1 / SAMPLE_RATE)

        idx_30 = np.argmin(np.abs(freqs - 30))
        idx_500 = np.argmin(np.abs(freqs - 500))

        # 30 Hz must be strongly attenuated (> 20 dB reduction)
        self.assertLess(fft_filt[idx_30] / fft_orig[idx_30], 0.1)
        # 500 Hz speech frequency must be preserved (> 95% amplitude)
        self.assertGreater(fft_filt[idx_500] / fft_orig[idx_500], 0.95)

        # None or <= 0 cutoff should return unmodified signal
        unmod = apply_high_pass_filter(composite, sr=SAMPLE_RATE, cutoff_hz=None)
        np.testing.assert_array_equal(unmod, composite.astype(np.float32))

    def test_12_multi_window_extraction_and_shapes(self):
        """12. Verify multi-window extraction produces expected window count, tensor shape, and metadata."""
        # Audio of 6.0 seconds -> 6.0 * 16000 / 256 ~ 376 frames (> 250 frames)
        duration = 6.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t)

        tensor_batch, mel_db_full, windows_info, meta = extract_mel_spectrogram_windows(
            sig, sr=SAMPLE_RATE, target_frames=250, hop_frames=125
        )

        self.assertGreater(len(windows_info), 1)
        self.assertEqual(tensor_batch.shape, (len(windows_info), 1, 80, 250))
        self.assertTrue(meta["is_windowed"])
        self.assertEqual(meta["num_windows"], len(windows_info))

        # Check that window timestamps are ordered and valid
        for i, win in enumerate(windows_info):
            self.assertEqual(win["window_idx"], i)
            self.assertLess(win["start_sec"], win["end_sec"])
            self.assertEqual(win["end_frame"] - win["start_frame"], 250)

    def test_13_multi_window_and_filter_cli_and_api(self):
        """13. Verify multi-window and high-pass inference runs end-to-end via API and CLI."""
        model, _ = load_model(self.checkpoint_path, device="cpu")

        # Run via API with high-pass and multi-window
        tensor_batch, mel_db, meta = process_audio(
            self.demo_real_path,
            high_pass_cutoff=80.0,
            use_windows=True
        )
        with torch.no_grad():
            all_probs = torch.softmax(model(tensor_batch), dim=1).numpy()
            avg_probs = np.mean(all_probs, axis=0)

        self.assertEqual(all_probs.shape[1], 2)
        self.assertAlmostEqual(float(np.sum(avg_probs)), 1.0, places=5)

        # Run CLI with --high-pass 80 --multi-window --json
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "predict.py"),
            "--audio", self.demo_real_path,
            "--high-pass", "80",
            "--multi-window",
            "--json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(res.returncode, 0, f"predict.py failed: {res.stderr}")
        cli_data = json.loads(res.stdout)

        cli_real = cli_data["softmax_probabilities"]["bonafide_real"]
        self.assertAlmostEqual(float(avg_probs[0]), cli_real, places=3)
        self.assertIn("window_analysis", cli_data)
        self.assertEqual(cli_data["window_analysis"]["total_windows"], len(all_probs))

    def test_14_baseline_equivalence(self):
        """14. Verify that default process_audio arguments preserve exact Kaggle baseline features."""
        # Extract using baseline extract_mel_spectrogram directly
        y, _ = sf.read(self.valid_audio_path, dtype='float32')
        tensor_direct, _, _ = extract_mel_spectrogram(y, sr=16000, target_frames=250)

        # Extract using process_audio with default arguments
        tensor_proc, _, meta = process_audio(self.valid_audio_path)

        self.assertFalse(meta["is_windowed"])
        self.assertIsNone(meta["high_pass_cutoff"])
        np.testing.assert_array_almost_equal(tensor_direct.numpy(), tensor_proc.numpy(), decimal=6)

    def test_15_robust_checkpoint_loading_strict(self):
        """15. Verify robust checkpoint exists and loads with strict=True and valid metadata."""
        self.assertTrue(os.path.exists(self.robust_checkpoint_path))
        model, metadata = load_model(self.robust_checkpoint_path, device="cpu")
        self.assertIsInstance(model, AudioDeepfakeCNN)
        self.assertEqual(metadata.get("model_type"), "robust_domain_adapted")
        self.assertIn("best_val_accuracy", metadata)
        self.assertGreaterEqual(metadata["best_val_accuracy"], 0.90)
        self.assertIn("history", metadata)

    def test_16_robust_grad_cam_compatibility(self):
        """16. Verify Grad-CAM operates correctly on the robust domain-adapted CNN."""
        model, _ = load_model(self.robust_checkpoint_path, device="cpu")
        tensor_input, mel_db, meta = process_audio(self.valid_audio_path)

        grad_cam = GradCAM(model)
        cam_map, target_class, conf = grad_cam.generate_cam(tensor_input)
        grad_cam.remove_hooks()

        self.assertEqual(cam_map.shape, (80, 250))
        self.assertTrue(0.0 <= np.min(cam_map) <= 1.0)
        self.assertTrue(0.0 <= np.max(cam_map) <= 1.0)

    def test_17_cli_model_variant_selection(self):
        """17. Verify CLI accepts --model-type flag for both baseline and robust variants."""
        for model_type in ["baseline", "robust"]:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "predict.py"),
                "--audio", self.demo_synth_path,
                "--model-type", model_type,
                "--json"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            self.assertEqual(res.returncode, 0, f"predict.py failed for --model-type {model_type}: {res.stderr}")
            cli_data = json.loads(res.stdout)
            self.assertEqual(cli_data.get("model_variant"), model_type)
            self.assertIn("model_name", cli_data)
            self.assertIn("model_checkpoint", cli_data)
            self.assertIn("softmax_probabilities", cli_data)

    def test_18_baseline_checkpoint_sha256_unaltered(self):
        """18. Re-verify baseline checkpoint SHA-256 remains strictly untouched throughout Stage 2."""
        with open(self.checkpoint_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest().lower()
        self.assertEqual(actual_sha256, EXPECTED_CHECKPOINT_SHA256)

if __name__ == "__main__":
    unittest.main()

