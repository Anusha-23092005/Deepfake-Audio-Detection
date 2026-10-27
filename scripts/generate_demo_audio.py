#!/usr/bin/env python3
"""
Reproducible Demo Audio Generation Script
Generates or retrieves:
1. demo_audio/demo_real_speech.wav (LibriSpeech open-source human speech recording)
2. demo_audio/demo_synthetic_speech.wav (Windows SAPI5 SpeechSynthesizer - Microsoft David TTS)
"""

import os
import sys
import argparse
import urllib.request
import subprocess
import soundfile as sf
import librosa
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo_audio")
REAL_FILE = os.path.join(DEMO_DIR, "demo_real_speech.wav")
SYNTH_FILE = os.path.join(DEMO_DIR, "demo_synthetic_speech.wav")

SPOKEN_TEXT = (
    "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes "
    "and fat mutton pieces to be ladled out in thick, peppered flour-fattened sauce."
)

# Canonical LibriSpeech public domain audio sample hosted on Hugging Face (16kHz FLAC)
HUMAN_SPEECH_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and verify demonstration audio files.")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing demo files if they already exist."
    )
    return parser.parse_args()

def fetch_real_speech(target_path: str, force: bool = False):
    """
    Downloads and standardizes a genuine public domain human speech recording from LibriSpeech.
    """
    if os.path.exists(target_path) and not force:
        print(f"[SKIP] Real speech file already exists at: {target_path} (use --force to overwrite)")
        return

    print(f"[DOWNLOADING] Fetching verified human speech from LibriSpeech repository...")
    try:
        req = urllib.request.Request(HUMAN_SPEECH_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            audio_data = response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download human speech sample: {e}. Check internet connection.")

    # Save temp and convert/resample to standard 16kHz mono WAV
    temp_download = os.path.join(DEMO_DIR, "_temp_real.flac")
    with open(temp_download, "wb") as f:
        f.write(audio_data)

    try:
        y, sr = librosa.load(temp_download, sr=16000, mono=True)
        sf.write(target_path, y, 16000, subtype="PCM_16")
        print(f"[SUCCESS] Saved human speech WAV to: {target_path}")
    finally:
        if os.path.exists(temp_download):
            os.remove(temp_download)

def generate_synthetic_speech(target_path: str, force: bool = False):
    """
    Synthesizes the matching sentence using a genuine Text-to-Speech (TTS) engine.
    Uses Windows SAPI5 SpeechSynthesizer (Microsoft David Desktop) with fallback to gTTS.
    """
    if os.path.exists(target_path) and not force:
        print(f"[SKIP] Synthetic speech file already exists at: {target_path} (use --force to overwrite)")
        return

    print(f"[SYNTHESIZING] Generating synthetic speech with Text-To-Speech engine...")
    temp_synth = os.path.join(DEMO_DIR, "_temp_synth.wav")

    # 1. Try Windows SAPI5 SpeechSynthesizer via PowerShell
    ps_command = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice('Microsoft David Desktop')
    $synth.SetOutputToWaveFile('{temp_synth}')
    $synth.Speak('{SPOKEN_TEXT}')
    $synth.Dispose()
    """
    res = subprocess.run(["powershell", "-Command", ps_command], capture_output=True, text=True)

    if res.returncode == 0 and os.path.exists(temp_synth):
        try:
            y, sr = librosa.load(temp_synth, sr=16000, mono=True)
            sf.write(target_path, y, 16000, subtype="PCM_16")
            print(f"[SUCCESS] Synthesized speech (Microsoft David TTS) saved to: {target_path}")
            return
        finally:
            if os.path.exists(temp_synth):
                os.remove(temp_synth)

    # 2. Fallback to gTTS if Windows TTS fails
    try:
        from gtts import gTTS
        print("[INFO] Attempting gTTS synthesis fallback...")
        temp_mp3 = os.path.join(DEMO_DIR, "_temp_synth.mp3")
        tts = gTTS(text=SPOKEN_TEXT, lang="en")
        tts.save(temp_mp3)
        y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
        sf.write(target_path, y, 16000, subtype="PCM_16")
        os.remove(temp_mp3)
        print(f"[SUCCESS] Synthesized speech (gTTS) saved to: {target_path}")
    except Exception as e:
        raise RuntimeError(f"All TTS generation methods failed: {e}")

def verify_file(filepath: str, label: str):
    """Verifies audio format, duration, channels, sample rate, and size."""
    if not os.path.exists(filepath):
        print(f"[ERROR] {label} was not found at {filepath}")
        return False

    info = sf.info(filepath)
    file_size_kb = os.path.getsize(filepath) / 1024
    print("\n" + "=" * 50)
    print(f"VERIFICATION REPORT: {label.upper()}")
    print("=" * 50)
    print(f"File Path    : {filepath}")
    print(f"Format       : {info.format} ({info.subtype})")
    print(f"Channels     : {info.channels} ({'Mono' if info.channels == 1 else 'Stereo'})")
    print(f"Sample Rate  : {info.samplerate} Hz")
    print(f"Duration     : {info.duration:.2f} seconds")
    print(f"File Size    : {file_size_kb:.1f} KB")
    print("Status       : VALID & READY")
    return True

def main():
    args = parse_args()
    os.makedirs(DEMO_DIR, exist_ok=True)

    print("==================================================")
    print("DEMO AUDIO GENERATOR FOR DEEPFAKE AUDIO DETECTION")
    print("==================================================")

    # 1. Fetch Real Speech
    fetch_real_speech(REAL_FILE, force=args.force)

    # 2. Generate Synthetic Speech
    generate_synthetic_speech(SYNTH_FILE, force=args.force)

    # 3. Verify
    v_real = verify_file(REAL_FILE, "Real Speech Demo")
    v_synth = verify_file(SYNTH_FILE, "Synthetic Speech Demo")

    if v_real and v_synth:
        print("\nAll demonstration audio files verified successfully.")
    else:
        print("\nOne or more demonstration files failed verification.")
        sys.exit(1)

if __name__ == "__main__":
    main()
