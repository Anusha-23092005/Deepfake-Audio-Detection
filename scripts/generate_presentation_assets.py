import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
import librosa

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

out_dir = os.path.join(PROJECT_ROOT, "scratch", "presentation_assets")
os.makedirs(out_dir, exist_ok=True)

from src.model import load_model, ROBUST_MODEL_PATH
from src.audio_processor import process_audio, extract_mel_spectrogram
from src.explainability import GradCAM

# Colors for Dark Theme
DARK_BG = "#0B1F3A"
CARD_BG = "#102A4C"
CYAN = "#38BDF8"
WHITE = "#FFFFFF"
MUTED = "#94A3B8"
GREEN = "#22C55E"
RED = "#EF4444"
BORDER_BLUE = "#1E4976"

# ==============================================================
# 1. Title / Waveform Graphic
# ==============================================================
fig, ax = plt.subplots(figsize=(10, 2), dpi=200)
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

t = np.linspace(0, 10, 1000)
w1 = 0.6 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.08 * (t - 5)**2)
w2 = 0.3 * np.sin(2 * np.pi * 4.2 * t) * np.exp(-0.05 * (t - 5)**2)
w3 = 0.15 * np.sin(2 * np.pi * 9.8 * t)
wave = w1 + w2 + w3

ax.plot(t, wave, color=CYAN, linewidth=2.0, alpha=0.95)
ax.fill_between(t, wave, 0, color=CYAN, alpha=0.2)
ax.plot(t, -wave * 0.7, color=GREEN, linewidth=1.5, alpha=0.85)
ax.fill_between(t, -wave * 0.7, 0, color=GREEN, alpha=0.15)

ax.axis('off')
plt.tight_layout()
wf_path = os.path.join(out_dir, 'title_waveform.png')
plt.savefig(wf_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print('Generated:', wf_path)

# ==============================================================
# 2. Spectrogram Comparison (LibriSpeech Genuine vs SAPI5 Synthetic)
# ==============================================================
real_path = r'demo_audio\demo_real_speech.wav'
synth_path = r'demo_audio\demo_synthetic_speech.wav'

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
fig.patch.set_facecolor(DARK_BG)
ax1.set_facecolor(CARD_BG)
ax2.set_facecolor(CARD_BG)

y_real, _ = sf.read(real_path)
y_synth, _ = sf.read(synth_path)

_, mel_real, _ = extract_mel_spectrogram(y_real[:16000*3], sr=16000, target_frames=250)
_, mel_synth, _ = extract_mel_spectrogram(y_synth[:16000*3], sr=16000, target_frames=250)

im1 = ax1.imshow(mel_real, aspect='auto', origin='lower', cmap='magma')
ax1.set_title('Genuine Human Speech (LibriSpeech)\nContinuous pitch contours & natural formants', fontsize=12, fontweight='bold', color=GREEN, pad=10)
ax1.set_ylabel('Mel Frequency Bins (0 - 80)', fontsize=10, color=WHITE)
ax1.set_xlabel('Time Frames (0 - 250)', fontsize=10, color=WHITE)
ax1.tick_params(colors=MUTED, labelsize=9)

im2 = ax2.imshow(mel_synth, aspect='auto', origin='lower', cmap='magma')
ax2.set_title('Synthetic AI Speech (Microsoft David TTS)\nRigid spectral tracks & boundary artifacts', fontsize=12, fontweight='bold', color=RED, pad=10)
ax2.set_ylabel('Mel Frequency Bins (0 - 80)', fontsize=10, color=WHITE)
ax2.set_xlabel('Time Frames (0 - 250)', fontsize=10, color=WHITE)
ax2.tick_params(colors=MUTED, labelsize=9)

plt.tight_layout()
comp_path = os.path.join(out_dir, 'spectrogram_comparison.png')
plt.savefig(comp_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print('Generated:', comp_path)

# ==============================================================
# 3. Grad-CAM Saliency Map Overlay
# ==============================================================
robust_model, _ = load_model(ROBUST_MODEL_PATH, device='cpu')
tensor_synth, mel_db_s, _ = process_audio(synth_path)
cam = GradCAM(robust_model)
cam_map_s, pred_idx_s, conf_s = cam.generate_cam(tensor_synth[:1], target_class=1)
cam.remove_hooks()

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, dpi=200)
fig.patch.set_facecolor(DARK_BG)
ax_top.set_facecolor(CARD_BG)
ax_bot.set_facecolor(CARD_BG)

time_sec = (250 * 256) / 16000

im_top = ax_top.imshow(mel_db_s[:, :250], aspect="auto", origin="lower", cmap="magma", extent=[0, time_sec, 0, 80])
ax_top.set_title("Input Log-Mel Spectrogram (dB) — Synthetic AI Speech", fontsize=11, fontweight="bold", color=WHITE)
ax_top.set_ylabel("Mel Bins", fontsize=10, color=WHITE)
ax_top.tick_params(colors=MUTED, labelsize=9)
cbar_top = fig.colorbar(im_top, ax=ax_top, pad=0.02)
cbar_top.set_label("dB", fontsize=9, color=WHITE)
cbar_top.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar_top.ax.yaxis.get_ticklabels(), color=MUTED)

ax_bot.imshow(mel_db_s[:, :250], aspect="auto", origin="lower", cmap="gray", alpha=0.35, extent=[0, time_sec, 0, 80])
im_bot = ax_bot.imshow(cam_map_s, aspect="auto", origin="lower", cmap="jet", alpha=0.7, extent=[0, time_sec, 0, 80])
ax_bot.set_title("Grad-CAM Saliency Overlay — Layer: features[8] (Attention on Vocoder Artifacts)", fontsize=11, fontweight="bold", color=CYAN)
ax_bot.set_xlabel("Time (seconds)", fontsize=10, color=WHITE)
ax_bot.set_ylabel("Mel Bins", fontsize=10, color=WHITE)
ax_bot.tick_params(colors=MUTED, labelsize=9)
cbar_bot = fig.colorbar(im_bot, ax=ax_bot, pad=0.02)
cbar_bot.set_label("Saliency", fontsize=9, color=WHITE)
cbar_bot.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar_bot.ax.yaxis.get_ticklabels(), color=MUTED)

fig.text(0.5, 0.01, "Note: Grad-CAM visualizes internal model feature activations. It provides explainability, not definitive proof.",
         ha="center", fontsize=8.5, color=MUTED, style="italic")

plt.tight_layout()
cam_path = os.path.join(out_dir, 'gradcam_saliency.png')
plt.savefig(cam_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print('Generated:', cam_path)

# ==============================================================
# 4. Evaluation Metric Comparison Chart
# ==============================================================
fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', '100% - FPR']
baseline_vals = [96.67, 93.75, 100.0, 96.77, 93.33]
robust_vals = [100.0, 100.0, 100.0, 100.0, 100.0]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline CNN (ASVspoof 2019)', color=CYAN, edgecolor=WHITE, linewidth=0.8)
rects2 = ax.bar(x + width/2, robust_vals, width, label='Robust Domain-Adapted CNN (Stage 2)', color=GREEN, edgecolor=WHITE, linewidth=0.8)

ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold', color=WHITE)
ax.set_title('Pilot Benchmark Evaluation Comparison (30 Samples, Disjoint Speakers)', fontsize=12, fontweight='bold', color=WHITE, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10, fontweight='bold', color=WHITE)
ax.set_ylim(85, 106)
ax.axhline(100, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.6)
ax.legend(frameon=True, facecolor=CARD_BG, edgecolor=BORDER_BLUE, fontsize=9, loc='lower right', labelcolor=WHITE)
ax.grid(axis='y', linestyle=':', alpha=0.3, color=MUTED)
ax.tick_params(colors=MUTED, labelsize=10)

for r in rects1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color=CYAN)
for r in rects2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color=GREEN)

plt.tight_layout()
eval_path = os.path.join(out_dir, 'evaluation_chart.png')
plt.savefig(eval_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
plt.close()
print('Generated:', eval_path)

# ==============================================================
# 5. Schematic App UI Illustration (Blue Theme)
# ==============================================================
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor(DARK_BG)

# Browser Bar
rect_browser = plt.Rectangle((1, 93), 98, 6, facecolor='#123B68', edgecolor=BORDER_BLUE)
ax.add_patch(rect_browser)
ax.text(3, 95.5, '● ● ●', color=CYAN, fontsize=12, fontweight='bold', va='center')
ax.text(12, 95.8, 'http://localhost:8501 — DeepFake Audio Detection (Streamlit Web App)', color=WHITE, fontsize=9.5, va='center')

# Sidebar
patch_side = plt.Rectangle((1, 1), 26, 92, facecolor='#0D223D', edgecolor=BORDER_BLUE, linewidth=1.2)
ax.add_patch(patch_side)
ax.text(3, 88, 'Model Selection', fontsize=12, fontweight='bold', color=WHITE)
ax.text(3, 84, 'Active Detection Model:', fontsize=9.5, fontweight='bold', color=CYAN)

# Radio 1 (Robust - Selected)
rect_r1 = plt.Rectangle((3, 75), 22, 7, facecolor='#0E382B', edgecolor=GREEN, linewidth=1.2)
ax.add_patch(rect_r1)
ax.text(5, 79.5, '(*) Robust Domain-Adapted', fontsize=8.5, fontweight='bold', color=GREEN)
ax.text(5, 76.5, '    Stage 2 Checkpoint (Active)', fontsize=7.5, color='#A7F3D0')

# Radio 2 (Baseline)
rect_r2 = plt.Rectangle((3, 66), 22, 7, facecolor='#102A4C', edgecolor=BORDER_BLUE, linewidth=0.8)
ax.add_patch(rect_r2)
ax.text(5, 70.5, '( ) Baseline CNN', fontsize=8.5, fontweight='bold', color=MUTED)
ax.text(5, 67.5, '    ASVspoof 2019 Checkpoint', fontsize=7.5, color='#64748B')

# App Header
ax.text(30, 88, 'DeepFake Audio Detection System', fontsize=16, fontweight='bold', color=WHITE)
ax.text(30, 84, 'Explainable AI for Synthetic Speech Detection | Active: Robust Domain-Adapted CNN', fontsize=9.5, color=CYAN, fontweight='bold')

# Active Model Banner
rect_abanner = plt.Rectangle((30, 77), 67, 5, facecolor='#0E382B', edgecolor=GREEN, linewidth=1)
ax.add_patch(rect_abanner)
ax.text(31.5, 79.5, 'ACTIVE MODEL: Robust Domain-Adapted CNN (Stage 2)', fontsize=8.5, fontweight='bold', color=GREEN, va='center')
ax.text(78, 79.5, 'Toggle model in sidebar <-', fontsize=7.5, color=CYAN, va='center')

# Left Column - Audio Input
rect_left = plt.Rectangle((30, 16), 32, 58, facecolor=CARD_BG, edgecolor=BORDER_BLUE, linewidth=1)
ax.add_patch(rect_left)
ax.text(32, 70, 'Analyze Audio', fontsize=12, fontweight='bold', color=WHITE)
ax.text(32, 65, 'Source: Upload Audio File', fontsize=9, color=MUTED)
rect_up = plt.Rectangle((32, 45), 28, 17, facecolor='#132E50', edgecolor=CYAN, linestyle='--', linewidth=1.2)
ax.add_patch(rect_up)
ax.text(46, 55, 'Drag & Drop Voice Note', ha='center', fontsize=9.5, color=WHITE)
ax.text(46, 49, 'WhatsApp Voice PTT (5.69s, Opus/OGG)', ha='center', fontsize=8.5, color=CYAN, fontweight='bold')

rect_btn = plt.Rectangle((32, 23), 28, 8, facecolor='#0284C7', edgecolor='#38BDF8', linewidth=1)
ax.add_patch(rect_btn)
ax.text(46, 27, 'Analyze Audio', ha='center', va='center', fontsize=11, fontweight='bold', color=WHITE)

# Right Column - Results
rect_right = plt.Rectangle((65, 16), 32, 58, facecolor=CARD_BG, edgecolor=BORDER_BLUE, linewidth=1)
ax.add_patch(rect_right)
ax.text(67, 70, 'Analysis Results', fontsize=12, fontweight='bold', color=WHITE)

# Evaluated Model Banner
rect_eval_b = plt.Rectangle((67, 59), 28, 8, facecolor='#0E382B', edgecolor=GREEN, linewidth=1)
ax.add_patch(rect_eval_b)
ax.text(68.5, 64.5, 'Evaluated: Robust Domain-Adapted CNN', fontsize=8, fontweight='bold', color=GREEN)
ax.text(68.5, 61.5, 'Checkpoint: robust_deepfake_audio_cnn.pth', fontsize=7.5, color='#A7F3D0')

# Verdict Card
rect_verd = plt.Rectangle((67, 43), 28, 13, facecolor='#0E382B', edgecolor=GREEN, linewidth=1.5)
ax.add_patch(rect_verd)
ax.text(81, 51.5, 'REAL SPEECH', ha='center', fontsize=14, fontweight='bold', color=GREEN)
ax.text(81, 46.5, 'Classified as Bona fide Human Speech', ha='center', fontsize=8.5, color='#A7F3D0')

# Softmax Bars
ax.text(67, 39, 'Model Scores (Softmax):', fontsize=9, fontweight='bold', color=CYAN)
ax.text(67, 34, 'Real: 95.22%', fontsize=8.5, fontweight='bold', color=GREEN)
rect_bar1_bg = plt.Rectangle((67, 30), 28, 3, facecolor='#1E4976')
rect_bar1_fg = plt.Rectangle((67, 30), 28 * 0.9522, 3, facecolor=GREEN)
ax.add_patch(rect_bar1_bg)
ax.add_patch(rect_bar1_fg)

ax.text(67, 26, 'Spoof: 4.78%', fontsize=8.5, fontweight='bold', color=RED)
rect_bar2_bg = plt.Rectangle((67, 22), 28, 3, facecolor='#1E4976')
rect_bar2_fg = plt.Rectangle((67, 22), 28 * 0.0478, 3, facecolor=RED)
ax.add_patch(rect_bar2_bg)
ax.add_patch(rect_bar2_fg)

# Tabs at bottom
rect_tabs = plt.Rectangle((30, 6), 67, 7, facecolor='#123B68', edgecolor=BORDER_BLUE, linewidth=1)
ax.add_patch(rect_tabs)
ax.text(33, 9.5, 'Tabs: [ Audio Analysis ]   [ Model Explanation (Grad-CAM) ]   [ How It Works ]   [ About & Benchmarks ]', fontsize=8.5, color=CYAN, fontweight='bold')

plt.tight_layout()
ui_path = os.path.join(out_dir, 'app_ui_mockup.png')
plt.savefig(ui_path, bbox_inches='tight', dpi=200)
plt.close()
print('Generated:', ui_path)
