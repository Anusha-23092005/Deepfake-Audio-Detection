import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "scratch", "presentation_assets")
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "DeepFake_Audio_Detection_Presentation.pptx")

# ==========================================
# CONSISTENT BLUE THEME PALETTE
# ==========================================
DARK_NAVY = RGBColor(11, 31, 58)      # #0B1F3A - Base Slide Background
MID_BLUE = RGBColor(18, 59, 104)      # #123B68 - Primary Container Blue
CARD_BLUE = RGBColor(16, 42, 76)      # #102A4C - Card Panel Background
BORDER_BLUE = RGBColor(30, 73, 118)   # #1E4976 - Subtle Border
CYAN_ACCENT = RGBColor(56, 189, 248)  # #38BDF8 - Bright Cyan Accent / Highlights
WHITE = RGBColor(255, 255, 255)       # #FFFFFF - Primary Text
TEXT_LIGHT = RGBColor(226, 232, 240)  # #E2E8F0 - Secondary Light Text
TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8 - Subtitle / Muted Text

# Green for positive results & genuine speech
GREEN_BG = RGBColor(14, 56, 43)       # #0E382B - Dark Green Card Fill
GREEN_BORDER = RGBColor(34, 197, 94)  # #22C55E - Bright Green Border
GREEN_TEXT = RGBColor(74, 222, 128)   # #4ADE80 - High-Contrast Green Text

# Red for spoof detection & critical warnings
RED_BG = RGBColor(59, 20, 30)         # #3B141E - Dark Red Card Fill
RED_BORDER = RGBColor(239, 68, 68)    # #EF4444 - Bright Red Border
RED_TEXT = RGBColor(248, 113, 113)    # #F87171 - High-Contrast Red Text

FONT_FAMILY = "Segoe UI"
TOTAL_SLIDES = 14

def set_slide_background(slide):
    """Sets consistent dark navy background on every slide."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_NAVY
    bg.line.fill.background()
    return bg

def add_header(slide, title_text, category_text="ACADEMIC PROJECT PRESENTATION"):
    """Adds consistent header with large white title and cyan category tag."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.733), Inches(1.05))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = FONT_FAMILY
    p_cat.font.size = Pt(13)
    p_cat.font.bold = True
    p_cat.font.color.rgb = CYAN_ACCENT

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_FAMILY
    p_title.font.size = Pt(30)
    p_title.font.bold = True
    p_title.font.color.rgb = WHITE
    p_title.space_before = Pt(2)

    # Accent divider line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(1.48), Inches(11.733), Inches(0.025)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_BLUE
    line.line.fill.background()

def add_footer(slide, slide_num):
    """Adds standard university project footer with 14-slide total counter."""
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.015)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_BLUE
    line.line.fill.background()

    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.06), Inches(11.733), Inches(0.35))
    tf = footer_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p = tf.paragraphs[0]
    p.font.name = FONT_FAMILY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED

    r_left = p.add_run()
    r_left.text = "Deep Learning-Based Explainable DeepFake Audio Detection System  |  "

    r_center = p.add_run()
    r_center.text = "College Project Review  |  "

    r_right = p.add_run()
    r_right.text = f"Slide {slide_num} of {TOTAL_SLIDES}"
    r_right.font.bold = True
    r_right.font.color.rgb = CYAN_ACCENT

def add_card(slide, left, top, width, height, bg_color=CARD_BLUE, border_color=BORDER_BLUE, border_width=1):
    """Creates a rounded container card for visual hierarchy."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_width)
    else:
        shape.line.fill.background()
    return shape

# ==========================================
# PRESENTATION INITIALIZATION
# ==========================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[6]

print(f"Building {TOTAL_SLIDES}-Slide Academic PowerPoint Presentation (Consistent Blue Theme)...")

# ==============================================================
# SLIDE 1: TITLE SLIDE (Cover Page & All 5 Team Members)
# ==============================================================
s1 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s1)

# Double Border
border_out = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(0.4), Inches(12.533), Inches(6.7))
border_out.fill.background()
border_out.line.color.rgb = CYAN_ACCENT
border_out.line.width = Pt(1.5)

border_in = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.46), Inches(0.46), Inches(12.413), Inches(6.58))
border_in.fill.background()
border_in.line.color.rgb = BORDER_BLUE
border_in.line.width = Pt(0.75)

# Institution & Department Header
header_tb = s1.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.733), Inches(0.85))
tf1 = header_tb.text_frame
tf1.word_wrap = True

p1 = tf1.paragraphs[0]
p1.alignment = PP_ALIGN.CENTER
p1.text = "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING"
p1.font.name = FONT_FAMILY
p1.font.size = Pt(13)
p1.font.bold = True
p1.font.color.rgb = CYAN_ACCENT

p2 = tf1.add_paragraph()
p2.alignment = PP_ALIGN.CENTER
p2.text = "ACADEMIC CAPSTONE PROJECT REVIEW  •  2025 – 2026"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(11)
p2.font.color.rgb = TEXT_LIGHT
p2.space_before = Pt(3)

# Project Title & Subtitle Box
title_tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.85))
tf_title = title_tb.text_frame
tf_title.word_wrap = True

p_t1 = tf_title.paragraphs[0]
p_t1.alignment = PP_ALIGN.CENTER
p_t1.text = "DEEP LEARNING-BASED EXPLAINABLE DEEPFAKE AUDIO DETECTION SYSTEM"
p_t1.font.name = FONT_FAMILY
p_t1.font.size = Pt(31)
p_t1.font.bold = True
p_t1.font.color.rgb = WHITE

p_t2 = tf_title.add_paragraph()
p_t2.alignment = PP_ALIGN.CENTER
p_t2.text = "Distinguishing Authentic and Synthetic Speech with 2D CNNs & Grad-CAM Saliency"
p_t2.font.name = FONT_FAMILY
p_t2.font.size = Pt(16)
p_t2.font.bold = True
p_t2.font.color.rgb = CYAN_ACCENT
p_t2.space_before = Pt(8)

# Embedded Waveform Accent
wf_path = os.path.join(ASSETS_DIR, "title_waveform.png")
if os.path.exists(wf_path):
    s1.shapes.add_picture(wf_path, Inches(2.66), Inches(3.6), Inches(8.0), Inches(1.0))

# Team Members Card
c_team = add_card(s1, Inches(1.5), Inches(4.75), Inches(10.333), Inches(2.0), bg_color=CARD_BLUE, border_color=CYAN_ACCENT, border_width=1.5)

tb_th = s1.shapes.add_textbox(Inches(1.7), Inches(4.85), Inches(9.933), Inches(0.4))
tf_th = tb_th.text_frame
pth = tf_th.paragraphs[0]
pth.alignment = PP_ALIGN.CENTER
pth.text = "PROJECT TEAM MEMBERS"
pth.font.name = FONT_FAMILY
pth.font.size = Pt(14)
pth.font.bold = True
pth.font.color.rgb = CYAN_ACCENT

# Column 1 (Members 1, 2, 3)
tb_col1 = s1.shapes.add_textbox(Inches(2.2), Inches(5.3), Inches(4.4), Inches(1.3))
tfc1 = tb_col1.text_frame
tfc1.word_wrap = True

members_c1 = [
    "SREENITHI K - 23MIS0382",
    "ANUSHA R - 23MIS0362",
    "SANJANA B - 23MIS0559"
]
for idx, m in enumerate(members_c1):
    p = tfc1.paragraphs[0] if idx == 0 else tfc1.add_paragraph()
    p.text = f"•  {m}"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if idx > 0:
        p.space_before = Pt(6)

# Column 2 (Members 4, 5)
tb_col2 = s1.shapes.add_textbox(Inches(7.2), Inches(5.3), Inches(4.4), Inches(1.3))
tfc2 = tb_col2.text_frame
tfc2.word_wrap = True

members_c2 = [
    "SAHAANA B - 23MIS0558",
    "SOWBERNEKA S - 23MIS0343"
]
for idx, m in enumerate(members_c2):
    p = tfc2.paragraphs[0] if idx == 0 else tfc2.add_paragraph()
    p.text = f"•  {m}"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if idx > 0:
        p.space_before = Pt(6)

print("Slide 1 generated.")

# ==============================================================
# SLIDE 2: INTRODUCTION
# ==============================================================
s2 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s2)
add_header(s2, "Introduction: The Rise of Audio Deepfakes", "Chapter 1: Background & Motivation")
add_footer(s2, 2)

intro_cards = [
    ("1. Voice Synthesis Advancements", "Modern neural vocoders generate highly natural speech mimicking genuine pitch, timbre, and human emotion.", WHITE, CARD_BLUE, BORDER_BLUE),
    ("2. Emerging Security Risks", "Cloned speech facilitates social engineering, voice phishing, financial fraud, and biometric authentication bypass.", RED_TEXT, RED_BG, RED_BORDER),
    ("3. Limits of Human Auditory Perception", "Human listeners cannot detect subtle mathematical vocoder phase artifacts or unnatural high-frequency boundaries.", CYAN_ACCENT, CARD_BLUE, BORDER_BLUE),
    ("4. Deep Learning Forensic Solution", "Extracts 80-bin log-Mel spectrograms and trains 2D CNNs with Grad-CAM explainability for automated detection.", GREEN_TEXT, GREEN_BG, GREEN_BORDER)
]

for idx, (title, desc, col_t, bg_c, bdr_c) in enumerate(intro_cards):
    y = Inches(1.65) + idx * Inches(1.28)
    card = add_card(s2, Inches(0.8), y, Inches(5.9), Inches(1.18), bg_color=bg_c, border_color=bdr_c)
    tb = s2.shapes.add_textbox(Inches(1.0), y + Inches(0.08), Inches(5.5), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = col_t
    
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.name = FONT_FAMILY; p2.font.size = Pt(15); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(3)

spec_comp_path = os.path.join(ASSETS_DIR, "spectrogram_comparison.png")
if os.path.exists(spec_comp_path):
    card_img = add_card(s2, Inches(7.0), Inches(1.65), Inches(5.533), Inches(5.15), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    s2.shapes.add_picture(spec_comp_path, Inches(7.1), Inches(1.78), Inches(5.333), Inches(3.2))

    tb_cap = s2.shapes.add_textbox(Inches(7.2), Inches(5.1), Inches(5.133), Inches(1.5))
    tfc = tb_cap.text_frame
    tfc.word_wrap = True
    pc = tfc.paragraphs[0]
    pc.text = "Visual Representation Differences:"
    pc.font.name = FONT_FAMILY; pc.font.size = Pt(18); pc.font.bold = True; pc.font.color.rgb = CYAN_ACCENT
    
    pc2 = tfc.add_paragraph()
    pc2.text = "• Genuine Speech: Continuous harmonic pitch tracks and smooth vocal formant transitions."
    pc2.font.name = FONT_FAMILY; pc2.font.size = Pt(15); pc2.font.color.rgb = GREEN_TEXT; pc2.space_before = Pt(4)
    
    pc3 = tfc.add_paragraph()
    pc3.text = "• Synthetic Speech: Rigid horizontal energy tracks, phase gaps, and vocoder anomalies."
    pc3.font.name = FONT_FAMILY; pc3.font.size = Pt(15); pc3.font.color.rgb = RED_TEXT; pc3.space_before = Pt(3)

print("Slide 2 generated.")

# ==============================================================
# SLIDE 3: PROBLEM STATEMENT
# ==============================================================
s3 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s3)
add_header(s3, "Problem Statement: Challenges in Audio Verification", "Chapter 1: Research Problem")
add_footer(s3, 3)

problems = [
    ("1. High Synthesizer Realism", "Commercial neural TTS produces realistic prosody, deceiving human ear inspection.", WHITE),
    ("2. Lossy Codec Distortion", "WhatsApp Opus/MP3 compression alters high frequencies and fools studio models.", CYAN_ACCENT),
    ("3. Black-Box Neural Networks", "Standard classifiers output binary numbers without showing acoustic evidence.", WHITE),
    ("4. Studio-to-Wild Domain Shift", "Models trained only on clean studio FLACs suffer extreme false alarm rates in the field.", RED_TEXT)
]

for idx, (title, desc, col_t) in enumerate(problems):
    y = Inches(1.65) + idx * Inches(1.28)
    card = add_card(s3, Inches(0.8), y, Inches(5.7), Inches(1.18), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    tb = s3.shapes.add_textbox(Inches(1.0), y + Inches(0.08), Inches(5.3), Inches(1.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]; p.text = title; p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = col_t
    p2 = tf.add_paragraph(); p2.text = desc; p2.font.name = FONT_FAMILY; p2.font.size = Pt(15); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(3)

card_central = add_card(s3, Inches(6.8), Inches(1.65), Inches(5.733), Inches(3.1), bg_color=MID_BLUE, border_color=CYAN_ACCENT, border_width=2)
tb_c = s3.shapes.add_textbox(Inches(7.1), Inches(1.85), Inches(5.133), Inches(2.7))
tfc = tb_c.text_frame
tfc.word_wrap = True

pc1 = tfc.paragraphs[0]; pc1.text = "CENTRAL RESEARCH QUESTION"
pc1.font.name = FONT_FAMILY; pc1.font.size = Pt(14); pc1.font.bold = True; pc1.font.color.rgb = CYAN_ACCENT

pc2 = tfc.add_paragraph()
pc2.text = "\"How can we identify synthetic speech reliably while explaining which audio features influenced the prediction?\""
pc2.font.name = FONT_FAMILY; pc2.font.size = Pt(21); pc2.font.bold = True; pc2.font.color.rgb = WHITE; pc2.space_before = Pt(8)

pc3 = tfc.add_paragraph()
pc3.text = "Requires automated feature learning, interpretable visual evidence, and resilience against lossy mobile codecs."
pc3.font.name = FONT_FAMILY; pc3.font.size = Pt(15); pc3.font.color.rgb = TEXT_LIGHT; pc3.space_before = Pt(8)

card_scope = add_card(s3, Inches(6.8), Inches(4.95), Inches(5.733), Inches(1.85), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_sc = s3.shapes.add_textbox(Inches(7.1), Inches(5.08), Inches(5.133), Inches(1.6))
tfsc = tb_sc.text_frame
tfsc.word_wrap = True

ps1 = tfsc.paragraphs[0]; ps1.text = "Academic Scope & Boundaries:"; ps1.font.name = FONT_FAMILY; ps1.font.size = Pt(18); ps1.font.bold = True; ps1.font.color.rgb = CYAN_ACCENT
ps2 = tfsc.add_paragraph()
ps2.text = "• Targets mathematical vocoder anomalies and spectral distortions.\n• Does not claim universal accuracy against all unseen commercial zero-shot cloners."
ps2.font.name = FONT_FAMILY; ps2.font.size = Pt(15); ps2.font.color.rgb = TEXT_LIGHT; ps2.space_before = Pt(5)

print("Slide 3 generated.")

# ==============================================================
# SLIDE 4: PROJECT OBJECTIVES
# ==============================================================
s4 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s4)
add_header(s4, "Project Objectives: Scope & Key Deliverables", "Chapter 1: Objectives")
add_footer(s4, 4)

objectives = [
    ("1", "Binary Audio Classifier", "Distinguish bona fide human speech from AI-synthesized or cloned speech with high accuracy."),
    ("2", "Log-Mel Feature Extraction", "Transform variable audio into standardized 80-bin Mel spectrograms across 250 frames with z-score normalization."),
    ("3", "Custom 2D CNN Architecture", "Design an efficient 3-stage convolutional network optimized for real-time CPU/GPU inference."),
    ("4", "Grad-CAM Saliency Integration", "Implement gradient backpropagation at features[8] to highlight suspicious time-frequency speech regions."),
    ("5", "Interactive Streamlit Web UI", "Deploy an intuitive dashboard supporting file upload, live waveform audio playback, and confidence scores."),
    ("6", "Domain Robustness", "Apply symmetrical Opus/MP3 augmentations to eliminate false positive alarms on compressed WhatsApp audio.")
]

for idx, (num, title, desc) in enumerate(objectives):
    col = 0 if idx < 3 else 1
    row = idx if col == 0 else idx - 3
    
    c_left = Inches(0.8) if col == 0 else Inches(6.8)
    c_top = Inches(1.65) + row * Inches(1.72)
    c_w = Inches(5.733)
    c_h = Inches(1.55)
    
    card = add_card(s4, c_left, c_top, c_w, c_h, bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    
    badge = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left + Inches(0.2), c_top + Inches(0.22), Inches(0.6), Inches(0.6))
    badge.fill.solid(); badge.fill.fore_color.rgb = CYAN_ACCENT; badge.line.fill.background()
    p_b = badge.text_frame.paragraphs[0]; p_b.text = num; p_b.font.name = FONT_FAMILY; p_b.font.size = Pt(18); p_b.font.bold = True; p_b.font.color.rgb = DARK_NAVY; p_b.alignment = PP_ALIGN.CENTER
    
    tb = s4.shapes.add_textbox(c_left + Inches(0.95), c_top + Inches(0.15), c_w - Inches(1.1), c_h - Inches(0.3))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = title; p.font.name = FONT_FAMILY; p.font.size = Pt(19); p.font.bold = True; p.font.color.rgb = WHITE
    p2 = tf.add_paragraph(); p2.text = desc; p2.font.name = FONT_FAMILY; p2.font.size = Pt(15); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(4)

print("Slide 4 generated.")

# ==============================================================
# SLIDE 5: EXISTING SYSTEM VS PROPOSED SYSTEM
# ==============================================================
s5 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s5)
add_header(s5, "Existing System vs. Proposed Framework", "Chapter 2: System Comparison")
add_footer(s5, 5)

table_shape = s5.shapes.add_table(5, 4, Inches(0.8), Inches(1.65), Inches(11.733), Inches(4.3))
tbl = table_shape.table
tbl.columns[0].width = Inches(2.2)
tbl.columns[1].width = Inches(3.2)
tbl.columns[2].width = Inches(3.4)
tbl.columns[3].width = Inches(2.933)

headers = ["Dimension / Aspect", "Existing Traditional Methods", "Proposed Deep Learning System", "Practical Benefit"]
tbl_data = [
    ("Detection Method", "Manual listening or handcrafted acoustic features (MFCC, CQCC)", "Automated 2D CNN trained on 80-bin log-Mel spectrograms", "Scalable, objective detection of subtle vocoder cues"),
    ("Model Transparency", "Black-box outputs; no insight into why audio was flagged", "Grad-CAM saliency heatmaps on Conv layer features[8]", "Verifiable visual evidence for security & forensic audit"),
    ("Channel Resilience", "Trained only on studio FLAC; fails on compressed phone audio", "Dual Checkpoints (Baseline + Symmetrical Robust Model)", "Eliminates false alarms on Opus/MP3 compressed WhatsApp speech"),
    ("User Interface", "Command-line scripts requiring manual terminal execution", "Streamlit Web Dashboard with live audio preview player", "Simple drag-and-drop testing for non-technical users")
]

for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.fill.solid(); cell.fill.fore_color.rgb = MID_BLUE
    p = cell.text_frame.paragraphs[0]; p.text = h; p.font.name = FONT_FAMILY; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

for i, row in enumerate(tbl_data):
    row_bg = CARD_BLUE if i % 2 == 0 else RGBColor(13, 34, 61)
    for j, val in enumerate(row):
        cell = tbl.cell(i+1, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = row_bg
        p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_FAMILY; p.font.size = Pt(15)
        if j == 0:
            p.font.bold = True; p.font.color.rgb = CYAN_ACCENT
        elif j == 2:
            p.font.bold = True; p.font.color.rgb = GREEN_TEXT
        else:
            p.font.color.rgb = TEXT_LIGHT

banner = add_card(s5, Inches(0.8), Inches(6.15), Inches(11.733), Inches(0.68), bg_color=GREEN_BG, border_color=GREEN_BORDER)
tb_b = s5.shapes.add_textbox(Inches(1.0), Inches(6.22), Inches(11.333), Inches(0.55))
tfb = tb_b.text_frame; tfb.word_wrap = True
p = tfb.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
p.text = "Key Takeaway: The proposed system unifies accurate detection, visual explainability, and real-world robustness."
p.font.name = FONT_FAMILY; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = GREEN_TEXT

print("Slide 5 generated.")

# ==============================================================
# SLIDE 6: SYSTEM ARCHITECTURE
# ==============================================================
s6 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s6)
add_header(s6, "System Architecture: End-to-End Processing Flow", "Chapter 3: System Design")
add_footer(s6, 6)

flow_nodes = [
    ("USER AUDIO", "Upload / Demo Speech", DARK_NAVY, CYAN_ACCENT, Inches(0.8), Inches(2.2)),
    ("STREAMLIT APP", "Interactive Web UI", MID_BLUE, WHITE, Inches(3.6), Inches(2.5)),
    ("PREPROCESSING", "16 kHz Mono | (80, 250)", CARD_BLUE, WHITE, Inches(6.7), Inches(2.7)),
    ("LOG-MEL GRID", "Tensor: (1, 1, 80, 250)", CARD_BLUE, CYAN_ACCENT, Inches(10.0), Inches(2.533))
]

for title, sub, bg_c, txt_c, x_pos, w_pos in flow_nodes:
    c = add_card(s6, x_pos, Inches(1.65), w_pos, Inches(1.05), bg_color=bg_c, border_color=BORDER_BLUE)
    p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; p.text = title; p.font.name = FONT_FAMILY; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = txt_c
    p2 = c.text_frame.add_paragraph(); p2.alignment = PP_ALIGN.CENTER; p2.text = sub; p2.font.name = FONT_FAMILY; p2.font.size = Pt(13); p2.font.color.rgb = TEXT_LIGHT

for x_arr in [Inches(3.1), Inches(6.2), Inches(9.5)]:
    a = s6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x_arr, Inches(2.02), Inches(0.4), Inches(0.28))
    a.fill.solid(); a.fill.fore_color.rgb = CYAN_ACCENT; a.line.fill.background()

# Dual Model Selector Card (Refined height to avoid clipping)
c_sel = add_card(s6, Inches(1.5), Inches(2.9), Inches(10.333), Inches(1.8), bg_color=MID_BLUE, border_color=CYAN_ACCENT, border_width=1.5)
tb_sel = s6.shapes.add_textbox(Inches(1.7), Inches(3.0), Inches(9.933), Inches(0.35))
tf_s = tb_sel.text_frame; tf_s.word_wrap = True
ps = tf_s.paragraphs[0]; ps.text = "MODEL ARCHITECTURE & DUAL CHECKPOINT SELECTION (AudioDeepfakeCNN)"; ps.font.name = FONT_FAMILY; ps.font.size = Pt(17); ps.font.bold = True; ps.font.color.rgb = WHITE

# Baseline Subcard
c_base = add_card(s6, Inches(1.8), Inches(3.45), Inches(4.7), Inches(1.1), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_b = s6.shapes.add_textbox(Inches(1.9), Inches(3.5), Inches(4.5), Inches(1.0))
tf_b = tb_b.text_frame; tf_b.word_wrap = True
pb = tf_b.paragraphs[0]; pb.text = "Option A: Baseline CNN Checkpoint"; pb.font.name = FONT_FAMILY; pb.font.size = Pt(16); pb.font.bold = True; pb.font.color.rgb = CYAN_ACCENT
pb2 = tf_b.add_paragraph(); pb2.text = "• best_deepfake_audio_cnn.pth\n• ASVspoof 2019 studio FLAC trained"; pb2.font.name = FONT_FAMILY; pb2.font.size = Pt(13.5); pb2.font.color.rgb = TEXT_LIGHT; pb2.space_before = Pt(2)

# Robust Subcard
c_rob = add_card(s6, Inches(6.8), Inches(3.45), Inches(4.7), Inches(1.1), bg_color=GREEN_BG, border_color=GREEN_BORDER)
tb_r = s6.shapes.add_textbox(Inches(6.9), Inches(3.5), Inches(4.5), Inches(1.0))
tf_r = tb_r.text_frame; tf_r.word_wrap = True
pr = tf_r.paragraphs[0]; pr.text = "Option B: Robust Domain-Adapted Checkpoint"; pr.font.name = FONT_FAMILY; pr.font.size = Pt(16); pr.font.bold = True; pr.font.color.rgb = GREEN_TEXT
pr2 = tf_r.add_paragraph(); pr2.text = "• robust_deepfake_audio_cnn.pth\n• Symmetrical Opus/MP3 & rumble trained"; pr2.font.name = FONT_FAMILY; pr2.font.size = Pt(13.5); pr2.font.color.rgb = TEXT_LIGHT; pr2.space_before = Pt(2)

# Dual Outputs
c_out1 = add_card(s6, Inches(1.0), Inches(4.88), Inches(5.4), Inches(1.35), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_o1 = s6.shapes.add_textbox(Inches(1.2), Inches(4.96), Inches(5.0), Inches(1.18))
tfo1 = tb_o1.text_frame; tfo1.word_wrap = True
po1 = tfo1.paragraphs[0]; po1.text = "CLASSIFICATION & CONFIDENCE SCORES"; po1.font.name = FONT_FAMILY; po1.font.size = Pt(16.5); po1.font.bold = True; po1.font.color.rgb = WHITE
po2 = tfo1.add_paragraph()
po2.text = "🟢 Class 0: Bona fide Human Speech (Real)\n🔴 Class 1: Synthetic or Cloned Speech (Spoof)\n• Reports raw uncalibrated Softmax distributions"
po2.font.name = FONT_FAMILY; po2.font.size = Pt(13.5); po2.font.color.rgb = TEXT_LIGHT; po2.space_before = Pt(3)

c_out2 = add_card(s6, Inches(6.9), Inches(4.88), Inches(5.4), Inches(1.35), bg_color=CARD_BLUE, border_color=CYAN_ACCENT)
tb_o2 = s6.shapes.add_textbox(Inches(7.1), Inches(4.96), Inches(5.0), Inches(1.18))
tfo2 = tb_o2.text_frame; tfo2.word_wrap = True
po3 = tfo2.paragraphs[0]; po3.text = "GRAD-CAM SALIENCY HEATMAP"; po3.font.name = FONT_FAMILY; po3.font.size = Pt(16.5); po3.font.bold = True; po3.font.color.rgb = CYAN_ACCENT
po4 = tfo2.add_paragraph()
po4.text = "• Traces backpropagation gradients to Conv features[8]\n• Overlays saliency heatmap onto the log-Mel spectrogram\n• Highlights influential spectral cue regions"
po4.font.name = FONT_FAMILY; po4.font.size = Pt(13.5); po4.font.color.rgb = TEXT_LIGHT; po4.space_before = Pt(3)

d_card = add_card(s6, Inches(0.8), Inches(6.34), Inches(11.733), Inches(0.52), bg_color=DARK_NAVY, border_color=BORDER_BLUE)
tb_d = s6.shapes.add_textbox(Inches(1.0), Inches(6.38), Inches(11.333), Inches(0.44))
tfd = tb_d.text_frame; tfd.word_wrap = True
pd = tfd.paragraphs[0]; pd.alignment = PP_ALIGN.CENTER
pd.text = "Important Design Principle: Grad-CAM provides visual model transparency. It explains internal convolutional feature activations and does not independently override the CNN prediction."
pd.font.name = FONT_FAMILY; pd.font.size = Pt(13); pd.font.color.rgb = TEXT_MUTED

print("Slide 6 generated.")

# ==============================================================
# SLIDE 7: DATASET AND AUDIO PREPROCESSING
# ==============================================================
s7 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s7)
add_header(s7, "Dataset and Audio Preprocessing Pipeline", "Chapter 3: Data Preparation")
add_footer(s7, 7)

# Dataset Overview Card (Adjusted height to 1.25")
c_ds = add_card(s7, Inches(0.8), Inches(1.65), Inches(11.733), Inches(1.25), bg_color=MID_BLUE, border_color=CYAN_ACCENT)
tb_ds = s7.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(11.333), Inches(1.05))
tfds = tb_ds.text_frame; tfds.word_wrap = True
p = tfds.paragraphs[0]; p.text = "PRIMARY BENCHMARK DATASET: ASVSPOOF 2019 LOGICAL ACCESS (LA)"; p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = WHITE
p2 = tfds.add_paragraph()
p2.text = "• Standardized benchmark containing authentic human speech and spoofs from state-of-the-art TTS & voice cloners.\n• Studio training data from ASVspoof 2019; supplemented with multi-domain speech for Stage 2 robustness."
p2.font.name = FONT_FAMILY; p2.font.size = Pt(15); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(3)

steps = [
    ("1. Signal Ingestion", "Ingests .wav, .mp3, .ogg, .flac. Downmixes to mono. Rejects silent clips (RMS < 1e-5)."),
    ("2. 16 kHz Resample", "Standardizes rate to 16,000 Hz, preserving frequencies up to 8 kHz Nyquist limit."),
    ("3. Log-Mel STFT", "STFT with n_fft=1024, hop=256, mapped across 80 Mel frequency filterbanks."),
    ("4. Normalization", "Converts power to dB (ref=max). Applies z-score: (x - μ) / (σ + 1e-6)."),
    ("5. Grid Alignment", "Standardizes time dimension to 250 frames (~4.0s). Padds silence or crops evenly.")
]

card_w = Inches(2.2)
card_h = Inches(3.7)
start_x = Inches(0.8)
gap = Inches(0.183)

for idx, (st_title, st_desc) in enumerate(steps):
    x = start_x + idx * (card_w + gap)
    card_step = add_card(s7, x, Inches(3.05), card_w, card_h, bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    
    badge = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(3.05), card_w, Inches(0.5))
    badge.fill.solid(); badge.fill.fore_color.rgb = MID_BLUE if idx != 4 else CYAN_ACCENT; badge.line.fill.background()
    pb = badge.text_frame.paragraphs[0]; pb.text = f"STEP {idx+1}"; pb.font.name = FONT_FAMILY; pb.font.size = Pt(15); pb.font.bold = True; pb.font.color.rgb = WHITE if idx != 4 else DARK_NAVY; pb.alignment = PP_ALIGN.CENTER
    
    tb_s = s7.shapes.add_textbox(x + Inches(0.12), Inches(3.65), card_w - Inches(0.24), card_h - Inches(0.7))
    tfs = tb_s.text_frame; tfs.word_wrap = True
    ps1 = tfs.paragraphs[0]; ps1.text = st_title; ps1.font.name = FONT_FAMILY; ps1.font.size = Pt(17); ps1.font.bold = True; ps1.font.color.rgb = CYAN_ACCENT
    ps2 = tfs.add_paragraph(); ps2.text = st_desc; ps2.font.name = FONT_FAMILY; ps2.font.size = Pt(14.5); ps2.font.color.rgb = TEXT_LIGHT; ps2.space_before = Pt(8)

print("Slide 7 generated.")

# ==============================================================
# SLIDE 8: CNN MODEL ARCHITECTURE
# ==============================================================
s8 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s8)
add_header(s8, "CNN Model Architecture: Layer Specifications", "Chapter 3: Model Architecture")
add_footer(s8, 8)

# Left Column: Layer Flowchart
c_arch = add_card(s8, Inches(0.8), Inches(1.65), Inches(6.8), Inches(5.15), bg_color=CARD_BLUE, border_color=CYAN_ACCENT)
tb_a = s8.shapes.add_textbox(Inches(1.0), Inches(1.78), Inches(6.4), Inches(4.9))
tfa = tb_a.text_frame; tfa.word_wrap = True

pa0 = tfa.paragraphs[0]; pa0.text = "AudioDeepfakeCNN Architecture Flowchart:"; pa0.font.name = FONT_FAMILY; pa0.font.size = Pt(18); pa0.font.bold = True; pa0.font.color.rgb = WHITE

blocks_desc = [
    ("Input Representation", "Log-Mel Spectrogram Tensor: (Batch, 1, 80, 250)", CYAN_ACCENT),
    ("Conv Block 1 (features.0-.3)", "Conv2d(1 -> 16, 3x3) + BatchNorm + ReLU + MaxPool  [16 × 40 × 125]", WHITE),
    ("Conv Block 2 (features.4-.7)", "Conv2d(16 -> 32, 3x3) + BatchNorm + ReLU + MaxPool  [32 × 20 × 62]", WHITE),
    ("Conv Block 3 (features.8-.10)", "Conv2d(32 -> 64, 3x3) + BatchNorm + ReLU  [64 × 20 × 62]  *Grad-CAM Target*", GREEN_TEXT),
    ("Adaptive Pooling (features.11)", "AdaptiveAvgPool2d((5, 10)) -> Compresses spatial size to [64, 5, 10]", WHITE),
    ("Classification Head", "Flatten() -> Linear(3200, 128) -> ReLU -> Dropout(0.4) -> Linear(128, 2)", CYAN_ACCENT)
]

for title, detail, col_t in blocks_desc:
    p_t = tfa.add_paragraph()
    p_t.text = f"• {title}:"
    p_t.font.name = FONT_FAMILY; p_t.font.size = Pt(15.5); p_t.font.bold = True; p_t.font.color.rgb = col_t; p_t.space_before = Pt(4)
    p_d = tfa.add_paragraph()
    p_d.text = f"   {detail}"
    p_d.font.name = FONT_FAMILY; p_d.font.size = Pt(14); p_d.font.color.rgb = TEXT_LIGHT; p_d.space_before = Pt(1)

# Right Column: Architectural Highlights Cards
cards_right = [
    ("Hierarchical Feature Learning", "Convolutional filters detect localized pitch variations, formant spacing, and vocoder spectral boundary cues.", WHITE),
    ("Spatial Pooling & Compression", "Successive Max Pooling and Adaptive Pooling compress features to (5, 10), ensuring temporal invariance.", CYAN_ACCENT),
    ("Overfitting Regularization", "Batch Normalization stabilizes gradients, while 40% Dropout prevents co-adapting to spurious cues.", WHITE),
    ("Grad-CAM Compatibility", "Conv Block 3 (features[8]) preserves spatial resolution, serving as the target layer for saliency mapping.", GREEN_TEXT)
]

for idx, (t, d, col) in enumerate(cards_right):
    cr = add_card(s8, Inches(7.8), Inches(1.65) + idx * Inches(1.3), Inches(4.733), Inches(1.18), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    tb_cr = s8.shapes.add_textbox(Inches(8.0), Inches(1.72) + idx * Inches(1.3), Inches(4.333), Inches(1.0))
    tfcr = tb_cr.text_frame; tfcr.word_wrap = True
    p1 = tfcr.paragraphs[0]; p1.text = t; p1.font.name = FONT_FAMILY; p1.font.size = Pt(17); p1.font.bold = True; p1.font.color.rgb = col
    p2 = tfcr.add_paragraph(); p2.text = d; p2.font.name = FONT_FAMILY; p2.font.size = Pt(14); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(3)

print("Slide 8 generated.")

# ==============================================================
# SLIDE 9: IMPLEMENTATION AND TECHNOLOGY STACK
# ==============================================================
s9 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s9)
add_header(s9, "Implementation: Technology Stack & Workflow", "Chapter 4: Implementation")
add_footer(s9, 9)

# Left Column: Tech Stack Table (Optimized row padding and font to clear footer)
table_shape = s9.shapes.add_table(7, 3, Inches(0.8), Inches(1.65), Inches(6.0), Inches(4.8))
tbl = table_shape.table
tbl.columns[0].width = Inches(1.6)
tbl.columns[1].width = Inches(1.7)
tbl.columns[2].width = Inches(2.7)

headers = ["Technology", "Module / Library", "Project Purpose & Role"]
tbl_data = [
    ("Python 3.10+", "Core Language", "Data pipelines, training scripts, and evaluation metrics"),
    ("PyTorch", "torch & torch.nn", "AudioDeepfakeCNN model, backprop, tensor inference"),
    ("Librosa & SoundFile", "librosa, soundfile", "STFT, 80-bin Mel spectrograms, audio decoding"),
    ("Streamlit", "streamlit 1.66+", "Interactive web application, model selector, results UI"),
    ("NumPy & SciPy", "numpy, scipy.signal", "Matrix operations, z-score norm, Butterworth filter"),
    ("Grad-CAM", "Hook Saliency", "Forward/backward hooks at features[8] for saliency maps")
]

for row in tbl.rows:
    for cell in row.cells:
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)

for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.fill.solid(); cell.fill.fore_color.rgb = MID_BLUE
    p = cell.text_frame.paragraphs[0]; p.text = h; p.font.name = FONT_FAMILY; p.font.size = Pt(14.5); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

for i, row in enumerate(tbl_data):
    row_bg = CARD_BLUE if i % 2 == 0 else RGBColor(13, 34, 61)
    for j, val in enumerate(row):
        cell = tbl.cell(i+1, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = row_bg
        p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_FAMILY; p.font.size = Pt(12)
        if j == 0: p.font.bold = True; p.font.color.rgb = WHITE
        elif j == 1: p.font.bold = True; p.font.color.rgb = CYAN_ACCENT
        else: p.font.color.rgb = TEXT_LIGHT

# Right Column: 6 Implementation Phases (Refined spacing)
card_stages = add_card(s9, Inches(7.1), Inches(1.65), Inches(5.433), Inches(5.15), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_st = s9.shapes.add_textbox(Inches(7.3), Inches(1.8), Inches(5.0), Inches(4.85))
tfst = tb_st.text_frame; tfst.word_wrap = True

pst0 = tfst.paragraphs[0]; pst0.text = "Key Implementation Phases:"; pst0.font.name = FONT_FAMILY; pst0.font.size = Pt(19); pst0.font.bold = True; pst0.font.color.rgb = WHITE

stages_list = [
    ("Stage 1: Preprocessing Pipeline", "Built 80-bin Mel spectrogram extraction & z-score norm."),
    ("Stage 2: CNN Model Definition", "Implemented AudioDeepfakeCNN with strict layer mappings."),
    ("Stage 3: Checkpoint Training", "Trained models on ASVspoof 2019 data and saved weights."),
    ("Stage 4: Prediction CLI", "Engineered predict.py supporting batch/single prediction."),
    ("Stage 5: Grad-CAM Explainability", "Integrated forward/backward PyTorch hooks at features[8]."),
    ("Stage 6: Streamlit Application", "Developed app.py with sidebar model selection & live audio UI.")
]

for st_name, st_det in stages_list:
    p_t = tfst.add_paragraph()
    p_t.text = f"✔ {st_name}: {st_det}"
    p_t.font.name = FONT_FAMILY; p_t.font.size = Pt(14.5); p_t.font.color.rgb = TEXT_LIGHT; p_t.space_before = Pt(8)
    # Highlight lead-in
    p_t.runs[0].font.bold = True
    p_t.runs[0].font.color.rgb = CYAN_ACCENT

print("Slide 9 generated.")

# ==============================================================
# SLIDE 10: ROBUST MODEL AND DOMAIN ADAPTATION
# ==============================================================
s10 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s10)
add_header(s10, "Improving Robustness to Real-World Audio", "Chapter 5: Domain Adaptation")
add_footer(s10, 10)

# Motivation Card (Top Left)
c_mot = add_card(s10, Inches(0.8), Inches(1.65), Inches(5.7), Inches(2.2), bg_color=RED_BG, border_color=RED_BORDER)
tb_m = s10.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(1.9))
tfm = tb_m.text_frame; tfm.word_wrap = True
pm1 = tfm.paragraphs[0]; pm1.text = "The Motivation: Out-of-Domain Failure"; pm1.font.name = FONT_FAMILY; pm1.font.size = Pt(18); pm1.font.bold = True; pm1.font.color.rgb = RED_TEXT
pm2 = tfm.add_paragraph()
pm2.text = "• The baseline model was trained exclusively on uncompressed studio FLACs.\n• When tested on a genuine WhatsApp recording, it failed severely (99.77% SPOOF).\n• Root Cause: Lossy Opus/MP3 compression patterns were misinterpreted as vocoder artifacts."
pm2.font.name = FONT_FAMILY; pm2.font.size = Pt(14.5); pm2.font.color.rgb = TEXT_LIGHT; pm2.space_before = Pt(4)

# Solution Card (Top Right - Refined bullets)
c_sol = add_card(s10, Inches(6.8), Inches(1.65), Inches(5.7), Inches(2.2), bg_color=GREEN_BG, border_color=GREEN_BORDER)
tb_s = s10.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(1.9))
tfs = tb_s.text_frame; tfs.word_wrap = True
ps1 = tfs.paragraphs[0]; ps1.text = "The Solution: Stage 2 Domain Adaptation"; ps1.font.name = FONT_FAMILY; ps1.font.size = Pt(18); ps1.font.bold = True; ps1.font.color.rgb = GREEN_TEXT
ps2 = tfs.add_paragraph()
ps2.text = "• Created robust checkpoint: models/robust_deepfake_audio_cnn.pth.\n• Applied symmetrical in-memory training augmentations (Opus/MP3/noise).\n• Preserved baseline checkpoint strictly untouched for controlled benchmarking."
ps2.font.name = FONT_FAMILY; ps2.font.size = Pt(14.5); ps2.font.color.rgb = TEXT_LIGHT; ps2.space_before = Pt(4)

# Bottom Section: Augmentations Breakdown
c_aug = add_card(s10, Inches(0.8), Inches(4.05), Inches(11.733), Inches(2.75), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_aug = s10.shapes.add_textbox(Inches(1.0), Inches(4.2), Inches(11.333), Inches(2.45))
tfaug = tb_aug.text_frame; tfaug.word_wrap = True

pa0 = tfaug.paragraphs[0]; pa0.text = "Symmetrical Multi-Domain Training Augmentations & Protocol Rules:"; pa0.font.name = FONT_FAMILY; pa0.font.size = Pt(18); pa0.font.bold = True; pa0.font.color.rgb = WHITE

augs = [
    ("Opus Codec Emulation", "In-memory Ogg/Opus compression (16–32 kbps) mimicking WhatsApp audio."),
    ("MP3 Codec Emulation", "In-memory MP3 compression (32–64 kbps) simulating diverse media codecs."),
    ("Sub-Speech Rumble", "Synthesized low-frequency rumble (15–65 Hz) from handheld mobile mics."),
    ("Ambient Room Noise", "Injected Gaussian pink/ambient noise (SNR 15–25 dB) for realistic room acoustics.")
]

for a_name, a_desc in augs:
    p_t = tfaug.add_paragraph()
    p_t.text = f"• {a_name}: {a_desc}"
    p_t.font.name = FONT_FAMILY; p_t.font.size = Pt(15); p_t.font.color.rgb = TEXT_LIGHT; p_t.space_before = Pt(4)

p_rule = tfaug.add_paragraph()
p_rule.text = "Crucial Protocol Integrity: Symmetrical augmentation applied to BOTH bonafide and spoof training speech. WhatsApp audio was 100% held out from training and tuning."
p_rule.font.name = FONT_FAMILY; p_rule.font.size = Pt(15); p_rule.font.bold = True; p_rule.font.color.rgb = CYAN_ACCENT; p_rule.space_before = Pt(6)

print("Slide 10 generated.")

# ==============================================================
# SLIDE 11: EXPLAINABILITY USING GRAD-CAM
# ==============================================================
s11 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s11)
add_header(s11, "Explainable AI: Grad-CAM Saliency Analysis", "Chapter 6: Model Explainability")
add_footer(s11, 11)

# Left Column: Grad-CAM Image
cam_img_path = os.path.join(ASSETS_DIR, "gradcam_saliency.png")
if os.path.exists(cam_img_path):
    card_cam = add_card(s11, Inches(0.8), Inches(1.65), Inches(5.8), Inches(5.15), bg_color=CARD_BLUE, border_color=CYAN_ACCENT)
    s11.shapes.add_picture(cam_img_path, Inches(0.9), Inches(1.78), Inches(5.6), Inches(4.3))
    
    tb_c_cap = s11.shapes.add_textbox(Inches(1.0), Inches(6.18), Inches(5.4), Inches(0.5))
    tfcc = tb_c_cap.text_frame; tfcc.word_wrap = True
    pcc = tfcc.paragraphs[0]; pcc.text = "Actual Grad-CAM Output: Spectrogram (top) & Saliency Overlay (bottom) for Synthetic Speech"
    pcc.font.name = FONT_FAMILY; pcc.font.size = Pt(13); pcc.font.bold = True; pcc.font.color.rgb = CYAN_ACCENT; pcc.alignment = PP_ALIGN.CENTER

# Right Column: Explainability Principles (Refined spacing)
card_exp = add_card(s11, Inches(6.9), Inches(1.65), Inches(5.633), Inches(5.15), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
tb_e = s11.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.233), Inches(4.85))
tfe = tb_e.text_frame; tfe.word_wrap = True

pe0 = tfe.paragraphs[0]; pe0.text = "How Grad-CAM Works in Audio Forensics:"; pe0.font.name = FONT_FAMILY; pe0.font.size = Pt(19); pe0.font.bold = True; pe0.font.color.rgb = WHITE

exp_points = [
    ("Gradient Attribution", "Backpropagates prediction scores to Conv layer features[8] to calculate spatial gradients."),
    ("Channel Weighting", "Global-average-pools gradients across width/height to assign positive importance weights."),
    ("Bilinear Interpolation", "Upsamples coarse map (20 × 62) back to (80 × 250) spectrogram scale and normalizes."),
    ("Interpreting Synthetic Saliency", "Warm colors (red/yellow) highlight vocoder boundary artifacts and rigid spectral energy."),
    ("Critical Forensic Limitation", "Grad-CAM visualizes model activations; it is an interpretive guide, not legal proof.")
]

for et, ed in exp_points:
    pet = tfe.add_paragraph()
    pet.text = f"• {et}: {ed}"
    pet.font.name = FONT_FAMILY; pet.font.size = Pt(14.5); pet.font.color.rgb = TEXT_LIGHT; pet.space_before = Pt(6)

print("Slide 11 generated.")

# ==============================================================
# SLIDE 12: WEB APPLICATION AND DEMONSTRATION
# ==============================================================
s12 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s12)
add_header(s12, "Web Application: Streamlit Research Interface", "Chapter 7: System Demonstration")
add_footer(s12, 12)

# Left Column: App UI Mockup Image
ui_img_path = os.path.join(ASSETS_DIR, "app_ui_mockup.png")
if os.path.exists(ui_img_path):
    card_ui = add_card(s12, Inches(0.8), Inches(1.65), Inches(6.8), Inches(5.15), bg_color=CARD_BLUE, border_color=CYAN_ACCENT)
    s12.shapes.add_picture(ui_img_path, Inches(0.9), Inches(1.75), Inches(6.6), Inches(4.45))
    
    tb_uicap = s12.shapes.add_textbox(Inches(1.0), Inches(6.26), Inches(6.4), Inches(0.45))
    tfuc = tb_uicap.text_frame; tfuc.word_wrap = True
    puc = tfuc.paragraphs[0]; puc.text = "Streamlit Research UI: Live Model Selector, Analysis Card, Verdict Badge & Softmax Distribution"
    puc.font.name = FONT_FAMILY; puc.font.size = Pt(12); puc.font.bold = True; puc.font.color.rgb = CYAN_ACCENT; puc.alignment = PP_ALIGN.CENTER

# Right Column: 4 UI Functional Callouts
ui_callouts = [
    ("Interactive Model Selector", "Sidebar radio control allows seamless toggling between Baseline CNN and Robust Stage 2 CNN with live cache invalidation.", CYAN_ACCENT),
    ("Multi-Format Audio Ingestion", "Supports drag-and-drop file upload for .wav, .mp3, .ogg, .flac, and .m4a with built-in audio preview player.", WHITE),
    ("Verdict & Uncalibrated Scores", "Displays clear color-coded classification badges (🟢 Real / 🔴 Spoof) accompanied by raw softmax distribution percentages.", GREEN_TEXT),
    ("Multi-Window Diagnostic Timeline", "For recordings > 4.0s, extracts overlapping 4-second windows (50% overlap) and presents a timestamped segment breakdown.", WHITE)
]

for idx, (ct, cd, ccol) in enumerate(ui_callouts):
    c_box = add_card(s12, Inches(7.8), Inches(1.65) + idx * Inches(1.3), Inches(4.733), Inches(1.18), bg_color=CARD_BLUE, border_color=BORDER_BLUE)
    tb_cbox = s12.shapes.add_textbox(Inches(8.0), Inches(1.72) + idx * Inches(1.3), Inches(4.333), Inches(1.0))
    tfcb = tb_cbox.text_frame; tfcb.word_wrap = True
    p1 = tfcb.paragraphs[0]; p1.text = ct; p1.font.name = FONT_FAMILY; p1.font.size = Pt(17); p1.font.bold = True; p1.font.color.rgb = ccol
    p2 = tfcb.add_paragraph(); p2.text = cd; p2.font.name = FONT_FAMILY; p2.font.size = Pt(14); p2.font.color.rgb = TEXT_LIGHT; p2.space_before = Pt(3)

print("Slide 12 generated.")

# ==============================================================
# SLIDE 13: RESULTS AND EVALUATION
# ==============================================================
s13 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s13)
add_header(s13, "Results and Evaluation: Baseline vs. Robust Model", "Chapter 8: Experimental Results")
add_footer(s13, 13)

# Left Column: Evaluation Chart Image
chart_img_path = os.path.join(ASSETS_DIR, "evaluation_chart.png")
if os.path.exists(chart_img_path):
    card_ch = add_card(s13, Inches(0.8), Inches(1.65), Inches(5.6), Inches(3.25), bg_color=CARD_BLUE, border_color=CYAN_ACCENT)
    s13.shapes.add_picture(chart_img_path, Inches(0.9), Inches(1.75), Inches(5.4), Inches(2.65))
    
    tb_chcap = s13.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(5.2), Inches(0.4))
    tfch = tb_chcap.text_frame; tfch.word_wrap = True
    pch = tfch.paragraphs[0]; pch.text = "Grouped Metric Comparison on Pilot Test Split (30 Samples, Disjoint Speakers)"
    pch.font.name = FONT_FAMILY; pch.font.size = Pt(12); pch.font.bold = True; pch.font.color.rgb = CYAN_ACCENT; pch.alignment = PP_ALIGN.CENTER

# Left Bottom: Essential Limitation Box (STRICT REQUIREMENT)
card_lim = add_card(s13, Inches(0.8), Inches(5.05), Inches(5.6), Inches(1.75), bg_color=RED_BG, border_color=RED_BORDER)
tb_lim = s13.shapes.add_textbox(Inches(1.0), Inches(5.15), Inches(5.2), Inches(1.55))
tflim = tb_lim.text_frame; tflim.word_wrap = True
pl1 = tflim.paragraphs[0]; pl1.text = "Essential Evaluation Limitation & Scientific Scope:"; pl1.font.name = FONT_FAMILY; pl1.font.size = Pt(15); pl1.font.bold = True; pl1.font.color.rgb = RED_TEXT
pl2 = tflim.add_paragraph()
pl2.text = "\"The pilot evaluation contains only 30 samples and includes the two demonstration recordings. These results show performance on this small evaluation set and do not establish 100% accuracy on all real-world audio.\""
pl2.font.name = FONT_FAMILY; pl2.font.size = Pt(13.5); pl2.font.bold = True; pl2.font.color.rgb = WHITE; pl2.space_before = Pt(4)

# Right Column: Comparative Results Table & Dev Benchmark Callout
table_shape = s13.shapes.add_table(7, 4, Inches(6.7), Inches(1.65), Inches(5.833), Inches(1.85))
tbl = table_shape.table
tbl.columns[0].width = Inches(1.833)
tbl.columns[1].width = Inches(0.95)
tbl.columns[2].width = Inches(0.95)
tbl.columns[3].width = Inches(2.1)

tbl_headers = ["Metric", "Baseline", "Robust", "Improvement / Note"]
pilot_rows = [
    ("Accuracy", "96.67%", "100.00%", "+3.33% (Pilot Set)"),
    ("Precision", "93.75%", "100.00%", "+6.25%"),
    ("Recall", "100.00%", "100.00%", "100% Maintained"),
    ("F1-Score", "0.9677", "1.0000", "+0.0323"),
    ("False Positive Rate", "6.67%", "0.00%", "-6.67% (Zero Alarms)"),
    ("False Negative Rate", "0.00%", "0.00%", "0.00% (Zero Misses)")
]

for row in tbl.rows:
    for cell in row.cells:
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.margin_left = Inches(0.04)
        cell.margin_right = Inches(0.04)

for j, h in enumerate(tbl_headers):
    cell = tbl.cell(0, j); cell.fill.solid(); cell.fill.fore_color.rgb = MID_BLUE
    p = cell.text_frame.paragraphs[0]; p.text = h; p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

for i, row in enumerate(pilot_rows):
    row_bg = CARD_BLUE if i % 2 == 0 else RGBColor(13, 34, 61)
    for j, val in enumerate(row):
        cell = tbl.cell(i+1, j); cell.fill.solid(); cell.fill.fore_color.rgb = row_bg
        p = cell.text_frame.paragraphs[0]; p.text = val; p.font.name = FONT_FAMILY; p.font.size = Pt(11)
        if j == 0: p.font.bold = True; p.font.color.rgb = WHITE
        elif j == 2: p.font.bold = True; p.font.color.rgb = GREEN_TEXT
        elif j == 3: p.font.bold = True; p.font.color.rgb = CYAN_ACCENT
        else: p.font.color.rgb = TEXT_LIGHT

# Dev Benchmark Callout Card (Bottom Right - Clean 0.35" separation from table)
card_dev = add_card(s13, Inches(6.7), Inches(3.80), Inches(5.833), Inches(3.0), bg_color=MID_BLUE, border_color=CYAN_ACCENT)
tb_dev = s13.shapes.add_textbox(Inches(6.88), Inches(3.92), Inches(5.47), Inches(2.78))
tfdev = tb_dev.text_frame; tfdev.word_wrap = True
pd1 = tfdev.paragraphs[0]; pd1.text = "Official ASVspoof 2019 Dev Split Benchmark:"; pd1.font.name = FONT_FAMILY; pd1.font.size = Pt(15.5); pd1.font.bold = True; pd1.font.color.rgb = CYAN_ACCENT
pd2 = tfdev.add_paragraph()
pd2.text = "• Large-Scale Benchmark: Evaluated on 24,844 development utterances (2,548 Real speech, 22,296 Spoof speech).\n• Baseline Verification: Accuracy = 98.65%  |  ROC-AUC = 0.9986  |  F1-Score = 99.24%.\n• Real-World Gap: Baseline misclassified genuine WhatsApp audio (99.77% spoof) due to lossy Opus compression.\n• Robust Adaptation: Robust model successfully resolved compression mismatch (95.22% Real)."
pd2.font.name = FONT_FAMILY; pd2.font.size = Pt(12.5); pd2.font.color.rgb = TEXT_LIGHT; pd2.space_before = Pt(4)

print("Slide 13 generated.")

# ==============================================================
# SLIDE 14: RESULTS DISCUSSION, LIMITATIONS, AND FUTURE WORK
# (FINAL SLIDE OF THE PRESENTATION)
# ==============================================================
s14 = prs.slides.add_slide(blank_slide_layout)
set_slide_background(s14)
add_header(s14, "Discussion, Limitations, & Future Directions", "Chapter 8: Critical Analysis")
add_footer(s14, 14)

col_w = Inches(3.75)
col_h = Inches(5.2)

# Column 1: Successful Findings (Green Card)
c_col1 = add_card(s14, Inches(0.8), Inches(1.60), col_w, col_h, bg_color=GREEN_BG, border_color=GREEN_BORDER)
tb_c1 = s14.shapes.add_textbox(Inches(0.95), Inches(1.72), col_w - Inches(0.3), col_h - Inches(0.24))
tfc1 = tb_c1.text_frame; tfc1.word_wrap = True
p1 = tfc1.paragraphs[0]; p1.text = "Key Observations"; p1.font.name = FONT_FAMILY; p1.font.size = Pt(18); p1.font.bold = True; p1.font.color.rgb = GREEN_TEXT

c1_points = [
    ("WhatsApp Generalization", "Robust model correctly identified genuine WhatsApp speech (95.22% Real), reversing baseline error."),
    ("Clean Speech Validation", "LibriSpeech speech correctly recognized as authentic human speech (99.70% Real)."),
    ("Synthetic Spoof Detection", "Synthetic TTS audio successfully identified as spoof (100.00% Spoof) with full sensitivity."),
    ("Balanced Pilot Results", "Zero false alarms and zero misses achieved across the 30-sample multi-domain pilot split.")
]
for t, d in c1_points:
    p = tfc1.add_paragraph(); p.text = f"✔ {t}: {d}"; p.font.name = FONT_FAMILY; p.font.size = Pt(11.8); p.font.color.rgb = TEXT_LIGHT; p.space_before = Pt(3.5)
    p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE

# Column 2: Current Limitations (Red Card)
c_col2 = add_card(s14, Inches(4.79), Inches(1.60), col_w, col_h, bg_color=RED_BG, border_color=RED_BORDER)
tb_c2 = s14.shapes.add_textbox(Inches(4.94), Inches(1.72), col_w - Inches(0.3), col_h - Inches(0.24))
tfc2 = tb_c2.text_frame; tfc2.word_wrap = True
p1 = tfc2.paragraphs[0]; p1.text = "Current Limitations"; p1.font.name = FONT_FAMILY; p1.font.size = Pt(18); p1.font.bold = True; p1.font.color.rgb = RED_TEXT

c2_points = [
    ("Small Pilot Evaluation", "Pilot contains 30 samples; does not statistically guarantee universal real-world performance."),
    ("Novel Voice Synthesis", "Emerging commercial diffusion models and zero-shot cloners may exhibit novel acoustic cues."),
    ("Uncalibrated Confidence", "Reported softmax outputs represent raw network activation energy rather than true probabilities."),
    ("Interpretability Scope", "Grad-CAM visualizes filter activations; it does not isolate specific manipulated syllables.")
]
for t, d in c2_points:
    p = tfc2.add_paragraph(); p.text = f"⚠ {t}: {d}"; p.font.name = FONT_FAMILY; p.font.size = Pt(11.8); p.font.color.rgb = TEXT_LIGHT; p.space_before = Pt(3.5)
    p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE

# Column 3: Future Research Directions (Blue Card)
c_col3 = add_card(s14, Inches(8.78), Inches(1.60), col_w, col_h, bg_color=MID_BLUE, border_color=CYAN_ACCENT)
tb_c3 = s14.shapes.add_textbox(Inches(8.93), Inches(1.72), col_w - Inches(0.3), col_h - Inches(0.24))
tfc3 = tb_c3.text_frame; tfc3.word_wrap = True
p1 = tfc3.paragraphs[0]; p1.text = "Future Directions"; p1.font.name = FONT_FAMILY; p1.font.size = Pt(18); p1.font.bold = True; p1.font.color.rgb = CYAN_ACCENT

c3_points = [
    ("Large-Scale Corpus Testing", "Expand testing to extensive cross-dataset benchmarks (In-the-Wild & FakeAVCeleb, > 10,000 clips)."),
    ("Commercial Cloner Testing", "Benchmark against contemporary commercial APIs (ElevenLabs, Bark, Tortoise, OpenVoice)."),
    ("Confidence Calibration", "Implement temperature scaling and Platt scaling to transform raw logits into posterior confidence."),
    ("Advanced XAI Frameworks", "Incorporate Layer-wise Relevance Propagation (LRP) and Integrated Gradients for forensic interpretability.")
]
for t, d in c3_points:
    p = tfc3.add_paragraph(); p.text = f"➔ {t}: {d}"; p.font.name = FONT_FAMILY; p.font.size = Pt(11.8); p.font.color.rgb = TEXT_LIGHT; p.space_before = Pt(3.5)
    p.runs[0].font.bold = True; p.runs[0].font.color.rgb = WHITE

print("Slide 14 generated.")

# ==========================================
# SAVE PRESENTATION
# ==========================================
prs.save(OUTPUT_PPTX)
print(f"\n==========================================")
print(f"SUCCESS: Generated exactly {TOTAL_SLIDES} slides!")
print(f"Presentation saved to: {OUTPUT_PPTX}")
print(f"==========================================")
