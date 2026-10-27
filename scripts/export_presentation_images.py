import os
import sys

pptx_path = os.path.abspath(r"C:\Users\Sreenithi\Desktop\Deepfake-Audio-Detection\DeepFake_Audio_Detection_Presentation.pptx")
out_dir = os.path.abspath(r"C:\Users\Sreenithi\Desktop\Deepfake-Audio-Detection\scratch\slide_renders")
os.makedirs(out_dir, exist_ok=True)

try:
    import win32com.client
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    # Open presentation in read-only mode without visible window
    presentation = powerpoint.Presentations.Open(pptx_path, ReadOnly=True, Untitled=False, WithWindow=False)
    
    total_slides = presentation.Slides.Count
    print(f"PowerPoint opened successfully. Total slides: {total_slides}")
    
    for i in range(1, total_slides + 1):
        slide = presentation.Slides(i)
        img_path = os.path.join(out_dir, f"slide_{i}.png")
        # Export slide to image at 1920x1080 resolution
        slide.Export(img_path, "PNG", 1920, 1080)
        print(f"Exported slide {i} -> {img_path}")
        
    presentation.Close()
    powerpoint.Quit()
    print("All slides exported successfully!")
except Exception as e:
    print(f"COM Export error: {e}")
    # Fallback to inspecting shapes via python-pptx

