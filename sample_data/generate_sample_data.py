"""
sample_data/generate_sample_data.py
Creates minimal dummy image and audio files for smoke-testing.
No external dataset required.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_dummy_images(n: int = 5):
    """Create n synthetic face-like images for each class."""
    for cls in ["real", "fake"]:
        folder = os.path.join(ROOT, "data", "images", cls)
        os.makedirs(folder, exist_ok=True)

        for i in range(n):
            img = Image.new("RGB", (224, 224),
                            color=tuple(np.random.randint(100, 200, 3).tolist()))
            draw = ImageDraw.Draw(img)
            # Draw a crude face outline
            draw.ellipse([40, 40, 184, 184], outline=(255, 220, 177), width=3)
            draw.ellipse([75, 85, 100, 110],  fill=(50, 50, 50))   # left eye
            draw.ellipse([124, 85, 149, 110], fill=(50, 50, 50))   # right eye
            draw.arc([85, 130, 139, 160], start=10, end=170, fill=(200, 50, 50), width=3)
            draw.text((80, 10), f"{cls.upper()} #{i+1}", fill=(255, 255, 255))
            path = os.path.join(folder, f"{cls}_{i+1:03d}.png")
            img.save(path)

    print(f"[SampleData] Created {n} images per class in data/images/")


def make_dummy_audio(n: int = 5, sr: int = 16_000, duration: float = 2.0):
    """Create n synthetic audio files for each class."""
    for cls in ["real", "fake"]:
        folder = os.path.join(ROOT, "data", "audio", cls)
        os.makedirs(folder, exist_ok=True)

        for i in range(n):
            t    = np.linspace(0, duration, int(sr * duration))
            freq = 220 * (1 + i * 0.1) if cls == "real" else 440 * (1 + i * 0.15)
            wave = 0.5 * np.sin(2 * np.pi * freq * t)
            # Add slight noise
            wave += 0.02 * np.random.randn(len(t))
            path = os.path.join(folder, f"{cls}_{i+1:03d}.wav")
            sf.write(path, wave.astype(np.float32), sr)

    print(f"[SampleData] Created {n} audio files per class in data/audio/")


if __name__ == "__main__":
    make_dummy_images(n=5)
    make_dummy_audio(n=5)
    print("[SampleData] Done! Run: python train.py --modality image --dummy")
