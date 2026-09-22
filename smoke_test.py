"""
smoke_test.py
Validates the entire pipeline using random tensors (no GPU / dataset needed).
Run: python smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("  DEEP FAKE DETECTION - SMOKE TEST")
print("=" * 60)

# 1. Image model
print("\n[1/5] Image model...")
from models.image_model import build_image_model
model = build_image_model()
x = torch.randn(2, 3, 224, 224)
out = model(x)
assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
label, conf = model.predict(x[:1])
assert label in (0, 1)
print(f"      OK output shape {out.shape}, label={label}, conf={conf:.4f}")

# 2. Audio model
print("\n[2/5] Audio model...")
from audio_processing.audio_model import build_audio_model
amodel = build_audio_model()
ax = torch.randn(2, 1, 128, 125)
aout = amodel(ax)
assert aout.shape == (2, 1)
alabel, aconf = amodel.predict(ax[:1])
print(f"      OK output shape {aout.shape}, label={alabel}, conf={aconf:.4f}")

# 3. Preprocessing - image
print("\n[3/5] Image preprocessing...")
from utils.preprocessing import preprocess_image, denormalize
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
t = preprocess_image(img)
assert t.shape == (1, 3, 224, 224)
arr = denormalize(t)
assert arr.shape == (224, 224, 3)
print(f"      OK tensor {t.shape}, denorm {arr.shape}")

# 4. Audio preprocessing
print("\n[4/5] Audio preprocessing...")
from utils.preprocessing import audio_to_melspectrogram, spectrogram_to_tensor
y = np.random.randn(16000 * 4).astype(np.float32)
spec = audio_to_melspectrogram(y)
assert spec.shape[0] == 128
t_spec = spectrogram_to_tensor(spec)
assert t_spec.shape[0] == 1
print(f"      OK spec {spec.shape}, tensor {t_spec.shape}")

# 5. Metrics
print("\n[5/5] Metrics...")
from utils.metrics import compute_metrics
y_true = [0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0]
y_prob = [0.1, 0.9, 0.6, 0.8, 0.4, 0.2]
m = compute_metrics(y_true, y_pred, y_prob)
assert "f1_score" in m
print(f"      OK F1={m['f1_score']:.4f}, Acc={m['accuracy']:.4f}")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
