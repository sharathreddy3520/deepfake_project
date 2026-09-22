"""
utils/preprocessing.py
Shared preprocessing utilities for image, video, and audio inputs.
"""

import os
from typing import List, Optional, Tuple

import cv2
import librosa
import librosa.display
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ──────────────────────────────────────────────────────────────────────────────
# Image constants
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference transform
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Training transform
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    """Load image as RGB."""
    return Image.open(path).convert("RGB")


def preprocess_image(
    img: Image.Image,
    train: bool = False
) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    tfm = TRAIN_TRANSFORM if train else INFERENCE_TRANSFORM
    return tfm(img).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor back to uint8 image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    img = tensor.cpu().squeeze(0)
    img = img * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Face detection (Streamlit + VS Code compatible)
# ──────────────────────────────────────────────────────────────────────────────

def _load_face_detector():
    """
    Load Haar Cascade safely.
    Returns an empty classifier instead of crashing.
    """
    try:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if not os.path.exists(haar_path):
            haar_path = os.path.join(
                os.path.dirname(cv2.__file__),
                "data",
                "haarcascade_frontalface_default.xml"
            )

        detector = cv2.CascadeClassifier(haar_path)

        if detector.empty():
            return cv2.CascadeClassifier()

        return detector

    except Exception:
        return cv2.CascadeClassifier()


_face_detector = _load_face_detector()


def detect_and_crop_face(
    img: Image.Image,
    margin: float = 0.3
) -> Optional[Image.Image]:
    """
    Detect largest face.
    Returns original image if detector is unavailable.
    """

    if _face_detector.empty():
        return img

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = _face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return img

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    mw = int(w * margin)
    mh = int(h * margin)

    H, W = bgr.shape[:2]

    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(W, x + w + mw)
    y2 = min(H, y + h + mh)

    cropped = np.array(img)[y1:y2, x1:x2]

    return Image.fromarray(cropped)


# ──────────────────────────────────────────────────────────────────────────────
# Video helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    max_frames: int = 16,
    resize: Tuple[int, int] = IMAGE_SIZE
) -> List[Image.Image]:
    """Extract evenly spaced frames from a video."""

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(
        0,
        total - 1,
        min(max_frames, total),
        dtype=int
    )

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize(resize)

        frames.append(frame)

    cap.release()

    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
N_MELS = 128
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
MAX_AUDIO_LEN = 4


def load_audio(
    path: str,
    sr: int = SAMPLE_RATE,
    max_len: float = MAX_AUDIO_LEN
) -> Tuple[np.ndarray, int]:
    """Load audio and pad/trim."""

    y, _ = librosa.load(path, sr=sr, mono=True)

    target = int(sr * max_len)

    if len(y) > target:
        y = y[:target]
    else:
        y = np.pad(y, (0, target - len(y)), mode="constant")

    return y, sr


def audio_to_melspectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Waveform → Mel Spectrogram."""

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    return librosa.power_to_db(mel, ref=np.max)


def audio_to_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Waveform → MFCC."""

    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


def spectrogram_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Normalize spectrogram for CNN."""

    spec = (spec - spec.mean()) / (spec.std() + 1e-8)

    return (
        torch.tensor(spec, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )