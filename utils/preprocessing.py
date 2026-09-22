"""
utils/preprocessing.py
Shared preprocessing utilities for image, video, and audio inputs.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Optional
import os


# ── Image constants ──────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Standard transform for inference
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Augmented transform for training
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Image helpers ─────────────────────────────────────────────────────────────
def load_image(path: str) -> Image.Image:
    """Load an image from disk as a PIL RGB image."""
    img = Image.open(path).convert("RGB")
    return img


def preprocess_image(img: Image.Image, train: bool = False) -> torch.Tensor:
    """
    Apply the appropriate transform and return a (1, C, H, W) tensor.

    Args:
        img:   PIL Image.
        train: Use augmented training transform if True.

    Returns:
        Tensor of shape (1, 3, 224, 224).
    """
    tfm = TRAIN_TRANSFORM if train else INFERENCE_TRANSFORM
    return tfm(img).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and return a uint8 HxWx3 numpy array.
    Useful for overlaying Grad-CAM heatmaps.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu().squeeze(0) * std + mean
    img  = img.permute(1, 2, 0).numpy()
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ── Face detection (works locally + Streamlit Cloud) ─────────────
try:
    HAAR_PATH = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
except AttributeError:
    # Fallback for some cloud environments
    HAAR_PATH = os.path.join(
        os.path.dirname(cv2.__file__),
        "data",
        "haarcascade_frontalface_default.xml"
    )

_face_detector = cv2.CascadeClassifier(HAAR_PATH)

if _face_detector.empty():
    raise RuntimeError(f"Could not load Haar Cascade: {HAAR_PATH}")


def detect_and_crop_face(img: Image.Image,
                         margin: float = 0.3) -> Optional[Image.Image]:
    """
    Detect the largest face in an image and return a cropped/padded version.
    Falls back to the full image if no face is found.

    Args:
        img:    PIL image (RGB).
        margin: Fractional margin to add around the detected bounding box.

    Returns:
        Cropped PIL image centred on the face, or the original image.
    """
    bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return img  # no face found – return original

    # Pick the largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    mw = int(w * margin)
    mh = int(h * margin)
    H, W = bgr.shape[:2]
    x1 = max(0, x - mw)
    y1 = max(0, y - mh)
    x2 = min(W, x + w + mw)
    y2 = min(H, y + h + mh)

    face_rgb = np.array(img)[y1:y2, x1:x2]
    return Image.fromarray(face_rgb)


# ── Video helpers ─────────────────────────────────────────────────────────────
def extract_frames(video_path: str,
                   max_frames: int = 16,
                   resize: Tuple[int, int] = IMAGE_SIZE) -> List[Image.Image]:
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to return.
        resize:     Target (width, height) for each frame.

    Returns:
        List of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb).resize(resize)
        frames.append(pil_img)

    cap.release()
    return frames


# ── Audio helpers ─────────────────────────────────────────────────────────────
import librosa
import librosa.display

# Audio constants
SAMPLE_RATE    = 16_000   # Hz
N_MELS         = 128
N_MFCC         = 40
HOP_LENGTH     = 512
N_FFT          = 2048
MAX_AUDIO_LEN  = 4        # seconds


def load_audio(path: str, sr: int = SAMPLE_RATE,
               max_len: float = MAX_AUDIO_LEN) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to `sr`, and trim/pad to `max_len` seconds.

    Args:
        path:    Path to the audio file.
        sr:      Target sample rate.
        max_len: Target duration in seconds (clip or zero-pad).

    Returns:
        Tuple of (waveform ndarray, sample_rate).
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * max_len)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y, sr


def audio_to_melspectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Convert a raw waveform to a log-Mel spectrogram (shape: n_mels x T).

    Returns:
        2-D numpy array of shape (N_MELS, time_steps).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def audio_to_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute MFCCs (shape: N_MFCC x T).

    Returns:
        2-D numpy array of shape (N_MFCC, time_steps).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc


def spectrogram_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """
    Normalise a spectrogram and convert to a (1, 1, H, W) tensor for CNN input.

    Args:
        spec: 2-D numpy array (H x W).

    Returns:
        Tensor of shape (1, 1, H, W).
    """
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)   # z-score normalise
    tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor
