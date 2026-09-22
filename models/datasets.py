"""
models/datasets.py
PyTorch Dataset classes for image and audio deepfake detection training.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional

from utils.preprocessing import (
    TRAIN_TRANSFORM, INFERENCE_TRANSFORM,
    detect_and_crop_face,
    load_audio, audio_to_melspectrogram, spectrogram_to_tensor,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Image dataset
# ─────────────────────────────────────────────────────────────────────────────
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ImageDeepfakeDataset(Dataset):
    """
    Loads images from two directories:
        root/real/  →  label 0
        root/fake/  →  label 1

    Args:
        root_dir:       Directory with `real/` and `fake/` sub-folders.
        train:          Use augmented training transform if True.
        use_face_crop:  Detect and crop faces before transforming.
        max_per_class:  Cap on samples per class (useful for quick tests).
    """

    def __init__(self, root_dir: str, train: bool = True,
                 use_face_crop: bool = False,
                 max_per_class: Optional[int] = None):
        self.train         = train
        self.use_face_crop = use_face_crop
        self.transform     = TRAIN_TRANSFORM if train else INFERENCE_TRANSFORM
        self.samples: List[Tuple[str, int]] = []

        for label, subfolder in [(0, "real"), (1, "fake")]:
            folder = os.path.join(root_dir, subfolder)
            if not os.path.isdir(folder):
                continue
            paths = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS
            ]
            if max_per_class:
                paths = paths[:max_per_class]
            self.samples.extend([(p, label) for p in paths])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.use_face_crop:
                img = detect_and_crop_face(img)
            tensor = self.transform(img)
        except Exception as e:
            # Return a zero tensor on corrupt files
            tensor = torch.zeros(3, 224, 224)
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  Audio dataset
# ─────────────────────────────────────────────────────────────────────────────
VALID_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioDeepfakeDataset(Dataset):
    """
    Loads audio files from two directories:
        root/real/  →  label 0
        root/fake/  →  label 1

    Each sample is converted to a log-Mel spectrogram and returned as a
    (1, N_MELS, T) float32 tensor.

    Args:
        root_dir:      Directory with `real/` and `fake/` sub-folders.
        max_per_class: Cap on samples per class.
    """

    def __init__(self, root_dir: str, max_per_class: Optional[int] = None):
        self.samples: List[Tuple[str, int]] = []

        for label, subfolder in [(0, "real"), (1, "fake")]:
            folder = os.path.join(root_dir, subfolder)
            if not os.path.isdir(folder):
                continue
            paths = [
                os.path.join(folder, f)
                for f in sorted(os.listdir(folder))
                if os.path.splitext(f)[1].lower() in VALID_AUDIO_EXTS
            ]
            if max_per_class:
                paths = paths[:max_per_class]
            self.samples.extend([(p, label) for p in paths])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            y, sr  = load_audio(path)
            spec   = audio_to_melspectrogram(y, sr)
            tensor = spectrogram_to_tensor(spec).squeeze(0)  # (1, n_mels, T)
        except Exception:
            tensor = torch.zeros(1, 128, 125)   # fallback silent spectrogram
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader builders
# ─────────────────────────────────────────────────────────────────────────────
def make_image_loaders(root_dir: str,
                       batch_size: int = 16,
                       val_split: float = 0.2,
                       num_workers: int = 0,
                       use_face_crop: bool = False,
                       max_per_class: Optional[int] = None):
    """
    Build train and validation DataLoaders for image data.

    Returns:
        (train_loader, val_loader)
    """
    full = ImageDeepfakeDataset(root_dir, train=True,
                                use_face_crop=use_face_crop,
                                max_per_class=max_per_class)
    n_val   = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    # Override transform for val set
    val_ds.dataset = ImageDeepfakeDataset(root_dir, train=False,
                                          use_face_crop=use_face_crop,
                                          max_per_class=max_per_class)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def make_audio_loaders(root_dir: str,
                       batch_size: int = 16,
                       val_split: float = 0.2,
                       num_workers: int = 0,
                       max_per_class: Optional[int] = None):
    """
    Build train and validation DataLoaders for audio data.

    Returns:
        (train_loader, val_loader)
    """
    full = AudioDeepfakeDataset(root_dir, max_per_class=max_per_class)
    n_val   = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
