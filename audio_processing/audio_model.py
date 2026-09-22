"""
audio_processing/audio_model.py
Lightweight CNN for deepfake audio detection using Mel Spectrograms.
"""
import torch
import torch.nn as nn
from typing import Tuple
import os


class SpectrogramCNN(nn.Module):
    """
    A small CNN that operates on log-Mel spectrograms.
    Input shape: (B, 1, N_MELS, T)
    Output:      (B, 1) logit
    """

    def __init__(self, n_mels: int = 128, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),   # fixed output size regardless of input T
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, time_steps)
        Returns:
            Logit (B, 1)
        """
        feat   = self.features(x)
        flat   = feat.view(feat.size(0), -1)
        logits = self.classifier(flat)
        return logits

    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> Tuple[int, float]:
        """
        Run inference on a single spectrogram tensor.

        Args:
            x:         Tensor (1, 1, n_mels, T).
            threshold: Decision threshold.

        Returns:
            (label, confidence) – label 0=Real, 1=Fake.
        """
        self.eval()
        with torch.no_grad():
            logit = self.forward(x)
            prob  = torch.sigmoid(logit).item()
        label = 1 if prob >= threshold else 0
        return label, float(prob)


def build_audio_model(n_mels: int = 128,
                      dropout: float = 0.3) -> SpectrogramCNN:
    """Factory function."""
    return SpectrogramCNN(n_mels=n_mels, dropout=dropout)


def save_audio_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[AudioModel] Saved → {path}")


def load_audio_model(path: str, device: torch.device,
                     n_mels: int = 128) -> SpectrogramCNN:
    model = build_audio_model(n_mels=n_mels)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[AudioModel] Loaded ← {path}")
    return model
