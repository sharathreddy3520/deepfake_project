"""
models/image_model.py
Lightweight EfficientNet-B0 fine-tuned for deepfake image classification.
Labels: 0 = Real, 1 = Fake
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Tuple
import os


class ImageDeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 backbone with a custom binary-classification head.
    The backbone is optionally frozen for low-resource environments.
    """

    def __init__(self, freeze_backbone: bool = False, dropout: float = 0.3):
        super().__init__()

        # Load pretrained backbone
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = tv_models.efficientnet_b0(weights=weights)

        # Remove the default classifier
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),   # single logit → sigmoid for probability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 3, 224, 224).
        Returns:
            Logit tensor (B, 1).
        """
        features = self.backbone(x)
        logits   = self.classifier(features)
        return logits

    def predict(self, x: torch.Tensor,
                threshold: float = 0.5) -> Tuple[int, float]:
        """
        Run inference on a single preprocessed image tensor.

        Args:
            x:         Tensor of shape (1, 3, 224, 224).
            threshold: Decision threshold (default 0.5).

        Returns:
            (label, confidence) where label=0 (Real) or 1 (Fake),
            and confidence is the probability of being Fake.
        """
        self.eval()
        with torch.no_grad():
            logit = self.forward(x)
            prob  = torch.sigmoid(logit).item()
        label = 1 if prob >= threshold else 0
        return label, float(prob)


def build_image_model(freeze_backbone: bool = False,
                      dropout: float = 0.3) -> ImageDeepfakeDetector:
    """Factory function for easy instantiation."""
    return ImageDeepfakeDetector(freeze_backbone=freeze_backbone, dropout=dropout)


def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ImageModel] Saved → {path}")


def load_model(path: str, device: torch.device,
               freeze_backbone: bool = False) -> ImageDeepfakeDetector:
    """Load model state dict from disk."""
    model = build_image_model(freeze_backbone=freeze_backbone)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[ImageModel] Loaded ← {path}")
    return model
