"""
models/explainability.py
Grad-CAM implementation for both image CNNs and audio spectrogram CNNs.
Produces visual saliency overlays that explain model decisions.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, Tuple
import io


# ─────────────────────────────────────────────────────────────────────────────
#  Core Grad-CAM engine
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(input_tensor)   # numpy (H, W) float32 [0,1]
        overlay = cam.overlay(original_image, heatmap)
    """

    def __init__(self, model: torch.nn.Module,
                 target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._hooks       = []
        self._register_hooks()

    # ── Private helpers ───────────────────────────────────────────────────────
    def _register_hooks(self):
        def save_activation(_, __, output):
            self._activations = output.detach()

        def save_gradient(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(save_activation)
        )
        self._hooks.append(
            self.target_layer.register_backward_hook(save_gradient)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(self, input_tensor: torch.Tensor,
                 class_idx: int = 0) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input.

        Args:
            input_tensor: (1, C, H, W) or (1, 1, H, W) for spectrograms.
            class_idx:    Output neuron to differentiate (default 0 for sigmoid).

        Returns:
            Normalised heatmap as a (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass w.r.t. the target class logit
        self.model.zero_grad()
        if output.shape[-1] == 1:
            score = output[0, 0]          # binary sigmoid head
        else:
            score = output[0, class_idx]  # softmax head
        score.backward()

        # Pool gradients across spatial dims → channel weights
        grads   = self._gradients            # (1, C, h, w)
        acts    = self._activations          # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input spatial size
        h_in, w_in = input_tensor.shape[-2:]
        cam = F.interpolate(cam, size=(h_in, w_in),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    @staticmethod
    def overlay(image_rgb: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Blend a Grad-CAM heatmap over an RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 array.
            heatmap:   (H, W) float32 in [0, 1].
            alpha:     Heatmap opacity.
            colormap:  OpenCV colormap constant.

        Returns:
            (H, W, 3) uint8 blended image.
        """
        # Resize heatmap to match image
        h, w = image_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap (expects 0-255 uint8)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap
        )
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        blended = (1 - alpha) * image_rgb.astype(np.float32) +                    alpha * heatmap_rgb.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  High-level wrappers
# ─────────────────────────────────────────────────────────────────────────────
def get_image_gradcam(model, input_tensor: torch.Tensor,
                      original_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM for an EfficientNet image model.

    Args:
        model:          ImageDeepfakeDetector instance.
        input_tensor:   (1, 3, 224, 224) tensor.
        original_image: (H, W, 3) uint8 numpy array (denormalised).

    Returns:
        (heatmap, overlay_image)  both as uint8 numpy arrays.
    """
    # Target: last Conv2d block of EfficientNet-B0 backbone
    target_layer = model.backbone.features[-1]

    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(input_tensor)
    cam.remove_hooks()

    overlay = GradCAM.overlay(original_image, heatmap)
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    return heatmap_vis, overlay


def get_audio_gradcam(model, spectrogram_tensor: torch.Tensor,
                      spec_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM saliency for the audio SpectrogramCNN.

    Args:
        model:              SpectrogramCNN instance.
        spectrogram_tensor: (1, 1, n_mels, T) tensor.
        spec_2d:            Raw spectrogram (n_mels, T) for visualisation background.

    Returns:
        (heatmap, overlay_image) as uint8 numpy arrays.
    """
    target_layer = model.features[-3]  # last Conv2d before AdaptiveAvgPool

    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(spectrogram_tensor)
    cam.remove_hooks()

    # Render spectrogram as RGB background
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(spec_2d, aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    bg = np.array(Image.open(buf).convert("RGB"))

    overlay = GradCAM.overlay(bg, heatmap, alpha=0.5)
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    return heatmap_vis, overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Spectrogram visualisation helper (standalone, no model needed)
# ─────────────────────────────────────────────────────────────────────────────
def visualise_spectrogram(spec: np.ndarray,
                          title: str = "Mel Spectrogram",
                          sr: int = 16_000,
                          hop_length: int = 512) -> Image.Image:
    """
    Render a Mel spectrogram to a PIL image for display in Streamlit.

    Args:
        spec:       2-D log-Mel spectrogram array (n_mels x T).
        title:      Figure title.
        sr:         Sample rate.
        hop_length: Hop length used when computing the spectrogram.

    Returns:
        PIL Image (RGB).
    """
    import librosa.display
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(spec, sr=sr, hop_length=hop_length,
                             x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
