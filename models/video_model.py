"""
models/video_model.py
Video deepfake detection by running the image model on extracted frames
and aggregating frame-level predictions.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image

from utils.preprocessing import (
    extract_frames, detect_and_crop_face, INFERENCE_TRANSFORM
)
from models.image_model import ImageDeepfakeDetector, load_model


class VideoDeepfakeDetector:
    """
    Wraps the image-level detector to produce a video-level verdict.

    Strategy:
        1. Extract evenly-spaced frames.
        2. (Optionally) detect and crop faces in each frame.
        3. Run the image model on every frame.
        4. Aggregate via mean-probability averaging.
    """

    def __init__(self,
                 model: ImageDeepfakeDetector,
                 device: torch.device,
                 max_frames: int = 16,
                 use_face_crop: bool = True,
                 threshold: float = 0.5):
        self.model        = model
        self.device       = device
        self.max_frames   = max_frames
        self.use_face_crop = use_face_crop
        self.threshold    = threshold

    def predict(self, video_path: str) -> Tuple[int, float, List[float]]:
        """
        Predict whether a video is real or fake.

        Args:
            video_path: Path to the video file.

        Returns:
            (label, confidence, frame_probs)
            label        – 0 (Real) or 1 (Fake)
            confidence   – Mean fake-probability across frames
            frame_probs  – Per-frame fake probabilities
        """
        frames = extract_frames(video_path, max_frames=self.max_frames)
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")

        frame_probs = []
        self.model.eval()

        for frame in frames:
            if self.use_face_crop:
                frame = detect_and_crop_face(frame)

            tensor = INFERENCE_TRANSFORM(frame).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self.model(tensor)
                prob  = torch.sigmoid(logit).item()

            frame_probs.append(prob)

        mean_prob = float(np.mean(frame_probs))
        label     = 1 if mean_prob >= self.threshold else 0
        return label, mean_prob, frame_probs

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str,
                        device: torch.device,
                        **kwargs) -> "VideoDeepfakeDetector":
        """Convenience constructor that loads weights from a checkpoint."""
        model = load_model(checkpoint_path, device)
        return cls(model=model, device=device, **kwargs)
