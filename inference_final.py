"""
inference.py
Command-line inference on a single file (image, video, or audio).

Usage:
    python inference.py --input path/to/image.jpg --modality image
    python inference.py --input path/to/video.mp4 --modality video
    python inference.py --input path/to/audio.wav --modality audio
    python inference.py --input path/to/image.jpg --modality image --explain
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils.logger_v2      import get_logger
from utils.preprocessing  import (
    load_image, preprocess_image, denormalize, detect_and_crop_face,
    load_audio, audio_to_melspectrogram, spectrogram_to_tensor,
)

logger = get_logger("inference")

LABEL_MAP = {0: "REAL", 1: "FAKE"}


# ─────────────────────────────────────────────────────────────────────────────
#  Model loaders (lazy – only import what is needed)
# ─────────────────────────────────────────────────────────────────────────────
def _load_image_model(checkpoint: str, device):
    from models.image_model import load_model
    return load_model(checkpoint, device)


def _load_audio_model(checkpoint: str, device):
    from audio_processing.audio_model import load_audio_model
    return load_audio_model(checkpoint, device)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference functions
# ─────────────────────────────────────────────────────────────────────────────
def run_image_inference(input_path: str, checkpoint: str,
                        device, explain: bool = False,
                        face_crop: bool = True) -> dict:
    """Run inference on a single image file."""
    model = _load_image_model(checkpoint, device)
    img   = load_image(input_path)
    if face_crop:
        img = detect_and_crop_face(img)

    tensor = preprocess_image(img).to(device)
    label, conf = model.predict(tensor)

    result = {
        "label":       LABEL_MAP[label],
        "confidence":  conf,
        "real_prob":   1 - conf,
        "fake_prob":   conf,
    }

    if explain:
        from models.explainability import get_image_gradcam
        img_np = denormalize(tensor.cpu())
        heatmap, overlay = get_image_gradcam(model, tensor, img_np)
        out_path = input_path.rsplit(".", 1)[0] + "_gradcam.png"
        Image.fromarray(overlay).save(out_path)
        result["gradcam_path"] = out_path
        logger.info(f"Grad-CAM saved → {out_path}")

    return result


def run_video_inference(input_path: str, checkpoint: str,
                        device, max_frames: int = 16) -> dict:
    """Run inference on a video file (frame-level aggregation)."""
    from models.image_model  import load_model
    from models.video_model  import VideoDeepfakeDetector

    model    = load_model(checkpoint, device)
    detector = VideoDeepfakeDetector(model=model, device=device,
                                     max_frames=max_frames)
    label, mean_prob, frame_probs = detector.predict(input_path)

    return {
        "label":       LABEL_MAP[label],
        "confidence":  mean_prob,
        "real_prob":   1 - mean_prob,
        "fake_prob":   mean_prob,
        "frame_probs": frame_probs,
        "n_frames":    len(frame_probs),
    }


def run_audio_inference(input_path: str, checkpoint: str,
                        device, explain: bool = False) -> dict:
    """Run inference on an audio file."""
    model    = _load_audio_model(checkpoint, device)
    y, sr    = load_audio(input_path)
    spec     = audio_to_melspectrogram(y, sr)
    tensor   = spectrogram_to_tensor(spec).to(device)

    label, conf = model.predict(tensor)
    result = {
        "label":      LABEL_MAP[label],
        "confidence": conf,
        "real_prob":  1 - conf,
        "fake_prob":  conf,
    }

    if explain:
        from models.explainability import get_audio_gradcam
        heatmap, overlay = get_audio_gradcam(model, tensor, spec)
        out_path = input_path.rsplit(".", 1)[0] + "_gradcam.png"
        Image.fromarray(overlay).save(out_path)
        result["gradcam_path"] = out_path
        logger.info(f"Audio Grad-CAM saved → {out_path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Detection – Inference")
    p.add_argument("--input",    required=True,  help="Path to input file")
    p.add_argument("--modality", required=True,
                   choices=["image", "video", "audio"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to model checkpoint (.pth). "
                        "Defaults to models/best_<modality>_model.pth")
    p.add_argument("--explain",  action="store_true",
                   help="Generate Grad-CAM explanation")
    p.add_argument("--face_crop", action="store_true",
                   help="Detect and crop face (image/video)")
    p.add_argument("--max_frames", type=int, default=16,
                   help="Max frames for video inference")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    checkpoint = args.checkpoint or f"models/best_{args.modality}_model.pth"
    if not os.path.isfile(checkpoint):
        logger.error(
            f"Checkpoint not found: {checkpoint}\n"
            "Train a model first using train.py, or place a .pth file there."
        )
        sys.exit(1)

    if args.modality == "image":
        result = run_image_inference(args.input, checkpoint, device,
                                     explain=args.explain,
                                     face_crop=args.face_crop)
    elif args.modality == "video":
        result = run_video_inference(args.input, checkpoint, device,
                                     max_frames=args.max_frames)
    else:
        result = run_audio_inference(args.input, checkpoint, device,
                                     explain=args.explain)

    print("\n" + "=" * 50)
    print(f"  File    : {args.input}")
    print(f"  Modality: {args.modality.upper()}")
    print(f"  Verdict : {result['label']}")
    print(f"  Fake prob: {result['fake_prob']:.4f}")
    print(f"  Real prob: {result['real_prob']:.4f}")
    if "gradcam_path" in result:
        print(f"  Grad-CAM: {result['gradcam_path']}")
    if "frame_probs" in result:
        print(f"  Frames  : {result['n_frames']} processed")
    print("=" * 50)


if __name__ == "__main__":
    main()
