"""
app/streamlit_app.py
Deep Fake Detection — Streamlit Web Application
Supports: Image | Video | Audio
Provides: Prediction, Confidence Score, Grad-CAM / Spectrogram Explanations
"""
import io
import os
import sys
import tempfile
import time
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.preprocessing import (
    load_image, preprocess_image, denormalize, detect_and_crop_face,
    load_audio, audio_to_melspectrogram, spectrogram_to_tensor,
    extract_frames, INFERENCE_TRANSFORM,
)
from models.explainability import (
    get_image_gradcam, get_audio_gradcam, visualise_spectrogram
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── App background ─────────────────────────────── */
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }

    /* ── Sidebar ─────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #0f3460;
    }

    /* ── Result cards ────────────────────────────────── */
    .result-card {
        padding: 20px 24px;
        border-radius: 12px;
        margin: 12px 0;
        font-size: 16px;
    }
    .fake-card  { background: linear-gradient(135deg,#3a0000,#5c1111);
                  border: 1px solid #ff4444; }
    .real-card  { background: linear-gradient(135deg,#003a1a,#115c30);
                  border: 1px solid #00cc66; }
    .info-card  { background: linear-gradient(135deg,#001a3a,#0f3460);
                  border: 1px solid #4488ff; }

    /* ── Metric widgets ──────────────────────────────── */
    [data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 12px;
    }

    /* ── Section headers ─────────────────────────────── */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #4fc3f7;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    /* ── Footer ──────────────────────────────────────── */
    .footer { text-align:center; color:#555; margin-top:40px; font-size:13px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = os.path.join(ROOT, "models")
LABEL_COLOR = {"FAKE": "#ff4444", "REAL": "#00cc66"}
CARD_CLASS  = {"FAKE": "fake-card", "REAL": "real-card"}


def _ckpt(modality: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"best_{modality}_model.pth")


def _model_exists(modality: str) -> bool:
    return os.path.isfile(_ckpt(modality))


@st.cache_resource(show_spinner=False)
def load_image_model():
    from models.image_model import load_model
    return load_model(_ckpt("image"), DEVICE)


@st.cache_resource(show_spinner=False)
def load_audio_model():
    from audio_processing.audio_model import load_audio_model as _lam
    return _lam(_ckpt("audio"), DEVICE)


def render_verdict(label: str, conf: float):
    """Render a big verdict banner + confidence bar."""
    real_pct = (1 - conf) * 100
    fake_pct = conf * 100

    st.markdown(f"""
    <div class="result-card {CARD_CLASS[label]}">
        <h2 style="margin:0; font-size:32px;">
            {'⚠️' if label=='FAKE' else '✅'}
            &nbsp; Verdict: <span style="color:{LABEL_COLOR[label]}">{label}</span>
        </h2>
        <p style="margin:8px 0 0 0; color:#ccc;">
            The media is classified as <strong>{label}</strong> with
            <strong>{conf*100:.1f}%</strong> fake confidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("✅ Real Probability", f"{real_pct:.1f}%")
    col2.metric("⚠️ Fake Probability", f"{fake_pct:.1f}%")


def fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib figure to a PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
#  Demo (untrained) model helper
# ─────────────────────────────────────────────────────────────────────────────
def _demo_image_result(img: Image.Image) -> tuple:
    """
    Return a plausible demo result when no trained checkpoint exists.
    Uses EfficientNet features without fine-tuning (random-ish output).
    """
    from models.image_model import build_image_model
    model = build_image_model().to(DEVICE)
    tensor = preprocess_image(img).to(DEVICE)
    label, conf = model.predict(tensor)
    return label, conf, tensor, model


def _demo_audio_result(y, sr) -> tuple:
    from audio_processing.audio_model import build_audio_model
    model = build_audio_model().to(DEVICE)
    spec   = audio_to_melspectrogram(y, sr)
    tensor = spectrogram_to_tensor(spec).to(DEVICE)
    label, conf = model.predict(tensor)
    return label, conf, tensor, spec, model


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("DeepFake Detector")
    st.markdown("---")

    modality = st.radio(
        "Select Input Type",
        ["🖼️ Image", "🎬 Video", "🎵 Audio"],
        index=0,
    )
    modality_key = modality.split()[1].lower()   # "image" | "video" | "audio"

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold   = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    show_explain = st.checkbox("Show Explanation (Grad-CAM)", value=True)
    face_crop    = st.checkbox("Auto Face Crop", value=True)

    if modality_key == "video":
        max_frames = st.slider("Max Frames to Analyse", 4, 32, 16, 4)
    else:
        max_frames = 16

    st.markdown("---")
    # Model status
    for m in ["image", "audio"]:
        exists = _model_exists(m)
        icon = "🟢" if exists else "🔴"
        st.markdown(f"{icon} **{m.capitalize()} model** {'loaded' if exists else '(demo mode)'}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#888;'>
    <b>Supported datasets:</b><br>
    FaceForensics++, Celeb-DF, DFDC,<br>
    ASVspoof, WaveFake
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#4fc3f7; margin-bottom:4px;'>
    🔍 Deep Fake Detection System
</h1>
<p style='text-align:center; color:#888; margin-bottom:24px;'>
    Multimodal AI · Image · Video · Audio · Grad-CAM Explainability
</p>
""", unsafe_allow_html=True)

tabs = st.tabs(["📤 Upload & Detect", "📊 How It Works", "ℹ️ About"])


# ╔═════════════════════════════════════════════════════════════════════════════
# ║  TAB 1 – Upload & Detect
# ╚═════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    if modality_key == "image":
        st.markdown('<div class="section-header">🖼️ Image Deepfake Detection</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )

        if uploaded:
            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.markdown("**Original Image**")
                img = Image.open(uploaded).convert("RGB")
                st.image(img, use_container_width=True)

            with col_res:
                with st.spinner("🔍 Analysing…"):
                    if face_crop:
                        img_proc = detect_and_crop_face(img)
                    else:
                        img_proc = img

                    if _model_exists("image"):
                        model  = load_image_model()
                        tensor = preprocess_image(img_proc).to(DEVICE)
                        prob   = torch.sigmoid(model(tensor)).item()
                        label  = 1 if prob >= threshold else 0
                        label_str = "FAKE" if label == 1 else "REAL"
                        conf = prob
                    else:
                        label_int, conf, tensor, model = _demo_image_result(img_proc)
                        label_str = "FAKE" if label_int == 1 else "REAL"
                        st.info("ℹ️ No trained model found – running in **demo mode** "
                                "(untrained weights, predictions are illustrative).")

                st.markdown("**Detection Result**")
                render_verdict(label_str, conf)

            # Grad-CAM
            if show_explain:
                st.markdown('<div class="section-header">🔥 Grad-CAM Explanation</div>',
                            unsafe_allow_html=True)

                with st.spinner("Generating Grad-CAM…"):
                    img_np = denormalize(tensor.cpu())
                    try:
                        heatmap, overlay = get_image_gradcam(model, tensor, img_np)
                        c1, c2, c3 = st.columns(3)
                        c1.image(img_proc, caption="Input (cropped)", use_container_width=True)
                        c2.image(heatmap,  caption="Heatmap",         use_container_width=True, clamp=True)
                        c3.image(overlay,  caption="Overlay",         use_container_width=True)

                        st.markdown("""
                        <div class="result-card info-card">
                        <b>🔬 How to read Grad-CAM:</b> Red/warm regions are the areas the
                        model focused on most. In deepfake images, these often highlight
                        inconsistencies around facial boundaries, eyes, or hair edges.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM failed: {e}")

    # ── VIDEO ─────────────────────────────────────────────────────────────────
    elif modality_key == "video":
        st.markdown('<div class="section-header">🎬 Video Deepfake Detection</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a video (MP4, AVI, MOV, MKV)",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded:
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded.name)[1], delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.video(tmp_path)

            with st.spinner(f"🎬 Extracting & analysing {max_frames} frames…"):
                frames = extract_frames(tmp_path, max_frames=max_frames)
                os.unlink(tmp_path)

                if not frames:
                    st.error("Could not extract frames from the video.")
                    st.stop()

                if _model_exists("image"):
                    model = load_image_model()
                else:
                    from models.image_model import build_image_model
                    model = build_image_model().to(DEVICE)
                    st.info("ℹ️ Running in demo mode (no trained checkpoint).")

                frame_probs = []
                for frame in frames:
                    if face_crop:
                        frame = detect_and_crop_face(frame)
                    t = INFERENCE_TRANSFORM(frame).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        p = torch.sigmoid(model(t)).item()
                    frame_probs.append(p)

                mean_prob = float(np.mean(frame_probs))
                label_str = "FAKE" if mean_prob >= threshold else "REAL"

            render_verdict(label_str, mean_prob)

            # Frame grid
            st.markdown('<div class="section-header">📽️ Frame Analysis</div>',
                        unsafe_allow_html=True)

            cols = st.columns(min(8, len(frames)))
            for i, (frame, prob) in enumerate(zip(frames, frame_probs)):
                col = cols[i % len(cols)]
                clr = "#ff4444" if prob >= threshold else "#00cc66"
                col.image(frame, use_container_width=True)
                col.markdown(
                    f"<div style='text-align:center; color:{clr}; font-size:11px;'>"
                    f"{'FAKE' if prob>=threshold else 'REAL'}<br>{prob*100:.0f}%</div>",
                    unsafe_allow_html=True,
                )

            # Frame probability chart
            st.markdown('<div class="section-header">📈 Frame-by-Frame Probabilities</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3),
                                   facecolor="#0f0f1a")
            ax.set_facecolor("#1a1a2e")
            x = range(len(frame_probs))
            ax.fill_between(x, frame_probs, alpha=0.3, color="#ff4444")
            ax.plot(x, frame_probs, "o-", color="#ff4444", linewidth=2, markersize=5)
            ax.axhline(threshold, color="#ffcc00", linestyle="--",
                       linewidth=1.5, label=f"Threshold ({threshold:.2f})")
            ax.set_xlim(0, len(frame_probs) - 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Frame", color="#aaa")
            ax.set_ylabel("Fake Probability", color="#aaa")
            ax.set_title("Per-Frame Fake Probability", color="#4fc3f7")
            ax.tick_params(colors="#aaa")
            ax.legend(facecolor="#1a1a2e", labelcolor="#eee")
            for spine in ax.spines.values():
                spine.set_color("#0f3460")
            st.pyplot(fig)
            plt.close(fig)

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    elif modality_key == "audio":
        st.markdown('<div class="section-header">🎵 Audio Deepfake Detection</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload an audio file (WAV, MP3, FLAC, OGG)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
        )

        if uploaded:
            st.audio(uploaded)

            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded.name)[1], delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("🎵 Analysing audio…"):
                y, sr   = load_audio(tmp_path)
                spec    = audio_to_melspectrogram(y, sr)
                tensor  = spectrogram_to_tensor(spec).to(DEVICE)
                os.unlink(tmp_path)

                if _model_exists("audio"):
                    model     = load_audio_model()
                    prob      = torch.sigmoid(model(tensor)).item()
                    label_int = 1 if prob >= threshold else 0
                    conf      = prob
                    label_str = "FAKE" if label_int else "REAL"
                else:
                    label_int, conf, tensor, spec, model = _demo_audio_result(y, sr)
                    label_str = "FAKE" if label_int else "REAL"
                    st.info("ℹ️ Running in demo mode.")

            render_verdict(label_str, conf)

            # Spectrogram visualisation
            st.markdown('<div class="section-header">🎼 Mel Spectrogram</div>',
                        unsafe_allow_html=True)
            spec_img = visualise_spectrogram(
                spec, title=f"Mel Spectrogram ({uploaded.name})", sr=sr
            )
            st.image(spec_img, use_container_width=True)

            # Grad-CAM on spectrogram
            if show_explain:
                st.markdown('<div class="section-header">🔥 Saliency Map (Grad-CAM on Spectrogram)</div>',
                            unsafe_allow_html=True)
                with st.spinner("Generating audio Grad-CAM…"):
                    try:
                        heatmap, overlay = get_audio_gradcam(model, tensor, spec)
                        c1, c2 = st.columns(2)
                        c1.image(heatmap, caption="Heatmap", use_container_width=True, clamp=True)
                        c2.image(overlay, caption="Overlay on Spectrogram",
                                 use_container_width=True)

                        st.markdown("""
                        <div class="result-card info-card">
                        <b>🔬 How to read the Audio Grad-CAM:</b> Warm regions highlight
                        time-frequency bands the model found suspicious. Synthesised (fake)
                        voices often have anomalies in specific frequency ranges or unnatural
                        temporal patterns.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Audio Grad-CAM failed: {e}")


# ╔═════════════════════════════════════════════════════════════════════════════
# ║  TAB 2 – How It Works
# ╚═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🧠 System Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🖼️ Image Pipeline
        1. **Upload** JPG/PNG image
        2. **Face Detection** (Haar Cascade)
        3. **Crop & Resize** → 224×224
        4. **EfficientNet-B0** (ImageNet pretrained)
        5. **Binary Classification** (Real / Fake)
        6. **Grad-CAM** heatmap overlay
        """)

    with col2:
        st.markdown("""
        ### 🎬 Video Pipeline
        1. **Upload** MP4/AVI video
        2. **Frame Extraction** (evenly-spaced)
        3. **Per-frame Image Model** inference
        4. **Mean-probability Aggregation**
        5. **Threshold** → verdict
        6. **Frame chart** visualisation
        """)

    with col3:
        st.markdown("""
        ### 🎵 Audio Pipeline
        1. **Upload** WAV/MP3/FLAC file
        2. **Load & Resample** → 16 kHz
        3. **Mel Spectrogram** (128 bands)
        4. **SpectrogramCNN** inference
        5. **Binary Classification** (Real / Fake)
        6. **Grad-CAM** on spectrogram
        """)

    st.markdown("---")
    st.markdown("## 📈 Model Details")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | **Image backbone** | EfficientNet-B0 |
        | **Audio model** | SpectrogramCNN (3 conv blocks) |
        | **Input size (image)** | 224 × 224 px |
        | **Input size (audio)** | 128 mel × ~125 time |
        | **Loss function** | Binary Cross-Entropy |
        | **Optimiser** | AdamW + Cosine LR |
        """)
    with col_b:
        st.markdown("""
        | Dataset | Modality |
        |---------|---------|
        | FaceForensics++ | Image/Video |
        | Celeb-DF | Image/Video |
        | DFDC | Image/Video |
        | ASVspoof 2019/2021 | Audio |
        | WaveFake | Audio |
        """)

    st.markdown("---")
    st.markdown("## 🔥 Explainability: Grad-CAM")
    st.markdown("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** works by:

    1. Running a forward pass through the model
    2. Computing gradients of the output score w.r.t. the last convolutional feature map
    3. Global-average-pooling the gradients to get channel importance weights
    4. Computing a weighted sum of activation maps → ReLU → resize to input resolution

    This highlights *which pixels/time-frequency regions* drove the fake/real decision,
    making the model transparent and debuggable.
    """)


# ╔═════════════════════════════════════════════════════════════════════════════
# ║  TAB 3 – About
# ╚═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    ## 🎯 About this Project

    This system is a **production-ready, multimodal deep fake detection** tool built with:

    - **PyTorch** – model training and inference
    - **EfficientNet-B0** – lightweight, accurate image backbone
    - **Grad-CAM** – gradient-based explainability
    - **librosa** – audio feature extraction
    - **OpenCV** – video frame extraction & face detection
    - **Streamlit** – interactive web interface

    ### 📂 Project Structure
    ```
    deepfake_project/
    ├── models/
    │   ├── image_model.py       # EfficientNet-B0 classifier
    │   ├── video_model.py       # Frame-level aggregation
    │   ├── datasets.py          # PyTorch Dataset classes
    │   └── explainability.py    # Grad-CAM engine
    ├── audio_processing/
    │   └── audio_model.py       # SpectrogramCNN
    ├── utils/
    │   ├── preprocessing.py     # Transforms, face crop, audio utils
    │   ├── metrics.py           # Evaluation metrics
    │   └── logger.py            # Centralised logging
    ├── app/
    │   └── streamlit_app.py     # This web app
    ├── train.py                 # Training entry-point
    ├── inference.py             # CLI inference
    └── README.md
    ```

    ### ⚡ Quick Start
    ```bash
    # Train image model (with dummy data)
    python train.py --modality image --dummy --epochs 2

    # Train audio model (with real data)
    python train.py --modality audio --data_dir data/audio --epochs 20

    # Run the web app
    streamlit run app/streamlit_app.py
    ```

    ### 🔒 Disclaimer
    This tool is intended for **research and educational purposes only**.
    It should not be used as the sole basis for legal or forensic decisions.
    """)

# Footer
st.markdown(
    '<div class="footer">Deep Fake Detection System · Built with PyTorch & Streamlit</div>',
    unsafe_allow_html=True,
)
