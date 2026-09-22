# 🔍 Deep Fake Detection System
### Multimodal AI — Image · Video · Audio · Grad-CAM Explainability

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## 📖 Project Overview

A production-ready deepfake detection system that classifies whether input
media (image, video, or audio) is **real or fake**, and provides visual
**explainability** via Grad-CAM heatmaps.

| Modality | Model | Explainability |
|----------|-------|----------------|
| Image    | EfficientNet-B0 (fine-tuned) | Grad-CAM overlay |
| Video    | Frame-level image model + mean aggregation | Per-frame chart |
| Audio    | SpectrogramCNN on Mel spectrograms | Grad-CAM on spectrogram |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web App                         │
│         (Upload → Predict → Confidence → Grad-CAM)          │
└────────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
     ┌───────▼──────┐ ┌───────▼──────┐ ┌──────▼────────┐
     │ Image Pipeline│ │ Video Pipeline│ │ Audio Pipeline│
     │               │ │               │ │               │
     │ • Face detect │ │ • Frame extract│ │ • Resample    │
     │ • Resize 224² │ │ • Face crop   │ │ • Mel spec    │
     │ • EfficientNet│ │ • Image model │ │ • SpectCNN    │
     │ • Grad-CAM    │ │ • Aggregation │ │ • Grad-CAM    │
     └───────────────┘ └───────────────┘ └───────────────┘
```

---

## 📂 Project Structure

```
deepfake_project/
│
├── app/
│   └── streamlit_app.py        # Streamlit web interface
│
├── models/
│   ├── image_model.py          # EfficientNet-B0 binary classifier
│   ├── video_model.py          # VideoDeepfakeDetector (frame aggregation)
│   ├── datasets.py             # PyTorch Dataset + DataLoader builders
│   └── explainability.py       # Grad-CAM engine + visualisation helpers
│
├── audio_processing/
│   └── audio_model.py          # SpectrogramCNN (3-block CNN)
│
├── utils/
│   ├── preprocessing.py        # Image / video / audio preprocessing
│   ├── metrics.py              # Accuracy, Precision, Recall, F1, ROC-AUC
│   └── logger.py               # Centralised logging (console + file)
│
├── data/
│   ├── images/{real,fake}/     # Image dataset (FaceForensics++, Celeb-DF…)
│   ├── videos/{real,fake}/     # Video dataset (DFDC…)
│   └── audio/{real,fake}/      # Audio dataset (ASVspoof, WaveFake…)
│
├── models/                     # Saved checkpoints (.pth files)
├── logs/                       # Training & inference logs
├── sample_data/
│   └── generate_sample_data.py # Creates dummy data for testing
│
├── train.py                    # Training entry-point (image + audio)
├── inference.py                # CLI single-file inference
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone / download the project

```bash
git clone <repo-url> deepfake_project
cd deepfake_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** If you have a CUDA GPU, install the matching
> `torch` wheel from https://pytorch.org/get-started/locally/

---

## 🚀 How to Run

### A. Quick smoke-test (no dataset needed)

```bash
# Generate synthetic dummy images & audio
python sample_data/generate_sample_data.py

# Train image model on random data (2 epochs)
python train.py --modality image --dummy --epochs 2

# Train audio model on random data
python train.py --modality audio --dummy --epochs 2
```

### B. Train on a real dataset

```bash
# Place your data:
#   data/images/real/  →  real face images
#   data/images/fake/  →  deepfake face images

python train.py \
    --modality    image \
    --data_dir    data/images \
    --epochs      20 \
    --batch_size  16 \
    --lr          1e-4 \
    --face_crop          # optional: crop faces before training

# Audio (same pattern)
python train.py \
    --modality  audio \
    --data_dir  data/audio \
    --epochs    30
```

### C. CLI inference on a single file

```bash
python inference.py --input path/to/image.jpg  --modality image  --explain
python inference.py --input path/to/video.mp4  --modality video
python inference.py --input path/to/audio.wav  --modality audio  --explain
```

### D. Launch the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📊 Supported Datasets

| Dataset | Modality | Link |
|---------|---------|------|
| **FaceForensics++** | Image/Video | [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF** | Image/Video | [github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** | Image/Video | [kaggle.com/c/deepfake-detection-challenge](https://www.kaggle.com/c/deepfake-detection-challenge) |
| **ASVspoof 2019/2021** | Audio | [asvspoof.org](https://www.asvspoof.org) |
| **WaveFake** | Audio | [github.com/RUB-SysSec/WaveFake](https://github.com/RUB-SysSec/WaveFake) |

Data directory structure expected:

```
data/
├── images/
│   ├── real/   ← real face images (any common format)
│   └── fake/   ← deepfake images
├── videos/
│   ├── real/
│   └── fake/
└── audio/
    ├── real/   ← genuine speech recordings
    └── fake/   ← synthesised / voice-converted speech
```

---

## 🔥 Grad-CAM Explainability

Grad-CAM highlights *which regions* of the input the model focused on:

- **Images/Video frames** → warm colours (red/orange) mark suspicious facial
  regions (boundary artefacts, eye inconsistencies, blending seams).
- **Audio spectrograms** → warm regions mark suspicious time-frequency bands
  (unnatural harmonics, missing noise floor, synthesis artefacts).

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True fakes / all flagged as fake |
| **Recall** | True fakes detected / all actual fakes |
| **F1-Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Area under the ROC curve |

Training curves (loss + accuracy) are saved automatically to `models/`.

---

## ☁️ Running on Replit

1. Fork this project on Replit (Python template).
2. In the **Packages** tab, install: `torch torchvision torchaudio streamlit librosa opencv-python-headless scikit-learn timm soundfile`.
3. Set the **Run** command: `streamlit run app/streamlit_app.py --server.port 8080`.
4. The app opens in the Replit webview.

> Replit has limited RAM; set `--dummy` flag and use small epoch counts for training.

---

## ⚡ Low-Resource Optimisations

- EfficientNet-B0 is one of the smallest high-accuracy backbones (~5.3 M params).
- Use `--freeze_backbone` to train only the classification head (faster, less RAM).
- Reduce `--batch_size` to 4–8 on CPU-only machines.
- Set `MAX_AUDIO_LEN = 2` in `utils/preprocessing.py` to halve audio memory.
- Video inference: reduce `max_frames` to 8 for faster processing.

---

## 📝 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [EfficientNet paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Grad-CAM paper](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [FaceForensics++](https://arxiv.org/abs/1901.08971) — Rössler et al., 2019
