# 🩺 Diabetic Retinopathy Grading — Clinical-Grade AI Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20MPS%20%7C%20CUDA%20%7C%20CPU-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Development-green)
![Research](https://img.shields.io/badge/Use-Research%20Only-orange)

**Ordinal 5-class DR grading from fundus photographs**  
EfficientNetV2-S · GeM Pooling · DeepCORAL · Temperature Scaling · Fundus Ensemble Validator

</div>

---

> ⚠️ **Research Use Only.** This system has not been validated for clinical deployment and must not be used for real patient diagnosis or treatment decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Clinical Context](#clinical-context)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Training Pipeline](#training-pipeline)
8. [Evaluation](#evaluation)
9. [Streamlit Deployment](#streamlit-deployment)
10. [Cross-Platform Notes](#cross-platform-notes)
11. [Roadmap](#roadmap)
12. [Disclaimer](#disclaimer)

---

## Overview

This project implements a **clinical-grade diabetic retinopathy (DR) grading pipeline** that classifies fundus photographs into five severity grades (0–4) as defined by the International Clinical DR Disease Severity Scale. The full system covers data ingestion, multi-phase training, post-hoc calibration, model explainability via Grad-CAM++, domain-shift robustness evaluation, and a Streamlit-based clinical UI.

**Target metric:** Quadratic Weighted Kappa (QWK) ≥ 0.90

| Component | Detail |
|-----------|--------|
| Backbone | EfficientNetV2-S (`tf_efficientnetv2_s`, 21M params) |
| Pooling | Generalised Mean Pooling (GeM, learnable exponent) |
| Head | DRHead: Linear → BN → SiLU → Dropout (×2 layers) |
| Loss | 0.5 × CE (label smooth=0.005) + 0.5 × Focal (γ=2) |
| Domain Adapt | DeepCORAL (covariance alignment, λ=0.0→0.10→0.20) |
| Calibration | Temperature Scaling (post-hoc, L-BFGS, ECE reported) |
| Augmentation | 7-tier pipeline (CLAHE, geometry, colour, occlusion, elastic) |
| Regularisation | Dropout 0.4 · Label Smoothing · CutMix · MixUp · SWA |
| Anti-Overfit | 3-phase progressive unfreeze + Stochastic Weight Averaging |
| Training | 6 epochs total (2 per phase), effective batch 16 |
| Resolution | 512 px or 1024 px (adaptive, auto-scaled batch) |
| Device | Apple MPS · CUDA · CPU (cross-platform, zero code changes) |
| Fundus Gate | MobileNetV3 + ConvNeXt-22k + ConvNeXt-12k ensemble + Mahalanobis |
| Explainability | Grad-CAM++ with multi-layer fusion |
| UI | Streamlit with live CLAHE preview, grade card, probability chart |

---

## Clinical Context

Diabetic retinopathy is the leading cause of preventable blindness worldwide. Timely grading of fundus photographs allows stratified management:

| Grade | Name | Clinical Action |
|-------|------|----------------|
| 0 | No DR | Routine follow-up every 1–2 years |
| 1 | Mild NPDR | Review in 12 months |
| 2 | Moderate NPDR | Review in 6 months |
| 3 | Severe NPDR | Urgent referral within 1–2 months |
| 4 | Proliferative DR | Immediate ophthalmology referral |

The pipeline targets **QWK ≥ 0.90**, the threshold commonly cited in clinical AI literature for grading equivalence with human experts.

---

## Architecture

```
Input (fundus RGB)
       │
       ▼
┌─────────────────────────────────────┐
│  Preprocessing                      │
│  CLAHE (LAB) → Retinal Crop →       │
│  Resize (512 / 1024 px)             │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  EfficientNetV2-S (frozen → partial │
│  unfreeze → full unfreeze)          │
│  Stage maps: 48 × 48 feature maps   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  GeM Pooling  (p learnable, ≈ 3.0)  │
│  → Flatten → 1280-dim feature vec  │
└─────────────────────────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌─────────────┐              ┌─────────────────┐
│  DRHead     │              │  DeepCORAL      │
│  1024 → 512 │              │  (domain align  │
│  → 5 logits │              │  source/target  │
└─────────────┘              │  covariance)    │
       │                     └─────────────────┘
       ▼
┌─────────────────────────────────────┐
│  Temperature Scaling (T post-hoc)  │
│  Final probabilities                │
└─────────────────────────────────────┘
       │
       ▼
   Grade 0–4

─────────────────────────────────────
Fundus Validator (inference gate only)
─────────────────────────────────────
Upload → MobileNetV3-L (w=0.25)
       → ConvNeXt-Tiny-22k  (w=0.45)  ← primary
       → ConvNeXt-Tiny-12k-ft(w=0.30)
       → Weighted ensemble prob + Mahalanobis distance
       → ACCEPT / REJECT
```

### Training Strategy — 3-Phase Progressive Unfreeze

| Phase | Layers Unfrozen | CORAL λ | Scheduler | Notes |
|-------|----------------|---------|-----------|-------|
| P1 (2 ep) | Head only | 0.00 | OneCycleLR (max_lr=3e-3) | CutMix + MixUp ON |
| P2 (2 ep) | Last 6 blocks + head | 0.10 | CosineAnnealingWarmRestarts | Differential LR (bb×0.05) |
| P3 (2 ep) | All layers | 0.20 | CosineAnnealing + SWA all epochs | bb LR×0.0017, head LR×0.017 |

**Effective batch = 16** at all resolutions:

| IMG_SIZE | BATCH_SIZE | GRAD_ACCUM | EffBatch |
|----------|------------|------------|----------|
| 1024 px  | 4 | 4 | 16 |
| 512 px   | 8 | 2 | 16 |

---

## Dataset

**APTOS 2019 Blindness Detection**

| Split | Samples | Source |
|-------|---------|--------|
| Train (balanced) | 5,000 (1,000/class) | APTOS 2019 |
| Validation | ~334 | APTOS 2019 |
| Test | ~334 | APTOS 2019 |
| Domain Shift (ext.) | ~334 | Test set + colour jitter |

- **Local path:** `/Users/gokulnathan/Desktop/ai_dr/aptos2019-blindness-detection`
- **Expected files:** `train_images/*.png` (≥ 3,000 images) + `train.csv`
- **Class distribution:** Highly imbalanced (G0 dominant) → balanced to 1,000/class before training

No Kaggle download is required. The dataset is read directly from the local path above.

---

## Project Structure

```
project/
│
├── DR_Grading_v19_1024_local.ipynb   ← Main training notebook (34 cells)
│
├── streamlit_app/
│   ├── app.py                        ← Streamlit UI
│   ├── model_loader.py               ← Model classes + cached loaders
│   ├── requirements.txt              ← Python dependencies
│   └── artifacts/                    ← Model checkpoints (copy from notebook)
│       ├── final_model.pt
│       └── fundus_validator_v3.pt
│
└── ~/DR_data_v15/artifacts_v15/      ← Notebook artifact output directory
    ├── final_model.pt                ← DR grading checkpoint (Cell 22)
    ├── fundus_validator_v3.pt        ← Fundus validator checkpoint (Cell 12)
    ├── best_model.pt                 ← Best checkpoint (saved during training)
    ├── phase1.pt / phase2.pt / phase3.pt
    ├── temperature_scaler.pt
    ├── splits.parquet
    ├── training_curves.png
    ├── gradcam_samples.png
    ├── domain_shift_report.csv
    └── _resume_state.json            ← Two-tier resume system state
```

### Notebook Cell Map

| # | Cell ID | Purpose |
|---|---------|---------|
| 0 | Header | Project summary + execution order |
| 1 | `step01_env` | Environment check (Python, PyTorch, device) |
| 2 | `step02_imports` | Imports · device · seeds · state management · `_master_recover()` |
| 3 | Dataset path check | Verify local APTOS path |
| 4 | `step04_download` | Local dataset load (no Kaggle) |
| 5 | `step05_splits` | 80/10/10 stratified splits → parquet cache |
| 6 | `step06_blur_qc` | Laplacian variance blur QC filter |
| 7 | `step07_preproc` | Image preprocessing (CLAHE + retinal crop) + resolution cache |
| 8 | `step08_cache` | Pre-cache all images at target resolution |
| 9 | `step09_eda` | Class distribution EDA + sample grid |
| 10 | `step10_augment` | Augmentation · Dataset · DataLoaders (resolution-aware) |
| 11 | `step_fundus_val_v3` | Fundus validator ensemble training (MobileNetV3 + ConvNeXt) |
| 12–13 | Model / losses | DRModel · GeM · DRHead · loss functions · CORAL |
| 14–16 | Training phases | P1 head-only · P2 partial unfreeze · P3 full + SWA |
| 17 | Curves | Training/validation curve plots |
| 18 | Evaluation | Metrics · confusion matrix · ROC · PR curves |
| 19 | Calibration | Temperature scaling · ECE |
| 20 | Grad-CAM++ | Multi-layer CAM visualisation |
| 21 | Domain shift | External evaluation under colour/gamma shift |
| 22 | `step22_save` | Save `final_model.pt` bundle |
| 23 | Artifact list | Print all saved artefacts |
| 24 | TTA evaluation | 8-view test-time augmentation |
| 25 | Threshold opt | Per-class threshold optimisation for ≥92% recall |
| 33 | Streamlit | App written to `streamlit_app/` (legacy inline writer) |

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- PyTorch 2.1+ (with MPS support for Apple Silicon, or CUDA for NVIDIA GPU)
- APTOS 2019 dataset extracted locally

### 1. Clone / set up environment

```bash
# Create a virtual environment
python3 -m venv dr_env
source dr_env/bin/activate      # macOS / Linux
# dr_env\Scripts\activate       # Windows

# Install dependencies
pip install -r streamlit_app/requirements.txt
```

### 2. Verify dataset path

The notebook expects the APTOS 2019 dataset at:

```
/Users/gokulnathan/Desktop/ai_dr/aptos2019-blindness-detection/
    ├── train_images/    (*.png, ≥ 3,000 files)
    └── train.csv
```

To use a different path, update `DS_DIR` in Cell 4 or set the env variable:

```bash
export DS_ROOT=/path/to/aptos2019-blindness-detection
```

### 3. Run the notebook

Open `DR_Grading_v19_1024_local.ipynb` in Jupyter and run all cells in order.

```bash
jupyter notebook DR_Grading_v19_1024_local.ipynb
```

**On kernel restart:** run Cell 2 only — `_master_recover()` restores all state from disk automatically.

---

## Training Pipeline

### Resume System

The notebook uses a two-tier resume system so training can be interrupted and restarted at any point:

1. **`.done` flag files** — each cell writes a flag on completion; skipped on re-run
2. **Per-phase `.pt` checkpoints** — `phase1.pt`, `phase2.pt`, `phase3.pt` + `best_model.pt`

### Key Hyperparameters

```python
SEED         = 42
IMG_SIZE     = 1024        # or 512 — set in Cell 2, honoured everywhere
BACKBONE     = "tf_efficientnetv2_s"
NUM_CLASSES  = 5

LR           = 3e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.005
EPOCHS_HEAD  = 2           # Phase 1
EPOCHS_FULL  = 2           # Phase 2
EPOCHS_FINE  = 2           # Phase 3  (+ SWA all epochs)

# CORAL weights per phase
P1_CORAL     = 0.00        # Head-only warmup — no domain alignment
P2_CORAL     = 0.10        # Fast diagonal CORAL
P3_CORAL     = 0.20        # Full CORAL + SWA

# Batch scaling (EffBatch = 16 always)
# 1024px → batch=4, accum=4
#  512px → batch=8, accum=2
```

### 7-Tier Augmentation Pipeline

| Tier | Transforms |
|------|-----------|
| T1 Geometry | 360° rotate · HFlip · VFlip · RandomRotate90 · ShiftScaleRotate |
| T2 Crop | RandomResizedCrop (scale 0.55–1.0) |
| T3 Colour | CLAHE · RandomBrightnessContrast · RandomGamma · Equalize · Posterize · Solarize · HueSaturationValue · ColorJitter · RGBShift |
| T4 Sharpness | Sharpen · UnsharpMask · GaussianBlur · GaussNoise · ISONoise · Downscale · MedianBlur |
| T5 Elastic | ElasticTransform · GridDistortion · OpticalDistortion |
| T6 Occlusion | CoarseDropout (scaled to IMG_SIZE) · GridDropout |
| T7 Camera | ToSepia · ChannelShuffle · ChannelDropout |

All transforms are validated for **albumentations ≥ 1.4.x** (no deprecated `AutoContrast` or old API calls).

---

## Evaluation

Reported on held-out test set (334 samples, 10% stratified split):

| Metric | Value |
|--------|-------|
| QWK (Quadratic Weighted Kappa) | Target ≥ 0.90 |
| Accuracy | — |
| F1 Macro | — |
| F1 Weighted | — |
| AUROC Macro | — |
| Binary AUROC (DR vs No-DR) | — |
| Sensitivity | — |
| Specificity | — |
| ECE (Expected Calibration Error) | — |

> Results shown after training completes. Interim results from earlier notebook versions are visible in the screenshots in the `docs/` folder.

### Confusion Matrix Interpretation

The model is evaluated on both **internal** (APTOS test split) and **external** (domain-shifted) test sets. Domain shift is simulated by applying random colour jitter, gamma, and Gaussian noise to the internal test images.

---

## Streamlit Deployment

### Option A — Point at notebook artifacts (fastest)

```bash
cd streamlit_app
pip install -r requirements.txt

# Point directly at notebook output — no file copying needed
export ARTIFACTS_PATH=~/DR_data_v15/artifacts_v15
streamlit run app.py
```

### Option B — Copy artifacts locally

```bash
mkdir -p streamlit_app/artifacts
cp ~/DR_data_v15/artifacts_v15/final_model.pt          streamlit_app/artifacts/
cp ~/DR_data_v15/artifacts_v15/fundus_validator_v3.pt  streamlit_app/artifacts/

cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Artifact Discovery Order

The app searches for model files in this order (first match wins):

```
1. $ARTIFACTS_PATH           (env variable)
2. ./artifacts/              (next to app.py)
3. ~/DR_data_v15/artifacts_v15/
4. ~/DR_data/artifacts/
5. ~/DR_data_v18/artifacts/
```

If no artifacts are found, the app shows a **Setup Required** page with exact copy commands.

### Prerequisites

Before deploying the Streamlit app, the following notebook cells must have completed successfully:

- **Cell 12** (`step_fundus_val_v3`) → generates `fundus_validator_v3.pt`
- **Cell 22** (`step22_save`) → generates `final_model.pt`

### App Features

- **Fundus gate** — rejects non-fundus images before inference (5-tier: heuristics → ensemble → Mahalanobis)
- **CLAHE preprocessing** — live side-by-side preview of raw vs. preprocessed image
- **Grade card** — colour-coded result with clinical follow-up advice
- **Probability chart** — interactive Altair bar chart of all 5 grade probabilities
- **Temperature calibration** — post-hoc scaling applied automatically
- **Device auto-detection** — CUDA → MPS → CPU, shown in sidebar

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY streamlit_app/ .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

```bash
docker build -t dr-grading .
docker run -p 8501:8501 \
  -v ~/DR_data_v15/artifacts_v15:/app/artifacts \
  dr-grading
```

---

## Cross-Platform Notes

The notebook and app are fully cross-platform. Device selection is automatic:

```python
DEVICE = "cuda"  # NVIDIA GPU
DEVICE = "mps"   # Apple Silicon (M1/M2/M3)
DEVICE = "cpu"   # fallback
```

| Feature | macOS MPS | Windows CUDA | Linux CUDA |
|---------|-----------|--------------|------------|
| `num_workers` | 0 (spawn safety) | 4+ | 4+ |
| `pin_memory` | False | True | True |
| `channels_last` | ✅ | ✅ | ✅ |
| `torch.compile` | ⚠️ Skipped | ✅ | ✅ |
| AMP autocast | ✅ (`torch.amp`) | ✅ | ✅ |
| `.cpu()` before `.numpy()` | Required | Optional | Optional |

---

## Roadmap

- [ ] Achieve QWK ≥ 0.90 on APTOS 2019 test set
- [ ] Multi-dataset training (APTOS + DDR + Messidor-2)
- [ ] 8-view TTA integration into final predictions
- [ ] Grad-CAM++ overlay in Streamlit UI
- [ ] ONNX export for edge deployment
- [ ] Streamlit Community Cloud deployment

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.16
albumentations>=1.4.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.35.0
altair>=5.0.0
```

---

## Disclaimer

This project is developed for **research and educational purposes only**. The model has been trained on a single dataset (APTOS 2019) and has not undergone clinical validation, regulatory review, or safety evaluation. It must not be used for patient screening, diagnosis, or any clinical decision-making. Always consult a qualified ophthalmologist for the assessment and management of diabetic retinopathy.

---

<div align="center">
Built with PyTorch · timm · albumentations · Streamlit<br>
Dataset: <a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection">APTOS 2019 Blindness Detection</a>
</div>
