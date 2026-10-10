---
title: DR AI - Diabetic Retinopathy Grading
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: DR grading with Grad-CAM++ explainability heatmaps
---

# 🩺 DR AI — Diabetic Retinopathy Grading

AI-powered severity classification of **Diabetic Retinopathy** from retinal fundus images with **medical-grade Grad-CAM++ visual explainability**.

Built with **Google Material Design 3** interface.

## Model Details

| Component | Detail |
|-----------|--------|
| **Backbone** | EfficientNetV2-S (tf_efficientnetv2_s, 21M params) |
| **Pooling** | Generalized Mean (GeM) |
| **Loss** | 0.5×CE(label_smooth=0.1, class-weighted) + 0.5×Focal(γ=2, class-weighted) |
| **Classes** | 5 grades (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR) |
| **Dataset** | APTOS 2019 Blindness Detection |
| **Explainability** | Multi-scale Grad-CAM++ (blocks[3] 32×32 + bn2 16×16) with SmoothGrad (N=2) |

## Features

- 🎨 **Material Design 3 UI** — Google premium dark theme with MD3 tokens, elevation, and motion
- 🔬 **Multi-Scale Grad-CAM++ Heatmaps** — Dual-layer fusion for fine vascular + semantic detail
- 📊 **Class Probabilities** — Full 5-class probability distribution with animated bars
- 🏥 **Clinical Preprocessing** — CLAHE enhancement + retinal disc cropping
- ⏱️ **Processing Time Metrics** — Real-time breakdown of preprocessing, inference, and postprocessing
- ⚡ **Optimized Inference** — 2-pass SmoothGrad for fast, accurate heatmaps

## Usage

1. Upload a retinal fundus image (PNG, JPG, JPEG)
2. Click "Analyze with Grad-CAM++"
3. View the predicted DR grade, confidence scores, processing time, and Grad-CAM++ heatmap

## API Endpoint

```
POST /predict
Content-Type: multipart/form-data
Body: file=<image_file>
```

**Response:**
```json
{
    "predicted_grade": 0,
    "grade_label": "No DR",
    "confidence": 0.95,
    "probabilities": [0.95, 0.02, 0.01, 0.01, 0.01],
    "gradcam_heatmap": "<base64_jpeg>",
    "original_image": "<base64_jpeg>",
    "timing": {
        "preprocessing_sec": 0.023,
        "inference_sec": 1.245,
        "postprocessing_sec": 0.089,
        "total_sec": 1.358,
        "device": "cpu",
        "smoothgrad_passes": 2
    }
}
```

## ⚠️ Disclaimer

**RESEARCH USE ONLY — NOT FOR CLINICAL DEPLOYMENT.**
Always consult a qualified ophthalmologist for diagnosis.
