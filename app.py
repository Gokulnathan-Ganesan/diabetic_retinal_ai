"""
🩺 Diabetic Retinopathy Grading — FastAPI Inference Server
EfficientNetV2-S + GeM Pooling | 5-Class (0–4) Classification
Multi-Scale Grad-CAM++ with Bounding Box Explainability
Google Material Design 3 Light Theme
"""

import os
import io
import cv2
import time
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 5
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRADE_MAP = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (PDR)"
}

# ── Device ───────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ── Model Architecture ──────────────────────────────────────────────────────────

class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class DRHead(nn.Module):
    """Classification head."""
    def __init__(self, in_features, num_classes=5, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class DRModel(nn.Module):
    """EfficientNetV2-S backbone + GeM pooling + DRHead."""
    def __init__(self, backbone="tf_efficientnetv2_s", num_classes=5, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )
        in_feat = self.backbone.num_features
        self.pool = GeM(p=3.0)
        self.head = DRHead(in_feat, num_classes)
        self._backbone_name = backbone

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════════
#  MULTI-SCALE GRAD-CAM++ (SINGLE PASS — FAST + ACCURATE)
# ══════════════════════════════════════════════════════════════════════════════════

class GradCAMPP_MultiScale:
    """
    Multi-scale Grad-CAM++ with single forward+backward pass.
    Fuses shallow (blocks[3], 32×32) + deep (bn2, 16×16) layers for
    both fine vascular detail and high-level semantic pathology detection.
    Includes region detection for bounding box annotations.
    """

    def __init__(self, model: DRModel):
        self.model = model
        self._hooks = []

        self.act_deep = None
        self.grad_deep = None
        self.act_shallow = None
        self.grad_shallow = None

        deep_layer = getattr(model.backbone, 'bn2')
        self._hooks.append(deep_layer.register_forward_hook(self._fwd_deep))
        self._hooks.append(deep_layer.register_full_backward_hook(self._bwd_deep))

        shallow_layer = model.backbone.blocks[3]
        self._hooks.append(shallow_layer.register_forward_hook(self._fwd_shallow))
        self._hooks.append(shallow_layer.register_full_backward_hook(self._bwd_shallow))

        print("  ✓ Grad-CAM++ hooks: backbone.bn2 (deep) + backbone.blocks[3] (shallow)")

    def _fwd_deep(self, m, i, o): self.act_deep = o.detach().clone()
    def _bwd_deep(self, m, gi, go): self.grad_deep = go[0].detach().clone()
    def _fwd_shallow(self, m, i, o): self.act_shallow = o.detach().clone()
    def _bwd_shallow(self, m, gi, go): self.grad_shallow = go[0].detach().clone()

    @staticmethod
    def _campp_weights(activations, gradients):
        """Grad-CAM++ exact paper formulation."""
        g2 = gradients.pow(2)
        g3 = gradients.pow(3)
        s = activations.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * g2 + s * g3 + 1e-8
        alpha = torch.where(gradients.abs() > 1e-10, g2 / denom, torch.zeros_like(g2))
        return (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

    @torch.enable_grad()
    def generate(self, input_tensor, target_class=None, shallow_weight=0.35):
        """Single-pass multi-scale Grad-CAM++ → heatmap + prediction + probs + regions."""
        self.model.eval()
        self.model.zero_grad()
        inp = input_tensor.clone().requires_grad_(True)

        logits = self.model(inp)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred = int(np.argmax(probs))
        if target_class is None:
            target_class = pred

        logits[0, target_class].backward(retain_graph=False)

        # Deep CAM (bn2: 1280ch × 16×16)
        w_d = self._campp_weights(self.act_deep, self.grad_deep)
        cam_d = F.relu((w_d * self.act_deep).sum(dim=1, keepdim=True))

        # Shallow CAM (blocks[3]: 128ch × 32×32)
        w_s = self._campp_weights(self.act_shallow, self.grad_shallow)
        cam_s = F.relu((w_s * self.act_shallow).sum(dim=1, keepdim=True))

        sz = (input_tensor.shape[2], input_tensor.shape[3])

        # Upsample both
        up_d = F.interpolate(cam_d, size=sz, mode='bicubic', align_corners=False
                             ).clamp(min=0).squeeze().cpu().numpy()
        up_s = F.interpolate(cam_s, size=sz, mode='bicubic', align_corners=False
                             ).clamp(min=0).squeeze().cpu().numpy()

        # Normalize each
        for arr in [up_d, up_s]:
            mn, mx = arr.min(), arr.max()
            if mx - mn > 1e-8:
                arr[:] = (arr - mn) / (mx - mn)

        # Fuse
        fused = (1.0 - shallow_weight) * up_d + shallow_weight * up_s

        # Post-process
        heatmap = self._postprocess(fused)

        return heatmap, pred, probs

    @staticmethod
    def _postprocess(cam):
        cam = cam.astype(np.float32)
        p_lo, p_hi = np.percentile(cam, 2), np.percentile(cam, 99.5)
        if p_hi - p_lo > 1e-8:
            cam = np.clip(cam, p_lo, p_hi)
            cam = (cam - p_lo) / (p_hi - p_lo)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=12, sigmaY=12)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=6, sigmaY=6)
        mx = cam.max()
        if mx > 1e-8:
            cam = cam / mx
        cam = np.power(cam, 0.55)
        mn, mx = cam.min(), cam.max()
        if mx - mn > 1e-8:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)
        return cam



    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Heatmap + Bounding Box Rendering ─────────────────────────────────────────────

def create_retinal_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask_f = mask.astype(np.float32) / 255.0
    return cv2.GaussianBlur(mask_f, (31, 31), sigmaX=8)


def create_heatmap_overlay(original_rgb, heatmap, alpha=0.45):
    """Heatmap overlay with retinal masking and intensity-adaptive blending."""
    h, w = original_rgb.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    retinal_mask = create_retinal_mask(original_rgb)
    hm = heatmap * retinal_mask
    mp = hm[retinal_mask > 0.3]
    if len(mp) > 0 and mp.max() > 1e-8:
        hm = hm / mp.max()
    hm = np.clip(hm, 0, 1)

    hm_u8 = (hm * 255).astype(np.uint8)
    try:
        colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    except AttributeError:
        colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    a = np.power(hm, 0.7) * alpha
    a = np.clip(a, 0.0, alpha)
    a3 = np.stack([a] * 3, axis=-1)
    overlay = colored_rgb.astype(np.float32) * a3 + original_rgb.astype(np.float32) * (1.0 - a3)
    return np.clip(overlay, 0, 255).astype(np.uint8)





def encode_image_to_base64(image_rgb, quality=90):
    img_pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Preprocessing ────────────────────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _clahe_lab(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def _retinal_mask(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(g, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

def _crop_retinal(rgb, pad=10):
    mask = _retinal_mask(rgb)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return rgb
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x, y = max(0, x - pad), max(0, y - pad)
    return rgb[y:min(rgb.shape[0], y + h + 2*pad), x:min(rgb.shape[1], x + w + 2*pad)]

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Cannot decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _clahe_lab(img)
    img = _crop_retinal(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    display = img.copy()
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    return tensor, display


# ── Load Model ───────────────────────────────────────────────────────────────────

def load_model():
    model = DRModel(backbone="tf_efficientnetv2_s", num_classes=NUM_CLASSES, pretrained=False)
    p = Path("/app/best_model.pt")
    if not p.exists(): raise FileNotFoundError(f"Model not found at {p}")
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded on {DEVICE}")
    return model


# ── FastAPI ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="🩺 DR AI", version="6.0.0")
model = None
gradcam = None

@app.on_event("startup")
async def startup():
    global model, gradcam
    model = load_model()
    gradcam = GradCAMPP_MultiScale(model)
    print("✅ Multi-scale Grad-CAM++ ready")


# ══════════════════════════════════════════════════════════════════════════════════
#  GOOGLE MATERIAL DESIGN 3 — LIGHT THEME + EXPLAINABILITY UI
# ══════════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DR AI — Diabetic Retinopathy Grading</title>
        <meta name="description" content="AI-powered DR severity classification with Grad-CAM++ explainability.">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #1a73e8;
                --on-primary: #fff;
                --primary-container: #d3e3fd;
                --on-primary-container: #062e6f;
                --secondary: #5f6368;
                --surface: #ffffff;
                --surface-dim: #f8f9fa;
                --surface-container: #f1f3f4;
                --surface-high: #e8eaed;
                --surface-highest: #dadce0;
                --on-surface: #202124;
                --on-surface-var: #5f6368;
                --outline: #dadce0;
                --outline-var: #e8eaed;
                --tertiary-container: #f3e5f5;
                --on-tertiary-container: #4a0072;
                --warning: #e8710a;
                --warning-container: #fef3e0;
                --on-warning-container: #7c4a03;
                --g0: #34a853; --g1: #fbbc04; --g2: #fa7b17; --g3: #ea4335; --g4: #9334e6;
                --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 28px; --radius-full: 9999px;
                --shadow-1: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
                --shadow-2: 0 1px 2px 0 rgba(60,64,67,.3), 0 2px 6px 2px rgba(60,64,67,.15);
                --ease: cubic-bezier(0.4,0,0.2,1);
            }
            *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
            body{font-family:'Google Sans','Roboto',system-ui,sans-serif;background:var(--surface-dim);color:var(--on-surface);min-height:100vh;-webkit-font-smoothing:antialiased}
            .material-symbols-outlined{font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;vertical-align:middle}
            .material-symbols-outlined.filled{font-variation-settings:'FILL' 1,'wght' 400,'GRAD' 0,'opsz' 24}

            .page{max-width:880px;margin:0 auto;padding:24px 16px 48px}

            /* Header */
            .header{text-align:center;padding:28px 16px 20px}
            .header__chip{display:inline-flex;align-items:center;gap:6px;height:32px;padding:0 16px;background:var(--primary-container);border-radius:var(--radius-full);margin-bottom:14px}
            .header__chip-dot{width:8px;height:8px;background:var(--primary);border-radius:50%;animation:pulse 2s ease-in-out infinite}
            @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.75)}}
            .header__chip-text{font-size:12px;font-weight:500;color:var(--on-primary-container);letter-spacing:.3px}
            .header__title{font-size:32px;font-weight:700;color:var(--on-surface);margin-bottom:4px;letter-spacing:-.5px}
            .header__sub{font-size:14px;color:var(--on-surface-var)}

            /* Card */
            .card{background:var(--surface);border-radius:var(--radius-lg);padding:24px;margin-bottom:16px;box-shadow:var(--shadow-1);transition:box-shadow .2s var(--ease)}
            .card:hover{box-shadow:var(--shadow-2)}

            /* Upload */
            .upload{border:2px dashed var(--outline);border-radius:var(--radius-lg);padding:40px 24px;text-align:center;cursor:pointer;transition:all .25s var(--ease)}
            .upload:hover{border-color:var(--primary);background:rgba(26,115,232,.04)}
            .upload.dragover{border-color:var(--primary);background:rgba(26,115,232,.08)}
            .upload.has-file{border-color:var(--g0);border-style:solid;padding:16px}
            .upload__icon{width:56px;height:56px;border-radius:var(--radius-xl);background:var(--primary-container);display:flex;align-items:center;justify-content:center;margin:0 auto 12px}
            .upload__icon .material-symbols-outlined{font-size:28px;color:var(--primary)}
            .upload__title{font-size:16px;font-weight:500;color:var(--on-surface);margin-bottom:4px}
            .upload__hint{font-size:12px;color:var(--on-surface-var)}
            input[type="file"]{display:none}

            .preview{display:none;margin-top:16px;background:var(--surface-container);border-radius:var(--radius-md);padding:8px;text-align:center}
            .preview img{width:100%;max-width:420px;height:auto;border-radius:var(--radius-sm);display:block;margin:0 auto}

            /* Google Blue Button */
            .btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;min-height:52px;padding:0 24px;margin-top:20px;background:var(--primary);color:var(--on-primary);border:none;border-radius:var(--radius-xl);font-family:inherit;font-size:15px;font-weight:500;cursor:pointer;position:relative;overflow:hidden;transition:all .2s var(--ease);box-shadow:var(--shadow-1)}
            .btn:hover:not(:disabled){box-shadow:var(--shadow-2);background:#1765cc}
            .btn:active:not(:disabled){background:#185abc}
            .btn:disabled{background:var(--surface-high);color:rgba(32,33,36,.38);box-shadow:none;cursor:not-allowed}
            .btn .material-symbols-outlined{font-size:20px}
            .ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,.3);transform:scale(0);animation:rip .5s ease-out forwards;pointer-events:none}
            @keyframes rip{to{transform:scale(4);opacity:0}}

            /* Loading */
            .loading{display:none;text-align:center;padding:48px 24px}
            .loading.active{display:block}
            .spinner{width:48px;height:48px;margin:0 auto 16px;animation:rot 1.4s linear infinite}
            .spinner circle{fill:none;stroke:var(--primary);stroke-width:4;stroke-dasharray:113;stroke-dashoffset:56;stroke-linecap:round;animation:dash 1.4s ease-in-out infinite}
            @keyframes rot{to{transform:rotate(360deg)}}
            @keyframes dash{0%{stroke-dashoffset:113}50%{stroke-dashoffset:28}100%{stroke-dashoffset:113}}
            .loading__title{font-size:16px;font-weight:500;color:var(--on-surface);margin-bottom:4px}
            .loading__sub{font-size:13px;color:var(--on-surface-var)}

            /* Results */
            .results{display:none}
            .results.active{display:block;animation:fadeUp .4s cubic-bezier(0,0,.2,1)}
            @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

            /* Timing */
            .timing{display:flex;gap:1px;background:var(--outline);border-radius:var(--radius-lg);overflow:hidden;margin-bottom:16px}
            .timing__item{flex:1;background:var(--surface);padding:14px 8px;text-align:center;display:flex;flex-direction:column;align-items:center;gap:4px}
            .timing__item .material-symbols-outlined{font-size:20px;color:var(--primary)}
            .timing__val{font-size:18px;font-weight:700;color:var(--on-surface);font-variant-numeric:tabular-nums}
            .timing__label{font-size:10px;font-weight:500;letter-spacing:.5px;color:var(--on-surface-var);text-transform:uppercase}

            /* Grade */
            .result-header{display:flex;align-items:center;gap:16px;margin-bottom:20px}
            .grade-badge{width:60px;height:60px;border-radius:var(--radius-xl);display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:700;flex-shrink:0}
            .result-info{flex:1}
            .result-label{font-size:20px;font-weight:500;margin-bottom:2px}
            .result-conf{font-size:14px;color:var(--on-surface-var)}

            /* Probs */
            .prob-section{margin-top:16px}
            .prob-hdr{display:flex;align-items:center;gap:8px;margin-bottom:14px}
            .prob-hdr .material-symbols-outlined{font-size:18px;color:var(--on-surface-var)}
            .prob-hdr__title{font-size:11px;font-weight:500;letter-spacing:.5px;color:var(--on-surface-var);text-transform:uppercase}
            .prob-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
            .prob-lbl{width:140px;font-size:13px;color:var(--on-surface-var);flex-shrink:0}
            .prob-track{flex:1;height:8px;background:var(--surface-container);border-radius:var(--radius-full);overflow:hidden}
            .prob-fill{height:100%;border-radius:var(--radius-full);transition:width .5s cubic-bezier(0,0,.2,1);min-width:3px}
            .prob-val{width:48px;text-align:right;font-size:13px;font-weight:500;font-variant-numeric:tabular-nums;flex-shrink:0}

            /* ═══ Explainability Section ═══ */
            .xai-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
            .xai-header__left{display:flex;align-items:center;gap:12px}
            .xai-icon{width:40px;height:40px;border-radius:var(--radius-md);background:var(--tertiary-container);display:flex;align-items:center;justify-content:center}
            .xai-icon .material-symbols-outlined{font-size:22px;color:var(--on-tertiary-container);font-variation-settings:'FILL' 1}
            .xai-title{font-size:16px;font-weight:500;color:var(--on-surface)}
            .xai-sub{font-size:12px;color:var(--on-surface-var)}
            .chip{display:inline-flex;align-items:center;height:28px;padding:0 12px;border-radius:var(--radius-full);border:1px solid var(--outline);font-size:12px;font-weight:500;color:var(--on-surface-var)}

            /* Image Grid — proper sizing */
            .img-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
            .img-panel{position:relative;border-radius:var(--radius-md);overflow:hidden;border:1px solid var(--outline);background:var(--surface-container)}
            .img-panel img{width:100%;height:auto;display:block}
            .img-panel__label{position:absolute;bottom:0;left:0;right:0;padding:20px 12px 8px;background:linear-gradient(to top,rgba(0,0,0,.55),transparent);font-size:11px;font-weight:500;letter-spacing:.5px;color:#fff;text-transform:uppercase}

            .legend{display:flex;align-items:center;justify-content:center;gap:12px;padding:8px 0}
            .legend-bar{width:180px;height:8px;border-radius:var(--radius-full);background:linear-gradient(90deg,#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04,#7a0403)}
            .legend-text{font-size:11px;color:var(--on-surface-var);font-weight:500}



            .method-info{margin-top:12px;padding:12px 16px;background:var(--surface-dim);border-radius:var(--radius-md);font-size:12px;color:var(--on-surface-var);line-height:1.8}
            .method-info strong{color:var(--on-surface);font-weight:500}

            /* Disclaimer — Orange */
            .disclaimer{display:flex;align-items:flex-start;gap:12px;margin-top:20px;padding:16px 20px;background:var(--warning-container);border-radius:var(--radius-md);border-left:4px solid var(--warning)}
            .disclaimer .material-symbols-outlined{font-size:20px;color:var(--warning);flex-shrink:0;margin-top:1px;font-variation-settings:'FILL' 1}
            .disclaimer__text{font-size:13px;line-height:20px;color:var(--on-warning-container)}
            .disclaimer__text strong{font-weight:700;color:var(--warning)}

            @media(max-width:640px){
                .page{padding:12px 8px 32px}
                .header__title{font-size:24px}
                .card{padding:16px}
                .img-grid{grid-template-columns:1fr}
                .upload{padding:32px 16px}
                .timing{flex-direction:column}
                .timing__item{flex-direction:row;justify-content:center;gap:10px;padding:10px}
                .result-header{flex-direction:column;text-align:center}
                .attn-stats{grid-template-columns:1fr}
            }
        </style>
    </head>
    <body>
    <div class="page">
        <header class="header">
            <div class="header__chip">
                <span class="header__chip-dot"></span>
                <span class="header__chip-text">EfficientNetV2-S · Multi-Scale Grad-CAM++</span>
            </div>
            <h1 class="header__title">🩺 DR AI</h1>
            <p class="header__sub">Diabetic Retinopathy Severity Classification with Visual Explainability</p>
        </header>

        <div class="card" id="uploadCard">
            <div class="upload" id="uploadZone" onclick="document.getElementById('fileInput').click()">
                <div class="upload__icon"><span class="material-symbols-outlined">add_photo_alternate</span></div>
                <p class="upload__title">Upload retinal fundus image</p>
                <p class="upload__hint">Click to browse or drag & drop · PNG, JPG, JPEG</p>
            </div>
            <input type="file" id="fileInput" accept="image/*">
            <div class="preview" id="previewBox">
                <img id="previewImage" src="" alt="Uploaded retinal image">
            </div>
            <button class="btn" id="analyzeBtn" disabled onclick="doAnalyze()">
                <span class="material-symbols-outlined">biotech</span>
                Analyze with Grad-CAM++
            </button>
        </div>

        <div class="loading" id="loadingSection">
            <svg class="spinner" viewBox="0 0 48 48"><circle cx="24" cy="24" r="18"/></svg>
            <p class="loading__title">Analyzing retinal image…</p>
            <p class="loading__sub">Running multi-scale Grad-CAM++ inference</p>
        </div>

        <div class="results" id="resultsWrapper">
            <!-- Timing -->
            <div class="timing" id="timingRow">
                <div class="timing__item">
                    <span class="material-symbols-outlined filled">image</span>
                    <span class="timing__val" id="tPre">—</span>
                    <span class="timing__label">Preprocess</span>
                </div>
                <div class="timing__item">
                    <span class="material-symbols-outlined filled">psychology</span>
                    <span class="timing__val" id="tInfer">—</span>
                    <span class="timing__label">Inference</span>
                </div>
                <div class="timing__item">
                    <span class="material-symbols-outlined filled">palette</span>
                    <span class="timing__val" id="tPost">—</span>
                    <span class="timing__label">Postprocess</span>
                </div>
                <div class="timing__item">
                    <span class="material-symbols-outlined filled">timer</span>
                    <span class="timing__val" id="tTotal">—</span>
                    <span class="timing__label">Total</span>
                </div>
            </div>

            <!-- Grade Result -->
            <div class="card">
                <div class="result-header">
                    <div class="grade-badge" id="gradeBadge"></div>
                    <div class="result-info">
                        <div class="result-label" id="resultLabel"></div>
                        <div class="result-conf" id="resultConf"></div>
                    </div>
                </div>
                <div class="prob-section">
                    <div class="prob-hdr">
                        <span class="material-symbols-outlined">bar_chart</span>
                        <span class="prob-hdr__title">Class Probabilities</span>
                    </div>
                    <div id="probBars"></div>
                </div>
            </div>

            <!-- Explainability Card -->
            <div class="card">
                <div class="xai-header">
                    <div class="xai-header__left">
                        <div class="xai-icon"><span class="material-symbols-outlined">local_fire_department</span></div>
                        <div>
                            <div class="xai-title">Grad-CAM++ Explainability</div>
                            <div class="xai-sub">Multi-scale visual attention map</div>
                        </div>
                    </div>
                    <span class="chip">XAI</span>
                </div>

                <!-- Images: Original | Heatmap -->
                <div class="img-grid" style="grid-template-columns:1fr 1fr">
                    <div class="img-panel">
                        <img id="origImg" src="" alt="Original">
                        <div class="img-panel__label">Original</div>
                    </div>
                    <div class="img-panel">
                        <img id="heatImg" src="" alt="Heatmap">
                        <div class="img-panel__label">Heatmap</div>
                    </div>
                </div>
                <div class="legend">
                    <span class="legend-text">Low</span>
                    <div class="legend-bar"></div>
                    <span class="legend-text">High</span>
                </div>

                <div class="method-info">
                    <strong>Method:</strong> Grad-CAM++ (Chattopadhay et al., 2018) &middot;
                    <strong>Layers:</strong> Multi-scale (blocks[3] 32&times;32 + bn2 16&times;16) &middot;
                    <strong>Colormap:</strong> TURBO
                </div>
            </div>
        </div>

        <div class="disclaimer">
            <span class="material-symbols-outlined">warning</span>
            <p class="disclaimer__text">
                <strong>Research Use Only</strong> — This tool is not intended for clinical deployment.
                Always consult a qualified ophthalmologist for diagnosis and treatment decisions.
            </p>
        </div>
    </div>

    <script>
        const uploadZone=document.getElementById('uploadZone'),fileInput=document.getElementById('fileInput'),
              previewBox=document.getElementById('previewBox'),previewImage=document.getElementById('previewImage'),
              analyzeBtn=document.getElementById('analyzeBtn'),loadingSection=document.getElementById('loadingSection'),
              resultsWrapper=document.getElementById('resultsWrapper');
        let selectedFile=null;
        const gc=['#34a853','#fbbc04','#fa7b17','#ea4335','#9334e6'],
              ge=['✅','⚠️','🔶','🔴','💜'],
              gn=['No DR','Mild DR','Moderate DR','Severe DR','Proliferative DR'];

        fileInput.addEventListener('change',e=>{
            if(e.target.files.length>0){
                selectedFile=e.target.files[0];
                const r=new FileReader();
                r.onload=ev=>{previewImage.src=ev.target.result;previewBox.style.display='block';uploadZone.classList.add('has-file');analyzeBtn.disabled=false;resultsWrapper.classList.remove('active')};
                r.readAsDataURL(selectedFile);
            }
        });
        ['dragover','dragenter'].forEach(ev=>uploadZone.addEventListener(ev,e=>{e.preventDefault();e.stopPropagation();uploadZone.classList.add('dragover')}));
        ['dragleave','drop'].forEach(ev=>uploadZone.addEventListener(ev,e=>{e.preventDefault();e.stopPropagation();uploadZone.classList.remove('dragover')}));
        uploadZone.addEventListener('drop',e=>{if(e.dataTransfer.files.length>0){fileInput.files=e.dataTransfer.files;fileInput.dispatchEvent(new Event('change'))}});
        analyzeBtn.addEventListener('click',function(e){const r=document.createElement('span');r.classList.add('ripple');const rect=this.getBoundingClientRect(),sz=Math.max(rect.width,rect.height);r.style.width=r.style.height=sz+'px';r.style.left=(e.clientX-rect.left-sz/2)+'px';r.style.top=(e.clientY-rect.top-sz/2)+'px';this.appendChild(r);setTimeout(()=>r.remove(),500)});

        function fmt(s){return s<1?(s*1000).toFixed(0)+'ms':s.toFixed(2)+'s'}

        async function doAnalyze(){
            if(!selectedFile)return;
            analyzeBtn.disabled=true;loadingSection.classList.add('active');resultsWrapper.classList.remove('active');
            const fd=new FormData();fd.append('file',selectedFile);
            try{
                const res=await fetch('/predict',{method:'POST',body:fd});
                const d=await res.json();
                if(d.error||d.detail){alert('Error: '+(d.error||d.detail));return}

                // Timing
                if(d.timing){
                    document.getElementById('tPre').textContent=fmt(d.timing.preprocessing_sec);
                    document.getElementById('tInfer').textContent=fmt(d.timing.inference_sec);
                    document.getElementById('tPost').textContent=fmt(d.timing.postprocessing_sec);
                    document.getElementById('tTotal').textContent=fmt(d.timing.total_sec);
                }

                // Grade
                const c=gc[d.predicted_grade],badge=document.getElementById('gradeBadge');
                badge.textContent=d.predicted_grade;badge.style.background=c+'18';badge.style.color=c;badge.style.border='2px solid '+c+'40';
                document.getElementById('resultLabel').textContent=ge[d.predicted_grade]+'  '+d.grade_label;
                document.getElementById('resultLabel').style.color=c;
                document.getElementById('resultConf').textContent='Confidence: '+(d.confidence*100).toFixed(1)+'%';

                // Probs
                const pb=document.getElementById('probBars');pb.innerHTML='';
                d.probabilities.forEach((p,i)=>{
                    const pct=(p*100).toFixed(1);
                    pb.innerHTML+=`<div class="prob-row"><span class="prob-lbl">Grade ${i}: ${gn[i]}</span><div class="prob-track"><div class="prob-fill" style="width:${pct}%;background:${gc[i]}"></div></div><span class="prob-val" style="color:${gc[i]}">${pct}%</span></div>`;
                });

                // Images
                if(d.original_image) document.getElementById('origImg').src='data:image/jpeg;base64,'+d.original_image;
                if(d.gradcam_heatmap) document.getElementById('heatImg').src='data:image/jpeg;base64,'+d.gradcam_heatmap;

                resultsWrapper.classList.add('active');
                resultsWrapper.scrollIntoView({behavior:'smooth',block:'start'});
            }catch(err){alert('Error: '+err.message)}
            finally{loadingSection.classList.remove('active');analyzeBtn.disabled=false}
        }
    </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """DR grade + multi-scale Grad-CAM++ + timing."""
    global model, gradcam
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    total_start = time.perf_counter()

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Read error: {str(e)}")

    # Preprocess
    t0 = time.perf_counter()
    try:
        tensor, display_img = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocess error: {str(e)}")
    preprocess_time = time.perf_counter() - t0

    # Inference (single-pass multi-scale Grad-CAM++)
    t1 = time.perf_counter()
    try:
        tensor = tensor.to(DEVICE)
        cam, pred_grade, probs = gradcam.generate(tensor)
        confidence = float(probs[pred_grade])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    inference_time = time.perf_counter() - t1

    # Postprocess: overlays + encoding
    t2 = time.perf_counter()
    try:
        overlay = create_heatmap_overlay(display_img, cam, alpha=0.45)

        original_b64 = encode_image_to_base64(display_img)
        heatmap_b64 = encode_image_to_base64(overlay)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postprocess error: {str(e)}")
    postprocess_time = time.perf_counter() - t2

    total_time = time.perf_counter() - total_start

    return JSONResponse(content={
        "predicted_grade": pred_grade,
        "grade_label": GRADE_MAP[pred_grade],
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in probs],
        "grade_descriptions": GRADE_MAP,
        "gradcam_heatmap": heatmap_b64,
        "original_image": original_b64,
        "timing": {
            "preprocessing_sec": round(preprocess_time, 4),
            "inference_sec": round(inference_time, 4),
            "postprocessing_sec": round(postprocess_time, 4),
            "total_sec": round(total_time, 4),
            "device": DEVICE,
        },
    })


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_ready": gradcam is not None,
        "device": DEVICE,
    }
