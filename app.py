import sys, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import streamlit as st
st.set_page_config(
    page_title="DR Grading — AI-Assisted Diabetic Retinopathy Detection",
    page_icon="🧪",
    layout="centered",
)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
GRADE_MAP = {0:"No DR", 1:"Mild DR", 2:"Moderate DR", 3:"Severe DR", 4:"Proliferative DR"}
GRADE_COLORS = ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#8e44ad"]
GRADE_ADVICE = {
    0: "No diabetic retinopathy detected. Recommended follow-up: 1-2 years.",
    1: "Mild NPDR. Ophthalmology follow-up in 12 months.",
    2: "Moderate NPDR. Ophthalmology review in 6 months.",
    3: "Severe NPDR. Urgent review within 1-2 months.",
    4: "Proliferative DR. Immediate ophthalmology referral required.",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS = Path(__file__).parent / "artifacts"

# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_clahe_lab(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def retinal_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def crop_retinal_disc(rgb):
    mask = retinal_mask(rgb)
    ys, xs = np.where(mask > 0)
    if len(ys) < 100: return rgb
    return rgb[int(ys.min()):int(ys.max()), int(xs.min()):int(xs.max())]

def preprocess(pil_img, size=512):
    rgb = np.array(pil_img.convert("RGB"))
    try:    rgb = apply_clahe_lab(crop_retinal_disc(rgb))
    except: pass
    return cv2.resize(rgb, (size, size))

# ── DR Model ──────────────────────────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(p)); self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0/self.p)

class DRHead(nn.Module):
    def __init__(self, in_f, nc=5, d=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(d),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(d/2),
            nn.Linear(512, nc))
    def forward(self, x): return self.net(x)

class DRModel(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_s", nc=5):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")
        self.pool = GeM(); self.head = DRHead(self.backbone.num_features, nc)
    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        return self.head(feat), feat

# ── Fundus Validator Ensemble ─────────────────────────────────────────────────
_FV_SPECS = [
    ("mobilenetv3_large_100.ra_in1k", "avg", 0.25),
    ("convnext_tiny.fb_in22k",         "avg", 0.45),
    ("convnext_tiny.in12k_ft_in1k",    "avg", 0.30),
]
FV_SIZE = 224

class _SingleFV(nn.Module):
    def __init__(self, name, pool="avg"):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0, global_pool=pool)
        for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.eval()
        with torch.no_grad():
            fd = self.backbone(torch.zeros(1, 3, FV_SIZE, FV_SIZE)).shape[1]
        self.head = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,2))
    def extract_features(self, x):
        with torch.no_grad(): return self.backbone(x)
    def forward(self, x): return self.head(self.extract_features(x))

class FVEnsemble(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.validators = nn.ModuleList([_SingleFV(n, p) for n, p, _ in specs])
        ws = [w for _,_,w in specs]; t = sum(ws); self.weights = [w/t for w in ws]
    def extract_primary_features(self, x):
        return self.validators[1].extract_features(x)
    def forward(self, x):
        avg = None
        for val, w in zip(self.validators, self.weights):
            p = F.softmax(val(x), dim=1)
            avg = p*w if avg is None else avg+p*w
        return torch.log(avg.clamp(min=1e-9))

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading DR grading model...")
def load_dr_model():
    ckpt_path = ARTIFACTS / "final_model.pt"
    if not ckpt_path.exists(): return None, None, None, {}
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m = DRModel(ckpt.get("backbone","tf_efficientnetv2_s")).to(DEVICE)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m, int(ckpt.get("img_size",512)), float(ckpt.get("temperature",1.0)), ckpt.get("val_metrics",{})

@st.cache_resource(show_spinner="Loading fundus validator ensemble...")
def load_fv_ensemble():
    ckpt_path = ARTIFACTS / "fundus_validator_v3.pt"
    if not ckpt_path.exists(): return None, None, None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    specs = ckpt.get("backbone_specs", _FV_SPECS)
    fv = FVEnsemble(specs).to(DEVICE)
    for val, hs in zip(fv.validators, ckpt["head_state_list"]): val.head.load_state_dict(hs)
    fv.eval()
    tfm = A.Compose([A.Resize(FV_SIZE,FV_SIZE), A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), ToTensorV2()])
    md  = {k: ckpt.get(k) for k in ["feat_mean","feat_cov_inv","mahal_threshold"]}
    return fv, tfm, md

# ── Heuristics + gate ─────────────────────────────────────────────────────────
STRONG_H = {"red_dominance":0.60,"disc_coverage":0.35,"edge_density":0.010}
WEAK_H   = {"red_dominance":0.35,"disc_coverage":0.15,"edge_density":0.008}

def fundus_heuristics(rgb):
    r,g,b  = rgb[...,0].astype(np.float32), rgb[...,1].astype(np.float32), rgb[...,2].astype(np.float32)
    bright = (r>15)|(g>15)|(b>15); nb = int(bright.sum())
    rd = float(((r>g)&(r>b))[bright].mean()) if nb>100 else float(((r>g)&(r>b)).mean())
    dc = float(retinal_mask(rgb).mean()/255.0)
    mag = np.sqrt(cv2.Sobel(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_32F,1,0,ksize=3)**2
                + cv2.Sobel(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_32F,0,1,ksize=3)**2)
    ed = float(mag[bright].mean()/255.0) if nb>100 else float(mag.mean()/255.0)
    return {"red_dominance":rd,"disc_coverage":dc,"edge_density":ed}

def gate_fundus_st(pil_img, fv, tfm, md):
    rgb_raw = np.array(pil_img.convert("RGB"))
    try:    rgb = apply_clahe_lab(crop_retinal_disc(rgb_raw))
    except: rgb = rgb_raw
    h = fundus_heuristics(rgb)
    x = tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pf = float(F.softmax(fv(x),dim=1).cpu().numpy()[0][1])
    s_ok = all(h[k]>=v for k,v in STRONG_H.items())
    w_ok = all(h[k]>=v for k,v in WEAK_H.items())
    if s_ok: return True, pf, h, "Strong visual signatures"
    if pf>=0.75: return True, pf, h, f"Ensemble confidence {pf*100:.1f}%"
    if pf>=0.50 and w_ok: return True, pf, h, "Ensemble + heuristics"
    if h["disc_coverage"]>=0.25 and pf>=0.35: return True, pf, h, f"Clear disc (cov={h['disc_coverage']:.2f})"
    if md and md.get("feat_mean") is not None:
        fm  = torch.from_numpy(np.asarray(md["feat_mean"])).to(DEVICE)
        fci = torch.from_numpy(np.asarray(md["feat_cov_inv"])).to(DEVICE)
        thr = float(md["mahal_threshold"])
        with torch.no_grad():
            feats = fv.extract_primary_features(x).squeeze(0)
        d = float(torch.sqrt(torch.clamp(torch.einsum("i,ij,j->",feats-fm,fci,feats-fm),min=0.)).cpu())
        if d<=thr: return True, pf, h, f"Feature similarity (d={d:.1f})"
    return False, pf, h, "Not a fundus image"

@torch.no_grad()
def predict_dr(pil_img, dr_model, img_size, temperature):
    rgb = preprocess(pil_img, img_size)
    tfm = A.Compose([A.Resize(img_size,img_size), A.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD), ToTensorV2()])
    x = tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    logits, _ = dr_model(x)
    probs = F.softmax(logits / max(temperature, 0.05), dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs, rgb

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧪 Diabetic Retinopathy Grading")
st.caption("AI-assisted grading | Research use only — not for clinical decisions")

dr_model, img_size, temperature, val_metrics = load_dr_model()
fv_result = load_fv_ensemble()

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**DR Grading v13**
- Backbone: EfficientNetV2-S + GeM
- Domain adapt: DeepCORAL
- Calibration: Temperature scaling
- Validator: MobileNetV3 + ConvNeXt ensemble
""")
    if dr_model is None:
        st.error("⚠️ final_model.pt not found, Copy artifacts/ folder here.")
    elif fv_result[0] is None:
        st.error("⚠️ fundus_validator_v3.pt missing")
    else:
        st.success("✅ Models loaded")
        if val_metrics: st.metric("Val QWK", f"{val_metrics.get('qwk',0):.4f}")
        st.metric("Temperature T", f"{temperature:.3f}")
    st.divider()
    st.error("⚠️ RESEARCH ONLY, Not validated for clinical use.")

uploaded = st.file_uploader("Upload a retinal fundus photograph", type=["jpg","jpeg","png","bmp"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    if dr_model is None or fv_result[0] is None:
        st.error("Models not loaded — see sidebar."); st.stop()

    fv, fv_tfm, mahal_data = fv_result
    with st.spinner("Validating fundus image..."):
        is_fundus, pf, heur, fv_reason = gate_fundus_st(pil_img, fv, fv_tfm, mahal_data)

    if not is_fundus:
        st.error(f"🚫 **Image rejected** — {fv_reason}")
        st.info("Please upload a colour retinal fundus photograph.")
        with st.expander("Validator details"):
            st.write(f"Ensemble score: {pf*100:.1f}%")
            for k,v in heur.items():
                thr = WEAK_H[k]
                icon = "✅" if v>=thr else "❌"
                st.write(f"{icon} {k}: {v:.4f} (need >= {thr})")
        st.stop()

    st.success(f"✅ Fundus confirmed — {fv_reason}")

    with st.spinner("Grading DR severity..."):
        pred, probs, proc_rgb = predict_dr(pil_img, dr_model, img_size, temperature)

    with col2:
        st.image(proc_rgb, caption="Preprocessed (CLAHE + retinal crop)", use_container_width=True)

    grade_color = GRADE_COLORS[pred]
    st.markdown(f"## Grade {pred}: {GRADE_MAP[pred]}")
    st.markdown(
        f'<div style="padding:16px;border-radius:8px;background:{grade_color}22;' +
        f'border-left:6px solid {grade_color};">' +
        f'<strong>{GRADE_ADVICE[pred]}</strong></div>',
        unsafe_allow_html=True)
    st.divider()

    st.subheader("Grade probabilities")
    import altair as alt, pandas as pd
    df_p = pd.DataFrame({
        "Grade": [f"G{i}: {GRADE_MAP[i]}" for i in range(5)],
        "Probability": probs.tolist(),
        "Color": GRADE_COLORS,
    })
    chart = (alt.Chart(df_p).mark_bar()
        .encode(x=alt.X("Probability:Q", scale=alt.Scale(domain=[0,1])),
                y=alt.Y("Grade:N", sort=None),
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=["Grade", alt.Tooltip("Probability", format=".3f")])
        .properties(height=180))
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Temperature: {temperature:.3f} | Confidence: {probs[pred]*100:.1f}%")

    with st.expander("Full probability table"):
        st.dataframe(pd.DataFrame({
            "Grade": [GRADE_MAP[i] for i in range(5)],
            "Probability (%)": [f"{p*100:.2f}" for p in probs]}))

    st.divider()
    st.caption("⚠️ Research use only — not for clinical diagnosis.")
