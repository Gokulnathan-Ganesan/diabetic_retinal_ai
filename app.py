"""
app.py  —  DR Grading Streamlit App  v19
─────────────────────────────────────────
Run:
    streamlit run app.py

Quick setup (copy artifacts from notebook):
    cp ~/DR_data_v15/artifacts_v15/final_model.pt          ./artifacts/
    cp ~/DR_data_v15/artifacts_v15/fundus_validator_v3.pt  ./artifacts/

Or set env var to point directly at the notebook artifacts:
    export ARTIFACTS_PATH=~/DR_data_v15/artifacts_v15
    streamlit run app.py
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import altair as alt
import pandas as pd
import streamlit as st

from model_loader import (
    load_dr_model,
    load_fv_ensemble,
    DEVICE,
    ARTIFACTS,
    _SEARCHED_PATHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    GRADE_MAP,
    GRADE_COLORS,
    GRADE_ADVICE,
    FV_SIZE,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Grading — AI-Assisted Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP PAGE — shown when artifacts are missing
# ─────────────────────────────────────────────────────────────────────────────
def show_setup_page():
    st.title("🩺 DR Grading — Setup Required")
    st.error("### ⚠️ Model artifacts not found")

    st.markdown("The app searched for `final_model.pt` and `fundus_validator_v3.pt` in:")
    for p in _SEARCHED_PATHS:
        st.code(p)

    st.markdown("---")
    st.markdown("### Fix — choose one option:")

    tab1, tab2 = st.tabs(["Option A — Copy artifacts here", "Option B — Set env variable"])

    with tab1:
        st.markdown("**Copy the two model files into an `artifacts/` folder next to `app.py`:**")
        st.code("""\
# Run these in your terminal:
mkdir -p artifacts
cp ~/DR_data_v15/artifacts_v15/final_model.pt          artifacts/
cp ~/DR_data_v15/artifacts_v15/fundus_validator_v3.pt  artifacts/
streamlit run app.py
""", language="bash")

    with tab2:
        st.markdown("**Point directly at the notebook artifacts folder:**")
        st.code("""\
export ARTIFACTS_PATH=~/DR_data_v15/artifacts_v15
streamlit run app.py
""", language="bash")

    st.markdown("---")
    st.markdown(
        "After the notebook finishes training, run **Cell 22** "
        "(`step22_save`) to generate `final_model.pt`, then follow one of the options above."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe_lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def retinal_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def crop_retinal_disc(rgb: np.ndarray) -> np.ndarray:
    mask = retinal_mask(rgb)
    ys, xs = np.where(mask > 0)
    if len(ys) < 100:
        return rgb
    return rgb[int(ys.min()): int(ys.max()), int(xs.min()): int(xs.max())]


def preprocess(pil_img: Image.Image, size: int) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    try:
        rgb = apply_clahe_lab(crop_retinal_disc(rgb))
    except Exception:
        pass
    return cv2.resize(rgb, (size, size))


# ─────────────────────────────────────────────────────────────────────────────
#  FUNDUS GATING
# ─────────────────────────────────────────────────────────────────────────────

STRONG_H = {"red_dominance": 0.60, "disc_coverage": 0.35, "edge_density": 0.010}
WEAK_H   = {"red_dominance": 0.35, "disc_coverage": 0.15, "edge_density": 0.008}


def fundus_heuristics(rgb: np.ndarray) -> dict:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    bright  = (r > 15) | (g > 15) | (b > 15)
    nb      = int(bright.sum())
    rd = float(((r > g) & (r > b))[bright].mean()) if nb > 100 else float(((r > g) & (r > b)).mean())
    dc = float(retinal_mask(rgb).mean() / 255.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mag  = np.sqrt(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    )
    ed = float(mag[bright].mean() / 255.0) if nb > 100 else float(mag.mean() / 255.0)
    return {"red_dominance": rd, "disc_coverage": dc, "edge_density": ed}


def gate_fundus(pil_img: Image.Image, fv, tfm, mahal_data: dict):
    """Returns (is_fundus, ensemble_prob, heuristics, reason)."""
    rgb_raw = np.array(pil_img.convert("RGB"))
    try:
        rgb = apply_clahe_lab(crop_retinal_disc(rgb_raw))
    except Exception:
        rgb = rgb_raw

    h = fundus_heuristics(rgb)
    x = tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pf = float(F.softmax(fv(x), dim=1).cpu().numpy()[0][1])

    s_ok = all(h[k] >= v for k, v in STRONG_H.items())
    w_ok = all(h[k] >= v for k, v in WEAK_H.items())

    if s_ok:
        return True, pf, h, "Strong visual signatures"
    if pf >= 0.75:
        return True, pf, h, f"Ensemble confidence {pf * 100:.1f}%"
    if pf >= 0.50 and w_ok:
        return True, pf, h, "Ensemble + heuristics"
    if h["disc_coverage"] >= 0.25 and pf >= 0.35:
        return True, pf, h, f"Clear disc (cov={h['disc_coverage']:.2f})"

    if mahal_data and "feat_mean" in mahal_data:
        fm  = mahal_data["feat_mean"]
        fci = mahal_data["feat_cov_inv"]
        thr = mahal_data["mahal_threshold"]
        with torch.no_grad():
            feats = fv.extract_primary_features(x).squeeze(0)
        dist = float(torch.sqrt(
            torch.clamp(torch.einsum("i,ij,j->", feats - fm, fci, feats - fm), min=0.0)
        ).cpu())
        if dist <= thr:
            return True, pf, h, f"Feature similarity (d={dist:.1f})"

    return False, pf, h, "Not a fundus image"


# ─────────────────────────────────────────────────────────────────────────────
#  DR PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_dr(pil_img: Image.Image, dr_model, img_size: int, temperature: float):
    """Returns (grade, probs, preprocessed_rgb)."""
    rgb = preprocess(pil_img, img_size)
    tfm = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    x = tfm(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    logits, _ = dr_model(x)
    probs = F.softmax(logits / max(temperature, 0.05), dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs, rgb


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# Show setup page immediately if artifacts not found
if ARTIFACTS is None:
    show_setup_page()

# Load models (cached — only runs once per session)
dr_model, img_size, temperature, val_metrics = load_dr_model()
fv, fv_tfm, mahal_data                       = load_fv_ensemble()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(f"""
**DR Grading v19**
- Backbone: EfficientNetV2-S + GeM
- Domain adapt: DeepCORAL
- Calibration: Temperature scaling
- Validator: MobileNetV3 + ConvNeXt ensemble
""")

    if dr_model is None:
        st.error(
            "⚠️ `final_model.pt` not found\n\n"
            f"Artifacts dir:\n`{ARTIFACTS or 'not found'}`"
        )
    elif fv is None:
        st.error("⚠️ `fundus_validator_v3.pt` missing")
    else:
        st.success("✅ Both models loaded")
        if val_metrics:
            qwk = val_metrics.get("qwk", val_metrics.get("val_qwk", 0))
            acc = val_metrics.get("acc", val_metrics.get("accuracy", 0))
            st.metric("Val QWK",      f"{qwk:.4f}")
            if acc:
                st.metric("Val Accuracy", f"{acc * 100:.1f}%" if acc <= 1 else f"{acc:.1f}%")
        st.metric("Image size",    f"{img_size} px")
        st.metric("Temperature T", f"{temperature:.3f}")
        st.metric("Device",        DEVICE.upper())
        st.caption(f"Artifacts: `{ARTIFACTS}`")

    st.divider()
    st.error("⚠️ RESEARCH ONLY\nNot validated for clinical use.")

# ── Main title ────────────────────────────────────────────────────────────────
st.title("🩺 Diabetic Retinopathy Grading")
st.caption("AI-assisted grading  |  ⚠️ Research use only — not for clinical decisions")

if dr_model is None or fv is None:
    st.warning(
        "Models not loaded — see the sidebar for details.\n\n"
        "Run the notebook to completion, then set up the `artifacts/` folder."
    )
    show_setup_page()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a retinal fundus photograph",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is None:
    st.info("👆 Upload a fundus photo to get started.")
    st.stop()

pil_img      = Image.open(uploaded)
col1, col2   = st.columns(2)

with col1:
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

# ── Fundus gate ───────────────────────────────────────────────────────────────
with st.spinner("Validating fundus image…"):
    is_fundus, pf, heur, fv_reason = gate_fundus(pil_img, fv, fv_tfm, mahal_data or {})

if not is_fundus:
    st.error(f"🚫 **Image rejected** — {fv_reason}")
    st.info("Please upload a colour retinal fundus photograph.")
    with st.expander("Validator details"):
        st.write(f"Ensemble score: {pf * 100:.1f}%")
        for k, v in heur.items():
            thr  = WEAK_H[k]
            icon = "✅" if v >= thr else "❌"
            st.write(f"{icon} **{k}**: {v:.4f}  (need ≥ {thr})")
    st.stop()

st.success(f"✅ Fundus confirmed — {fv_reason}")

# ── DR grading ────────────────────────────────────────────────────────────────
with st.spinner("Grading DR severity…"):
    pred, probs, proc_rgb = predict_dr(pil_img, dr_model, img_size, temperature)

with col2:
    st.image(proc_rgb, caption="Preprocessed (CLAHE + retinal crop)", use_container_width=True)

# ── Result card ───────────────────────────────────────────────────────────────
grade_color = GRADE_COLORS[pred]
st.markdown(f"## Grade {pred}: {GRADE_MAP[pred]}")
st.markdown(
    f'<div style="padding:16px;border-radius:8px;background:{grade_color}22;'
    f'border-left:6px solid {grade_color};">'
    f"<strong>{GRADE_ADVICE[pred]}</strong></div>",
    unsafe_allow_html=True,
)
st.divider()

# ── Probability chart ─────────────────────────────────────────────────────────
st.subheader("Grade probabilities")
df_p = pd.DataFrame({
    "Grade":       [f"G{i}: {GRADE_MAP[i]}" for i in range(5)],
    "Probability": probs.tolist(),
    "Color":       GRADE_COLORS,
})
chart = (
    alt.Chart(df_p)
    .mark_bar()
    .encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Grade:N", sort=None),
        color=alt.Color("Color:N", scale=None, legend=None),
        tooltip=["Grade", alt.Tooltip("Probability", format=".3f")],
    )
    .properties(height=180)
)
st.altair_chart(chart, use_container_width=True)
st.caption(
    f"Temperature: {temperature:.3f}  |  "
    f"Confidence: {probs[pred] * 100:.1f}%  |  "
    f"Device: {DEVICE.upper()}"
)

with st.expander("Full probability table"):
    st.dataframe(
        pd.DataFrame({
            "Grade":           [GRADE_MAP[i] for i in range(5)],
            "Probability (%)": [f"{p * 100:.2f}" for p in probs],
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption("⚠️ Research use only — not for clinical diagnosis.")
