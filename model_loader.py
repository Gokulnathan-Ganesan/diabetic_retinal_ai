"""
model_loader.py
───────────────
All model definitions + cached loaders for the DR Grading Streamlit app.

Models expected inside  artifacts/  (same directory as app.py):
  artifacts/final_model.pt          ← DR grading checkpoint
  artifacts/fundus_validator_v3.pt  ← Fundus ensemble validator checkpoint
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st

# ── Runtime constants ─────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "mps" if (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
ARTIFACTS = Path(__file__).parent / "artifacts"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Grade metadata ────────────────────────────────────────────────────────────
GRADE_MAP = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}
GRADE_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
GRADE_ADVICE = {
    0: "No diabetic retinopathy detected. Recommended follow-up: 1–2 years.",
    1: "Mild NPDR. Ophthalmology follow-up in 12 months.",
    2: "Moderate NPDR. Ophthalmology review in 6 months.",
    3: "Severe NPDR. Urgent review within 1–2 months.",
    4: "Proliferative DR. Immediate ophthalmology referral required.",
}

# ─────────────────────────────────────────────────────────────────────────────
#  DR GRADING MODEL  (EfficientNetV2-S + GeM + dual head)
# ─────────────────────────────────────────────────────────────────────────────

class GeM(nn.Module):
    """Generalised Mean Pooling with learnable exponent."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class DRHead(nn.Module):
    """Classification head: Linear → BN → SiLU → Dropout × 2."""
    def __init__(self, in_features: int, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRModel(nn.Module):
    """EfficientNetV2-S + GeM + classification head."""
    def __init__(self, backbone: str = "tf_efficientnetv2_s", num_classes: int = 5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool=""
        )
        self.pool = GeM()
        self.head = DRHead(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.pool(self.backbone(x)).flatten(1)
        return self.head(feat), feat


# ─────────────────────────────────────────────────────────────────────────────
#  FUNDUS VALIDATOR ENSEMBLE  (MobileNetV3 + ConvNeXt-22k + ConvNeXt-12k)
# ─────────────────────────────────────────────────────────────────────────────

_FV_SPECS_DEFAULT = [
    ("mobilenetv3_large_100.ra_in1k", "avg", 0.25),
    ("convnext_tiny.fb_in22k",        "avg", 0.45),
    ("convnext_tiny.in12k_ft_in1k",   "avg", 0.30),
]
FV_SIZE = 224


class _SingleFV(nn.Module):
    """One backbone + lightweight classification head for fundus gating."""
    def __init__(self, name: str, pool: str = "avg"):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=False, num_classes=0, global_pool=pool
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        with torch.no_grad():
            fd = self.backbone(torch.zeros(1, 3, FV_SIZE, FV_SIZE)).shape[1]
        self.head = nn.Sequential(
            nn.LayerNorm(fd),
            nn.Linear(fd, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(x))


class FVEnsemble(nn.Module):
    """Weighted ensemble of fundus validators."""
    def __init__(self, specs=None):
        super().__init__()
        specs = specs or _FV_SPECS_DEFAULT
        self.validators = nn.ModuleList([_SingleFV(n, p) for n, p, _ in specs])
        raw_w = [w for _, _, w in specs]
        total = sum(raw_w)
        self.weights = [w / total for w in raw_w]

    def extract_primary_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ConvNeXt-22k features (index 1) for Mahalanobis gating."""
        return self.validators[1].extract_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = None
        for val, w in zip(self.validators, self.weights):
            p = F.softmax(val(x), dim=1)
            avg = p * w if avg is None else avg + p * w
        return torch.log(avg.clamp(min=1e-9))


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS  (called once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DR grading model…")
def load_dr_model():
    """
    Returns (model, img_size, temperature, val_metrics).
    All four are None / {} if the checkpoint is missing.
    """
    ckpt_path = ARTIFACTS / "final_model.pt"
    if not ckpt_path.exists():
        return None, None, None, {}

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    backbone = ckpt.get("backbone", "tf_efficientnetv2_s")
    model = DRModel(backbone).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_size    = int(ckpt.get("img_size", 512))
    temperature = float(ckpt.get("temperature", 1.0))
    val_metrics = ckpt.get("val_metrics", {})
    return model, img_size, temperature, val_metrics


@st.cache_resource(show_spinner="Loading fundus validator ensemble…")
def load_fv_ensemble():
    """
    Returns (fv_model, transform, mahal_data).
    All three are None if the checkpoint is missing.
    """
    ckpt_path = ARTIFACTS / "fundus_validator_v3.pt"
    if not ckpt_path.exists():
        return None, None, None

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    specs = ckpt.get("backbone_specs", _FV_SPECS_DEFAULT)
    fv    = FVEnsemble(specs).to(DEVICE)

    for val, head_state in zip(fv.validators, ckpt["head_state_list"]):
        val.head.load_state_dict(head_state)
    fv.eval()

    tfm = A.Compose([
        A.Resize(FV_SIZE, FV_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    mahal_data = {
        k: ckpt.get(k)
        for k in ("feat_mean", "feat_cov_inv", "mahal_threshold")
    }
    return fv, tfm, mahal_data
