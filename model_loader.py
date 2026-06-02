"""
model_loader.py  —  DR Grading  v19
────────────────────────────────────
Artifact discovery order (first path that contains both .pt files wins):
  1. ARTIFACTS_PATH env variable          export ARTIFACTS_PATH=/abs/path/to/dir
  2. ./artifacts/                         (next to app.py — standard deployment)
  3. ~/DR_data_v15/artifacts_v15/         (notebook v15 default)
  4. ~/DR_data/artifacts/                 (notebook v14 / generic)
  5. ~/DR_data_v18/artifacts/             (v18 variant)

Checkpoint keys expected (written by notebook Cell 22 / 12):
  final_model.pt        → model_state, backbone, img_size, temperature, val_metrics
  fundus_validator_v3.pt→ head_state_list, backbone_specs, feat_mean, feat_cov_inv,
                           mahal_threshold, img_size
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _best_device()

# ─────────────────────────────────────────────────────────────────────────────
#  ARTIFACT PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent

_CANDIDATE_DIRS = [
    # 1. Explicit env var
    Path(os.environ["ARTIFACTS_PATH"]) if "ARTIFACTS_PATH" in os.environ else None,
    # 2. Next to app.py (standard deployment / Docker)
    _APP_DIR / "artifacts",
    # 3. Notebook default paths
    Path.home() / "DR_data_v15" / "artifacts_v15",
    Path.home() / "DR_data" / "artifacts",
    Path.home() / "DR_data_v18" / "artifacts",
    Path.home() / "DR_data_v19" / "artifacts",
]

_NEEDED = {"final_model.pt", "fundus_validator_v3.pt"}

def _find_artifacts() -> Tuple[Optional[Path], list]:
    """Return (artifacts_dir or None, list of searched paths)."""
    searched = []
    for d in _CANDIDATE_DIRS:
        if d is None:
            continue
        searched.append(str(d))
        if d.exists() and all((d / f).exists() for f in _NEEDED):
            return d, searched
    return None, searched

ARTIFACTS, _SEARCHED_PATHS = _find_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

GRADE_MAP = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}
GRADE_COLORS  = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
GRADE_ADVICE  = {
    0: "No diabetic retinopathy detected. Recommended follow-up: 1–2 years.",
    1: "Mild NPDR. Ophthalmology follow-up in 12 months.",
    2: "Moderate NPDR. Ophthalmology review in 6 months.",
    3: "Severe NPDR. Urgent review within 1–2 months.",
    4: "Proliferative DR. Immediate ophthalmology referral required.",
}

# ─────────────────────────────────────────────────────────────────────────────
#  DR MODEL  (EfficientNetV2-S + GeM + DRHead)
# ─────────────────────────────────────────────────────────────────────────────

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


class DRHead(nn.Module):
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
#  FUNDUS VALIDATOR ENSEMBLE  (matches notebook FundusValidatorEnsemble)
# ─────────────────────────────────────────────────────────────────────────────

_FV_DEFAULT_SPECS = [
    ("mobilenetv3_large_100.ra_in1k", "avg", 0.25),
    ("convnext_tiny.fb_in22k",        "avg", 0.45),
    ("convnext_tiny.in12k_ft_in1k",   "avg", 0.30),
]
FV_SIZE = 224


class _SingleFundusValidator(nn.Module):
    """Frozen backbone + small MLP head — matches notebook _SingleFundusValidator."""

    def __init__(self, backbone_name: str, pool: str = "avg"):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool=pool
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        with torch.no_grad():
            fd = self.backbone(torch.zeros(1, 3, FV_SIZE, FV_SIZE)).shape[1]
        self.feat_dim = fd

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self


class FundusValidatorEnsemble(nn.Module):
    """Weighted ensemble — matches notebook FundusValidatorEnsemble."""

    def __init__(self, specs=None):
        super().__init__()
        specs = specs or _FV_DEFAULT_SPECS
        self.validators = nn.ModuleList(
            [_SingleFundusValidator(n, p) for n, p, _ in specs]
        )
        raw_w  = [w for _, _, w in specs]
        total  = sum(raw_w)
        self.weights = [w / total for w in raw_w]

    def extract_primary_features(self, x: torch.Tensor) -> torch.Tensor:
        """ConvNeXt-22k features (index 1) for Mahalanobis gating."""
        return self.validators[1].extract_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = None
        for val, w in zip(self.validators, self.weights):
            p   = F.softmax(val(x), dim=1)
            avg = p * w if avg is None else avg + p * w
        return torch.log(avg.clamp(min=1e-9))


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DR grading model…")
def load_dr_model() -> Tuple[Optional[DRModel], Optional[int], float, Dict[str, Any]]:
    """Returns (model, img_size, temperature, val_metrics). Model is None if not found."""
    if ARTIFACTS is None:
        return None, None, 1.0, {}

    ckpt_path = ARTIFACTS / "final_model.pt"
    if not ckpt_path.exists():
        return None, None, 1.0, {}

    try:
        ckpt        = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        backbone    = ckpt.get("backbone", "tf_efficientnetv2_s")
        num_classes = int(ckpt.get("num_classes", 5))
        model       = DRModel(backbone, num_classes).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        img_size    = int(ckpt.get("img_size", 512))
        temperature = float(ckpt.get("temperature", 1.0))
        val_metrics = ckpt.get("val_metrics", {})
        return model, img_size, temperature, val_metrics
    except Exception as e:
        st.error(f"❌ Failed to load final_model.pt: {e}")
        return None, None, 1.0, {}


@st.cache_resource(show_spinner="Loading fundus validator ensemble…")
def load_fv_ensemble():
    """Returns (fv_model, transform, mahal_data). fv_model is None if not found."""
    if ARTIFACTS is None:
        return None, None, None

    ckpt_path = ARTIFACTS / "fundus_validator_v3.pt"
    if not ckpt_path.exists():
        return None, None, None

    try:
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        specs = ckpt.get("backbone_specs", _FV_DEFAULT_SPECS)
        fv    = FundusValidatorEnsemble(specs).to(DEVICE)

        for val, head_state in zip(fv.validators, ckpt["head_state_list"]):
            val.head.load_state_dict(head_state)
        fv.eval()

        size = int(ckpt.get("img_size", FV_SIZE))
        tfm  = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        mahal_data: Dict[str, Any] = {}
        if "feat_mean" in ckpt:
            mahal_data["feat_mean"]       = torch.from_numpy(
                np.asarray(ckpt["feat_mean"], dtype=np.float32)).to(DEVICE)
            mahal_data["feat_cov_inv"]    = torch.from_numpy(
                np.asarray(ckpt["feat_cov_inv"], dtype=np.float32)).to(DEVICE)
            mahal_data["mahal_threshold"] = float(ckpt["mahal_threshold"])
        return fv, tfm, mahal_data
    except Exception as e:
        st.error(f"❌ Failed to load fundus_validator_v3.pt: {e}")
        return None, None, None
