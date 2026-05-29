"""
model_loader.py  —  Drop this file next to app.py in your GitHub repo.

It auto-downloads final_model.pt (and optional companion files) from
Hugging Face Hub at startup, so Streamlit Community Cloud never needs
the artifacts/ folder on disk.

SETUP (one-time, ~5 min):
  1. Create a free account at https://huggingface.co
  2. Create a NEW MODEL REPO  (e.g. "gokulnathan/dr-grading-v13")
     → New → Model → set to Private (recommended for medical models)
  3. Upload your artifacts from your Mac:
       pip install huggingface_hub
       python -c "
         from huggingface_hub import HfApi
         api = HfApi()
         api.upload_file(
             path_or_fileobj='/Users/gokulnathan/DR_data/artifacts/final_model.pt',
             path_in_repo='final_model.pt',
             repo_id='YOUR_HF_USERNAME/dr-grading-v13',
             repo_type='model',
         )
         # Repeat for each artifact file you need:
         # best_model.pt, fundus_validator_ensemble.pt,
         # _resume_state.json, calibration.pt, etc.
       "
  4. If repo is Private, create a READ token at
     https://huggingface.co/settings/tokens
     and add it to Streamlit Cloud secrets:
       Dashboard → your app → Settings → Secrets → paste:
         HF_TOKEN = "hf_xxxxxxxxxxxx"
  5. In your app.py, replace the old artifact-loading block with:
       from model_loader import ensure_artifacts
       ARTIFACT_DIR = ensure_artifacts()
"""

import os
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this to your actual HuggingFace repo: "username/repo-name"
HF_REPO_ID = "YOUR_HF_USERNAME/dr-grading-v13"

# Files to download from HF repo → local path (relative to ARTIFACT_DIR)
# Add / remove entries to match what your app.py actually loads.
ARTIFACT_FILES = [
    "final_model.pt",
    "best_model.pt",
    "fundus_validator_ensemble.pt",
    "calibration.pt",
    "_resume_state.json",
]

# Local cache directory (writable on Streamlit Cloud)
_LOCAL_ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/tmp/dr_artifacts"))


def ensure_artifacts(
    repo_id: str = HF_REPO_ID,
    files: list[str] = ARTIFACT_FILES,
    local_dir: Path = _LOCAL_ARTIFACT_DIR,
    force: bool = False,
) -> Path:
    """
    Download all required artifact files from HuggingFace Hub if not already
    cached locally.  Returns the local artifact directory Path.

    Call this ONCE at the top of app.py:
        from model_loader import ensure_artifacts
        ARTIFACT_DIR = ensure_artifacts()
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Collect only files that are missing (or force=True)
    missing = [f for f in files if force or not (local_dir / f).exists()]

    if not missing:
        print(f"✅ All artifacts already cached at {local_dir}")
        return local_dir

    print(f"📥 Downloading {len(missing)} artifact(s) from {repo_id} ...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "Add  huggingface-hub  to your requirements.txt and redeploy."
        )

    # Optional: read HF token from Streamlit secrets or env
    token = _get_hf_token()

    for fname in missing:
        dest = local_dir / fname
        print(f"   ⬇  {fname}")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="model",
                local_dir=str(local_dir),
                token=token,
            )
            print(f"      → {path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{fname}' from '{repo_id}'.\n"
                f"Error: {e}\n\n"
                "Checklist:\n"
                "  • Repo ID is correct (username/repo-name)\n"
                "  • File exists in the HF repo\n"
                "  • HF_TOKEN secret is set in Streamlit Cloud (if repo is private)\n"
                "  • huggingface-hub is in requirements.txt"
            ) from e

    print(f"✅ All artifacts downloaded to {local_dir}")
    return local_dir


def _get_hf_token() -> str | None:
    """
    Try to read HF token from:
      1. Streamlit secrets  (st.secrets["HF_TOKEN"])
      2. Environment variable HF_TOKEN
    Returns None if not found (works for public repos).
    """
    # Try Streamlit secrets first (only available when running in Streamlit)
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass

    # Fallback to env var
    return os.environ.get("HF_TOKEN", None)


# ── Quick self-test (run this file directly to verify) ────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        HF_REPO_ID = sys.argv[1]   # pass repo_id as arg

    print(f"Testing download from: {HF_REPO_ID}")
    artifact_dir = ensure_artifacts(force=True)
    print("\nFiles in artifact dir:")
    for p in sorted(artifact_dir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
