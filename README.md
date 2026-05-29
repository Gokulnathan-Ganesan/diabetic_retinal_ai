# 🔧 Fix: `final_model.pt found` on Streamlit Cloud

## Root Cause
GitHub blocks files > 100 MB. Your `final_model.pt` lives only on your Mac
(`~/DR_data/artifacts/`) and was never pushed. Streamlit Cloud clones your repo,
so it never gets the model.

## Solution: Host model on Hugging Face Hub (free)

---

## Step 1 — Upload model to Hugging Face (on your Mac, one-time)

```bash
pip install huggingface_hub

python3 << 'EOF'
from huggingface_hub import HfApi, login

login()   # opens browser → paste your HF token

api = HfApi()

# Create the repo (skip if already exists)
api.create_repo("dr-grading-v13", repo_type="model", private=True)

ARTIFACTS = "/Users/gokulnathan/DR_data/artifacts"
FILES_TO_UPLOAD = [
    "final_model.pt",
    "best_model.pt",
    "fundus_validator_ensemble.pt",
    "calibration.pt",
    "_resume_state.json",
]

for fname in FILES_TO_UPLOAD:
    path = f"{ARTIFACTS}/{fname}"
    import os
    if os.path.exists(path):
        print(f"Uploading {fname} ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=fname,
            repo_id="YOUR_HF_USERNAME/dr-grading-v13",  # ← change this
            repo_type="model",
        )
        print(f"  ✅ {fname} done")
    else:
        print(f"  ⚠️  {fname} not found, skipping")
EOF
```

---

## Step 2 — Add model_loader.py to your GitHub repo

Copy `model_loader.py` (provided) into your repo root next to `app.py`.

Edit line 12 in `model_loader.py`:
```python
HF_REPO_ID = "YOUR_HF_USERNAME/dr-grading-v13"   # ← your actual HF username
```

---

## Step 3 — Patch app.py (add 3 lines at the top of the file)

Find where `app.py` sets up `ARTIFACT_DIR` (look for something like
`ARTIFACT_DIR = Path("artifacts")`), and replace that block with:

```python
# ── Model loader (downloads from HuggingFace Hub on Streamlit Cloud) ──────────
from model_loader import ensure_artifacts
ARTIFACT_DIR = ensure_artifacts()   # returns Path("/tmp/dr_artifacts")
```

That's it. Every other reference to `ARTIFACT_DIR` in app.py stays the same.

---

## Step 4 — Add HF token to Streamlit Cloud secrets (private repo only)

1. Go to https://huggingface.co/settings/tokens → **New token** → Read → Copy it
2. Open https://share.streamlit.io → your app → **Settings** → **Secrets**
3. Paste:
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```
4. Save → Streamlit Cloud automatically restarts the app.

---

## Step 5 — Add huggingface-hub to requirements.txt

```
huggingface-hub>=0.23.0
```

Commit and push. Streamlit Cloud will redeploy, download the model on first
cold start (~1-2 min), and cache it for subsequent requests.

---

## What happens at runtime

```
[startup]  📥 Downloading 2 artifact(s) from YOUR_HF_USERNAME/dr-grading-v13
            ⬇  final_model.pt  →  /tmp/dr_artifacts/final_model.pt
            ⬇  fundus_validator_ensemble.pt  →  /tmp/dr_artifacts/...
           ✅ All artifacts downloaded to /tmp/dr_artifacts

[2nd visit] ✅ All artifacts already cached at /tmp/dr_artifacts
```

> ⚠️  `/tmp` is cleared on Streamlit Cloud restart. The download runs once per
> cold start (not per user). Each cold start takes ~1-2 min on first load.
> After that, warm requests are instant.

---

## Files to commit to GitHub

```
diabetic_retinal_ai/
├── app.py              ← patched (add 3 lines at top)
├── model_loader.py     ← NEW (provided)
├── requirements.txt    ← add huggingface-hub>=0.23.0
└── README.md
```

**Do NOT commit `final_model.pt` or the `artifacts/` folder to GitHub.**
Add them to `.gitignore`:
```
artifacts/
*.pt
*.pth
```
