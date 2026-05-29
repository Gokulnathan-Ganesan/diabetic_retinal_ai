# DR Grading Streamlit App

## Setup
```bash
cd streamlit_app
pip install -r requirements.txt
```

## Copy model artifacts
```bash
cp -r ~/DR_data/artifacts streamlit_app/artifacts
```
The app expects `artifacts/` inside this folder with:
- `final_model.pt`            -- DR grading model (EfficientNetV2-S)
- `fundus_validator_v3.pt`    -- Fundus ensemble validator (v3)

## Run locally
```bash
streamlit run app.py
```
Then open http://localhost:8501

## Deploy to Streamlit Community Cloud (free)
1. Push this folder (with `artifacts/`) to a GitHub repo
2. https://share.streamlit.io -> New app -> select repo & app.py
3. Done — no secrets needed

## Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Notes
- GPU: automatic if CUDA available (DEVICE auto-detected)
- Research use only — NOT for clinical deployment
