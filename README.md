# ecg-arrhythmia-detection-xai
Deep learning–based ECG arrhythmia detection using the MIT-BIH dataset with Explainable AI techniques to interpret and visualize model predictions.

## FastAPI Web App

This project now includes a simple web interface for inference:

1. Upload ECG signal file (`.csv`, `.txt`, or `.npy`)
2. Select model (`CNN`, `LSTM`, or `Hybrid`)
3. Get predicted class, confidence, ECG plot, and explainability heatmap

### Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.
