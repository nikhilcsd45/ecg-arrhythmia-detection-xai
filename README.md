# ecg-arrhythmia-detection-xai
Deep learning–based ECG arrhythmia detection using the MIT-BIH dataset with Explainable AI techniques to interpret and visualize model predictions.

## FastAPI Web App

This project now includes a simple web interface for inference:

1. Upload ECG signal file (`.csv`, `.txt`, `.npy`, or WFDB `.dat` + `.hea`)
2. Select any available trained model (`CNN Weighted`, `CNN Baseline`, `LSTM`, `Transformer`, `CNN-Transformer`, or `Hybrid`)
3. Get predicted class, confidence, ECG plot, and explainability heatmap

### Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Groq AI Explanation

The app can optionally generate a richer natural-language explanation for each prediction using the Groq API.

1. Create a `.env` file in the project root
2. Add your Groq key:

```bash
GROQ_API_KEY=> I put into .env file 
GROQ_MODEL=llama-3.3-70b-versatile
```

If `GROQ_API_KEY` is missing or the API request fails, the app automatically falls back to the built-in explanation from `CLASS_EXPLANATIONS`, so prediction still works normally.
