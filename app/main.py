from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wfdb
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.explainability.gradcam import GradCAM
from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel
from src.preprocessing.preprocess import normalize

import wfdb
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATASET_DIR = ROOT_DIR / "dataset" / "mitdb"
TEMPLATES_DIR = ROOT_DIR / "app" / "templates"
STATIC_DIR = ROOT_DIR / "app" / "static"

MITDB_EXTENSIONS = {".dat", ".hea", ".atr", ".xws", ".at_"}

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}

CLASS_EXPLANATIONS: Dict[int, str] = {
    0: "The model focuses on regular rhythm and stable waveform morphology.",
    1: "The model highlights subtle timing changes and atrial-origin irregularities.",
    2: "The model emphasizes abnormal QRS-like deformation and stronger morphology shifts.",
    3: "The model detects mixed characteristics between normal and abnormal beat patterns.",
    4: "The signal does not strongly match a single known arrhythmia pattern.",
}

app = FastAPI(title="ECG Arrhythmia Detection")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _find_existing_model_path(candidates: Tuple[str, ...]) -> Path:
    for name in candidates:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of these model files exist: {candidates}")


def _load_models() -> Tuple[CNNModel, LSTMModel, GradCAM]:
    device = torch.device("cpu")

    cnn_path = _find_existing_model_path(("cnn_weighted.pth", "cnn_baseline.pth"))
    lstm_path = _find_existing_model_path(("lstm.pth",))

    cnn_model = CNNModel(num_classes=5).to(device)
    lstm_model = LSTMModel(num_classes=5).to(device)

    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))

    cnn_model.eval()
    lstm_model.eval()

    gradcam = GradCAM(cnn_model, cnn_model.conv2)
    return cnn_model, lstm_model, gradcam


CNN_MODEL, LSTM_MODEL, CNN_GRADCAM = _load_models()




def _parse_uploaded_signal(upload: UploadFile, lead_idx: int = 0, extra_files: dict = None) -> np.ndarray:
    """
    Accepts:
    - CSV/TXT: one column (or two: time,value), with/without header, any delimiter
    - WFDB: .dat/.hea pair (both must be uploaded together)
    - NPY: numpy array
    - Multi-lead: uses lead_idx (default 0)
    """
    import tempfile
    import shutil
    suffix = Path(upload.filename or "").suffix.lower()
    name = Path(upload.filename or "").stem

    # Try WFDB: require both .hea and .dat to be uploaded together
    if suffix in {".hea", ".dat"}:
        if extra_files is None or not any(f for f in extra_files if f != upload.filename and Path(f).stem == name and Path(f).suffix.lower() in {".hea", ".dat"}):
            raise ValueError("For WFDB records, please upload both .dat and .hea files together.")
        tempdir = tempfile.mkdtemp()
        try:
            # Save both files
            for fname, fobj in extra_files.items():
                if Path(fname).stem == name and Path(fname).suffix.lower() in {".hea", ".dat"}:
                    with open(Path(tempdir) / fname, "wb") as out:
                        out.write(fobj.file.read())
            record = wfdb.rdrecord(str(Path(tempdir) / name))
            if record.p_signal is None or record.p_signal.size == 0:
                raise ValueError(f"Record '{name}' does not contain usable signal data.")
            if record.p_signal.ndim == 1:
                arr = np.asarray(record.p_signal, dtype=np.float32)
            else:
                arr = np.asarray(record.p_signal[:, lead_idx], dtype=np.float32)
            arr = arr[~np.isnan(arr)]
            if arr.size < 8:
                raise ValueError(f"Record '{name}' has too few usable ECG points.")
            return arr
        finally:
            shutil.rmtree(tempdir)

    # Try NPY
    raw = upload.file.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")
    if suffix == ".npy":
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size < 8:
            raise ValueError("Too few ECG points found. Please upload a longer signal.")
        return arr

    # Try CSV/TXT: auto-detect delimiter, skip header, allow single-column, skip bad lines
    text = raw.decode("utf-8", errors="ignore")
    arr = None
    for delim in [",", "\t", ";", " "]:
        try:
            # Read all lines, skip empty or bad lines
            lines = [line for line in text.splitlines() if line.strip()]
            parsed = []
            for idx, line in enumerate(lines):
                parts = [p for p in line.strip().split(delim) if p]
                # Accept lines with 1 or 2 columns (signal or time,signal)
                if len(parts) == 1:
                    try:
                        parsed.append(float(parts[0]))
                    except Exception:
                        continue
                elif len(parts) >= 2:
                    try:
                        parsed.append(float(parts[-1]))
                    except Exception:
                        continue
                # else: skip lines with 0 columns
            arr = np.asarray(parsed, dtype=np.float32)
            arr = arr[~np.isnan(arr)]
            if arr.size >= 8:
                break
        except Exception:
            continue
    if arr is None or arr.size < 8:
        raise ValueError("No valid ECG signal found in file. Please upload a file with at least 8 numeric values in one column.")
    return arr
    return arr


def _resample_to_180(signal: np.ndarray) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=180, dtype=np.float32)
    resized = np.interp(x_new, x_old, signal).astype(np.float32)
    return normalize(resized).astype(np.float32)


def _to_b64_png(fig: plt.Figure) -> str:
    import base64

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_signal(signal: np.ndarray, pred_name: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(signal, color="#134074", linewidth=1.8)
    ax.set_title(f"Uploaded ECG (Predicted: {pred_name})")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    return _to_b64_png(fig)


def _plot_explanation(signal: np.ndarray, heatmap: np.ndarray, model_name: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(signal.shape[0])
    ax.plot(x, signal, color="#0f172a", linewidth=1.5, label="ECG")

    h = np.clip(heatmap, 0.0, 1.0)
    for i in range(len(x) - 1):
        ax.axvspan(x[i], x[i + 1], color=plt.cm.OrRd(h[i]), alpha=0.25, linewidth=0)

    ax.set_title(f"Explainability Heatmap ({model_name.upper()})")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    return _to_b64_png(fig)


def _upsample_to_180(values: np.ndarray) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=180, dtype=np.float32)
    upsampled = np.interp(x_new, x_old, values).astype(np.float32)
    vmin = float(upsampled.min())
    vmax = float(upsampled.max())
    return (upsampled - vmin) / (vmax - vmin + 1e-8)


def _cnn_predict_and_explain(signal180: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    x = torch.from_numpy(signal180).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = CNN_MODEL(x)
    pred_idx = int(torch.argmax(logits, dim=1).item())

    cam_raw = CNN_GRADCAM.generate(x, pred_idx)
    cam = _upsample_to_180(np.asarray(cam_raw, dtype=np.float32))
    return logits.squeeze(0).detach().numpy(), pred_idx, cam


def _lstm_predict_and_explain(signal180: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    x = torch.from_numpy(signal180).unsqueeze(0).unsqueeze(-1)
    x = x.clone().detach().requires_grad_(True)

    logits = LSTM_MODEL(x)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[:, pred_idx]

    LSTM_MODEL.zero_grad()
    score.backward()

    saliency = x.grad.detach().abs().squeeze(0).squeeze(-1).numpy().astype(np.float32)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return logits.squeeze(0).detach().numpy(), pred_idx, saliency


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
   
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
        },
    )



@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    model_type: str = Form(...),
    ecg_file: List[UploadFile] = File(...)
):

    result = None
    error = None
    try:
        # If multiple files, build a dict for WFDB support
        extra_files = {f.filename: f for f in ecg_file}
        # Use the first file as the main one (for CSV, TXT, NPY, or WFDB)
        main_file = ecg_file[0]
        raw_signal = _parse_uploaded_signal(main_file, extra_files=extra_files)
        signal180 = _resample_to_180(raw_signal)

        cnn_logits, cnn_idx, cnn_map = _cnn_predict_and_explain(signal180)
        lstm_logits, lstm_idx, lstm_map = _lstm_predict_and_explain(signal180)

        if model_type == "cnn":
            final_logits = cnn_logits
            pred_idx = cnn_idx
            explain_map = cnn_map
            used_model = "CNN"
        elif model_type == "lstm":
            final_logits = lstm_logits
            pred_idx = lstm_idx
            explain_map = lstm_map
            used_model = "LSTM"
        elif model_type == "hybrid":
            probs = (F.softmax(torch.tensor(cnn_logits), dim=0) + F.softmax(torch.tensor(lstm_logits), dim=0)) / 2.0
            final_logits = probs.numpy()
            pred_idx = int(np.argmax(final_logits))
            explain_map = 0.5 * (cnn_map + lstm_map)
            used_model = "Hybrid (CNN + LSTM)"
        else:
            raise ValueError("Model selection must be cnn, lstm, or hybrid.")

        pred_name = CLASS_NAMES.get(pred_idx, "Unknown")
        confidence = float(F.softmax(torch.tensor(final_logits), dim=0)[pred_idx].item())

        result = {
            "model": used_model,
            "class_id": pred_idx,
            "class_name": pred_name,
            "confidence": f"{confidence * 100:.2f}%",
            "explanation": CLASS_EXPLANATIONS.get(pred_idx, "No explanation available."),
            "signal_plot_b64": _plot_signal(signal180, pred_name),
            "explain_plot_b64": _plot_explanation(signal180, explain_map, used_model),
            "file_name": os.path.basename(main_file.filename or "uploaded_signal"),
        }
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": error,
            "selected_model": model_type,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
