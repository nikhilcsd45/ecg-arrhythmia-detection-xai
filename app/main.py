from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

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
from src.explainability.groq_report import (
    format_report_for_ui,
    generate_groq_report,
    load_env_file,
)
from src.models.cnn import CNNModel
from src.models.cnn_trans import CNNTransformer
from src.models.lstm import LSTMModel
from src.models.transformer import ECGTransformer
from src.preprocessing.preprocess import normalize

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
TEMPLATES_DIR = ROOT_DIR / "app" / "templates"
STATIC_DIR = ROOT_DIR / "app" / "static"

load_env_file(ROOT_DIR / ".env")

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}

# Built-in fallback text used when the Groq explanation is unavailable or empty.
CLASS_EXPLANATIONS: Dict[int, str] = {
    0: "The model focuses on regular rhythm and stable waveform morphology.",
    1: "The model highlights subtle timing changes and atrial-origin irregularities.",
    2: "The model emphasizes abnormal QRS-like deformation and stronger morphology shifts.",
    3: "The model detects mixed characteristics between normal and abnormal beat patterns.",
    4: "The signal does not strongly match a single known arrhythmia pattern.",
}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    model_class: type[torch.nn.Module]
    weight_files: Tuple[str, ...]
    input_layout: str
    explainer: str
    train_command: str


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "cnn_weighted": ModelConfig(
        key="cnn_weighted",
        label="CNN Weighted",
        model_class=CNNModel,
        weight_files=("cnn_weighted.pth",),
        input_layout="cnn",
        explainer="gradcam",
        train_command="python -m src.training.train_cnn",
    ),
    "cnn_baseline": ModelConfig(
        key="cnn_baseline",
        label="CNN Baseline",
        model_class=CNNModel,
        weight_files=("cnn_baseline.pth",),
        input_layout="cnn",
        explainer="gradcam",
        train_command="python -m src.training.train_cnn",
    ),
    "lstm": ModelConfig(
        key="lstm",
        label="LSTM",
        model_class=LSTMModel,
        weight_files=("lstm.pth",),
        input_layout="sequence",
        explainer="saliency",
        train_command="python -m src.training.train_lstm",
    ),
    "transformer": ModelConfig(
        key="transformer",
        label="Transformer",
        model_class=ECGTransformer,
        weight_files=("transformer.pth",),
        input_layout="sequence",
        explainer="saliency",
        train_command="python -m src.training.train_transformer",
    ),
    "cnn_transformer": ModelConfig(
        key="cnn_transformer",
        label="CNN-Transformer",
        model_class=CNNTransformer,
        weight_files=("cnn_trans.pth",),
        input_layout="sequence",
        explainer="saliency",
        train_command="python -m src.training.train_cnn_trans",
    ),
}

MODEL_ALIASES = {
    "cnn": "cnn_weighted",
    "cnn_trans": "cnn_transformer",
}

HYBRID_KEY = "hybrid"
HYBRID_LABEL = "Hybrid (CNN + LSTM)"
HYBRID_CNN_CANDIDATES = ("cnn_weighted", "cnn_baseline")
DEVICE = torch.device("cpu")
MODEL_CACHE: Dict[str, torch.nn.Module] = {}
GRADCAM_CACHE: Dict[str, GradCAM] = {}

app = FastAPI(title="ECG Arrhythmia Detection")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _canonical_model_key(model_key: str) -> str:
    return MODEL_ALIASES.get(model_key, model_key)


def _find_existing_model_path(candidates: Tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = MODELS_DIR / name
        if candidate.exists():
            return candidate
    return None


def _model_missing_message(config: ModelConfig) -> str:
    expected = ", ".join(str(MODELS_DIR / name) for name in config.weight_files)
    return (
        f"{config.label} weights are missing. Expected: {expected}. "
        f"Train this model with: {config.train_command}"
    )


def _load_model(model_key: str) -> torch.nn.Module:
    canonical_key = _canonical_model_key(model_key)
    config = MODEL_CONFIGS.get(canonical_key)
    if config is None:
        available = ", ".join(list(MODEL_CONFIGS.keys()) + [HYBRID_KEY])
        raise ValueError(f"Unknown model '{model_key}'. Choose one of: {available}.")

    if canonical_key in MODEL_CACHE:
        return MODEL_CACHE[canonical_key]

    model_path = _find_existing_model_path(config.weight_files)
    if model_path is None:
        raise FileNotFoundError(_model_missing_message(config))

    model = config.model_class(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    MODEL_CACHE[canonical_key] = model
    return model


def _has_weights(model_key: str) -> bool:
    config = MODEL_CONFIGS[model_key]
    return _find_existing_model_path(config.weight_files) is not None


def _hybrid_cnn_key() -> str | None:
    for model_key in HYBRID_CNN_CANDIDATES:
        if _has_weights(model_key):
            return model_key
    return None


def _hybrid_available() -> bool:
    return _hybrid_cnn_key() is not None and _has_weights("lstm")


def _default_model_key() -> str:
    for model_key in MODEL_CONFIGS:
        if _has_weights(model_key):
            return model_key
    return next(iter(MODEL_CONFIGS))


def _model_options() -> List[Dict[str, str | bool]]:
    options: List[Dict[str, str | bool]] = []
    for config in MODEL_CONFIGS.values():
        has_weights = _has_weights(config.key)
        status = "Ready" if has_weights else f"Missing weights. Run: {config.train_command}"
        options.append(
            {
                "value": config.key,
                "label": config.label,
                "available": has_weights,
                "status": status,
            }
        )

    hybrid_status = "Ready" if _hybrid_available() else "Needs at least one CNN model and LSTM weights."
    options.append(
        {
            "value": HYBRID_KEY,
            "label": HYBRID_LABEL,
            "available": _hybrid_available(),
            "status": hybrid_status,
        }
    )
    return options


def _parse_uploaded_signal(
    upload: UploadFile,
    lead_idx: int = 0,
    extra_files: dict | None = None,
) -> np.ndarray:
    """
    Accepts:
    - CSV/TXT: one column (or two: time,value), with/without header, any delimiter
    - WFDB: .dat/.hea pair (both must be uploaded together)
    - NPY: numpy array
    - Multi-lead: uses lead_idx (default 0)
    """
    suffix = Path(upload.filename or "").suffix.lower()
    name = Path(upload.filename or "").stem

    # Try WFDB: require both .hea and .dat to be uploaded together
    if suffix in {".hea", ".dat"}:
        has_partner = extra_files is not None and any(
            filename != upload.filename
            and Path(filename).stem == name
            and Path(filename).suffix.lower() in {".hea", ".dat"}
            for filename in extra_files
        )
        if not has_partner:
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
        lines = [line for line in text.splitlines() if line.strip()]
        parsed = []
        for line in lines:
            parts = [part for part in line.strip().split(delim) if part]
            if len(parts) == 1:
                try:
                    parsed.append(float(parts[0]))
                except ValueError:
                    continue
            elif len(parts) >= 2:
                try:
                    parsed.append(float(parts[-1]))
                except ValueError:
                    continue
        arr = np.asarray(parsed, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size >= 8:
            break
    if arr is None or arr.size < 8:
        raise ValueError("No valid ECG signal found in file. Please upload a file with at least 8 numeric values in one column.")
    return arr


def _resample_to_180(signal: np.ndarray) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=signal.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=180, dtype=np.float32)
    resized = np.interp(x_new, x_old, signal).astype(np.float32)
    return normalize(resized).astype(np.float32)


def _to_b64_png(fig: plt.Figure) -> str:
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
        ax.axvspan(
            x[i],
            x[i + 1],
            color=matplotlib.colormaps["OrRd"](float(h[i])),
            alpha=0.25,
            linewidth=0,
        )

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


def _class_probabilities(values: np.ndarray, already_probabilities: bool = False) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float32)
    if already_probabilities:
        total = float(probs.sum())
        return probs / (total + 1e-8)
    return F.softmax(torch.tensor(probs), dim=0).detach().numpy()


def _top_class_predictions(probabilities: np.ndarray, limit: int = 3) -> List[Dict[str, float | int | str]]:
    ranked_indices = np.argsort(probabilities)[::-1][:limit]
    top_predictions: List[Dict[str, float | int | str]] = []
    for idx in ranked_indices:
        top_predictions.append(
            {
                "class_id": int(idx),
                "class_name": CLASS_NAMES.get(int(idx), "Unknown"),
                "probability": round(float(probabilities[idx]), 4),
            }
        )
    return top_predictions


def _summarize_explain_map(heatmap: np.ndarray, max_regions: int = 3) -> List[Dict[str, float | int]]:
    values = np.asarray(heatmap, dtype=np.float32)
    if values.size == 0:
        return []

    threshold = max(float(values.max()) * 0.75, 0.55)
    active = values >= threshold

    regions: List[Tuple[int, int, float]] = []
    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            segment = values[start:idx]
            regions.append((start, idx - 1, float(segment.mean())))
            start = None

    if start is not None:
        segment = values[start:]
        regions.append((start, len(values) - 1, float(segment.mean())))

    if not regions:
        peak = int(np.argmax(values))
        start = max(0, peak - 6)
        end = min(len(values) - 1, peak + 6)
        regions.append((start, end, float(values[start:end + 1].mean())))

    regions.sort(key=lambda item: item[2], reverse=True)
    return [
        {
            "start": start_idx,
            "end": end_idx,
            "importance": round(score, 3),
        }
        for start_idx, end_idx, score in regions[:max_regions]
    ]


def _build_explanation_payload(
    signal180: np.ndarray,
    explain_map: np.ndarray,
    used_model: str,
    pred_idx: int,
    probabilities: np.ndarray,
    file_name: str,
) -> Dict[str, object]:
    return {
        "file_name": file_name,
        "model": used_model,
        "predicted_class_id": pred_idx,
        "predicted_class_name": CLASS_NAMES.get(pred_idx, "Unknown"),
        "confidence": round(float(probabilities[pred_idx]), 4),
        "top_predictions": _top_class_predictions(probabilities),
        "highlighted_regions": _summarize_explain_map(explain_map),
        "signal_length": int(signal180.shape[0]),
        "signal_summary": {
            "mean": round(float(np.mean(signal180)), 4),
            "std": round(float(np.std(signal180)), 4),
            "min": round(float(np.min(signal180)), 4),
            "max": round(float(np.max(signal180)), 4),
        },
    }


def _fallback_explanation_points(pred_idx: int, confidence: float) -> List[Dict[str, str]]:
    return [
        {
            "label": "Summary",
            "text": f"The model predicts {CLASS_NAMES.get(pred_idx, 'Unknown')} with {confidence * 100:.2f}% confidence.",
        },
        {
            "label": "Why",
            "text": CLASS_EXPLANATIONS.get(pred_idx, "No explanation available."),
        },
    ]


def _format_report_points(report: Dict[str, str] | None) -> List[Dict[str, str]]:
    if not report:
        return []

    sections = [
        ("Summary", report.get("summary")),
        ("Why", report.get("why_model_thinks_this")),
        ("Important Regions", report.get("important_regions")),
        ("Limitations", report.get("limitations")),
        ("Clinical Warning", report.get("clinical_warning")),
    ]

    return [
        {"label": label, "text": text.strip()}
        for label, text in sections
        if text and text.strip()
    ]


def _model_input(signal180: np.ndarray, input_layout: str) -> torch.Tensor:
    x = torch.from_numpy(signal180).float()
    if input_layout == "cnn":
        return x.unsqueeze(0).unsqueeze(0)
    return x.unsqueeze(0).unsqueeze(-1)


def _get_gradcam(model_key: str, model: torch.nn.Module) -> GradCAM:
    if model_key not in GRADCAM_CACHE:
        if not hasattr(model, "conv2"):
            raise ValueError(f"Grad-CAM is not configured for {MODEL_CONFIGS[model_key].label}.")
        GRADCAM_CACHE[model_key] = GradCAM(model, model.conv2)
    return GRADCAM_CACHE[model_key]


def _gradient_saliency(model: torch.nn.Module, x: torch.Tensor, pred_idx: int) -> np.ndarray:
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    logits = model(x)
    logits[:, pred_idx].backward()

    grad = x.grad.detach().abs().squeeze(0)
    if grad.ndim == 2:
        saliency = grad.mean(dim=-1).cpu().numpy().astype(np.float32)
    else:
        saliency = grad.cpu().numpy().astype(np.float32)
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


def _predict_and_explain(model_key: str, signal180: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, str]:
    canonical_key = _canonical_model_key(model_key)
    config = MODEL_CONFIGS[canonical_key]
    model = _load_model(canonical_key)
    x = _model_input(signal180, config.input_layout)

    with torch.no_grad():
        logits = model(x)
    pred_idx = int(torch.argmax(logits, dim=1).item())

    if config.explainer == "gradcam":
        gradcam = _get_gradcam(canonical_key, model)
        explain_raw = gradcam.generate(x, pred_idx)
        explain_map = _upsample_to_180(np.asarray(explain_raw, dtype=np.float32))
    else:
        explain_map = _gradient_saliency(model, x, pred_idx)

    return logits.squeeze(0).detach().numpy(), pred_idx, explain_map, config.label


def _hybrid_predict_and_explain(signal180: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, str]:
    cnn_key = _hybrid_cnn_key()
    if cnn_key is None:
        raise FileNotFoundError(
            "Hybrid prediction needs a CNN weight file. Train CNN with: python -m src.training.train_cnn"
        )

    cnn_logits, _, cnn_map, cnn_label = _predict_and_explain(cnn_key, signal180)
    lstm_logits, _, lstm_map, _ = _predict_and_explain("lstm", signal180)

    probs = (
        F.softmax(torch.tensor(cnn_logits), dim=0)
        + F.softmax(torch.tensor(lstm_logits), dim=0)
    ) / 2.0
    probabilities = probs.numpy()
    pred_idx = int(np.argmax(probabilities))
    explain_map = 0.5 * (cnn_map + lstm_map)
    return probabilities, pred_idx, explain_map, f"{HYBRID_LABEL} using {cnn_label}"


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "model_options": _model_options(),
            "selected_model": _default_model_key(),
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

        selected_model = _canonical_model_key(model_type)
        if selected_model == HYBRID_KEY:
            final_scores, pred_idx, explain_map, used_model = _hybrid_predict_and_explain(signal180)
            probabilities = _class_probabilities(final_scores, already_probabilities=True)
        elif selected_model in MODEL_CONFIGS:
            final_scores, pred_idx, explain_map, used_model = _predict_and_explain(selected_model, signal180)
            probabilities = _class_probabilities(final_scores)
        else:
            available = ", ".join(list(MODEL_CONFIGS.keys()) + [HYBRID_KEY])
            raise ValueError(f"Unknown model '{model_type}'. Choose one of: {available}.")

        pred_name = CLASS_NAMES.get(pred_idx, "Unknown")
        confidence = float(probabilities[pred_idx].item())
        file_name = os.path.basename(main_file.filename or "uploaded_signal")
        explanation_report = generate_groq_report(
            _build_explanation_payload(
                signal180=signal180,
                explain_map=explain_map,
                used_model=used_model,
                pred_idx=pred_idx,
                probabilities=probabilities,
                file_name=file_name,
            )
        )
        llm_explanation = format_report_for_ui(explanation_report)
        explanation_points = _format_report_points(explanation_report)
        if not explanation_points:
            explanation_points = _fallback_explanation_points(pred_idx, confidence)

        result = {
            "model": used_model,
            "class_id": pred_idx,
            "class_name": pred_name,
            "confidence": f"{confidence * 100:.2f}%",
            "explanation": llm_explanation or CLASS_EXPLANATIONS.get(pred_idx, "No explanation available."),
            "explanation_points": explanation_points,
            "explanation_source": "Groq AI" if llm_explanation else "Built-in fallback",
            "signal_plot_b64": _plot_signal(signal180, pred_name),
            "explain_plot_b64": _plot_explanation(signal180, explain_map, used_model),
            "file_name": file_name,
        }
    except Exception as exc:
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": error,
            "model_options": _model_options(),
            "selected_model": _canonical_model_key(model_type),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
