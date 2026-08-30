"""Flask web application for the real-vs-AI image detector.

Provides:
  GET  /            -> serves the frontend UI
  POST /predict     -> accepts one or more uploaded images, runs inference,
                       returns JSON with per-image statistics
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import local modules
# even when the app is launched from a different working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision import transforms

from config import BEST_WEIGHTS, EVAL_CFG, MODEL_CFG
from model import build_model, load_best_weights
from metrics import classification_metrics, optimal_threshold

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

# ---------------------------------------------------------------------------
# Error handlers – always return JSON so the frontend can parse the response
# regardless of what goes wrong server-side.
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Upload too large. Maximum total size is 500 MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500

@app.errorhandler(Exception)
def unhandled(e):
    return jsonify({"error": str(e)}), 500

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Model – loaded once at startup and reused for every request
# ---------------------------------------------------------------------------
_model = None
_device = None
_threshold = 0.5


def get_model():
    """Lazy-load the model on the first request."""
    global _model, _device, _threshold

    if _model is not None:
        return _model, _device, _threshold

    # Device selection: cuda > mps > cpu
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    _model = build_model(MODEL_CFG, _device)

    weights = BEST_WEIGHTS
    if not Path(weights).exists():
        # Try the outputs/checkpoints path used by some training runs
        alt = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
        if alt.exists():
            weights = alt

    load_best_weights(_model, weights, _device)
    _model.eval()

    _threshold = EVAL_CFG.threshold
    return _model, _device, _threshold


# ---------------------------------------------------------------------------
# Preprocessing – identical to what training / inference.py use
# ---------------------------------------------------------------------------
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    return _transform(pil_image.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def confidence_label(prob_ai: float) -> str:
    """Map AI probability to a human-readable confidence band."""
    p = max(prob_ai, 1 - prob_ai)   # distance from 0.5
    if p >= 0.90:
        return "Very High"
    if p >= 0.75:
        return "High"
    if p >= 0.60:
        return "Moderate"
    return "Low"


def verdict(prob_ai: float, threshold: float) -> str:
    return "AI-Generated" if prob_ai >= threshold else "Real / Human"


# Label 1 = AI, 0 = Real. Inferred from the closest parent folder whose name
# matches one of the known class tokens in the browser-reported relative path.
_AI_TOKENS   = {"ai", "ai_generated", "fake", "synthetic"}
_REAL_TOKENS = {"real", "human", "authentic", "genuine"}

def infer_label(relative_path: str) -> int | None:
    """Return 1 (AI), 0 (real), or None (unknown) from the file's relative path."""
    parts = Path(relative_path.replace("\\", "/")).parts
    # Walk from the innermost folder outward (skip the filename itself)
    for part in reversed(parts[:-1]):
        token = part.strip().lower()
        if token in _AI_TOKENS:
            return 1
        if token in _REAL_TOKENS:
            return 0
    return None


def pil_to_data_uri(img: Image.Image, max_side: int = 200) -> str:
    """Resize image to a thumbnail and return a base-64 data URI."""
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    fmt = "JPEG" if thumb.mode == "RGB" else "PNG"
    thumb.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/eval-report/exists", methods=["GET"])
def eval_report_exists():
    """Lightweight check — returns {exists: true/false} without reading the file."""
    candidates = [
        Path(BEST_WEIGHTS).with_name("eval_report.json"),
        PROJECT_ROOT / "outputs" / "checkpoints" / "eval_report.json",
        PROJECT_ROOT / "models" / "eval_report.json",
    ]
    return jsonify({"exists": any(p.exists() for p in candidates)})


@app.route("/eval-report", methods=["GET"])
def eval_report():
    """Return the eval_report.json produced by evaluate.py, searching known locations."""
    candidates = [
        # Primary: next to the configured weights file
        Path(BEST_WEIGHTS).with_name("eval_report.json"),
        # Fallback: outputs/checkpoints (used by some training runs)
        PROJECT_ROOT / "outputs" / "checkpoints" / "eval_report.json",
        # Any .json named eval_report anywhere under models/
        PROJECT_ROOT / "models" / "eval_report.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["_source"] = str(path.relative_to(PROJECT_ROOT))
            return jsonify(data)

    return jsonify({"error": (
        "eval_report.json not found. "
        "Run  python evaluate.py  first to generate it."
    )}), 404


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded."}), 400

    # Relative paths sent by the browser (webkitRelativePath or bare filename).
    # Index-aligned with the 'images' list so we can infer labels per file.
    rel_paths = request.form.getlist("paths[]")

    model, device, threshold = get_model()
    results = []

    for idx, f in enumerate(files):
        if not f or f.filename == "":
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "id": str(uuid.uuid4())[:8],
                "filename": f.filename,
                "error": f"Unsupported file type '{ext}'.",
            })
            continue

        try:
            raw_bytes = f.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            width, height = img.size
            file_size_kb = round(len(raw_bytes) / 1024, 1)

            # Infer ground-truth label from the browser-reported relative path
            # (populated when the user picks a folder via webkitdirectory).
            rel_path = rel_paths[idx] if idx < len(rel_paths) else f.filename
            true_label = infer_label(rel_path)

            # --- Inference ---
            t0 = time.perf_counter()
            with torch.no_grad():
                x = preprocess(img).to(device)
                prob_ai = float(torch.sigmoid(model(x)).squeeze(-1).cpu().item())
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            prob_real  = round(1.0 - prob_ai, 4)
            prob_ai_r  = round(prob_ai, 4)
            pred_label = 1 if prob_ai >= threshold else 0

            row = {
                "id":             str(uuid.uuid4())[:8].upper(),
                "filename":       f.filename,
                "thumbnail":      pil_to_data_uri(img),
                "prob_ai":        prob_ai_r,
                "prob_real":      prob_real,
                "verdict":        verdict(prob_ai, threshold),
                "confidence":     confidence_label(prob_ai),
                "pred_label":     pred_label,
                "true_label":     true_label,   # 0, 1, or null
                "width":          width,
                "height":         height,
                "file_size_kb":   file_size_kb,
                "inference_ms":   elapsed_ms,
                "threshold_used": threshold,
            }
            # Correctness only meaningful when ground truth is known
            if true_label is not None:
                row["correct"] = (pred_label == true_label)

            results.append(row)

        except Exception as exc:
            results.append({
                "id":       str(uuid.uuid4())[:8],
                "filename": f.filename,
                "error":    str(exc),
            })

    # ------------------------------------------------------------------
    # Aggregate metrics – computed only when at least one label is known
    # ------------------------------------------------------------------
    ok = [r for r in results if "error" not in r]
    labeled = [(r["true_label"], r["prob_ai"]) for r in ok if r["true_label"] is not None]

    aggregate = None
    if len(labeled) >= 2:
        import numpy as np
        y_true = [l for l, _ in labeled]
        y_prob = [p for _, p in labeled]

        # Use the Youden-optimal threshold when enough labels are present,
        # otherwise fall back to the model's configured threshold.
        try:
            best_thr = optimal_threshold(y_true, y_prob)
        except Exception:
            best_thr = threshold

        metrics, y_pred = classification_metrics(y_true, y_prob, best_thr)

        y_true_arr = [int(v) for v in y_true]
        y_pred_arr = [int(v) for v in y_pred]
        fp = sum(1 for t, p in zip(y_true_arr, y_pred_arr) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true_arr, y_pred_arr) if t == 1 and p == 0)

        fp_confs = [y_prob[i] for i, (t, p) in enumerate(zip(y_true_arr, y_pred_arr)) if t == 0 and p == 1]
        fn_confs = [y_prob[i] for i, (t, p) in enumerate(zip(y_true_arr, y_pred_arr)) if t == 1 and p == 0]

        aggregate = {
            "n":         len(labeled),
            "threshold": round(best_thr, 4),
            "metrics":   {k: round(v, 4) for k, v in metrics.items()},
            "error_analysis": {
                "false_positives":    fp,
                "false_negatives":    fn,
                "fp_mean_confidence": round(float(np.mean(fp_confs)), 4) if fp_confs else None,
                "fn_mean_confidence": round(float(np.mean(fn_confs)), 4) if fn_confs else None,
            },
        }

    return jsonify({"results": results, "aggregate": aggregate})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting AI Image Detector web interface...")
    print(f"  Model weights : {BEST_WEIGHTS}")
    print(f"  Open browser  : http://127.0.0.1:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
