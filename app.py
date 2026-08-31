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

import json

import torch

from config import EVAL_CFG, MODEL_CFG, weights_path
from datasets import build_transform
from model import build_model, load_best_weights
from metrics import classification_metrics, optimal_threshold

# ---------------------------------------------------------------------------
# Which trained model this web demo serves. Change this one line (and restart)
# to swap models. Its checkpoint is models/best_model_<MODEL_NAME>.pth, and its
# eval report (from `python evaluate.py --model <MODEL_NAME>`) sits next to it.
#   "hybrid_effb1" -> CLIP + EfficientNet-B1 low-level branch (4-channel input)
#   "hybrid_clip"  -> CLIP + small-CNN low-level branch
#   "efficientnet_b0" / ... -> the plain EfficientNet baseline (3-channel input)
# ---------------------------------------------------------------------------
MODEL_NAME = "hybrid_effb1"
WEIGHTS = weights_path(MODEL_NAME)

# Decision threshold for the per-image "AI-Generated / Real" verdict. Lower it
# if this model's scores skew low (see the cross-eval calibration note).
DECISION_THRESHOLD = 0.672

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
# Upload limits for one /predict request, and how many images to push through
# the model per forward pass.
#   MAX_IMAGES    - hard cap on files per upload (see MAX_FORM_PARTS below)
#   MAX_UPLOAD_MB - total request-body size cap
#   INFER_BATCH   - images per model forward pass (drop it if inference OOMs,
#                   raise it for speed if the GPU has spare memory)
MAX_IMAGES = 10000
MAX_UPLOAD_MB = 3000
INFER_BATCH = 32
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# The frontend sends 2 form parts per image (the file + its relative path), so
# Flask's default MAX_FORM_PARTS=1000 rejects any upload over ~500 images.
app.config["MAX_FORM_PARTS"] = MAX_IMAGES * 2 + 100
# Room for the (small, text) paths[] fields of a large upload; the default is
# 500 KB, which thousands of path strings can exceed.
app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# Error handlers – always return JSON so the frontend can parse the response
# regardless of what goes wrong server-side.
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"Upload too large. Maximum total size is {MAX_UPLOAD_MB} MB."}), 413

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
_threshold = DECISION_THRESHOLD


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

    # build_model dispatches on MODEL_CFG.name (EfficientNet vs. a hybrid variant).
    MODEL_CFG.name = MODEL_NAME
    _model = build_model(MODEL_CFG, _device)

    weights = WEIGHTS
    if not Path(weights).exists():
        # Try the outputs/checkpoints path used by some training runs
        alt = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
        if alt.exists():
            weights = alt

    load_best_weights(_model, weights, _device)
    _model.eval()

    _threshold = DECISION_THRESHOLD
    return _model, _device, _threshold


# ---------------------------------------------------------------------------
# Preprocessing – identical to what training / evaluate.py use for this model
# (4-channel RGB+FFT tensor for a hybrid model, ImageNet-normalised 3-channel
# for an EfficientNet). Picked from MODEL_NAME via datasets.build_transform.
# ---------------------------------------------------------------------------
_transform = build_transform(EVAL_CFG.image_size, train=False, model_name=MODEL_NAME)


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
        Path(WEIGHTS).with_name("eval_report.json"),
        PROJECT_ROOT / "outputs" / "checkpoints" / "eval_report.json",
        PROJECT_ROOT / "models" / "eval_report.json",
    ]
    return jsonify({"exists": any(p.exists() for p in candidates)})


@app.route("/eval-report", methods=["GET"])
def eval_report():
    """Return the eval_report.json produced by evaluate.py, searching known locations."""
    candidates = [
        # Primary: next to the configured weights file
        Path(WEIGHTS).with_name("eval_report.json"),
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

    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Too many images ({len(files)}). Limit is {MAX_IMAGES} per upload."}), 400

    model, device, threshold = get_model()

    # Results are kept in upload order (indexed list). Images are decoded,
    # preprocessed and run through the model in chunks of INFER_BATCH so memory
    # stays bounded no matter how large the upload is.
    results = [None] * len(files)
    buf_idx, buf_row, buf_tensor = [], [], []

    def flush():
        if not buf_tensor:
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.stack(buf_tensor).to(device)
            probs = torch.sigmoid(model(x)).squeeze(-1).cpu().tolist()
        per_img_ms = round((time.perf_counter() - t0) * 1000 / max(1, len(buf_tensor)), 1)
        for ri, row, prob_ai in zip(buf_idx, buf_row, probs):
            prob_ai = float(prob_ai)
            pred_label = 1 if prob_ai >= threshold else 0
            row.update({
                "prob_ai":      round(prob_ai, 4),
                "prob_real":    round(1.0 - prob_ai, 4),
                "verdict":      verdict(prob_ai, threshold),
                "confidence":   confidence_label(prob_ai),
                "pred_label":   pred_label,
                "inference_ms": per_img_ms,
            })
            if row["true_label"] is not None:
                row["correct"] = (pred_label == row["true_label"])
            results[ri] = row
        buf_idx.clear(); buf_row.clear(); buf_tensor.clear()

    for idx, f in enumerate(files):
        if not f or f.filename == "":
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results[idx] = {
                "id": str(uuid.uuid4())[:8],
                "filename": f.filename,
                "error": f"Unsupported file type '{ext}'.",
            }
            continue

        try:
            raw_bytes = f.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            width, height = img.size
            file_size_kb = round(len(raw_bytes) / 1024, 1)

            rel_path = rel_paths[idx] if idx < len(rel_paths) else f.filename
            true_label = infer_label(rel_path)

            row = {
                "id":             str(uuid.uuid4())[:8].upper(),
                "filename":       f.filename,
                "thumbnail":      pil_to_data_uri(img),
                "true_label":     true_label,   # 0, 1, or null
                "width":          width,
                "height":         height,
                "file_size_kb":   file_size_kb,
                "threshold_used": threshold,
            }
            buf_idx.append(idx)
            buf_row.append(row)
            buf_tensor.append(preprocess(img).squeeze(0))
            if len(buf_tensor) >= INFER_BATCH:
                flush()

        except Exception as exc:
            results[idx] = {
                "id":       str(uuid.uuid4())[:8],
                "filename": f.filename,
                "error":    str(exc),
            }

    flush()
    results = [r for r in results if r is not None]   # drop skipped blanks

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
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Model weights : {WEIGHTS}")
    print(f"  Open browser  : http://127.0.0.1:5001")
    app.run(debug=True, host="0.0.0.0", port=5001, threaded=True)
