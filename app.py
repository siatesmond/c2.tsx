"""Flask web application for the real-vs-AI image detector.

Provides:
  GET  /                  -> serves the frontend UI
  POST /predict           -> scores ONE chunk of uploaded images and returns
                             per-image statistics
  POST /aggregate         -> folder-level metrics over every chunk of a run
  GET  /eval-report       -> the offline eval_report.json from evaluate.py
  GET  /eval-report/exists

A folder of any size is supported because the frontend slices the selection
into chunks and posts them one at a time; /predict is stateless per chunk and
/aggregate stitches the run back together at the end.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
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

from dataclasses import replace

from config import BEST_WEIGHTS, EVAL_CFG, MODEL_CFG
from model import build_model, load_best_weights
from metrics import classification_metrics, optimal_threshold

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Upload sizing
#
# The frontend never posts a whole folder in one request: it slices the
# selection into small chunks and posts them one after another (see
# CHUNK_FILES / CHUNK_BYTES in templates/index.html). That is what makes an
# arbitrarily large folder work -- total upload size is unbounded, only the
# size of a single chunk matters here.
#
# MAX_UPLOAD_MB therefore caps ONE chunk, not the job. Set it to 0 (the
# default) to disable the per-request cap entirely.
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "0"))
app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB * 1024 * 1024) or None

# How many images are stacked into a single forward pass. Larger = faster on
# GPU, more memory. Overridable for low-RAM machines.
BATCH_SIZE = max(1, int(os.environ.get("PREDICT_BATCH_SIZE", "16")))

# ---------------------------------------------------------------------------
# Error handlers – always return JSON so the frontend can parse the response
# regardless of what goes wrong server-side.
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": (
        f"Upload chunk too large (limit {MAX_UPLOAD_MB} MB per request). "
        "Lower CHUNK_BYTES in the frontend, or raise/disable the cap with "
        "the MAX_UPLOAD_MB environment variable."
    )}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": f"Internal server error: {e}"}), 500

@app.errorhandler(Exception)
def unhandled(e):
    # Let real HTTP errors (404, 405, ...) keep their own status code; only
    # genuine crashes should be reported as 500.
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": str(e)}), 500

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Checkpoint discovery
#
# This repo contains TWO independently-trained stacks that save incompatible
# checkpoints, so the app must pick the right one rather than the first one
# that happens to exist:
#
#   * root stack   (train.py + model.py, torchvision EfficientNet)
#       -> keys look like "backbone.features.0.0.weight"
#       -> THIS is what app.py / model.py can load
#   * src/ stack   (src/train.py + src/model.py, timm EfficientNet)
#       -> keys look like "conv_stem.weight", "blocks.0.0..."
#       -> loading it into the torchvision model raises a key-mismatch error
#
# Candidates are tried in order; any checkpoint in the timm layout is skipped
# with an explanatory message instead of blowing up mid-request.
# ---------------------------------------------------------------------------
def _weight_candidates():
    """Checkpoints to try, best first. WEIGHTS env var overrides everything."""
    override = os.environ.get("WEIGHTS")
    if override:
        return [Path(override)]
    return [
        Path(BEST_WEIGHTS),                                    # models/best_model_<variant>.pth
        # Root-level variant checkpoints, newest architecture first. A run of
        # `train.py --model efficientnet_b1` writes this name, and it will not
        # load into a b0, so the variant has to be read from the file rather
        # than assumed from config.
        PROJECT_ROOT / "best_model_efficientnet_b2.pth",
        PROJECT_ROOT / "best_model_efficientnet_b1.pth",
        PROJECT_ROOT / "best_model.pth",                       # original b0
        PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt",  # src/ stack (timm) - rejected
    ]


# NOTE: deliberately not evaluated at import time. The WEIGHTS override can
# be set after this module is imported (by __main__, or by a test), and a
# frozen list would silently ignore it.
WEIGHT_CANDIDATES = _weight_candidates()   # import-time snapshot, for reference only


def _state_dict_of(path):
    """Return the raw parameter dict from a checkpoint, whatever wrapper it uses."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    return obj


def is_torchvision_checkpoint(path) -> bool:
    """True if the checkpoint matches model.EfficientNetDetector's parameter names."""
    try:
        keys = list(_state_dict_of(path).keys())
    except Exception:
        return False
    return any(k.startswith("backbone.features.") for k in keys)


def resolve_weights() -> Path:
    """First candidate that exists AND is loadable by the torchvision model."""
    seen = []
    # Re-read rather than using the import-time snapshot, so a WEIGHTS
    # override set later still takes effect.
    for path in _weight_candidates():
        if not path.exists():
            continue
        seen.append(path)
        if is_torchvision_checkpoint(path):
            return path
        print(f"[app] skipping {path}: not a torchvision-format checkpoint "
              f"(looks like a timm checkpoint from src/train.py)")

    raise FileNotFoundError(
        "No compatible checkpoint found. app.py serves the root stack "
        "(train.py + model.py, torchvision EfficientNet). "
        f"Looked in {[str(p) for p in WEIGHT_CANDIDATES]}; "
        f"existing but incompatible: {[str(p) for p in seen]}. "
        "Train one with 'python train.py', or point BEST_WEIGHTS at a "
        "checkpoint whose keys start with 'backbone.features.'."
    )

def checkpoint_meta(path):
    """What a checkpoint says about itself: (model_name, image_size, threshold).

    train.py records these alongside the weights, which matters because they
    are NOT interchangeable between runs: efficientnet_b1 weights will not
    load into a b0, and a model trained at 240px scored at 224px quietly
    loses accuracy with no error to notice. Any field the checkpoint does not
    carry comes back None and the caller falls back to config.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None, None
    if not isinstance(obj, dict):
        return None, None, None

    name = obj.get("model_name")
    cfg = obj.get("config")
    size = cfg.get("image_size") if isinstance(cfg, dict) else None
    metrics = obj.get("metrics")
    # The threshold train.py fitted on its own validation split. Specific to
    # this model's output distribution, so it beats any report written for a
    # different checkpoint.
    threshold = metrics.get("threshold") if isinstance(metrics, dict) else None
    return name, size, threshold


# ---------------------------------------------------------------------------
# Decision threshold
#
# 0.5 is the right default only for a well-calibrated model. This one is not:
# on realworld_eval_2k.json (4,000 images) its ROC-AUC is 0.972, but the
# ROC-optimal cut is 0.0019 and even the most AI-looking REAL images only
# score ~0.06. The whole probability mass sits near zero, so a 0.5 cutoff
# labels essentially everything "Real / Human" -- the ranking is good, the
# calibration is not.
#
# So prefer a threshold that was actually fitted to this model's outputs
# (Youden's J, written by test_realworld.py / evaluate.py) and fall back to
# the configured constant only when no report exists. THRESHOLD env var
# overrides everything.
# ---------------------------------------------------------------------------
THRESHOLD_REPORTS = [
    PROJECT_ROOT / "realworld_eval_2k.json",
    PROJECT_ROOT / "realworld_eval.json",
]


def calibrated_threshold(from_checkpoint=None, weights_name=None):
    """Fitted decision threshold, or EVAL_CFG.threshold if none is available.

    Order matters. A threshold fitted to a DIFFERENT checkpoint is worse than
    useless: the b0 sits at 0.0019 and the b1 at 0.69, so applying one to the
    other flips essentially every verdict. The checkpoint's own value wins.
    """
    override = os.environ.get("THRESHOLD")
    if override:
        return float(override), "THRESHOLD env var"

    if from_checkpoint is not None:
        return float(from_checkpoint), f"threshold recorded in {weights_name}"

    for path in THRESHOLD_REPORTS + _eval_report_candidates():
        if not path.exists():
            continue
        try:
            with open(path) as f:
                value = json.load(f).get("threshold")
        except Exception:
            continue
        if isinstance(value, (int, float)) and 0.0 < value < 1.0:
            return float(value), str(path.relative_to(PROJECT_ROOT))

    return EVAL_CFG.threshold, "config default (uncalibrated)"


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

    weights = resolve_weights()
    name, image_size, ckpt_threshold = checkpoint_meta(weights)

    # Build the architecture the checkpoint was trained as, not the one the
    # config happens to name -- a mismatch fails loudly here, or worse, the
    # resolution mismatch fails silently at inference time.
    cfg = replace(MODEL_CFG, name=name) if name else MODEL_CFG
    # pretrained=False: load_best_weights replaces every parameter below, so
    # fetching ImageNet weights first is wasted work -- and on a network with
    # strict TLS it fails with SSL CERTIFICATE_VERIFY_FAILED before any image
    # is scored.
    _model = build_model(replace(cfg, pretrained=False), _device)

    print(f"[app] loading weights: {weights}")
    print(f"[app] architecture   : {cfg.name}" + ("" if name else " (from config - checkpoint did not say)"))
    load_best_weights(_model, weights, _device)
    _model.eval()

    global _transform
    _transform = build_transform(image_size or EVAL_CFG.image_size)
    print(f"[app] input size     : {image_size or EVAL_CFG.image_size}px"
          + ("" if image_size else " (from config - checkpoint did not say)"))

    _threshold, source = calibrated_threshold(ckpt_threshold, weights.name)
    print(f"[app] decision threshold: {_threshold:g}  (source: {source})")
    return _model, _device, _threshold


# ---------------------------------------------------------------------------
# Preprocessing – identical to what training / inference.py use
# ---------------------------------------------------------------------------
def build_transform(image_size):
    """Preprocessing must match what the model was TRAINED with.

    Not a constant any more: efficientnet_b0 was trained at 224 and b1 at
    240, and scoring at the wrong size costs accuracy without raising
    anything. get_model() rebuilds this from the checkpoint's own config.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# Replaced at model-load time with the size the checkpoint specifies.
_transform = build_transform(EVAL_CFG.image_size)


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    return _transform(pil_image.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def calibrated_score(prob_ai: float, threshold: float) -> float:
    """Remap the raw score onto a 0-1 scale where the THRESHOLD sits at 0.5.

    The raw score cannot be read as a probability when the model is poorly
    calibrated: efficientnet_b0's outputs cluster near zero with a decision
    boundary at 0.0019, so a raw 0.28% is genuinely AI while "99.7% real" is
    the same number restated against the wrong boundary.

    The remap is piecewise-linear between the boundary and each extreme:
    [0, threshold] -> [0, 0.5] and [threshold, 1] -> [0.5, 1]. It answers
    "how far from the boundary towards certainty is this?", which holds
    whatever the threshold is. That matters because thresholds vary hugely
    between checkpoints -- 0.0019 for the b0, 0.688 for the b1 -- and an
    earlier log-space version with a fixed span silently collapsed the b1's
    entire AI range into 50-55%, reporting a raw score of 1.0 as borderline.
    """
    t = min(max(threshold, 1e-9), 1.0 - 1e-9)
    p = min(max(prob_ai, 0.0), 1.0)
    if p < t:
        return 0.5 * (p / t)
    return 0.5 + 0.5 * (p - t) / (1.0 - t)


def confidence_label(calibrated: float) -> str:
    """Confidence band, measured from the DECISION BOUNDARY.

    Takes the calibrated score, not the raw one: measuring from a hardcoded
    0.5 would call every image "Very High" simply because every raw score
    sits far below 0.5, including the ones sitting right on the boundary.
    """
    p = max(calibrated, 1 - calibrated)
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


# ---------------------------------------------------------------------------
# eval_report.json discovery
#
# evaluate.py writes the report NEXT TO the weights file it was given
# (Path(weights).with_name("eval_report.json")), so where it lands depends on
# which checkpoint was evaluated:
#
#   python evaluate.py                              -> models/eval_report.json
#   python evaluate.py --weights best_model.pth     -> eval_report.json (repo root)
#
# The repo-root case is the common one here, because the committed checkpoint
# is best_model.pth at the root rather than under models/. All the locations
# are searched so the tab finds the report whichever way it was produced.
# ---------------------------------------------------------------------------
def _eval_report_candidates():
    return [
        # Next to the configured weights file (models/, by default)
        Path(BEST_WEIGHTS).with_name("eval_report.json"),
        # Next to the committed root-stack checkpoint
        PROJECT_ROOT / "eval_report.json",
        # Used by some training runs
        PROJECT_ROOT / "outputs" / "checkpoints" / "eval_report.json",
        PROJECT_ROOT / "models" / "eval_report.json",
    ]


@app.route("/eval-report/exists", methods=["GET"])
def eval_report_exists():
    """Lightweight check — returns {exists: true/false} without reading the file."""
    candidates = _eval_report_candidates()
    return jsonify({"exists": any(p.exists() for p in candidates)})


@app.route("/eval-report", methods=["GET"])
def eval_report():
    """Return the eval_report.json produced by evaluate.py, searching known locations."""
    candidates = _eval_report_candidates()
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


# ---------------------------------------------------------------------------
# Inference core
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_batch(model, device, tensors):
    """One forward pass over a stack of preprocessed images.

    Batching matters at folder scale: a single 16-image forward pass costs far
    less than 16 one-image passes, which is the difference between a 5k-image
    folder finishing in minutes rather than tens of minutes.
    """
    x = torch.cat(tensors, dim=0).to(device)
    return torch.sigmoid(model(x)).squeeze(-1).reshape(-1).cpu().tolist()


def compute_aggregate(y_true, y_prob, fallback_threshold):
    """Dataset-level metrics + error analysis, or None when labels are unusable.

    Kept separate from the request handlers because it is needed twice: once
    for a one-shot /predict call, and once over the *whole* folder after the
    frontend has streamed every chunk through /predict.
    """
    if len(y_true) < 2:
        return None

    import numpy as np

    # Use the Youden-optimal threshold when enough labels are present,
    # otherwise fall back to the model's configured threshold.
    try:
        best_thr = optimal_threshold(y_true, y_prob)
    except Exception:
        best_thr = fallback_threshold

    metrics, y_pred = classification_metrics(y_true, y_prob, best_thr)

    y_true_arr = [int(v) for v in y_true]
    y_pred_arr = [int(v) for v in y_pred]
    fp_idx = [i for i, (t, q) in enumerate(zip(y_true_arr, y_pred_arr)) if t == 0 and q == 1]
    fn_idx = [i for i, (t, q) in enumerate(zip(y_true_arr, y_pred_arr)) if t == 1 and q == 0]

    fp_confs = [y_prob[i] for i in fp_idx]
    fn_confs = [y_prob[i] for i in fn_idx]

    return {
        "n":         len(y_true),
        "threshold": round(float(best_thr), 4),
        "metrics":   {k: round(v, 4) for k, v in metrics.items()},
        "error_analysis": {
            "false_positives":    len(fp_idx),
            "false_negatives":    len(fn_idx),
            "fp_mean_confidence": round(float(np.mean(fp_confs)), 4) if fp_confs else None,
            "fn_mean_confidence": round(float(np.mean(fn_confs)), 4) if fn_confs else None,
            # Index positions so the caller can point back at the offending rows.
            "fp_indices":         fp_idx[:50],
            "fn_indices":         fn_idx[:50],
        },
    }


# ---------------------------------------------------------------------------
# Routes - prediction
# ---------------------------------------------------------------------------
def thumbnail_bytes(img, max_side=320):
    """JPEG thumbnail bytes for an already-open image."""
    thumb = img.convert("RGB")
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.route("/model-info")
def model_info():
    """What this server is actually serving.

    Worth an endpoint rather than trusting config: BEST_WEIGHTS names a file
    that may not exist, the checkpoint decides the architecture and input
    size, and the model loads lazily on the first prediction. So report the
    live state when it is loaded, and what WOULD load when it is not.
    """
    try:
        weights = resolve_weights()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    name, image_size, ckpt_threshold = checkpoint_meta(weights)
    threshold, source = calibrated_threshold(ckpt_threshold, weights.name)

    return jsonify({
        "loaded": _model is not None,
        "weights": str(weights.relative_to(PROJECT_ROOT)
                       if PROJECT_ROOT in weights.parents else weights),
        "architecture": name or MODEL_CFG.name,
        "architecture_source": "checkpoint" if name else "config default",
        "image_size": image_size or EVAL_CFG.image_size,
        "image_size_source": "checkpoint" if image_size else "config default",
        "threshold": threshold,
        "threshold_source": source,
        "device": str(_device) if _device else "not yet selected",
        "batch_size": BATCH_SIZE,
    })


@app.route("/eval-image")
def eval_image():
    """Serve a thumbnail for a path listed in eval_report.json.

    The report records absolute paths from whichever machine ran evaluate.py,
    so the Evaluate Report tab needs a way to actually display them. Serving a
    caller-supplied filesystem path is a path-traversal risk, so the resolved
    path must sit inside this project directory AND carry an image extension.
    Anything else is refused rather than read.
    """
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "missing path"}), 400

    try:
        target = Path(raw)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        # strict=True so a non-existent path fails here rather than later.
        target = target.resolve(strict=True)
    except (OSError, ValueError):
        return jsonify({"error": "not found"}), 404

    if PROJECT_ROOT not in target.parents:
        return jsonify({"error": "path is outside the project directory"}), 403
    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "not an image file"}), 403

    try:
        with Image.open(target) as img:
            data = thumbnail_bytes(img)
    except Exception:
        return jsonify({"error": "unreadable image"}), 404

    response = app.response_class(data, mimetype="image/jpeg")
    # These files do not change during a session; let the browser keep them.
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response


@app.route("/predict", methods=["POST"])
def predict():
    """Score one chunk of uploaded images.

    Form fields:
      images     - the image files (repeated)
      paths[]    - browser-reported relative path per file, index-aligned with
                   `images`; used to infer the ground-truth label from folder
                   names such as .../real/ or .../ai/
      thumbs     - "0" to skip base-64 previews. The frontend turns these off
                   past the first few hundred images so a 10k-image folder does
                   not build a multi-hundred-MB JSON response.
      aggregate  - "0" to skip per-chunk metrics. Chunked runs pass 0 and call
                   /aggregate once at the end instead, because metrics over a
                   16-image slice say nothing about the folder.
    """
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded."}), 400

    rel_paths      = request.form.getlist("paths[]")
    want_thumbs    = request.form.get("thumbs", "1") != "0"
    want_aggregate = request.form.get("aggregate", "1") != "0"

    model, device, threshold = get_model()

    results = []   # rows in upload order
    pending = []   # (index into `results`, preprocessed tensor) awaiting a pass

    # --- Pass 1: decode + preprocess -------------------------------------
    for idx, f in enumerate(files):
        if not f or f.filename == "":
            continue

        # Prefer the browser-reported relative path: inside a folder upload it
        # is the only thing that distinguishes real/cat.jpg from ai/cat.jpg.
        rel_path = rel_paths[idx] if idx < len(rel_paths) else f.filename

        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "id":         str(uuid.uuid4())[:8],
                "filename":   f.filename,
                "image_path": rel_path,
                "error":      f"Unsupported file type '{ext}'.",
            })
            continue

        try:
            raw_bytes = f.read()
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            width, height = img.size

            row = {
                "id":             str(uuid.uuid4())[:8].upper(),
                "filename":       f.filename,
                # Deliverable field (Section 5.5): the JSON export pairs
                # image_path with pred straight out of these rows.
                "image_path":     rel_path,
                "true_label":     infer_label(rel_path),   # 0, 1, or None
                "width":          width,
                "height":         height,
                "file_size_kb":   round(len(raw_bytes) / 1024, 1),
                "threshold_used": threshold,
            }
            # Build the thumbnail now, while the decoded image is still around,
            # so the full-resolution copy can be released immediately after.
            if want_thumbs:
                row["thumbnail"] = pil_to_data_uri(img)

            pending.append((len(results), preprocess(img)))
            results.append(row)
            del img

        except Exception as exc:
            results.append({
                "id":         str(uuid.uuid4())[:8],
                "filename":   f.filename,
                "image_path": rel_path,
                "error":      str(exc),
            })

    # --- Pass 2: batched forward passes ----------------------------------
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]

        t0 = time.perf_counter()
        probs = run_batch(model, device, [t for _, t in batch])
        # Batched inference has no meaningful per-image timing; report the
        # batch's per-image share so the column stays comparable.
        per_image_ms = round((time.perf_counter() - t0) * 1000 / len(batch), 1)

        for (row_idx, _), prob_ai in zip(batch, probs):
            prob_ai = float(prob_ai)
            row = results[row_idx]
            # 6dp, not 4: this model's AI scores cluster near zero, and
            # rounding to 4dp flattens a real 0.0003 into a meaningless 0.0.
            cal = calibrated_score(prob_ai, threshold)
            row["prob_ai"]      = round(prob_ai, 6)
            row["prob_real"]    = round(1.0 - prob_ai, 6)
            # Threshold-relative view of the same score, so the number shown
            # next to the verdict agrees with it.
            row["calibrated_ai"] = round(cal, 4)
            row["verdict"]      = verdict(prob_ai, threshold)
            row["confidence"]   = confidence_label(cal)
            row["pred_label"]   = 1 if prob_ai >= threshold else 0
            row["inference_ms"] = per_image_ms
            # Correctness only means anything when ground truth is known.
            if row["true_label"] is not None:
                row["correct"] = (row["pred_label"] == row["true_label"])

    # --- Optional per-chunk metrics --------------------------------------
    aggregate = None
    if want_aggregate:
        labeled = [(r["true_label"], r["prob_ai"]) for r in results
                   if "error" not in r and r["true_label"] is not None]
        aggregate = compute_aggregate([l for l, _ in labeled],
                                      [p for _, p in labeled],
                                      threshold)

    return jsonify({"results": results, "aggregate": aggregate})


@app.route("/aggregate", methods=["POST"])
def aggregate_route():
    """Score the WHOLE job once every chunk has been predicted.

    The frontend posts back just the (true_label, prob_ai) pairs it collected
    across all chunks - a few bytes per image - so folder-level accuracy, F1,
    ROC-AUC and the error counts are computed over the entire upload rather
    than over whichever 16 images happened to share a request.

    Body: {"items": [{"true_label": 0|1, "prob_ai": 0.97}, ...]}
    """
    payload = request.get_json(silent=True) or {}
    items = payload.get("items") or []

    y_true, y_prob = [], []
    for it in items:
        label = it.get("true_label")
        prob = it.get("prob_ai")
        if label is None or prob is None:
            continue
        y_true.append(int(label))
        y_prob.append(float(prob))

    _, _, threshold = get_model()
    return jsonify({"aggregate": compute_aggregate(y_true, y_prob, threshold)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="Serve the real-vs-AI detector UI.")
    _ap.add_argument("--weights", help="Checkpoint to serve (overrides auto-detection)")
    _ap.add_argument("--threshold", type=float, help="Override the decision threshold")
    _ap.add_argument("--port", type=int, default=5001)
    _args = _ap.parse_args()

    # Feed the flags through the same env-var path the rest of the module
    # already honours, so there is one mechanism rather than two.
    if _args.weights:
        os.environ["WEIGHTS"] = _args.weights
    if _args.threshold is not None:
        os.environ["THRESHOLD"] = str(_args.threshold)

    print("Starting AI Image Detector web interface...")
    # Resolve for real rather than echoing BEST_WEIGHTS, which names a
    # per-variant path that often does not exist and is not what gets loaded.
    try:
        _w = resolve_weights()
        _name, _size, _thr = checkpoint_meta(_w)
        _t, _src = calibrated_threshold(_thr, _w.name)
        print(f"  Model weights : {_w}")
        print(f"  Architecture  : {_name or MODEL_CFG.name}"
              f"{'' if _name else ' (config default)'}")
        print(f"  Input size    : {_size or EVAL_CFG.image_size}px"
              f"{'' if _size else ' (config default)'}")
        print(f"  Threshold     : {_t:g}  ({_src})")
    except FileNotFoundError as exc:
        print(f"  Model weights : NONE FOUND - {exc}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Upload cap    : "
          f"{str(MAX_UPLOAD_MB) + ' MB per chunk' if MAX_UPLOAD_MB else 'unlimited'}")
    print(f"  Open browser  : http://127.0.0.1:{_args.port}")
    # threaded=False: one model on one device, and the frontend already sends
    # chunks in a stream - serialising requests avoids concurrent GPU access.
    app.run(debug=True, host="0.0.0.0", port=_args.port, threaded=False)
