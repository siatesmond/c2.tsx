# c2.tsx - Real vs AI Image Detector

## Project Overview
This project trains and evaluates a binary image classifier (EfficientNet backbone)
that predicts whether an image is a **real photograph** or **AI-generated**. It also
includes a fixed 15-transform **get_robustness_transforms()** (across 6 families: JPEG compression,
blur, resize, noise, color jitter, center crop) to measure how well the model holds
up under realistic image degradations (e.g. social-media re-uploads, compression,
thumbnailing).

The detector fine-tunes a pretrained EfficientNet (default: `efficientnet_b0`) with a dropout + linear classification head that outputs a single logit. A sigmoid converts this to the probability that an image is AI-generated.

Key design decisions:

- **Robustness-aware training** - augmentation at training time deliberately samples the same 6 corruption families used in evaluation (JPEG compression, Gaussian blur, downscale/upscale, Gaussian noise, color jitter, center crop), but at random severities kept away from the exact evaluation grid points. This tests generalization rather than memorization.
- **Class-imbalance handling** - `BCEWithLogitsLoss` is weighted by the real/AI ratio automatically.
- **Configurable backbone** - any TorchVision EfficientNet variant (`b0`–`b7`) can be trained; each saves to its own checkpoint so variants never overwrite each other.
- **Automatic device selection** - defaults to CUDA → MPS → CPU.

&nbsp;
## Setup

**Requirements:** Python 3.9+

1. Clone the repository and create a virtual environment:

```bash
git clone <repo-url>
cd c2.tsx
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the data directory layout:

```
data/
  train/
    real/    # real photograph images
    ai/      # AI-generated images
  val/
    real/
    ai/
  test/
    real/
    ai/
```

If you only have a `train/` split, run `splitfolder.py` to carve out a validation set (moves 25% of each class):

```bash
python splitfolder.py
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

&nbsp;

## Reproduction Steps

### 1. Train

```bash
python train.py
```

Common overrides:

```bash
python train.py --model efficientnet_b1 --epochs 40 --batch_size 64 --lr 1e-4
```

All CLI flags and their defaults:

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | TorchVision EfficientNet variant |
| `--epochs` | `30` | Maximum training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--augment_level` | `1` | 0 = no augmentation, 1 = enabled |
| `--scheduler` | `cosine_warm_restarts` | LR schedule (`cosine_warm_restarts`, `reduce_on_plateau`, `step`) |
| `--monitor` | `f1` | Validation metric for checkpointing and early stopping |
| `--patience` | `8` | Early stopping patience (epochs) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--seed` | `42` | Random seed |

The best checkpoint is saved to `models/best_model_<variant>.pth`. Training history is written to `models/best_model_<variant>.history.json`.
&nbsp;

### 2. Evaluate

```bash
python evaluate.py
```

Override the weights or data path:

```bash
python evaluate.py --weights models/best_model_efficientnet_b1.pth --threshold 0.4
```

&nbsp;

## Typical Workflow

```bash
# 1. Train a model (e.g. efficientnet_b0) 
# take note that 30 epochs may take the entire day so do 15 instead
python train.py --epochs 30 --model efficientnet_b0

# 2. Evaluate clean + robustness metrics on data/test/{real,ai}
python evaluate.py --weights models/best_model_efficientnet_b0.pth

# 3. Run inference on new images (NOT YET)
python inference.py path/to/image_or_folder --weights models/best_model_efficientnet_b0.pth
```


This runs two evaluations:

- **Clean metrics** - accuracy, precision, recall, F1, and ROC-AUC on the unmodified test set.
- **Robustness score** - mean accuracy across 15 transforms (the 6-family spec defined in `config.ROBUSTNESS_SPEC`).

A full JSON report is written next to the weights file (`eval_report.json`), including per-transform error counts and false-positive / false-negative examples.


&nbsp;
## Limitations

- **Binary classifier only.** The model outputs a single real/AI probability. It does not identify the generative model used, image regions that are AI-edited, or degrees of manipulation.
- **Distribution shift.** Performance degrades on AI generators or photography styles not represented in the training data. The robustness transforms cover common post-processing artifacts but not adversarial attacks.
- **Resolution assumption.** Images are resized to 224×224. Very low-resolution inputs or images with extreme aspect ratios may lose discriminative detail after resizing.
- **No confidence calibration.** The raw sigmoid output is not calibrated. Use threshold tuning (`--threshold`) and the error analysis section of `eval_report.json` to adjust the operating point for your use case.
- **Robustness eval is not exhaustive.** The 15-transform evaluation suite covers the 6 families in the TikTok TechJam brief. It does not cover blur types other than Gaussian, film grain, geometric distortions, or steganographic watermarking.
- **Hardware.** Training on CPU is functional but slow. A CUDA or Apple Silicon (MPS) GPU is recommended for runs beyond ~10 epochs.

&nbsp;

## `config.py`

Every other module imports its settings from here rather than hardcoding values (pretty self explanatory).

**Paths**
- `PROJECT_ROOT`, `DATA_ROOT` (`data/{train,val,test}/{real,ai}`), `RAW_ROOT`, `MODELS_DIR`.
- `weights_path(model_name)` - builds a **per-model-variant** checkpoint filename
  (e.g. `models/best_model_efficientnet_b0.pth` vs `..._b1.pth`) so training a
  different backbone never overwrites another variant's saved weights.
- `SPLIT_RATIOS = (0.7, 0.15, 0.15)` is the default train : val : test.

**Dataclasses**
- `ModelConfig` class - backbone name (`efficientnet_b0`), use ImageNet pretrained, number of output classes (1 = single-logit binary), and dropout.
- `TrainConfig` - image size, batch size, epoch count, learning rate, weight
  decay (with a per-epoch decay factor), optimizer/scheduler choice, cosine
  warm-restart parameters, plateau-scheduler parameters, early-stopping
  patience, the metric to monitor (`f1`), the decision threshold, the
  augmentation on/off switch (`augment_level`), worker count, seed, device, and
  AMP (mixed precision) toggle.
- `EvalConfig` - image size, batch size, threshold, workers, device for evaluation.
- `RobustnessConfig` - legacy `severity` knob (kept for API compatibility) and a
  `weight_by_severity` flag (default `False`, meaning the robustness score is a
  plain unweighted mean over the 15 fixed transforms).
- `TrainAugConfig` - controls the **on-the-fly training augmentation sampler**
  (used by `augmentations.py`):
  - `num_transforms_weights` - probability of applying 0, 1, or 2 corruption
    families to a given image (defaults: 15% none, 65% single, 20% compound).
  - `generalization_prob` - probability of instead drawing a "generalization-only"
    transform (slight rotation or grayscale) that is **not** one of the evaluated
    families.
  - Per-family severity *ranges* used at training time (JPEG quality, blur sigma,
    resize scale, noise sigma, color factor, crop fraction). These ranges span
    (and slightly exceed) the fixed evaluation grid, but individual draws are
    pushed away from the exact evaluation points (see `augmentations.py`).

**Fixed robustness spec**
- `ROBUSTNESS_SPEC` - human-readable description of the 6 families and their
  exact parameter levels (15 transforms total), matching the evaluation brief.
- `EVAL_SEVERITY_POINTS` - the same information as a dict keyed by short family
  name (`jpeg`, `blur`, `resize`, `noise`, `color`, `crop`), used only to keep
  training-time random severities away from these exact values so the
  robustness score reflects genuine generalization rather than memorized
  eval parameters.

**Singletons**
- `MODEL_CFG`, `TRAIN_CFG`, `EVAL_CFG`, `ROBUSTNESS_CFG`, `TRAIN_AUG_CFG` - imported everywhere.
- `BEST_WEIGHTS` - default checkpoint path for the current configured model variant (b0, b1, ...), used as the default `--weights` argument in `evaluate.py` and `inference.py`.

&nbsp;

## `augmentations.py` - Image Transforms & Augmentation Sampling

Two distinct roles:

### 1. Core image transform primitives
Reusable PIL/NumPy-based functions, each simulating a realistic real-world
degradation:
- `jpeg_compress(img, quality)` - simulates messaging/social-media re-upload compression.
- `gaussian_blur(img, sigma)` - stmulates out-of-focus cameras.
- `resize_down_up(img, scale)` - downscale -> upscale (thumbnail generation artifacts).
- `gaussian_noise(img, sigma)` - add Gaussian noise to simulate low light sensor noise.
- `color_jitter(img, factor)` - shift brightness/contrast/saturation by factor (simulate auto-enhance filters).
- `center_crop(img, frac)` - crops to the central fraction (simulates profile-picture
  framing).
- `slight_rotate(img, degrees)` - small rotation with fill color (simulates rotation).
- `grayscale(img)` - simulate reshares that strip color or greyscale filter.

### 2. `get_robustness_transforms()` - The 15-Transform 
Returns the exact, unchanging 15-transform dictionary used by `evaluate.py` to compute the robustness score:
- JPEG quality ∈ {90, 70, 50, 30} (4 entries)
- Blur sigma ∈ {0.5, 1.0, 2.0} (3 entries)
- Resize scale ∈ {0.5, 0.25} (2 entries)
- Noise sigma ∈ {0.02, 0.05, 0.10} (3 entries)
- Color jitter ∈ {+20%, −20%} (2 entries)
- Center crop @ 80% (1 entry)

**do-not-change** this benchmark that model variants are compared against.

### 3. Training-time augmentation sampler
`apply_training_augmentation(img, rng, aug_cfg)` implement strategies:

1. **Same 6 families as the eval set** (`FAMILY_FUNCS` / `FAMILIES`: jpeg, blur,
   resize, noise, color, crop) so training exposure roughly matches the
   evaluation space.
2. **Random severity per family per image** rather than one fixed value, drawn
   via `_draw_severity()`, which samples uniformly within the ranges defined in
   `TrainAugConfig` for a fair broad coverage.
3. **Avoids the exact evaluation grid points.** `_sample_away_from_grid()`
   repeatedly re-samples (up to 25 tries) until the drawn value is at least
   `margin` away from every exact point in `EVAL_SEVERITY_POINTS` for that
   family. This is the key anti-leakage mechanism: it prevents the model from
   ever training on the literal parameter values used for scoring, so the
   robustness score measures genuine generalization rather than memorization
   of the eval harness's specific settings.
4. **A separate, non-overlapping "generalization" pool.** With probability
   `generalization_prob`, instead of a robustness-family transform, one of
   `GENERALIZATION_POOL` (small random rotation ±15°, or grayscale) is applied.
   These add useful invariances but can **never** touch the evaluated
   parameter space, so they can't inflate or deflate the robustness score in
   either direction. Horizontal/vertical flips are intentionally *excluded*
   from this pool - for AI-detection, image orientation can itself carry
   useful signal (e.g. asymmetric artifacts or mirrored text), so the model is
   deliberately not made invariant to it.

`get_augment_names()` returns families + generalization pool for logging purposes.

&nbsp;

## `datasets.py` - PyTorch Dataset

`RealAIDataset(Dataset)` loads images from `data/{split}/{real,ai,full_synthetic_part2}/`
and returns `(image_tensor, label)` pairs. 
**image tensor: array representing an image as numbers

- **Label encoding:** `real` → 0, anything else (`ai`, `full_synthetic_part2`) → 1.
- **Folder scanning:** walks each class folder, keeping only recognized image
  extensions (`.jpg .jpeg .png .bmp .webp`). Any leftover pre-generated
  augmentation files from an older static-augmentation pipeline (named with
  `_l1_` or `_l2_` in the stem) are explicitly excluded so they can't
  accidentally be double-counted as extra samples - augmentation is now
  generated fresh on-the-fly every epoch instead.
- **Augmentation:** only applied when `split == "train"` and `augment_level > 0`;
  delegates the actual transform choice/severity to
  `augmentations.apply_training_augmentation`, driven by a seeded
  `random.Random` instance for per-run determinism.
- **Base transform** (`build_base_transform`): resize to `image_size`² → tensor →
  ImageNet mean/std normalization. Applied to every image regardless of split.
- **Robust loading:** `__getitem__` retries up to 10 times on a different random
  index if an image fails to load (corrupt file, etc.), and falls back to a
  blank black image with label 0 if all retries fail - so a single bad file
  can't crash a training run.

&nbsp;

## `model.py` - Model Definition

`EfficientNetDetector(nn.Module)` wraps a TorchVision EfficientNet backbone
(default `efficientnet_b0`, configurable via `MODEL_CFG.name`) and swaps out its
classifier head with `Dropout → Linear(in_features, num_classes)`, where
`num_classes=1` produces a single logit for binary real-vs-AI classification.

"EfficientNet backbone with a binary real-vs-AI head"

- `build_model(cfg, device)` - constructs the model and (if a device is given) moves it there.
- `load_best_weights(model, path, device)` - loads a checkpoint saved by `train.py`. Accepts either the full checkpoint dict (`{"model_state": ...}`) or a raw state_dict directly, then sets the model to `eval()` mode for inference.
- `predict_proba(x)` - inference helper that return sigmoid probabilities directly.
- Unsupported model names (anything not present in `torchvision.models`) raise a clear `ValueError` instead of a cryptic `AttributeError`.

&nbsp;

## `train.py` - Training Loop

**Setup**
- Resolves device via `resolve_device` (Pick the compute device. "auto" prefers the fastest available backend).
- Derives the checkpoint output path from the *current* model variant via `weights_path(MODEL_CFG.name)`, so training `efficientnet_b0` and later `efficientnet_b1` never clobber each other's saved weights.
- Seeds `torch`, `numpy`, and `random` for reproducibility.

**Data / model / optimization**
- `build_loaders(cfg)` constructs train/val `RealAIDataset`s and wraps them in
  `DataLoader`s (train shuffles + drops last partial batch; val does not
  shuffle). Augmentation is only ever applied to the training loader.
- `compute_pos_weight(train_ds)` computes `#negatives / #positives` for the
  positive ("AI") class to counter class imbalance, fed into
  `nn.BCEWithLogitsLoss(pos_weight=...)` (numerically safer than manual
  sigmoid + BCE).
- `AdamW` optimizer; `make_scheduler` builds one of three LR schedules based on
  `cfg.scheduler`: `cosine_warm_restarts` (`CosineAnnealingWarmRestarts`),
  `reduce_on_plateau` (`ReduceLROnPlateau`, tracks the monitored metric), or
  `step` (`StepLR`).
- `GradScaler` is enabled only when AMP is requested **and** running on CUDA
  (mixed precision has no benefit on CPU/MPS here).

**Per-epoch loop**
- `train_epoch()` - standard train step; under AMP, uses
  `torch.cuda.amp.autocast()` + gradient scaling; otherwise full precision.
  Returns the loss averaged over the whole dataset.
- `validation_confidence()` - like a standard eval pass but also tracks mean
  predicted probability ("confidence_score") and separately the mean
  confidence on correct vs. incorrect predictions - a simple calibration
  diagnostic (e.g. is the model overconfident when wrong?).
- Steps the scheduler (`ReduceLROnPlateau` needs the monitored value passed in;
  others step unconditionally).
- Optionally decays `weight_decay` each epoch by `cfg.wd_decay_factor` (extra
  regularization schedule on top of the LR schedule).
- **Checkpointing:** saves `{"model_state", "model_name", "config", "epoch",
  "metrics"}` to the variant-specific weights path whenever the monitored
  metric (`cfg.monitor`, default `f1`) improves.
- **Early stopping:** halts training after `cfg.early_stopping_patience` epochs
  with no improvement.
- Writes the full per-epoch history to `<weights>.history.json` at the end.

**CLI** (`parse_args`): epochs, batch size, LR, augment level, scheduler,
monitor metric, patience, device, workers, seed, and `--model` (selects the
EfficientNet variant, e.g. `b0`/`b1`/`b2` - each saved to its own checkpoint).

&nbsp;

## `evaluate.py` - Clean Metrics + Robustness Evaluation

The main evaluation entry point (`python evaluate.py [args]`), run against
`data/test/{real,ai}`.

- `TestImageDataset` - loads test images with the standard (non-augmented)
  preprocessing pipeline, also excluding stale `_l1_`/`_l2_` files.
- `predict_probs()` - runs the model over a loader and collects probabilities,
  labels, and file paths.
- **Clean metrics:** `metrics.classification_metrics()` computes accuracy,
  precision, recall, F1, and ROC-AUC on the unmodified test set.
- **Robustness evaluation:** `evaluate_robustness()` applies each of the 15
  fixed transforms from `get_robustness_transforms()` (via `_TransformedView`,
  which re-loads and re-transforms each image on the fly) and records per-transform
  accuracy and per-transform error counts.
- `metrics.robustness_score()` aggregates the 15 per-transform accuracies into
  a single score - by default the plain mean (`weight_by_severity=False`).
- Results are also grouped **by family** (`robustness_by_family`, e.g. mean
  accuracy across all JPEG-quality levels) using the category labels from
  `ROBUSTNESS_SPEC`/`get_robustness_transforms`.
- **Error analysis:** `error_analysis()` breaks down false positives
  (real misclassified as AI) vs. false negatives (AI misclassified as real),
  their mean confidence scores, and up to 20 example paths of each; `_interpret()`
  generates a short natural-language summary of which error mode dominates.
- Writes a full JSON report (`eval_report.json`, saved next to the weights
  file) containing clean metrics, robustness score, per-family robustness,
  error analysis, and per-transform error counts. Also prints a
  human-readable summary to the console, including the 3 weakest transforms.
- **CLI:** `--weights`, `--data_root`, `--threshold`, `--severity` (legacy
  robustness severity knob), `--device`.

&nbsp;

## `inference.py` - Single-Image / Batch Prediction

The deployment-facing script for running the trained detector on new images
(`python inference.py <file_or_folder> [args]`).

- `load_image_tensor()` - applies the exact same preprocessing used during
  training/eval (resize → tensor → ImageNet normalization).
- `predict()` - runs the model over a list of paths and returns, for each
  image: `image_path`, `pred` (the AI-likelihood probability, rounded to 4
  decimals - the required deliverable field), plus convenience fields
  `label` (`"ai"`/`"real"`), `is_ai` (bool), and the `threshold` used.
- `gather_paths()` - accepts either a single image file or a directory
  (in which case all recognized image files inside are collected and sorted).
- Always writes a JSON predictions file (default `predictions.json`) containing
  `image_path` and `pred` for every image, per the deliverable spec, regardless
  of the `--quiet` flag. `--quiet` only suppresses the per-image console log.
- **CLI:** positional `input` (file or folder), `--weights`, `--threshold`,
  `--image_size`, `--device`, `--output`, `--quiet`.

&nbsp;

## `metrics.py` - Metric Computation

Small, dependency-light module used by both `train.py` and `evaluate.py`:

- `classification_metrics(y_true, y_prob, threshold)` - thresholds probabilities
  into hard predictions, then computes accuracy, precision, recall, and F1
  (all with `zero_division=0` to avoid crashes on degenerate batches), plus
  ROC-AUC (only when both classes are present in `y_true`; otherwise `NaN`).
  Returns both the metrics dict and the hard predictions array.
- `robustness_score(per_transform_acc, weight_by_severity, severity_weights)` -
  aggregates a dict of `{transform_name: accuracy}` into a single robustness
  score. By default this is an unweighted mean across all transforms; if
  `weight_by_severity=True` and `severity_weights` are supplied, a weighted
  mean is computed instead. Also reports mean/min/max accuracy and echoes back
  the full per-transform breakdown.



&nbsp;

## Web Interface (`app.py`)

A browser-based frontend with two tabs:

- **Analyze Images** - upload images and get per-image AI detection results
- **Evaluate Report** - view and download the full evaluation report produced by `evaluate.py`

Routes:

| Route | Purpose |
|---|---|
| `GET /` | The UI |
| `POST /predict` | Scores **one chunk** of uploaded images; stateless per chunk |
| `POST /aggregate` | Folder-level metrics once every chunk has been scored |
| `GET /eval-report` | The `eval_report.json` written by `evaluate.py` |
| `GET /eval-report/exists` | Cheap existence check for the above |

Metrics are deliberately **not** computed per chunk - accuracy over whichever 16 images shared a request says nothing about the folder. The frontend collects the `(true_label, prob_ai)` pairs across every chunk and posts them to `/aggregate` once at the end, so accuracy, F1, ROC-AUC, the Youden-optimal threshold and the false positive/negative counts are all computed over the entire upload.

### Setup

```bash
pip install -r requirements.txt
```

### Running

```bash
python app.py
```

Then open **http://127.0.0.1:5001** in your browser.

---

### Tab 1 - Analyze Images

#### Input

The interface accepts images three ways:

- **Browse Files** - standard file picker; hold `Ctrl`/`Cmd` to select multiple images at once.
- **Browse Folder** - folder picker that automatically includes every image inside, including nested subfolders.
- **Drag & drop** - drag image files or an entire folder onto the drop zone. The drop handler walks the directory tree recursively via the browser's `FileSystemEntry` API.

All three methods feed into the same queue. Duplicates are detected by **relative path + size**, not by filename alone - a dataset folder normally holds both `real/0001.jpg` and `ai/0001.jpg`, and keying on the name would drop one of every such pair. With 8 or fewer files queued the individual filenames are shown as removable chips; with more than 8 a compact summary bar is shown with the image count, total size, batch count and a "Clear all" button.

Click **Analyze Images** to submit. Rows stream into the table as each batch comes back.

#### Folder size

There is **no limit on how many images (or how many bytes) a folder may contain**. The selection is never posted in one request: the frontend slices it into chunks and posts them one after another, so peak memory on both sides is bounded by a single chunk rather than by the folder.

| Knob | Where | Default | Meaning |
|---|---|---|---|
| `CHUNK_FILES` | `templates/index.html` | 16 | Images per request |
| `CHUNK_BYTES` | `templates/index.html` | 24 MB | Byte ceiling per request, whichever limit is hit first |
| `THUMB_LIMIT` | `templates/index.html` | 300 | Previews are requested only for the first N images; past that the response carries no base-64 payload |
| `TABLE_LIMIT` | `templates/index.html` | 200 | Rows painted into the DOM at a time, extended by a **Show more** button |
| `PREDICT_BATCH_SIZE` | env var, `app.py` | 16 | Images stacked into one forward pass |
| `MAX_UPLOAD_MB` | env var, `app.py` | 0 (off) | Per-request size cap. Caps one chunk, never the job |

A progress bar reports `done / total` across the whole run, with a **Cancel** button that stops after the current chunk and keeps whatever has already been scored. A chunk that fails is recorded as an error against its own files and the run continues.

#### Summary strip

The strip above the table always shows four base cards:

| Card | Description |
|---|---|
| Analyzed | Total images successfully processed |
| AI-Generated | Count the model predicted as AI |
| Real / Human | Count the model predicted as real |
| Threshold | Decision cutoff. Auto-tuned to the ROC-optimal value (Youden's J) when labels are present; otherwise the calibrated threshold loaded at startup (see **Calibration** below) |

A second row of metric cards appears **only when ground-truth labels are detected** (images loaded from a folder named `ai/` or `real/`):

| Card | Description |
|---|---|
| Accuracy | Percentage of all images the model got right |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | How well the model ranks AI images above real ones across all thresholds. 1.0 = perfect, 0.5 = random |
| Precision | Of all images called AI, the fraction that actually were |
| Recall | Of all actual AI images, the fraction the model caught |
| False Positives | Real images misclassified as AI, with mean confidence shown |
| False Negatives | AI images misclassified as real, with mean confidence shown |

#### Per-image table

| Column | Description |
|---|---|
| # | Row index |
| Image ID | Short random hex ID assigned at inference time |
| Preview | 200 px thumbnail of the uploaded image, for the first `THUMB_LIMIT` images; `—` beyond that |
| Filename | Original filename |
| Verdict | AI-Generated or Real / Human |
| True Label | *(label mode only)* Ground truth inferred from folder name |
| Correct? | *(label mode only)* ✓ Yes (green) or ✗ No (red). Rows are tinted accordingly |
| AI Score | **Raw** model output — the exact value exported as `pred` in predictions.json |
| vs Threshold | The same score measured against the decision threshold, where **50% sits exactly on the boundary**. This is the number that agrees with the verdict |
| Confidence | Distance from the **decision threshold** (not from 50%) - **Very High** (≥ 90%), **High** (≥ 75%), **Moderate** (≥ 60%), **Low** |
| Dimensions | Original image width × height in pixels |
| File Size | Upload size in KB |
| Inference | Model time per image (ms) - the batch's total divided by its size, since images are scored in batches |

Two download buttons sit in the results header:

- **predictions.json** - the Section 5.5 deliverable exactly: a JSON array of `{ image_path, pred }`, one entry per scored image, where `pred` is the AI-generated likelihood in `[0, 1]`. This is the same schema `inference.py` writes from the command line.
- **Full report** - everything else: every per-image field plus the folder-level `aggregate` block (metrics + error analysis), as a timestamped JSON file. Thumbnails are stripped to keep the file small.

```json
[
  { "image_path": "dataset/real/0000.jpg", "pred": 0.0527 },
  { "image_path": "dataset/ai/0000.jpg",   "pred": 0.9412 }
]
```

#### Calibration

**The raw score is not a probability, and 0.5 is the wrong cutoff for this model.**

`realworld_eval_2k.json` (4,000 images) records a ROC-AUC of **0.972** alongside a ROC-optimal
threshold of **0.0019**. The model ranks AI above real very well, but its outputs are compressed
against zero — even the real images it finds *most* AI-looking only reach a score of 0.063. Judged
at 0.5, essentially every image on earth is classified "Real / Human".

So `app.py` does not hardcode 0.5. At startup it loads a threshold that was actually fitted to this
model's outputs, searching in order:

1. the `THRESHOLD` environment variable, if set
2. `realworld_eval_2k.json` / `realworld_eval.json` (written by `test_realworld.py`)
3. `eval_report.json` (written by `evaluate.py`)
4. `EVAL_CFG.threshold` — the uncalibrated 0.5 fallback, used only when no report exists

The chosen value and its source are printed on startup. To experiment without editing anything:

```bash
THRESHOLD=0.01 python app.py
```

Because the raw scores span orders of magnitude, the UI also shows a **vs Threshold** column: the
same score remapped in log space so that 50% sits on the decision boundary, saturating three decades
out. Without it a genuine AI detection reads as "AI score 0.28% / Real score 99.7%", which looks
like the app contradicting itself. **Confidence** is likewise measured from the threshold, not from
0.5 — otherwise every image scores "Very High" purely because every raw score is far below 0.5.

The raw value is still what `predictions.json` exports, unchanged.

#### Label detection

Labels are inferred from the folder name in the file's relative path - `webkitRelativePath` for the folder picker, the walked `FileSystemEntry.fullPath` for drag-and-drop:

- `ai`, `ai_generated`, `fake`, `synthetic` → label 1 (AI)
- `real`, `human`, `authentic`, `genuine` → label 0 (Real)
- Anything else or individually picked files → no label, metrics strip hidden

Dropping your existing `data/test/ai/` and `data/test/real/` folders directly onto the drop zone will automatically produce the full metrics report.

---

### Tab 2 - Evaluate Report

```bash
python evaluate.py
```

Refresh the page after running `evaluate.py`.

The report is divided into four sections:

**Score banner**
The headline final score (`0.5 × Clean AUC + 0.5 × Robust AUC`) shown large, alongside clean ROC-AUC, robustness score, and the Youden's J optimal threshold.

**Clean test metrics**
Accuracy, precision, recall, F1, and ROC-AUC from the unmodified test set, displayed as cards.

**Robustness**
Two side-by-side panels:
- *By family* - one bar per transform family (JPEG, blur, noise, resize, color, crop), sorted worst-first
- *Per transform* - all 15 individual transforms sorted worst-first

Bars are colour-coded: green ≥ 80%, amber 60–80%, red < 60%. The three weakest transforms are called out in a red highlight box below.

**Error analysis**
False positive and false negative counts with mean model confidence on each error type, the plain-English interpretation text from `evaluate.py`, and collapsible lists of up to 20 example filenames per error type.

A **Download JSON** button downloads the raw `eval_report.json` file. A **Reload** button re-fetches the file from disk without refreshing the page (useful after re-running `evaluate.py`).

---

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the UI |
| `POST` | `/predict` | Runs inference on uploaded images |
| `GET` | `/eval-report` | Returns `eval_report.json` as JSON (404 if not found) |
| `GET` | `/eval-report/exists` | Returns `{"exists": true/false}` - lightweight file presence check used on page load |

`POST /predict` - request and response shapes:

- **Request:** `multipart/form-data` with files under `images` and matching relative paths under `paths[]`. Accepted: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Max: 500 MB.
- **Response:** `{ "results": [...], "aggregate": {...} | null }`

`results` - one object per file:

```json
{
  "id": "A3F2C1B0",
  "filename": "photo.jpg",
  "thumbnail": "data:image/jpeg;base64,…",
  "prob_ai": 0.9123,
  "prob_real": 0.0877,
  "verdict": "AI-Generated",
  "confidence": "Very High",
  "pred_label": 1,
  "true_label": 1,
  "correct": true,
  "width": 1024,
  "height": 768,
  "file_size_kb": 214.5,
  "inference_ms": 38.2,
  "threshold_used": 0.5
}
```

`aggregate` - present only when at least two labelled images were submitted:

```json
{
  "n": 40,
  "threshold": 0.002,
  "metrics": {
    "accuracy": 0.8872,
    "precision": 0.8901,
    "recall": 0.8875,
    "f1": 0.8887,
    "roc_auc": 0.9564
  },
  "error_analysis": {
    "false_positives": 12,
    "false_negatives": 10,
    "fp_mean_confidence": 0.7341,
    "fn_mean_confidence": 0.3102
  }
}
```

On a per-file error the result object contains `"error": "<message>"` instead of the prediction fields.
On a per-file error (unsupported type, corrupt image, etc.) the result object contains `"error": "<message>"` instead of the prediction fields.