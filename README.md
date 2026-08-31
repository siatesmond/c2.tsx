# c2.tsx — Robust Real vs. AI Image Detector

## 1. Project Overview

This project detects **AI-generated images** vs. **authentic photographs**, with an
explicit focus on staying accurate after the kind of post-processing an image
picks up when shared online — JPEG re-compression, blur, thumbnail rescaling,
sensor noise, colour filters and cropping. A fixed 15-transform robustness suite
(6 families: JPEG compression, blur, resize, noise, colour jitter, centre crop)
scores how well a model holds up under those degradations.

The final model, **`hybrid_effb1`**, is a two-branch detector:

- **Semantic branch — frozen CLIP ViT-B/32.** A pretrained CLIP vision encoder,
  never fine-tuned. It contributes a fixed "does this look like a plausible real
  scene" prior. Because it does not train, it cannot memorise a generator's
  fingerprint or a dataset's capture style, and its features are largely
  invariant to compression, blur, cropping and colour shifts — so this branch
  keeps accuracy up when an image is degraded.
- **Low-level branch — two fused sub-streams:**
  - *Spatial:* a fixed high-pass residual (`image − gaussian_blur(image)`) fed to
    a CNN. The high-pass step removes scene layout, colour cast and exposure —
    easy dataset shortcuts — leaving texture and noise. For `hybrid_effb1` this
    CNN is a pretrained **EfficientNet-B1** feature extractor; the lighter
    `hybrid_clip` variant uses a ~0.3 M-param CNN.
  - *Frequency:* the image is split into a 7×7 grid of patches, each patch's 2D
    FFT log-magnitude spectrum is reassembled into a single-channel spectrogram
    image, and a small CNN reads it — to catch the periodic upsampling /
    checkerboard artifacts common to generative models.
- **Fusion head.** `[CLIP projection ‖ low-level vector] → MLP → 1 logit →
  sigmoid → P(AI-generated)`. Only ~7 M parameters train; CLIP's ~87 M stay
  frozen. The head learns to lean on the low-level branch when an image is clean
  and fall back on CLIP semantics when it is not.

The hybrid input is a 4-channel tensor (RGB in `[0, 1]` + the FFT spectrogram
channel). A plain `EfficientNetDetector` baseline (`--model efficientnet_b0/b1/b2`)
is also included and takes an ordinary ImageNet-normalised 3-channel tensor.

**Key design decisions**

- **Robustness-aware training:** augmentation samples the same 6 corruption
  families used in evaluation, **at most one per image, never stacked**, at
  random severities kept off the exact evaluation grid points — measuring
  generalisation, not memorisation of the scoring harness.
- **Data-level anti-leakage:** the demonstration set (COCO val2017 + DALL·E
  Advanced) is a subset of WildFake, so `holdout_index.py` indexes those images
  by content hash *and* perceptual hash and `prepare_wildfake.py` drops any
  training candidate that matches.
- **Threshold-independent model selection:** checkpointing / early stopping
  monitor **ROC-AUC**, not F1.
- Per-variant checkpoints (`models/best_model_<name>.pth`); automatic device
  selection (CUDA → MPS → CPU); class-imbalance-weighted `BCEWithLogitsLoss`.

## 2. Setup and Installation

**Requirements:** Python 3.10+, PyTorch ≥ 2.0 built for your accelerator.

```bash
git clone <repo-url>
cd c2.tsx
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

- **GPU:** `pip install torch` may pull a CPU-only wheel. For an NVIDIA GPU
  install a CUDA build, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
- **First hybrid run** downloads CLIP ViT-B/32 weights (~600 MB) to the Hugging
  Face cache — needs internet once. The EfficientNet-only path needs neither
  `transformers` nor internet.

**Data layout**:

```
data/
  train/{real, ai}      # any folder other than real/ is treated as label 1
  val/{real, ai}
  test/{real, ai}
```

Supported formats: `.jpg .jpeg .png .bmp .webp`.

## 3. Steps to Reproduce

### 3.1 Prepare the data

```bash
# 1. Index the benchmark images so they can never leak into training
python holdout_index.py --add path/to/coco_val2017 --add path/to/dalle_advanced \
                        --output holdout_index.json

# 2. Download the WildFake training subset, filtered against the index (DO NOT USE COCO/DALLE IMAGES FOR TRAINING)
Link for dataset:
• https://modelscope.cn/datasets/hy2628982280/WildFake/file/view/master/Images%2FDiffusion_based%2FDALLE.zip?id=72967&status=2
• https://modelscope.cn/datasets/hy2628982280/WildFake/file/view/master/Images%2FReal%2Fcoco.zip?id=72967&status=2

python prepare_wildfake.py --inspect                             # check the schema first
python prepare_wildfake.py --exclude-index holdout_index.json    # into data/<split>/{real,ai}

# 3. Download HuggingFace & CIFAKE data

Link for training dataset (place synthetic images into the 'ai' folder & place real images into the 'real' folder):
• https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download
• https://drive.google.com/file/d/15xlLIHWfBbzj0QnKH5Co5v9Dvzv_rc-N/view?usp=sharing
• https://drive.google.com/file/d/1EvPmNcMDoUCiJdLe2aXad9xrLBRVSiVz/view?usp=sharing
• https://drive.google.com/file/d/1GdTLnujfVGBqvx8AjZ3zdrKVXDJrldsF/view?usp=sharing

Link for validation dataset (place ai images into the 'ai' folder & place real images into the 'real' folder): 
• https://drive.google.com/drive/folders/1sFZxSrDibjpvzTrHeNVS1fTf-qIue744

Link for testing dataset (place ai images into the 'ai' folder & place real images into the 'real' folder): 
• https://drive.usercontent.google.com/download?id=1_ivsEV5e14efuv93tJgXjWOondYnEC2G&export=download&authuser=0

# 4. Split the data
python splitfolder.py                                            # run this to move 25% of each class into the respective folders

```
<img width="281" height="200" alt="image" src="https://github.com/user-attachments/assets/d11dcced-a2d9-439a-a26d-8f3701cffdf4" />

This is how the dataset should be organised.

### 3.2 Train

```bash
python train.py --model hybrid_effb1 --device cuda --batch_size 16 --robust_eval_every 5 --no_prompt
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | `efficientnet_b0…b7`, or `hybrid_clip` / `hybrid_effb0` / `hybrid_effb1` |
| `--epochs` | `8` | Max training epochs |
| `--batch_size` | `32` | `hybrid_effb1` fits at 16 on a 4 GB GPU |
| `--lr` | `3e-4` | Learning rate |
| `--augment_level` | `1` | 0 = off, 1 = on-the-fly augmentation |
| `--scheduler` | `cosine_warm_restarts` | `cosine_warm_restarts` / `reduce_on_plateau` / `step` |
| `--monitor` | `roc_auc` | Validation metric for checkpointing / early stopping |
| `--patience` | `8` | Early-stopping patience (epochs) |
| `--robust_eval_every` | `1` | Run the 15-transform validation robustness eval every N epochs |
| `--no_prompt` | off | Skip the interactive "continue?" prompt after each epoch |
| `--data_root` | `data` | Dataset root (point at a subset for a smoke test) |
| `--limit` | none | Cap training samples for a fast end-to-end run |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |

Best checkpoint → `models/best_model_<variant>.pth`; per-epoch history →
`models/best_model_<variant>.history.json`. Only trainable parameters are
optimised (frozen CLIP excluded).

### 3.3 Evaluate (clean metrics + robustness)

```bash
python evaluate.py --model hybrid_effb1 --device cuda
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | Must match the checkpoint |
| `--weights` | auto | `models/best_model_<model>.pth` |
| `--data_root` | `data` | Test data at `<data_root>/test/{real,ai}` |
| `--threshold` | ROC-optimal | Force a fixed threshold; default is Youden's J on the clean test set |
| `--limit` | none | Evaluate on a balanced random subset |
| `--severity` | `0.5` | Legacy knob; the 15-transform set is fixed |
| `--device` | `auto` | |

Writes `eval_report.json` next to the weights: clean metrics, per-transform and
per-family robustness **ROC-AUC**, `final_score` (`0.5·clean_AUC + 0.5·robust_AUC`),
error analysis, per-transform error counts; prints a summary + the 3 weakest
transforms.

### 3.4 Cross-source generalisation test

```bash
python test_realworld.py --root . --model hybrid_effb1 --device cuda \
                         --max_per_class 2001 --output crosseval_effb1.json
```

Points at any folder holding `real/` and `ai/` sub-folders (for held-out sets
from other sources). Flags: `--root`, `--model`, `--weights`, `--threshold`,
`--batch_size`, `--max_per_class`, `--device`, `--output`.

### 3.5 Inference

```bash
python inference.py path/to/image_or_folder --model hybrid_effb1 --output predictions.json
```

Writes a JSON array of `{image_path, pred}` per image, where `pred` is P(AI) to 4
decimals. Flags: positional `input`, `--model`, `--weights`, `--threshold`
(`0.5`), `--image_size` (`224`), `--device`, `--output` (`predictions.json`),
`--quiet`.

## 4. Limitations & What We'd Improve

**Limitations**

- **CIFAKE shortcut.** CIFAKE separates real (CIFAR-10 upscaled) from fake (SD @
  32 px upscaled) largely via a low-frequency upscaling artifact, so high CIFAKE
  scores partly reflect that rather than genuine AIGC detection.
- **In-distribution robustness only.** `data/test` is a held-out CIFAKE split —
  same generator and real source as most of training. The robustness table
  measures transform-robustness on one distribution, not cross-generator
  robustness.
- **Cross-source test is only partly held out.** The set mixes WildFake sources
  overlapping training with novel DALL·E content, and we lack per-source labels
  to isolate the novel portion — suggestive, not conclusive. No run on the
  official COCO/DALL·E benchmark; no leave-one-generator-out.
- **Narrow generator coverage.** AI training images = SD (CIFAKE) + WildFake's
  DDIM/DDPM — no GAN, autoregressive, Midjourney, or SD-variant data. Real
  sources skew to faces/objects. SID_Set was evaluated but not integrated.
- **Miscalibration under domain shift.** The ROC-optimal decision threshold
  drops from ~0.65 in-distribution to ~0.003 cross-source (scores compress
  toward 0). ROC-AUC is stable, but a fixed threshold fails; no temperature
  scaling is implemented (`app.py` works around it by loading a fitted
  threshold).
- **The frequency branch is fragile where it is needed** — blur/resize/JPEG
  destroy the high-frequency cues it uses, so under the worst corruptions
  robustness rests on the frozen CLIP branch.
- **Binary output only** — no generator ID, no localisation of edited regions.
- **Compute-constrained.** Single 4 GB GPU: batch 16, subsampled training, 8
  epochs (loss still falling at cut-off), single run per model, no
  hyperparameter search — reported numbers are a lower bound.

**Given more time**

1. Per-source AUC on the cross-source set; leave-one-generator-out.
2. Broader training generators (GAN, Midjourney, SD variants, autoregressive)
3. Temperature scaling / per-domain threshold calibration.
4. Full-dataset training, more epochs, multiple seeds, a hyperparameter sweep.
5. Evaluate on the official COCO/DALL·E benchmark.

## 5. Team Member Contributions

Claire: Set up CLIPS + Low Frequency Patch & Perform Hybrid (CLIPS integrated with EfficientNet B1) training + evaluation (including cross evaluation)

Clarice: Performed EfficientNet B0 Baseline training + evaluation (including cross evaluation) & wrote README

Sabrina: Perform Testing & analyse outputs & wrote README & recorded DEMO video

Tesmond: Perform EfficientNet B2 training + evaluation (including cross evaluation) & analyse outputs

Xin Yin: Set up base code & perform EfficientNet B0 training + evaluation (including cross evaluation)

---

# Additional Information

## Development Tools, Models, Libraries & Datasets

**Tools:** VS Code, Python virtualenv, Git/GitHub. Training on a single consumer
NVIDIA GPU (GTX 1650 SUPER, 4 GB) and Apple-Silicon MPS.

**Models / APIs:** CLIP `openai/clip-vit-base-patch32` (frozen vision encoder)
via Hugging Face `transformers`; TorchVision EfficientNet-B0/B1/B2
(ImageNet-pretrained).

**Libraries:** PyTorch, TorchVision, Hugging Face Transformers, scikit-learn,
NumPy, Pillow, pandas, tqdm, Matplotlib, Flask.

**Datasets:**
- **CIFAKE** (Kaggle) — CIFAR-10 upscaled vs. Stable Diffusion @ 32 px upscaled.
- **WildFake** (ModelScope `hy2628982280/WildFake`) — a capped, balanced subset:
  real from ImageNet / LSUN-Church / FFHQ / AFHQ / CelebA-HQ; AI from DDIM and
  DDPM. **COCO and DALL·E partitions excluded** (the benchmark is drawn from
  them).
- **SID_Set** (Hugging Face)

## Results Summary

ROC-AUC unless noted. Baselines trained on **CIFAKE only**; hybrids on **CIFAKE +
WildFake** — not one progression.

**EfficientNet baselines (CIFAKE test, 8 k subset)**

| | Clean AUC | Robustness mean AUC | Final score |
|---|---|---|---|
| B0 | 0.9977 | 0.9906 | 0.9942 |
| **B1** | **0.9988** | **0.9954** | **0.9971** |
| B2 | 0.9985 | 0.9954 | 0.9969 |

B0 underfits robustness; B2 adds parameters over B1 for no gain and worse
calibration. **B1 is the sweet spot** — hence its use as the hybrid spatial
backbone.

**Hybrid models**

| | CIFAKE test AUC | CIFAKE robustness mean AUC | Final score | FP / FN | Cross-source AUC |
|---|---|---|---|---|---|
| `hybrid_clip` | 0.9935 | 0.9775 | 0.9855 | 382 / 389 | 0.9773 |
| **`hybrid_effb1`** | **0.9961** | **0.9840** | **0.9901** | **229 / 355** | **0.9848** |

`hybrid_effb1` wins on every clean metric, all six robustness families, all 15
transforms, and out-of-distribution AUC. Its advantage is *larger* cross-source
(+0.0076) than on clean CIFAKE (+0.0027) — evidence the added capacity
generalises rather than memorises. Weakest transforms for both: `resize_0.25`
and `blur_s2.0`.

## Data Preparation (details)

- **`holdout_index.py`** — builds `holdout_index.json`: SHA-256 (byte-identical
  copies) + dHash (same image after resize / re-compression / format change) of
  every benchmark image. `--add DIR` (repeatable), `--output`, `--merge`.
- **`prepare_wildfake.py`** — downloads WildFake from ModelScope into
  `data/<split>/{real,ai}`, dropping anything that matches the hold-out index
  (running without `--exclude-index` warns loudly). `--inspect` prints the
  schema first. Flags: `--dataset`, `--subset`, `--dataset-split`, `--split`,
  `--out`, `--exclude-index`, `--image-column`, `--label-column`, `--limit`,
  `--inspect`.
- **`prep_wildfake.py`** — alternative: extracts a capped, balanced subset from
  an already-cloned `../wildfake_raw` LFS checkout. `--src`, `--dst`,
  `--per_class`, `--val_frac`, `--seed`, `--delete_zips`.
- **`splitfolder.py`** — moves 25% of each class from `train/` to `val/`.

## Module Reference

### `config.py`
Central settings. `weights_path()` → per-variant checkpoint paths.
`TrainConfig` now defaults `monitor="roc_auc"` and adds `robust_eval_every`.
`HybridConfig` — CLIP id, feature/projection widths, `freeze_clip`; high-pass
residual params; `spatial_backbone` (`smallcnn` / `efficientnet_b0` /
`efficientnet_b1`, set from the `--model` name), `spatial_pretrained`,
`spatial_trainable`; FFT `freq_grid` (7); fusion widths.
`TrainAugConfig` — `apply_prob` (0.85, chance a single robustness-family
transform is applied — never stacked), `generalization_prob` (0.15,
rotate/grayscale from a non-evaluated pool), per-family severity *ranges*.
`ROBUSTNESS_SPEC` / `EVAL_SEVERITY_POINTS` define the fixed 15-transform grid.
`HYBRID_SPATIAL_BACKBONE` + `is_hybrid()` do the model-name dispatch.
Singletons: `MODEL_CFG`, `TRAIN_CFG`, `EVAL_CFG`, `ROBUSTNESS_CFG`,
`TRAIN_AUG_CFG`, `HYBRID_CFG`, `BEST_WEIGHTS`.

### `augmentations.py`
(1) PIL/NumPy primitives — `jpeg_compress`, `gaussian_blur`, `resize_down_up`,
`gaussian_noise`, `color_jitter`, `center_crop`, `slight_rotate`, `grayscale`.
(2) `get_robustness_transforms()` — the fixed 15-transform eval set (JPEG q∈{90,
70,50,30}; blur σ∈{0.5,1,2}; resize {0.5,0.25}; noise σ∈{0.02,0.05,0.10}; colour
±20%; centre crop 80%). **Do not change.**
(3) `apply_training_augmentation()` — one robustness-family transform per image
at most, at a random severity kept ≥ `margin` from every `EVAL_SEVERITY_POINTS`
value (`_sample_away_from_grid`, the anti-leakage mechanism), or a single
rotate/grayscale from the generalization pool. Flips are deliberately excluded
(orientation can carry signal).

### `frequency.py`
`compute_freq_spectrogram(rgb01, grid)` — per-patch 2D FFT log-magnitude
spectrogram (224/7 = 32 px patches). `stack_image_and_spectrogram()` — packs it
as the 4th channel of the hybrid model input.

### `datasets.py`
`RealAIDataset` — loads `data/{split}/{real, ai, full_synthetic_part2}/`, label
`real`→0 / else→1, on-the-fly augmentation for `split=="train"`,
retry-on-bad-file (10 tries then a blank tensor).
`build_transform(image_size, train, model_name)` — 4-channel RGB+FFT
`HybridTransform` for `hybrid_*` models (via `is_hybrid`), ImageNet-normalised
3-channel `Compose` otherwise. Stale `_l1_`/`_l2_` files excluded.

### `model.py`
`EfficientNetDetector` — torchvision backbone + `Dropout → Linear(·,1)` head;
unsupported names raise `ValueError`.
`HybridDetector` — frozen CLIP + high-pass spatial CNN/EfficientNet + FFT CNN +
fusion head; keeps frozen sub-nets in `eval()` under `.train()`.
`build_model(cfg, device)` — dispatches on the name via `is_hybrid`.
`load_best_weights()` — loads a checkpoint dict or raw state_dict, sets `eval()`.

### `train.py`
Device resolve, seeded RNG, per-variant checkpoint path. `build_loaders(cfg,
root, limit)` (+ balanced `--limit` subsample). Class-imbalance `pos_weight` →
`BCEWithLogitsLoss`. `AdamW` over trainable params only; three LR schedulers;
per-epoch `weight_decay` decay; AMP on CUDA. Per epoch: `validation_confidence`
(Youden's-J threshold + hard metrics + calibration diagnostics), error-analysis
print, and every `robust_eval_every` epochs a 15-transform validation robustness
pass with `final_score`. Checkpoints on `monitor` improvement; early stopping;
optional interactive stop (`--no_prompt` disables). Writes
`<weights>.history.json`.

### `evaluate.py`
Model-aware `TestImageDataset` (+ `--limit`); `checkpoint_meta()` reads
`(model_name, image_size, threshold)` from the checkpoint. Clean
`classification_metrics` (acc/precision/recall/F1/ROC-AUC). `evaluate_robustness`
— **per-transform ROC-AUC** and error counts via `_TransformedView`.
`robustness_score` → mean/min/max AUC, grouped by family; `final_score` =
½ clean AUC + ½ robust AUC. `error_analysis` — FP/FN counts, mean confidence, up
to 20 examples each, plain-English interpretation. Writes `eval_report.json`.
CLI: `--model`, `--weights`, `--data_root`, `--threshold`, `--limit`,
`--severity`, `--device`.

### `test_realworld.py`
Cross-source evaluation against any `real/`+`ai/` folder. Model-aware
preprocessing; ROC-AUC primary; hard metrics + error analysis at a forced or
ROC-optimal threshold; writes a report JSON. CLI: `--root`, `--model`,
`--weights`, `--threshold`, `--batch_size`, `--max_per_class`, `--device`,
`--output`.

### `inference.py`
`load_image_tensor()` (model-aware transform); `predict()` returns per image
`image_path`, `pred` (P(AI), 4 dp) + `label`, `is_ai`, `threshold`;
`gather_paths()` (file or directory). Always writes the `{image_path, pred}`
JSON (default `predictions.json`); `--quiet` only silences the console.

### `metrics.py`
`classification_metrics` (acc/precision/recall/F1/ROC-AUC, `zero_division=0`).
`optimal_threshold` — Youden's J with degenerate-case fallbacks.
`robustness_score` — mean/min/max over per-transform AUCs.
`final_score(clean_auc, robust_auc)` = `0.5·clean + 0.5·robust`.

## Web Interface (`app.py`)

A Flask app for interactive analysis and for viewing evaluation reports.

```bash
python app.py                                              # http://127.0.0.1:5001
python app.py --weights models/best_model_hybrid_effb1.pth --threshold 0.05 --port 5001
```

The model served is set by `MODEL_NAME` near the top of `app.py`
(`"hybrid_effb1"`). Checkpoint, input size and a calibrated threshold are then
read from the checkpoint and from any `eval_report.json` / `realworld_eval*.json`
present; a `THRESHOLD` env var / `--threshold` overrides.

> **Merge state on `main`:** `app.py` currently has unresolved conflicts from the
> frontend rewrite — `NameError: name 'BEST_WEIGHTS' is not defined` at import
> and a duplicated `@app.route("/eval-report/exists")`. Fix: add
> `BEST_WEIGHTS = weights_path(MODEL_CFG.name)` after the `from config import …`
> line and delete the duplicate route.

**Routes**

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The UI |
| `POST` | `/predict` | Score one chunk of uploaded images; per-image results (+ per-chunk `aggregate` unless `aggregate=0`) |
| `POST` | `/aggregate` | Folder-level metrics over the whole upload from the collected `(true_label, prob_ai)` pairs |
| `GET` | `/model-info` | What checkpoint / architecture / input size / threshold is being served |
| `GET` | `/eval-image` | Serve a project-local image by path (report viewer) |
| `GET` | `/eval-report` | The `eval_report.json` written by `evaluate.py` |
| `GET` | `/eval-report/exists` | Existence check used on page load |

Folder metrics are computed by `/aggregate` over the **whole upload**, not per
chunk.

**Tab 1 — Analyze Images.** Input via file picker (multi-select), folder picker
(nested), or drag-and-drop (walks the tree via `FileSystemEntry`). Duplicates
keyed by **relative path + size** (a dataset folder has both `real/0001.jpg` and
`ai/0001.jpg`). The selection is sliced into chunks and posted sequentially, so
peak memory is bounded by one chunk — **no limit on folder size**.

Knobs: `CHUNK_FILES` (16) / `CHUNK_BYTES` (24 MB) in `templates/index.html`;
`THUMB_LIMIT` (300, previews only for the first N); `TABLE_LIMIT` (200 rows,
"Show more"); `INFER_BATCH` (16, images per forward pass); `MAX_UPLOAD_MB` (3000);
`MAX_IMAGES` (10000, also raises Flask's `MAX_FORM_PARTS`). A progress bar +
**Cancel** (stops after the current chunk, keeps what's scored); a failed chunk
is an error against its own files and the run continues.

*Summary strip* — always: Analyzed, AI-Generated, Real/Human, Threshold. Second
row **only with ground-truth labels** (folders named `ai/`/`real/`): Accuracy,
F1, ROC-AUC, Precision, Recall, False Positives, False Negatives (each with mean
confidence).

*Per-image table* — `#`, Image ID, Preview, Filename, Verdict, True Label /
Correct? *(label mode)*, **AI Score** (raw output = `pred`), **vs Threshold**
(score remapped so the boundary sits at 50 %), Confidence (from the threshold,
not from 50 %), Dimensions, File Size, Inference ms. Downloads: **predictions.json**
(the deliverable — `[{image_path, pred}]`) and **Full report** (all per-image
fields + `aggregate`, thumbnails stripped).

*Calibration.* The raw score is not a probability and 0.5 is the wrong cutoff
(cross-source ROC-optimal threshold ≈ 0.002). `app.py` loads a fitted threshold,
searching: `THRESHOLD` env / `--threshold` → `realworld_eval*.json` →
`eval_report.json` → `EVAL_CFG.threshold` (0.5 fallback). The chosen value and
its source print at startup. The **vs Threshold** column and **Confidence** are
measured from that threshold; `predictions.json` exports the raw value.

*Label detection.* From the folder name in each file's relative path:
`ai / ai_generated / fake / synthetic` → AI; `real / human / authentic /
genuine` → real; anything else → unlabelled (metrics strip hidden).

**Tab 2 — Evaluate Report.** Run `python evaluate.py`, refresh. Sections: score
banner (final score, clean ROC-AUC, robustness score, Youden's-J threshold);
clean metric cards; robustness (by-family + per-transform, worst-first,
colour-coded, 3 weakest highlighted); error analysis (FP/FN counts + mean
confidence + interpretation + example filenames). **Download JSON** / **Reload**.

**`POST /predict`** — `multipart/form-data`: files under `images`, index-aligned
relative paths under `paths[]`; optional `thumbs=0`, `aggregate=0`. Response:
`{ "results": [...], "aggregate": {...} | null }`. Per-file result:

```json
{
  "id": "A3F2C1B0", "filename": "photo.jpg", "thumbnail": "data:image/jpeg;base64,…",
  "prob_ai": 0.9123, "prob_real": 0.0877, "verdict": "AI-Generated",
  "confidence": "Very High", "pred_label": 1, "true_label": 1, "correct": true,
  "width": 1024, "height": 768, "file_size_kb": 214.5, "inference_ms": 38.2,
  "threshold_used": 0.05
}
```

`aggregate` (≥ 2 labelled images, or from `/aggregate` for the whole job):

```json
{
  "n": 40, "threshold": 0.002,
  "metrics": { "accuracy": 0.8872, "precision": 0.8901, "recall": 0.8875, "f1": 0.8887, "roc_auc": 0.9564 },
  "error_analysis": { "false_positives": 12, "false_negatives": 10, "fp_mean_confidence": 0.7341, "fn_mean_confidence": 0.3102 }
}
```

On a per-file error the object carries `"error": "<message>"` instead of the
prediction fields.
