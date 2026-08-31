# c2.tsx — Robust Real vs. AI Image Detector

## Project Overview

This project detects **AI-generated images** vs. **authentic photographs**, with an
explicit focus on staying accurate after the kind of post-processing an image
picks up when it is shared online — JPEG re-compression, blur, thumbnail
rescaling, sensor noise, colour filters and cropping. It includes a fixed
15-transform robustness suite (`get_robustness_transforms()`, 6 families: JPEG
compression, blur, resize, noise, colour jitter, centre crop) that scores how
well a model holds up under those degradations.

The final model, **`hybrid_effb1`**, is a two-branch detector:

- **Semantic branch — frozen CLIP ViT-B/32.** A pretrained CLIP vision encoder,
  never fine-tuned. It contributes a fixed "does this look like a plausible real
  scene" prior. Because it does not train, it cannot memorise a particular
  generator's fingerprint or a dataset's capture style, and its features are
  largely invariant to compression, blur, cropping and colour shifts — so this
  branch keeps accuracy up when an image is degraded.
- **Low-level branch — two fused sub-streams:**
  - *Spatial:* a fixed high-pass residual (`image − gaussian_blur(image)`) fed to
    a CNN. The high-pass step removes scene layout, colour cast and exposure —
    easy dataset shortcuts — leaving texture and noise. For `hybrid_effb1` this
    CNN is a pretrained **EfficientNet-B1** feature extractor; the lighter
    `hybrid_clip` variant uses a ~0.3 M-param CNN.
  - *Frequency:* the image is split into a 7×7 grid of patches, each patch's 2D
    FFT log-magnitude spectrum is reassembled into a single-channel "spectrogram
    image", and a small CNN reads it — to catch the periodic upsampling /
    checkerboard artifacts common to generative models.
  - The two sub-streams fuse into one low-level vector.
- **Fusion head.** `[CLIP projection ‖ low-level vector] → MLP → 1 logit →
  sigmoid → P(AI-generated)`. Only ~7 M parameters train (`hybrid_effb1`);
  CLIP's ~87 M stay frozen. The head learns to lean on the low-level branch when
  an image is clean and fall back on CLIP semantics when it is not.

The hybrid input is a 4-channel tensor: RGB in `[0, 1]` plus the FFT spectrogram
channel. A plain `EfficientNetDetector` baseline (`--model efficientnet_b0/b1/b2`,
dropout + linear head → single logit) is also included and takes an ordinary
ImageNet-normalised 3-channel tensor.

### Key design decisions

- **Robustness-aware training.** Training-time augmentation samples the *same 6
  corruption families* used in evaluation, **at most one per image, never
  stacked**, at *random* severities kept deliberately away from the exact
  evaluation grid points. This measures generalisation, not memorisation of the
  scoring harness.
- **Data-level anti-leakage.** The TechJam demonstration set (COCO val2017 +
  DALL·E Advanced) is a subset of WildFake. `holdout_index.py` indexes those
  images by content hash **and** perceptual hash, and `prepare_wildfake.py`
  drops any training candidate that matches — filename checks alone miss
  re-encoded copies.
- **Threshold-independent model selection.** Checkpointing and early stopping
  monitor **ROC-AUC** (`monitor="roc_auc"`), which is stable, rather than F1,
  which swings with the per-epoch decision threshold.
- **Class-imbalance handling.** `BCEWithLogitsLoss` is weighted by the
  real/AI ratio automatically.
- **Per-variant checkpoints.** `weights_path()` names every model's checkpoint
  after its variant (`models/best_model_<name>.pth`), so training `efficientnet_b1`
  then `hybrid_effb1` never overwrites the other.
- **Automatic device selection.** CUDA → MPS → CPU.

&nbsp;

## Development Tools, Models, Libraries & Datasets

**Tools:** VS Code, Python virtualenv, Git/GitHub. Training on a single consumer
NVIDIA GPU (GTX 1650 SUPER, 4 GB) and Apple-Silicon MPS.

**Models / APIs:**
- CLIP `openai/clip-vit-base-patch32` (frozen vision encoder) via Hugging Face
  `transformers`.
- TorchVision EfficientNet-B0/B1/B2 (ImageNet-pretrained), used both as the
  standalone baseline and as the `hybrid_effb*` spatial backbone.

**Libraries / frameworks:** PyTorch, TorchVision, Hugging Face Transformers,
scikit-learn, NumPy, Pillow, pandas, tqdm, Matplotlib, Flask.

**Datasets:**
- **CIFAKE** (Kaggle) — real = CIFAR-10 upscaled, fake = Stable Diffusion @ 32 px
  upscaled. Baseline experiments and part of the hybrid training mix.
- **WildFake** (ModelScope `hy2628982280/WildFake`) — a capped, balanced subset:
  real from ImageNet / LSUN-Church / FFHQ / AFHQ / CelebA-HQ; AI from DDIM and
  DDPM. **COCO and DALL·E partitions are excluded** (the demonstration benchmark
  is drawn from them).
- **SID_Set** (Hugging Face)

&nbsp;

## Setup

**Requirements:** Python 3.10+, PyTorch ≥ 2.0 built for your accelerator.

1. Clone and create a virtual environment:

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

- **GPU note:** `pip install torch` may give a CPU-only wheel. For an NVIDIA GPU
  install a CUDA build, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
- **First hybrid run** downloads the CLIP ViT-B/32 weights (~600 MB) to the
  Hugging Face cache — needs internet access once. The EfficientNet-only path
  needs neither `transformers` nor internet.

3. Data directory layout (all scripts assume this):

```
data/
  train/{real, ai}      # any folder other than real/ is treated as label 1;
  val/{real, ai}        # a second AI-source folder is fine
  test/{real, ai}
```

Supported formats: `.jpg .jpeg .png .bmp .webp`.

&nbsp;

## Data Preparation

### 1. Build the hold-out exclusion index (first)

```bash
python holdout_index.py --add path/to/coco_val2017 --add path/to/dalle_advanced \
                        --output holdout_index.json
```

Indexes each benchmark image by SHA-256 (byte-identical copies) and dHash (same
image after resize / re-compression / format change). `--merge` extends an
existing index.

### 2. Pull the WildFake training subset

```bash
python prepare_wildfake.py --inspect                              # see the schema first
python prepare_wildfake.py --exclude-index holdout_index.json     # download + filter
python prepare_wildfake.py --exclude-index holdout_index.json --limit 20000
```

Downloads from ModelScope straight into `data/<split>/{real,ai}`, dropping any
image that matches the hold-out index. Running without `--exclude-index` is
allowed but warns loudly. Flags: `--dataset`, `--subset`, `--dataset-split`,
`--split`, `--out`, `--image-column`, `--label-column`, `--limit`, `--inspect`.

*(Alternative:* `prep_wildfake.py` extracts a capped, balanced subset from an
already-cloned `../wildfake_raw` LFS checkout — `--src --dst --per_class
--val_frac --seed --delete_zips`.)*

### 3. Carve a validation split (if you only have `train/`)

```bash
python splitfolder.py     # moves 25% of each class from train/ to val/
```

&nbsp;

## Reproducing the Results

### 1. Train

```bash
# final hybrid model
python train.py --model hybrid_effb1 --device cuda --batch_size 16 --robust_eval_every 5 --no_prompt

# lighter hybrid variant
python train.py --model hybrid_clip

# plain EfficientNet baseline
python train.py --model efficientnet_b1
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | `efficientnet_b0…b7`, or a hybrid: `hybrid_clip` (small-CNN spatial branch), `hybrid_effb0` / `hybrid_effb1` (EfficientNet spatial branch) |
| `--epochs` | `8` | Max training epochs (early stopping may end sooner) |
| `--batch_size` | `32` | Batch size (`hybrid_effb1` fits at 16 on 4 GB) |
| `--lr` | `3e-4` | Learning rate |
| `--augment_level` | `1` | 0 = off, 1 = on-the-fly augmentation enabled |
| `--scheduler` | `cosine_warm_restarts` | `cosine_warm_restarts`, `reduce_on_plateau`, `step` |
| `--monitor` | `roc_auc` | Validation metric for checkpointing / early stopping |
| `--patience` | `8` | Early-stopping patience (epochs) |
| `--robust_eval_every` | `1` | Run the 15-transform validation robustness eval every N epochs (raise for slow models) |
| `--no_prompt` | off | Skip the interactive "continue training?" prompt after each epoch |
| `--data_root` | `data` | Dataset root; point at a subset folder for a quick smoke test |
| `--limit` | none | Cap training samples (val scaled down too) for a fast end-to-end run |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |

Best checkpoint → `models/best_model_<variant>.pth`; per-epoch history →
`models/best_model_<variant>.history.json`. Only `requires_grad=True` parameters
are optimised, so the frozen CLIP encoder is excluded from the optimiser.

### 2. Evaluate (clean metrics + robustness)

```bash
python evaluate.py --model hybrid_effb1 --device cuda
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | Must match the checkpoint's architecture |
| `--weights` | auto | Defaults to `models/best_model_<model>.pth` |
| `--data_root` | `data` | Test data at `<data_root>/test/{real,ai}` |
| `--threshold` | ROC-optimal | Force a fixed threshold; default is Youden's J on the clean test set |
| `--limit` | none | Evaluate on a balanced random subset (the robustness pass runs the full set 15×) |
| `--severity` | `0.5` | Legacy knob; the 15-transform set is fixed |
| `--device` | `auto` | |

Writes `eval_report.json` next to the weights: clean metrics, per-transform and
per-family robustness **ROC-AUC**, `final_score` (`0.5·clean_AUC + 0.5·robust_AUC`),
error analysis, per-transform error counts. Console prints a summary plus the 3
weakest transforms.

### 3. Cross-source generalisation test

```bash
python test_realworld.py --root . --model hybrid_effb1 --device cuda \
                         --max_per_class 2001 --output crosseval_effb1.json
```

Points at any folder containing `real/` and `ai/` sub-folders (not the
`data/test/` layout). Reports ROC-AUC, accuracy, F1 and an error breakdown at
either a forced `--threshold` or the ROC-optimal one for that set. Flags:
`--root`, `--model`, `--weights`, `--threshold`, `--batch_size`,
`--max_per_class`, `--device`, `--output`.

### 4. Inference

```bash
python inference.py path/to/image_or_folder --model hybrid_effb1 --output predictions.json
```

| Flag | Default | Description |
|---|---|---|
| `input` | — | Image file or directory (positional) |
| `--model` | `efficientnet_b0` | Architecture to build; must match the checkpoint |
| `--weights` | auto | Defaults to `models/best_model_<model>.pth` |
| `--threshold` | `0.5` | Decision boundary for the human-readable label |
| `--image_size` | `224` | |
| `--device` | `auto` | |
| `--output` | `predictions.json` | JSON of `{image_path, pred}` per image (`pred` = P(AI), 4 dp), always written |
| `--quiet` | off | Suppress the per-image console log (file still written) |

### Typical workflow

```bash
python train.py --model hybrid_effb1 --robust_eval_every 5 --no_prompt   # ~hours on GPU
python evaluate.py --model hybrid_effb1                                    # clean + robustness
python test_realworld.py --root . --model hybrid_effb1 --max_per_class 2001
python inference.py path/to/images --model hybrid_effb1
```

&nbsp;

## Results Summary

All figures are ROC-AUC unless noted. Baselines were trained on **CIFAKE only**;
the hybrids on **CIFAKE + WildFake** — do not read the two groups as one
progression.

### EfficientNet baselines (CIFAKE test, 8 k subset)

| | Clean AUC | Robustness mean AUC | Final score |
|---|---|---|---|
| B0 | 0.9977 | 0.9906 | 0.9942 |
| **B1** | **0.9988** | **0.9954** | **0.9971** |
| B2 | 0.9985 | 0.9954 | 0.9969 |

B0 underfits robustness (worst: `blur_s2.0` 0.96, `resize_0.25` 0.96); B2 adds
parameters over B1 for no accuracy gain and worse calibration (ROC-optimal
threshold 0.98, highly overconfident errors). **B1 is the sweet spot** — hence
its use as the hybrid spatial backbone.

### Hybrid models

| | CIFAKE test AUC | CIFAKE robustness mean AUC | Final score | FP / FN | Cross-source AUC |
|---|---|---|---|---|---|
| `hybrid_clip` | 0.9935 | 0.9775 | 0.9855 | 382 / 389 | 0.9773 |
| **`hybrid_effb1`** | **0.9961** | **0.9840** | **0.9901** | **229 / 355** | **0.9848** |

`hybrid_effb1` wins on every clean metric, all six robustness families, all 15
transforms, and out-of-distribution AUC. Its advantage is *larger* cross-source
(+0.0076) than on clean CIFAKE (+0.0027) — evidence the added capacity
generalises rather than memorises. Weakest transforms for both: `resize_0.25`
and `blur_s2.0` (they destroy the frequency signal, so robustness there rests on
the CLIP branch).

&nbsp;

## Limitations & Reflection

**Data**
- **CIFAKE is a weak proxy.** Its real/fake split is separable by a low-frequency
  resolution/upscaling artifact, so high CIFAKE scores partly reflect that
  shortcut rather than genuine AIGC detection.
- **Narrow generator coverage.** Training AI images come from SD (CIFAKE) plus
  WildFake's DDIM and DDPM — no GANs, autoregressive models, Midjourney or
  fine-tuned SD variants. Real sources skew toward faces and objects.
- SID_Set was not integrated; only ~18 GB of WildFake's ~1.6 TB was used.

**Model & scope**
- **Binary classifier only.** No identification of the generator used, no
  localisation of AI-edited regions, no degree-of-manipulation output.
- **Resolution assumption.** Images are resized to 224×224; extreme aspect
  ratios or very low resolution lose discriminative detail.
- Frozen CLIP cannot adapt to domains far from its pretraining (medical,
  satellite, documents).
- The **frequency branch is fragile exactly where it is needed** — blur, resize
  and JPEG destroy the high-frequency cues it depends on, so under the worst
  corruptions robustness rests almost entirely on the frozen CLIP branch.
- The EfficientNet-B1 spatial branch (~7 M trainable params) has capacity to
  memorise dataset-specific artifacts; the frozen CLIP encoder and high-pass
  residual mitigate this only partially.

**Evaluation**
- **In-distribution robustness only.** `data/test` is a held-out CIFAKE split —
  same generator and real source as most of training. The robustness table
  measures transform-robustness on one distribution, not cross-generator
  robustness.
- **The cross-source set is only partially held out** — it mixes WildFake
  sources overlapping training with novel DALL·E content, and we lack per-source
  labels to isolate the novel portion. Suggestive, not conclusive.
- No run on the official COCO/DALL·E benchmark; no leave-one-generator-out.
- Single training run per model; some evals on subsets; GPU non-determinism
  makes sub-0.002 AUC differences noise.
- The robustness suite covers the 6 brief families only — no non-Gaussian blur,
  film grain, geometric distortion, watermarking, or adversarial attacks.

**Calibration**
- **Miscalibration under domain shift** — the ROC-optimal threshold is ~0.65
  in-distribution vs ~0.003 cross-source; scores compress toward zero on
  unfamiliar data. ROC-AUC (ranking) is stable but a fixed threshold fails. No
  temperature scaling / recalibration is implemented; `app.py` works around it
  by loading a fitted threshold (see **Web Interface → Calibration**).

**Compute & time**
- Single 4 GB GPU: batch 16, `--limit` subsampling, 8 epochs — training loss was
  still falling at the cutoff, so reported metrics are a lower bound.
- No hyperparameter search, no multi-seed variance estimates; architecture
  exploration limited to backbone size and the spatial-branch swap.
- The single-process evaluation pipeline makes a full 20 k × 16-pass robustness
  run take ~1 hour, forcing subsampled runs.

### Given more time

1. Per-source AUC on the cross-source set; leave-one-generator-out.
2. Broader training generators (GAN, Midjourney, SD variants, autoregressive) +
   SID_Set.
3. Temperature scaling / per-domain threshold calibration.
4. Full-dataset training, more epochs, multiple seeds, a hyperparameter sweep.
5. Evaluate on the official COCO/DALL·E benchmark.
6. Grad-CAM / integrated-gradients attribution on FP/FN examples.
7. Region-level localisation using SID_Set masks.

&nbsp;

## `config.py`

Every module reads its settings from here.

**Paths**
- `PROJECT_ROOT`, `DATA_ROOT` (`data/{train,val,test}/{real,ai}`), `RAW_ROOT`,
  `MODELS_DIR`.
- `weights_path(model_name)` — per-variant checkpoint filename
  (`models/best_model_<name>.pth`) so variants never overwrite each other.
- `SPLIT_RATIOS = (0.7, 0.15, 0.15)`.

**Dataclasses**
- `ModelConfig` — EfficientNet backbone name, ImageNet-pretrained flag,
  `num_classes` (1 = single-logit binary), dropout.
- `TrainConfig` — image size, batch size, `num_epochs` (8), LR, weight decay
  (+ per-epoch decay factor), optimizer / scheduler, cosine-warm-restart and
  plateau params, `early_stopping_patience`, **`monitor="roc_auc"`**,
  **`robust_eval_every`** (validation robustness eval cadence), `threshold`,
  `augment_level`, workers, seed, device, AMP toggle.
- `EvalConfig` — image size, batch size, threshold, workers, device.
- `RobustnessConfig` — legacy `severity` knob and `weight_by_severity` (default
  `False` → the robustness score is a plain unweighted mean over the 15
  transforms).
- **`HybridConfig`** — the two-branch detector's config: CLIP checkpoint id
  (`openai/clip-vit-base-patch32`), CLIP feature / projection widths,
  `freeze_clip`; high-pass residual blur sigma & kernel; **`spatial_backbone`**
  (`"smallcnn"` or `"efficientnet_b0/b1"`, set automatically from the `--model`
  name), `spatial_pretrained`, `spatial_trainable`; FFT `freq_grid` (7) and
  feature width; fused `lowlevel_dim`; fusion-head hidden width and dropout.
- `TrainAugConfig` — the on-the-fly augmentation sampler:
  - **`apply_prob`** (0.85) — probability a robustness-family transform is
    applied at all. When one is applied, **exactly one** family is drawn;
    transforms are **never combined**. With probability `1 − apply_prob` the
    image is left clean.
  - `generalization_prob` (0.15) — probability of instead drawing a single
    rotate/grayscale transform from a pool that is **not** one of the evaluated
    families.
  - Per-family severity *ranges* used at training time (JPEG quality, blur sigma,
    resize scale, noise sigma, colour factor, crop fraction). They span (and
    slightly exceed) the evaluation grid, but individual draws are pushed off the
    exact evaluation points.

**Fixed robustness spec**
- `ROBUSTNESS_SPEC` — the 6 families and their exact parameter levels
  (15 transforms), matching the brief.
- `EVAL_SEVERITY_POINTS` — the same, keyed by short family name, used only to
  keep training-time severities off the exact scoring values.

**Hybrid dispatch**
- `HYBRID_MODEL_NAME = "hybrid_clip"`; `HYBRID_SPATIAL_BACKBONE` maps
  `hybrid_clip → smallcnn`, `hybrid_effb0 → efficientnet_b0`,
  `hybrid_effb1 → efficientnet_b1`; `is_hybrid(name)` tests membership.

**Singletons**
- `MODEL_CFG`, `TRAIN_CFG`, `EVAL_CFG`, `ROBUSTNESS_CFG`, `TRAIN_AUG_CFG`,
  **`HYBRID_CFG`** — imported everywhere.
- `BEST_WEIGHTS` — default checkpoint path for the currently configured variant.
  (`evaluate.py` / `inference.py` now derive `--weights` from `--model` via
  `weights_path()`; `BEST_WEIGHTS` is still used as a fallback and by `app.py`.)

&nbsp;

## `augmentations.py` — Image Transforms & Augmentation Sampling

### 1. Core transform primitives
PIL/NumPy functions, each simulating a real-world degradation: `jpeg_compress`,
`gaussian_blur`, `resize_down_up`, `gaussian_noise`, `color_jitter`,
`center_crop`, `slight_rotate`, `grayscale`.

### 2. `get_robustness_transforms()` — the fixed 15-transform eval set
- JPEG quality ∈ {90, 70, 50, 30} — 4
- Blur sigma ∈ {0.5, 1.0, 2.0} — 3
- Resize scale ∈ {0.5, 0.25} — 2
- Noise sigma ∈ {0.02, 0.05, 0.10} — 3
- Colour jitter ∈ {+20%, −20%} — 2
- Centre crop @ 80% — 1

**Do not change** — this is the benchmark variants are compared against.

### 3. `apply_training_augmentation(img, rng, aug_cfg)`
1. **Same 6 families as the eval set** (`FAMILIES`: jpeg, blur, resize, noise,
   colour, crop) so training exposure matches the evaluation space.
2. **At most one family per image, never stacked** — a single degradation step,
   matching how the brief evaluates each transform individually.
3. **Random severity** per draw (`_draw_severity`), and **off the eval grid** —
   `_sample_away_from_grid()` re-samples (up to 25 tries) until the value is at
   least `margin` away from every `EVAL_SEVERITY_POINTS` value for that family.
   This is the key anti-leakage mechanism.
4. **A separate, non-overlapping "generalization" pool.** With probability
   `generalization_prob`, one of `GENERALIZATION_POOL` (small rotation ±15°, or
   grayscale) is applied instead. These add invariances without ever touching
   the evaluated parameter space. Horizontal/vertical flips are **excluded** —
   orientation can carry signal (asymmetric artifacts, mirrored text).

`get_augment_names()` lists the families + generalization pool for logging.

&nbsp;

## `frequency.py` — Low-Level Frequency Features

- `compute_freq_spectrogram(rgb01, grid)` — splits the image into `grid × grid`
  patches, takes each patch's 2D FFT log-magnitude, and reassembles them into a
  single-channel spectrogram image (224 / 7 = 32 px patches).
- `stack_image_and_spectrogram(rgb01, grid)` — concatenates the `[0,1]` RGB
  image with its spectrogram along the channel axis, producing the 4-channel
  tensor the hybrid models consume.

&nbsp;

## `datasets.py` — PyTorch Dataset

`RealAIDataset(Dataset)` loads images from `data/{split}/{real, ai,
full_synthetic_part2}/` and returns `(image_tensor, label)`.

- **Label encoding:** `real` → 0, anything else → 1.
- **Folder scanning:** recognised extensions only; stale `_l1_`/`_l2_` files
  from an older static-augmentation pipeline are excluded.
- **Augmentation:** applied only when `split == "train"` and `augment_level > 0`,
  delegating to `augmentations.apply_training_augmentation` via a seeded
  `random.Random` for per-run determinism.
- **`build_transform(image_size, train, model_name)`** — chooses the pipeline
  for the model: the 4-channel RGB+FFT `HybridTransform` for `hybrid_*` models
  (via `is_hybrid`), or the ImageNet-normalised 3-channel `Compose` otherwise.
- **Robust loading:** `__getitem__` retries up to 10 times on a different random
  index if an image fails to load, then falls back to a blank tensor with label
  0 — a single bad file can't crash a run.

&nbsp;

## `model.py` — Model Definitions

- **`EfficientNetDetector`** — a TorchVision EfficientNet backbone with its
  classifier swapped for `Dropout → Linear(in_features, 1)`. Unsupported names
  raise a clear `ValueError`.
- **`HybridDetector`** — frozen CLIP encoder + high-pass spatial CNN (or
  EfficientNet backbone) + per-patch FFT CNN + fusion head. Keeps frozen
  sub-networks in `eval()` even when the detector is put in `train()`. Input is
  the 4-channel tensor; the module splits the spectrogram channel back out
  internally.
- `build_model(cfg, device)` — dispatches on the name via `is_hybrid`
  (`hybrid_clip` / `hybrid_effb0` / `hybrid_effb1` → `HybridDetector` with the
  matching `spatial_backbone`; otherwise `EfficientNetDetector`).
- `load_best_weights(model, path, device)` — loads a checkpoint dict
  (`{"model_state": ...}`) or a raw state_dict, then sets `eval()` mode.

&nbsp;

## `train.py` — Training Loop

**Setup** — resolves device (`auto` = CUDA → MPS → CPU); derives the checkpoint
path from the current variant; seeds `torch` / `numpy` / `random`.

**Data / model / optimization**
- `build_loaders(cfg, root, limit)` — train/val `RealAIDataset`s + `DataLoader`s
  (train shuffles + drops the last partial batch); `--limit` takes a balanced
  random subsample.
- `compute_pos_weight` → `nn.BCEWithLogitsLoss(pos_weight=...)`.
- `AdamW` over **trainable parameters only** (frozen CLIP excluded);
  `make_scheduler` builds `cosine_warm_restarts` / `reduce_on_plateau` / `step`.
- `GradScaler` only when AMP **and** CUDA.

**Per-epoch loop**
- `train_epoch()` — standard step (AMP autocast + scaling on CUDA), returns the
  mean loss.
- `validation_confidence()` — collects probabilities, picks a **Youden's-J
  threshold**, computes hard metrics at it, and tracks mean predicted
  probability and mean confidence on correct vs. wrong predictions (calibration
  diagnostic).
- Per-epoch **error analysis** print (FP/FN counts + mean confidence), and every
  `robust_eval_every` epochs a **15-transform validation robustness pass**
  (`evaluate_val_robustness`) with `final_score`.
- Steps the scheduler; optionally decays `weight_decay` by `wd_decay_factor`.
- **Checkpointing** on `monitor` (`roc_auc`) improvement — saves
  `{"model_state", "model_name", "config", "epoch", "metrics"}`.
- **Early stopping** after `early_stopping_patience` idle epochs; an optional
  interactive "continue?" prompt after each epoch (`--no_prompt` disables).
- Writes `<weights>.history.json` at the end.

**CLI:** see the Train table above.

&nbsp;

## `evaluate.py` — Clean Metrics + Robustness Evaluation

Run against `data/test/{real,ai}` (or `--data_root`).

- `TestImageDataset` — model-aware preprocessing (`build_transform`), `--limit`
  balanced subsample, excludes stale `_l1_`/`_l2_` files.
- `checkpoint_meta()` — reads `(model_name, image_size, threshold)` recorded in
  the checkpoint, so a b1 checkpoint isn't silently loaded into a b0.
- **Clean metrics** — accuracy, precision, recall, F1, ROC-AUC.
- **Robustness** — `evaluate_robustness()` applies each of the 15 fixed
  transforms (via `_TransformedView`, re-transforming every image on the fly)
  and records **per-transform ROC-AUC** and per-transform error counts.
- `metrics.robustness_score()` → mean / min / max AUC; grouped **by family**;
  `metrics.final_score()` → `0.5·clean_AUC + 0.5·robust_AUC`.
- `error_analysis()` — FP (real→AI) vs FN (AI→real) counts, mean confidence, up
  to 20 example paths each, and a plain-English interpretation of the dominant
  error mode.
- Writes `eval_report.json` next to the weights and prints a summary (incl. the
  3 weakest transforms).
- **CLI:** `--model`, `--weights`, `--data_root`, `--threshold`, `--limit`,
  `--severity`, `--device`.

&nbsp;

## `test_realworld.py` — Cross-Source Evaluation

Runs the trained model against any folder holding `real/` and `ai/`
sub-folders (not the `data/test/` layout — for held-out sets from other
sources). Model-aware preprocessing; ROC-AUC as the primary metric; hard
metrics + error analysis at a forced `--threshold` or the ROC-optimal one for
that set; writes a report JSON. **CLI:** `--root`, `--model`, `--weights`,
`--threshold`, `--batch_size`, `--max_per_class`, `--device`, `--output`.

&nbsp;

## `inference.py` — Single-Image / Batch Prediction

The deployment-facing script (`python inference.py <file_or_folder> [args]`).

- `load_image_tensor()` — the same model-aware preprocessing as training/eval.
- `predict()` — per image returns `image_path`, `pred` (P(AI), 4 dp — the
  deliverable field), plus `label` (`"ai"`/`"real"`), `is_ai`, `threshold`.
- `gather_paths()` — a single file or a directory (recognised images, sorted).
- Always writes `{image_path, pred}` JSON (default `predictions.json`),
  regardless of `--quiet` (which only silences the console log).
- **CLI:** positional `input`, `--model`, `--weights`, `--threshold`,
  `--image_size`, `--device`, `--output`, `--quiet`.

&nbsp;

## `metrics.py` — Metric Computation

- `classification_metrics(y_true, y_prob, threshold)` — accuracy, precision,
  recall, F1 (all `zero_division=0`), and ROC-AUC (NaN if one class). Returns
  the dict and the hard predictions.
- `optimal_threshold(y_true, y_prob)` — the ROC-optimal decision threshold
  (Youden's J), with sensible fallbacks for degenerate cases.
- `robustness_score(per_transform_auc, weight_by_severity, severity_weights)` —
  aggregates `{transform: ROC-AUC}` into a single score (mean by default) and
  reports mean / min / max plus the full breakdown.
- `final_score(clean_auc, robust_auc)` — `0.5·clean + 0.5·robust`.

&nbsp;

## Web Interface (`app.py`)

A Flask app for interactive analysis and for viewing evaluation reports.

```bash
python app.py                                              # http://127.0.0.1:5001
python app.py --weights models/best_model_hybrid_effb1.pth --threshold 0.05 --port 5001
```

Which model it serves is set by `MODEL_NAME` near the top of `app.py`
(`"hybrid_effb1"` by default). Checkpoint, input size and a calibrated decision
threshold are then read from the checkpoint and from any `eval_report.json` /
`realworld_eval*.json` present; a `THRESHOLD` env var or `--threshold` overrides.

> **Merge state on `main`:** `app.py` currently has unresolved conflicts from the
> frontend rewrite — it raises `NameError: name 'BEST_WEIGHTS' is not defined` at
> import and defines `@app.route("/eval-report/exists")` twice. Fix: add
> `BEST_WEIGHTS = weights_path(MODEL_CFG.name)` after the `from config import …`
> line and delete the duplicate route. Reconcile any `INFER_BATCH` / `BATCH_SIZE`
> naming while you're there.

### Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The UI |
| `POST` | `/predict` | Score one chunk of uploaded images; returns per-image results (and a per-chunk `aggregate` unless `aggregate=0`) |
| `POST` | `/aggregate` | Folder-level metrics from the `(true_label, prob_ai)` pairs the frontend collected across all chunks |
| `GET` | `/model-info` | What checkpoint / architecture / input size / threshold the server is actually serving |
| `GET` | `/eval-image` | Serve a project-local image by path (report viewer) |
| `GET` | `/eval-report` | The `eval_report.json` written by `evaluate.py` (404 if absent) |
| `GET` | `/eval-report/exists` | Lightweight existence check used on page load |

Folder-level accuracy/F1/ROC-AUC and error counts are computed by `/aggregate`
over the **whole upload**, not per chunk — accuracy over whichever ~16 images
shared a request says nothing about the folder.

### Tab 1 — Analyze Images

**Input:** *Browse Files* (Ctrl/Cmd for multi-select), *Browse Folder* (includes
nested sub-folders), or *drag & drop* (walks the directory tree via the
browser's `FileSystemEntry` API). Duplicates are detected by **relative path +
size**, not filename alone (a dataset folder has both `real/0001.jpg` and
`ai/0001.jpg`). ≤ 8 files show as removable chips; more show a compact summary
bar. Rows stream into the table as each chunk returns.

**Folder size:** no limit. The selection is sliced into chunks and posted one
after another, so peak memory is bounded by one chunk.

| Knob | Where | Meaning |
|---|---|---|
| `CHUNK_FILES` (16) | `templates/index.html` | Images per request |
| `CHUNK_BYTES` (24 MB) | `templates/index.html` | Byte ceiling per request (whichever limit hits first) |
| `THUMB_LIMIT` (300) | `templates/index.html` | Previews requested only for the first N images |
| `TABLE_LIMIT` (200) | `templates/index.html` | Rows painted at once; **Show more** extends |
| `INFER_BATCH` (16) | `app.py` | Images stacked into one forward pass |
| `MAX_UPLOAD_MB` (3000) | `app.py` | Per-request body-size cap |
| `MAX_IMAGES` (10000) | `app.py` | Hard cap on files per upload; also raises Flask's `MAX_FORM_PARTS` |

A progress bar reports `done / total`; **Cancel** stops after the current chunk
and keeps what has been scored. A failed chunk is recorded as an error against
its own files and the run continues.

**Summary strip** — always: *Analyzed*, *AI-Generated*, *Real / Human*,
*Threshold* (auto-tuned to ROC-optimal when labels are present; otherwise the
calibrated startup threshold). A second row appears **only with ground-truth
labels** (folders named `ai/` or `real/`): *Accuracy*, *F1*, *ROC-AUC*,
*Precision*, *Recall*, *False Positives*, *False Negatives* (each with mean
confidence).

**Per-image table:** `#`, Image ID, Preview (first `THUMB_LIMIT`), Filename,
Verdict, True Label / Correct? *(label mode)*, **AI Score** (the raw output
exported as `pred`), **vs Threshold** (score remapped so the decision boundary
sits at 50 %), Confidence (distance from the *threshold*, not from 50 %),
Dimensions, File Size, Inference ms.

Two downloads: **predictions.json** (the Section 5.5 deliverable — a JSON array
of `{image_path, pred}`) and **Full report** (every per-image field + the
folder-level `aggregate`, thumbnails stripped, timestamped).

```json
[
  { "image_path": "dataset/real/0000.jpg", "pred": 0.0527 },
  { "image_path": "dataset/ai/0000.jpg",   "pred": 0.9412 }
]
```

#### Calibration

**The raw score is not a probability, and 0.5 is the wrong cutoff for this
model.** A cross-source report (`realworld_eval_2k.json`, 4,000 images) records
ROC-AUC ≈ 0.97 alongside a ROC-optimal threshold near **0.002** — the model ranks
AI above real well, but its outputs compress against zero, so at 0.5 essentially
everything is labelled "Real / Human". `app.py` therefore loads a fitted
threshold rather than hardcoding 0.5, searching in order:

1. the `THRESHOLD` env var / `--threshold`,
2. `realworld_eval*.json` (from `test_realworld.py`),
3. `eval_report.json` (from `evaluate.py`),
4. `EVAL_CFG.threshold` — the uncalibrated 0.5 fallback, only if no report exists.

The chosen value and its source are printed at startup. The UI's **vs Threshold**
column remaps the score in log space so the boundary sits at 50 % (otherwise a
genuine detection reads as "AI 0.28 % / Real 99.7 %"); **Confidence** is likewise
measured from the threshold. `predictions.json` still exports the raw value.

#### Label detection

From the folder name in each file's relative path (`webkitRelativePath`, or the
walked `FileSystemEntry.fullPath` for drag-and-drop): `ai / ai_generated / fake /
synthetic` → AI, `real / human / authentic / genuine` → real; anything else →
unlabelled and the metrics strip is hidden. Dropping `data/test/ai/` and
`data/test/real/` onto the drop zone produces the full metrics report.

### Tab 2 — Evaluate Report

Run `python evaluate.py` then refresh. Four sections:

- **Score banner** — the headline final score (`0.5·Clean AUC + 0.5·Robust AUC`),
  clean ROC-AUC, robustness score, Youden's-J threshold.
- **Clean test metrics** — accuracy, precision, recall, F1, ROC-AUC as cards.
- **Robustness** — two panels: *by family* and *per transform*, both sorted
  worst-first; bars green ≥ 0.8 / amber 0.6–0.8 / red < 0.6; the 3 weakest
  transforms highlighted.
- **Error analysis** — FP/FN counts with mean confidence, the interpretation
  text, and collapsible lists of up to 20 example filenames per error type.

**Download JSON** saves the raw `eval_report.json`; **Reload** re-fetches it
without a page refresh.

### `POST /predict` — request / response

- **Request:** `multipart/form-data`, files under `images`, index-aligned
  relative paths under `paths[]`. Optional `thumbs=0` (skip base-64 previews),
  `aggregate=0` (skip per-chunk metrics). Accepted: `.jpg .jpeg .png .bmp
  .webp`. Size cap: `MAX_UPLOAD_MB`.
- **Response:** `{ "results": [...], "aggregate": {...} | null }`.

`results` — one object per file:

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
  "threshold_used": 0.05
}
```

`aggregate` (present with ≥ 2 labelled images, or from `/aggregate` for the whole
job):

```json
{
  "n": 40,
  "threshold": 0.002,
  "metrics": { "accuracy": 0.8872, "precision": 0.8901, "recall": 0.8875, "f1": 0.8887, "roc_auc": 0.9564 },
  "error_analysis": { "false_positives": 12, "false_negatives": 10, "fp_mean_confidence": 0.7341, "fn_mean_confidence": 0.3102 }
}
```

On a per-file error (unsupported type, corrupt image, …) the result object
carries `"error": "<message>"` instead of the prediction fields.

&nbsp;

## Team Member Contributions

Claire: Set up CLIPS + Low Frequency Patch & Perform Hybrid (CLIPS integrated with EfficientNet B1) training + evaluation (including cross evaluation)

Clarice: Performed EfficientNet B0 Baseline training + evaluation (including cross evaluation) & wrote README

Sabrina: Perform Testing & analyse outputs & wrote README & recorded DEMO video

Tesmond: Perform EfficientNet B2 training + evaluation (including cross evaluation) & analyse outputs

Xin Yin: Set up base code & perform EfficientNet B0 training + evaluation (including cross evaluation)
