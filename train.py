# Standard library imports
import argparse
import json
import random
import sys
from pathlib import Path

# Third-party ML / numerical libraries
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                                       ReduceLROnPlateau, StepLR)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # progress bar helper

# Project-local modules
from config import (BEST_WEIGHTS, DATA_ROOT, EVAL_CFG, TRAIN_CFG, ROBUSTNESS_CFG,
                    MODEL_CFG, TrainConfig, weights_path)
from datasets import RealAIDataset  # custom dataset that loads real vs AI images
from metrics import (classification_metrics, robustness_score, final_score,
                     optimal_threshold)
from augmentations import get_robustness_transforms
from evaluate import predict_probs, error_analysis
from model import build_model  # factory that constructs the detector network


def resolve_device(pref):
    # Pick the compute device. "auto" prefers the fastest available backend.
    if pref != "auto":
        # User explicitly requested a device string such as "cpu" or "cuda:0".
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU acceleration
    return torch.device("cpu")  # fallback when no accelerator is present


def make_scheduler(optimizer, cfg):
    # Build the learning-rate schedule object based on the chosen strategy.
    if cfg.scheduler == "cosine_warm_restarts":
        # LR follows a cosine curve and "restarts" every T_0 epochs (T_mult grows each cycle).
        return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.t_max, T_mult=cfg.t_mult)
    if cfg.scheduler == "reduce_on_plateau":
        # LR is reduced by `factor` when the monitored metric stops improving.
        return ReduceLROnPlateau(optimizer, mode="max", factor=cfg.plateau_factor,
                                 patience=cfg.plateau_patience, verbose=True)
    if cfg.scheduler == "step":
        # LR is multiplied by `gamma` every `step_size` epochs.
        return StepLR(optimizer, step_size=cfg.t_max, gamma=cfg.plateau_factor)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def _subsample(ds, n, seed):
    # Shrink a dataset in place to ~n samples, kept label-balanced where
    # possible. Used by the --limit smoke-test flag so a full pipeline run
    # (data -> CLIP -> train step -> checkpoint -> eval) takes minutes.
    if not n or n >= len(ds.samples):
        return
    rng = random.Random(seed)
    pos = [s for s in ds.samples if s[1] == 1]
    neg = [s for s in ds.samples if s[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = max(1, n // 2)
    picked = pos[:half] + neg[:half]
    rng.shuffle(picked)
    ds.samples = picked


def build_loaders(cfg, root=DATA_ROOT, limit=None):
    # Create the training and validation datasets, then wrap them in DataLoaders.
    # Training uses the configured augmentation level; validation never augments.
    # `model_name` picks the image pipeline (hybrid_clip needs the 4-channel
    # RGB+FFT tensor; everything else the ImageNet-normalised 3-channel one).
    train_ds = RealAIDataset(root, "train", cfg.image_size, cfg.augment_level,
                             seed=cfg.seed, model_name=MODEL_CFG.name)
    val_ds = RealAIDataset(root, "val", cfg.image_size, 0,
                           seed=cfg.seed, model_name=MODEL_CFG.name)
    if limit:
        # Cap train at `limit`, val at a fifth of it (min 10) -- enough to get a
        # real validation metric without slowing the smoke test down.
        _subsample(train_ds, limit, cfg.seed)
        _subsample(val_ds, max(10, limit // 5), cfg.seed + 1)
        print(f"--limit {limit}: using {len(train_ds)} train / {len(val_ds)} val samples")
    if len(train_ds) == 0:
        # Guard so we fail loudly instead of training on an empty dataset.
        raise SystemExit(f"Training set is empty. Expected images under {Path(root) / 'train'}/{{real,ai}}.")
    # Training loader shuffles and drops the last partial batch for stable batch stats.
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, drop_last=True)
    # Validation loader is only built if validation data exists; otherwise it is None.
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers) if len(val_ds) else None
    return train_dl, val_dl


def compute_pos_weight(train_ds):
    # Compute a per-class weight for the positive ("AI") class to counter class imbalance.
    # pos_weight = (#negatives) / (#positives): larger when AI images are rarer.
    n_pos = sum(1 for _, label, _ in train_ds.samples if label == 1)
    n_neg = len(train_ds.samples) - n_pos
    if n_pos == 0:
        # No positive samples -> weighting is meaningless, return None.
        return None
    return torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)


@torch.no_grad()  # disable gradient tracking -> less memory, faster, no training
def evaluate_loader(model, loader, device, threshold):
    # Run the model over an entire loader and collect ground-truth labels + predicted probabilities.
    model.eval()  # switch to evaluation mode (affects dropout / batchnorm)
    y_true, y_prob = [], []  # lists of true labels and predicted probabilities
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        # model outputs raw logits -> sigmoid converts to a probability in [0, 1]
        prob = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        y_true.extend(y.numpy().astype(int).tolist())
        y_prob.extend(np.atleast_1d(prob).tolist())
    # Compute precision/recall/F1/accuracy from the collected predictions.
    return classification_metrics(y_true, y_prob, threshold)


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    # Train the model for a single epoch and return the average loss over the dataset.
    model.train()  # enable training-mode behaviour (dropout / batchnorm update)
    running_loss = 0.0
    seen = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device).unsqueeze(1)  # move batch to device, shape labels (B,1)
        optimizer.zero_grad()  # clear gradients from the previous step
        if use_amp and device.type == "cuda":
            # Automatic Mixed Precision: run parts in float16 to speed up + save memory.
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            # Gradient scaling keeps float16 gradients numerically stable.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard full-precision forward/backward/update path.
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        # Accumulate loss weighted by the number of samples in this batch.
        running_loss += loss.item() * x.size(0)
        seen += x.size(0)
        # Live running-mean loss on the progress bar.
        pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")
    # Divide by total samples to get the mean loss for the epoch.
    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validation_confidence(model, loader, device):
    # ROC-AUC is threshold-independent and stays the headline validation metric;
    # the hard metrics (accuracy/F1/...) and the correct-vs-wrong confidence
    # breakdown are computed at the ROC-optimal decision threshold (Youden's J).
    model.eval()
    y_true, y_prob = [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        prob = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        y_true.extend(y.numpy().astype(int).tolist())
        y_prob.extend(np.atleast_1d(prob).tolist())
    best_thr = optimal_threshold(y_true, y_prob)
    metrics, y_pred = classification_metrics(y_true, y_prob, best_thr)
    correct_conf, wrong_conf = [], []
    for t, p, pr in zip(y_true, y_pred, y_prob):
        (correct_conf if p == t else wrong_conf).append(pr)
    metrics["threshold"] = best_thr  # record the operating point that was used
    metrics["confidence_score"] = float(np.mean(y_prob))  # average predicted probability
    metrics["mean_conf_correct"] = float(np.mean(correct_conf)) if correct_conf else float("nan")
    metrics["mean_conf_wrong"] = float(np.mean(wrong_conf)) if wrong_conf else float("nan")
    return metrics, y_true, y_prob


class _ValSampleDataset(Dataset):
    # Wraps the validation samples so they can be fed through a custom transform
    # (used for the per-epoch robustness evaluation on the val split).
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, _cls = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), label, str(p)


class _TransformedView(Dataset):
    # Applies a single robustness transform (on top of the base pipeline) to each
    # validation image, mirroring evaluate.py's robustness evaluation.
    def __init__(self, samples, fn, base_tf):
        self.samples = samples
        self.fn = fn
        self.base_tf = base_tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label, _cls = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.fn(img)
        return self.base_tf(img), label, str(p)


@torch.no_grad()
def evaluate_val_robustness(model, val_ds, device, batch_size, severity,
                            max_samples=2000):
    # Per-transform ROC-AUC on the validation set. This runs 15 full passes, so
    # on a large val split we sample a fixed label-balanced subset (default
    # 2000) -- enough for a stable per-epoch robustness signal without turning
    # the check into a multi-hour job. Set max_samples=0 to use all of val.
    base_tf = val_ds.transform
    samples = val_ds.samples
    if max_samples and len(samples) > max_samples:
        rng = random.Random(0)  # fixed subset so the metric is comparable across epochs
        pos = [s for s in samples if s[1] == 1]
        neg = [s for s in samples if s[1] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        half = max(1, max_samples // 2)
        samples = pos[:half] + neg[:half]
    rob = get_robustness_transforms(severity)
    per_transform_auc = {}
    for name, (category, fn) in rob.items():
        loader = DataLoader(_TransformedView(samples, fn, base_tf),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        probs, labels, _ = predict_probs(model, loader, device, desc=name)
        m, _ = classification_metrics(labels, probs, 0.5)
        per_transform_auc[name] = m["roc_auc"]
    return per_transform_auc


def ask_continue(epoch, num_epochs, enabled=True):
    # Interactive early-stop check. Skipped entirely when disabled (--no_prompt)
    # or when stdin is not a real terminal, so piped/non-interactive runs keep
    # going automatically.
    if not enabled or not sys.stdin.isatty():
        return True
    try:
        ans = input(f"\nEpoch {epoch}/{num_epochs}: continue training? "
                    f"[Enter/Y = yes, n = stop now, b = stop & keep best] ").strip().lower()
    except EOFError:
        return True
    if ans in ("n", "no", "b"):
        return False
    return True


def main(cfg=TRAIN_CFG, data_root=DATA_ROOT, limit=None, prompt=True):
    # --- Setup ---------------------------------------------------------------
    device = resolve_device(cfg.device)
    # Checkpoint path is derived from the model variant (e.g. efficientnet_b0
    # vs efficientnet_b1) so training different variants never overwrites
    # each other's saved weights.
    weights_out = weights_path(MODEL_CFG.name)
    print(f"Device: {device} | model: {MODEL_CFG.name} | augment_level: {cfg.augment_level}")
    print(f"Checkpoint will be saved to: {weights_out}")
    # Fix all seeds so runs are reproducible.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # --- Data, model, loss, optimizer ---------------------------------------
    train_dl, val_dl = build_loaders(cfg, root=data_root, limit=limit)
    model = build_model(MODEL_CFG, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Weight the loss to handle imbalanced real/AI classes.
    pos_weight = compute_pos_weight(train_dl.dataset)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        print(f"Class imbalance -> pos_weight (ai): {pos_weight.item():.1f}")
    # BCEWithLogitsLoss expects raw logits and applies sigmoid internally (numerically safer).
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Only optimise parameters that require grad -- the hybrid model's CLIP
    # encoder is frozen, so its ~87M params must be excluded from the optimiser.
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_trainable:,}")
    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    # GradScaler is only meaningful with AMP on CUDA; disabled elsewhere.
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    # --- Training loop bookkeeping ------------------------------------------
    best_metric = -float("inf")  # best value of the monitored validation metric
    epochs_no_improve = 0  # counter for early stopping
    history = []  # per-epoch records for later analysis

    # --- Main training loop --------------------------------------------------
    for epoch in range(1, cfg.num_epochs + 1):
        loss = train_epoch(model, train_dl, optimizer, criterion, device, scaler, cfg.amp)
        if val_dl is not None:
            # Evaluate on the validation set. ROC-AUC is threshold-independent and
            # is the primary metric; hard metrics use the ROC-optimal threshold.
            val_metrics, y_true, y_prob = validation_confidence(model, val_dl, device)
            # The metric we track for "best model" / scheduling decisions.
            monitor_val = val_metrics.get(cfg.monitor, -float("inf"))
            print(f"Epoch {epoch:02d}/{cfg.num_epochs} | loss {loss:.4f} | "
                  f"val_roc_auc {val_metrics['roc_auc']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
                  f"thr {val_metrics['threshold']:.3f} | "
                  f"conf {val_metrics['confidence_score']:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

            # --- Per-epoch diagnostics: error analysis + robustness + final score ---
            err = error_analysis(np.array(y_prob), np.array(y_true), None, val_metrics["threshold"])
            print(f"  errors: FP(real->AI)={err['false_positives']} "
                  f"(conf {err['fp_mean_confidence']}) | "
                  f"FN(AI->real)={err['false_negatives']} (conf {err['fn_mean_confidence']})")
            print(f"  -> {err['interpretation']}")

            if (epoch % cfg.robust_eval_every) == 0:
                per_t_auc = evaluate_val_robustness(model, val_dl.dataset, device,
                                                    cfg.batch_size, ROBUSTNESS_CFG.severity)
                rob = robustness_score(per_t_auc, ROBUSTNESS_CFG.weight_by_severity)
                score = final_score(val_metrics["roc_auc"], rob["robustness_score"])
                worst = sorted(per_t_auc.items(), key=lambda kv: kv[1])[:3]
                print(f"  robustness AUC: {rob['robustness_score']:.4f} | "
                      f"final score (0.5*AUC_clean+0.5*AUC_robust): {score:.4f}")
                print(f"  weakest transforms: " + ", ".join(f"{n}={a:.3f}" for n, a in worst))
        else:
            # No validation set: treat negative training loss as the monitor signal.
            monitor_val = -loss
            val_metrics = {"loss": loss}
            print(f"Epoch {epoch:02d}/{cfg.num_epochs} | loss {loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        # Step the LR scheduler. ReduceLROnPlateau needs the monitored value.
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(monitor_val)
        else:
            scheduler.step()

        # Optionally decay weight_decay each epoch to regularize later training.
        if cfg.wd_decay_factor < 1.0:
            for g in optimizer.param_groups:
                g["weight_decay"] = max(1e-6, g["weight_decay"] * cfg.wd_decay_factor)

        history.append({"epoch": epoch, "loss": loss, **val_metrics})

        # --- Checkpointing / early stopping --------------------------------
        if monitor_val > best_metric:
            best_metric = monitor_val
            epochs_no_improve = 0  # reset the no-improvement counter
            # Persist the best model plus metadata needed to resume/infer later.
            torch.save({
                "model_state": model.state_dict(),
                "model_name": MODEL_CFG.name,
                "config": cfg.__dict__,
                "epoch": epoch,
                "metrics": val_metrics,
            }, weights_out)
            print(f"  -> saved best weights (monitor={cfg.monitor}: {best_metric:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                # Stop early once validation stops improving for `patience` epochs.
                print(f"Early stopping: no improvement for {epochs_no_improve} epochs.")
                break

        # --- Interactive early stop ------------------------------------------
        # After each epoch, let the user stop once results look acceptable.
        if not ask_continue(epoch, cfg.num_epochs, enabled=prompt):
            print("Stopping early by user request; best weights are kept.")
            break

    # --- Teardown -----------------------------------------------------------
    # Write the full training history to a JSON file next to the weights.
    with open(weights_out.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best {cfg.monitor}: {best_metric:.4f}")


def parse_args():
    # Build a CLI so users can override any default training hyperparameter.
    ap = argparse.ArgumentParser(description="Train real-vs-AI image detector.")
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.num_epochs)
    ap.add_argument("--batch_size", type=int, default=TRAIN_CFG.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN_CFG.learning_rate)
    ap.add_argument("--augment_level", type=int, default=TRAIN_CFG.augment_level)
    ap.add_argument("--scheduler", type=str, default=TRAIN_CFG.scheduler)
    ap.add_argument("--monitor", type=str, default=TRAIN_CFG.monitor)
    ap.add_argument("--patience", type=int, default=TRAIN_CFG.early_stopping_patience)
    ap.add_argument("--device", type=str, default=TRAIN_CFG.device)
    ap.add_argument("--num_workers", type=int, default=TRAIN_CFG.num_workers)
    ap.add_argument("--seed", type=int, default=TRAIN_CFG.seed)
    ap.add_argument("--robust_eval_every", type=int, default=TRAIN_CFG.robust_eval_every,
                    help="Run the validation robustness eval every N epochs (1 = every "
                         "epoch). It is a full val pass per transform (15), so raise "
                         "this for hybrid_clip on MPS/CPU.")
    ap.add_argument("--no_prompt", action="store_true",
                    help="Disable the interactive continue/stop prompt after each epoch.")
    ap.add_argument("--model", type=str, default=MODEL_CFG.name,
                    help="TorchVision EfficientNet variant (e.g. efficientnet_b0/b1/b2) "
                         "or a hybrid variant: hybrid_clip (small-CNN spatial branch), "
                         "hybrid_effb0 / hybrid_effb1 (EfficientNet spatial branch). "
                         "Each variant is saved to its own checkpoint file.")
    ap.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                    help="Dataset root containing {train,val}/{real,ai}. Point at a small "
                         "subset folder for a quick pipeline smoke test.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of training samples (val scaled down too) for a "
                         "fast end-to-end test run. Omit for a full run.")
    return ap.parse_args()


if __name__ == "__main__":
    # Entry point: parse CLI args, build the config object, and start training.
    a = parse_args()
    # MODEL_CFG is a shared singleton object imported by model.py's build_model()
    # default argument, so mutating it here propagates without needing to pass
    # it explicitly everywhere.
    MODEL_CFG.name = a.model
    cfg = TrainConfig(
        num_epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr,
        augment_level=a.augment_level, scheduler=a.scheduler, monitor=a.monitor,
        early_stopping_patience=a.patience, device=a.device, seed=a.seed,
        num_workers=a.num_workers, robust_eval_every=a.robust_eval_every,
    )
    main(cfg, data_root=Path(a.data_root), limit=a.limit, prompt=not a.no_prompt)
