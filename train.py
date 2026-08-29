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
from torch.optim import AdamW
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                                       ReduceLROnPlateau, StepLR)
from torch.utils.data import DataLoader
from tqdm import tqdm  # progress bar helper

# Project-local modules
from config import (BEST_WEIGHTS, DATA_ROOT, EVAL_CFG, TRAIN_CFG,
                    MODEL_CFG, TrainConfig, weights_path)
from datasets import RealAIDataset  # custom dataset that loads real vs AI images
from metrics import classification_metrics  # computes F1/accuracy/precision/recall
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


def build_loaders(cfg, root=DATA_ROOT):
    # Create the training and validation datasets, then wrap them in DataLoaders.
    # Training uses the configured augmentation level; validation never augments.
    train_ds = RealAIDataset(root, "train", cfg.image_size, cfg.augment_level, seed=cfg.seed)
    val_ds = RealAIDataset(root, "val", cfg.image_size, 0, seed=cfg.seed)
    if len(train_ds) == 0:
        # Guard so we fail loudly instead of training on an empty dataset.
        raise SystemExit("Training set is empty. Add images to data/train/real and data/train/ai.")
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
    for x, y in loader:
        x = x.to(device)
        # model outputs raw logits -> sigmoid converts to a probability in [0, 1]
        prob = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        y_true.extend(y.numpy().astype(int).tolist())
        y_prob.extend(prob.tolist())
    # Compute precision/recall/F1/accuracy from the collected predictions.
    return classification_metrics(y_true, y_prob, threshold)


def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    # Train the model for a single epoch and return the average loss over the dataset.
    model.train()  # enable training-mode behaviour (dropout / batchnorm update)
    running_loss = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
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
    # Divide by total samples to get the mean loss for the epoch.
    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def validation_confidence(model, loader, device, threshold):
    # Like evaluate_loader, but also tracks the model's average confidence on
    # correct vs. incorrect predictions (a useful calibration signal).
    model.eval()
    y_true, y_prob, correct_conf, wrong_conf = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        yt = y.numpy().astype(int)
        pred = (prob >= threshold).astype(int)  # threshold the probability into a class
        for t, p, pr in zip(yt, pred, prob):
            if p == t:
                correct_conf.append(pr)  # store confidence for correct predictions
            else:
                wrong_conf.append(pr)  # store confidence for wrong predictions
        y_true.extend(yt.tolist())
        y_prob.extend(prob.tolist())
    metrics, _ = classification_metrics(y_true, y_prob, threshold)
    # Extra diagnostic fields describing how confident the model generally is.
    metrics["confidence_score"] = float(np.mean(y_prob))  # average predicted probability
    metrics["mean_conf_correct"] = float(np.mean(correct_conf)) if correct_conf else float("nan")
    metrics["mean_conf_wrong"] = float(np.mean(wrong_conf)) if wrong_conf else float("nan")
    return metrics


def main(cfg=TRAIN_CFG):
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
    train_dl, val_dl = build_loaders(cfg)
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
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
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
            # Evaluate on the validation set using the configured threshold.
            val_metrics = validation_confidence(model, val_dl, device, cfg.threshold)
            # The metric we track for "best model" / scheduling decisions.
            monitor_val = val_metrics.get(cfg.monitor, -float("inf"))
            print(f"Epoch {epoch:02d}/{cfg.num_epochs} | loss {loss:.4f} | "
                  f"val_f1 {val_metrics['f1']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
                  f"conf {val_metrics['confidence_score']:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")
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
    ap.add_argument("--model", type=str, default=MODEL_CFG.name,
                    help="TorchVision EfficientNet variant, e.g. efficientnet_b0/b1/b2. "
                         "Each variant is saved to its own checkpoint file.")
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
        num_workers=a.num_workers,
    )
    main(cfg)
