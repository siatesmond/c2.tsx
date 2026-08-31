import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from augmentations import get_robustness_transforms
from config import (BEST_WEIGHTS, DATA_ROOT, EVAL_CFG, MODEL_CFG,
                    ROBUSTNESS_CFG, weights_path)
from datasets import build_transform
from metrics import (classification_metrics, final_score, optimal_threshold,
                     robustness_score)
from model import build_model, load_best_weights


class TestImageDataset(Dataset):
    def __init__(self, root, image_size, model_name=None, limit=None, seed=42):
        self.root = Path(root)
        # Same pipeline the training set uses for this model (4-channel
        # RGB+FFT tensor for the hybrid detector, ImageNet-normalised 3-channel
        # otherwise).
        self.transform = build_transform(image_size, train=False,
                                         model_name=model_name)
        self.items = []
        for cls_idx, cls in enumerate(["real", "ai"]):
            folder = self.root / "test" / cls
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    if "_l1_" in p.stem or "_l2_" in p.stem:
                        continue
                    self.items.append((p, cls_idx))
        if limit and limit < len(self.items):
            # Keep a label-balanced random subset -- the robustness pass runs
            # the whole set 15 times, so a smoke test needs a small `limit`.
            rng = random.Random(seed)
            pos = [it for it in self.items if it[1] == 1]
            neg = [it for it in self.items if it[1] == 0]
            rng.shuffle(pos)
            rng.shuffle(neg)
            half = max(1, limit // 2)
            self.items = pos[:half] + neg[:half]
            rng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, label = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), label, str(p)


def resolve_device(pref):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_probs(model, loader, device, desc=None):
    model.eval()
    probs, paths, labels = [], [], []
    for x, y, ps in tqdm(loader, desc=desc or "predict", leave=False):
        x = x.to(device)
        p = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        probs.extend(np.atleast_1d(p).tolist())
        labels.extend(y.tolist())
        paths.extend(ps)
    return np.array(probs), np.array(labels), paths


def evaluate_robustness(model, dataset, device, batch_size, threshold, severity):
    # Per-transform score is ROC-AUC (threshold-independent, our primary metric);
    # the error count is still reported at the chosen decision `threshold`.
    base_tf = dataset.transform
    rob = get_robustness_transforms(severity)
    per_transform_auc = {}
    per_transform_error = {}
    n = len(rob)
    # Each transform is a full pass over the test set, so this loop does n
    # extra inference passes -- print which one is running so the process
    # visibly progresses instead of looking hung.
    for i, (name, (category, fn)) in enumerate(rob.items(), 1):
        print(f"  robustness [{i:2d}/{n}] {name} ...", flush=True)
        loader = DataLoader(
            _TransformedView(dataset, fn, base_tf),
            batch_size=batch_size, shuffle=False, num_workers=0)
        probs, labels, _ = predict_probs(model, loader, device, desc=name)
        m, preds = classification_metrics(labels, probs, threshold)
        per_transform_auc[name] = m["roc_auc"]
        per_transform_error[name] = int((preds != labels).sum())
    return per_transform_auc, per_transform_error


class _TransformedView(Dataset):
    def __init__(self, base, fn, base_tf):
        self.base = base
        self.fn = fn
        self.base_tf = base_tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        p, label = self.base.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.fn(img)
        return self.base_tf(img), label, str(p)


def error_analysis(probs, labels, paths=None, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]
    conf_fp = probs[fp_idx] if len(fp_idx) else np.array([])
    conf_fn = probs[fn_idx] if len(fn_idx) else np.array([])
    fp_examples = [paths[i] for i in fp_idx[:20]] if paths is not None else []
    fn_examples = [paths[i] for i in fn_idx[:20]] if paths is not None else []
    return {
        "false_positives": int(len(fp_idx)),
        "false_negatives": int(len(fn_idx)),
        "fp_mean_confidence": float(conf_fp.mean()) if len(conf_fp) else None,
        "fn_mean_confidence": float(conf_fn.mean()) if len(conf_fn) else None,
        "fp_examples": fp_examples,
        "fn_examples": fn_examples,
        "interpretation": _interpret(len(fp_idx), len(fn_idx), len(labels)),
    }


def _interpret(n_fp, n_fn, n):
    parts = []
    if n == 0:
        return "No test samples available."
    if n_fp >= n_fn:
        parts.append("Model leans toward flagging real images as AI "
                     "(false positives dominate), suggesting it over-triggers on artifacts.")
    else:
        parts.append("Model leans toward missing AI images "
                     "(false negatives dominate), suggesting AI fakes are becoming too realistic to spot.")
    if n_fp and n_fn:
        parts.append("Both error types are present: consider threshold tuning and the robustness report "
                     "to find which transformations expose the model.")
    return " ".join(parts)


def main(weights=BEST_WEIGHTS, data_root=DATA_ROOT, cfg=EVAL_CFG,
         rob_cfg=ROBUSTNESS_CFG, limit=None, threshold_override=None):
    device = resolve_device(cfg.device)
    model = build_model(device=device)
    load_best_weights(model, weights, device)

    ds = TestImageDataset(data_root, cfg.image_size, model_name=MODEL_CFG.name,
                          limit=limit)
    if len(ds) == 0:
        raise SystemExit("Test set is empty. Add images to data/test/{real,ai}.")
    rob_passes = len(get_robustness_transforms(rob_cfg.severity))
    print(f"Test samples: {len(ds)} | device: {device} | "
          f"{1 + rob_passes} inference passes (1 clean + {rob_passes} robustness)",
          flush=True)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print("clean pass ...", flush=True)
    probs, labels, paths = predict_probs(model, loader, device, desc="clean")

    # ROC-AUC is the primary metric and is threshold-independent. Pick the
    # ROC-optimal decision threshold (Youden's J) for all threshold-dependent
    # metrics and error counts so they reflect the best operating point --
    # unless the caller forced one with --threshold.
    best_thr = (threshold_override if threshold_override is not None
                else optimal_threshold(labels, probs))
    metrics, _ = classification_metrics(labels, probs, best_thr)
    metrics["threshold"] = best_thr

    rob_spec = get_robustness_transforms(rob_cfg.severity)
    per_t_auc, per_t_err = evaluate_robustness(model, ds, device, cfg.batch_size,
                                               best_thr, rob_cfg.severity)
    rob = robustness_score(per_t_auc, rob_cfg.weight_by_severity)

    by_family = {}
    for name, auc in per_t_auc.items():
        cat = rob_spec.get(name, ("other", None))[0]
        by_family.setdefault(cat, []).append(auc)
    robustness_by_family = {c: float(np.mean(v)) for c, v in by_family.items()}

    errors = error_analysis(probs, labels, paths, best_thr)

    score = final_score(metrics["roc_auc"], rob["robustness_score"])

    report = {
        "threshold": best_thr,
        "clean_metrics": metrics,
        "robustness": rob,
        "robustness_by_family": robustness_by_family,
        "final_score": score,
        "error_analysis": errors,
        "per_transform_errors": per_t_err,
    }

    out = Path(weights).with_name("eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== ROC-optimal decision threshold (Youden's J): {best_thr:.4f} ===")
    print("=== Clean test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"=== Robustness score: {rob['robustness_score']:.4f} (mean AUC {rob['mean_auc']:.4f}) ===")
    for fam, auc in sorted(robustness_by_family.items()):
        print(f"  {fam}: {auc:.4f}")
    worst = sorted(per_t_auc.items(), key=lambda kv: kv[1])[:3]
    print("  weakest transforms:", ", ".join(f"{n}={a:.3f}" for n, a in worst))
    print(f"=== Final score (0.5*AUC_clean + 0.5*AUC_robust): {score:.4f} ===")
    print("=== Error analysis ===")
    print(f"  false positives (real->AI): {errors['false_positives']} (mean conf {errors['fp_mean_confidence']})")
    print(f"  false negatives (AI->real): {errors['false_negatives']} (mean conf {errors['fn_mean_confidence']})")
    print(f"  {errors['interpretation']}")
    print(f"Report written to {out}")
    return report


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate real-vs-AI detector.")
    ap.add_argument("--model", type=str, default=MODEL_CFG.name,
                    help="Model variant to build (must match the checkpoint): "
                         "an EfficientNet name, or hybrid_clip / hybrid_effb0 / hybrid_effb1.")
    ap.add_argument("--weights", default=None,
                    help="Checkpoint path. Defaults to the standard path for --model.")
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="Force this decision threshold. Default: the ROC-optimal "
                         "threshold (Youden's J) computed on the clean test set.")
    ap.add_argument("--severity", type=float, default=ROBUSTNESS_CFG.severity)
    ap.add_argument("--device", type=str, default=EVAL_CFG.device)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate on a label-balanced random subset of this many "
                         "test images. Use for a fast check -- the robustness pass "
                         "runs the full set 15 times otherwise.")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    # Set the shared model name before build_model() / dataset construction.
    MODEL_CFG.name = a.model
    weights = a.weights or str(weights_path(a.model))
    EVAL_CFG.device = a.device
    ROBUSTNESS_CFG.severity = a.severity
    main(Path(weights), Path(a.data_root), EVAL_CFG, ROBUSTNESS_CFG,
         limit=a.limit, threshold_override=a.threshold)
