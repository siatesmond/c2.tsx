import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

from augmentations import get_robustness_transforms
from config import BEST_WEIGHTS, DATA_ROOT, EVAL_CFG, ROBUSTNESS_CFG
from metrics import (classification_metrics, robustness_score, final_score,
                     optimal_threshold)
from model import build_model, load_best_weights


class TestImageDataset(Dataset):
    def __init__(self, root, image_size):
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
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
    # This script makes 16 full passes over the test set (1 clean + 15
    # robustness transforms) and prints nothing until the very end, which is
    # indistinguishable from a hang on a large set. Show progress per pass.
    for x, y, ps in tqdm(loader, desc=desc, unit="batch", leave=False, disable=desc is None):
        x = x.to(device)
        p = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(y.tolist())
        paths.extend(ps)
    return np.array(probs), np.array(labels), paths


def evaluate_robustness(model, dataset, device, batch_size, threshold, severity):
    base_tf = dataset.transform
    rob = get_robustness_transforms(severity)
    per_transform_auc = {}
    per_transform_error = {}
    total = len(dataset)
    for i, (name, (category, fn)) in enumerate(rob.items(), 1):
        loader = DataLoader(
            _TransformedView(dataset, fn, base_tf),
            batch_size=batch_size, shuffle=False, num_workers=0)
        probs, labels, _ = predict_probs(
            model, loader, device, desc=f"robustness {i}/{len(rob)}: {name}")
        preds = (probs >= threshold).astype(int)
        m, _ = classification_metrics(labels, probs, threshold)
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
         rob_cfg=ROBUSTNESS_CFG):
    device = resolve_device(cfg.device)
    model = build_model(device=device)
    load_best_weights(model, weights, device)

    ds = TestImageDataset(data_root, cfg.image_size)
    if len(ds) == 0:
        raise SystemExit("Test set is empty. Add images to data/test/{real,ai}.")

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    print(f"Scoring {len(ds)} test images (1 clean pass + 15 robustness passes)...")
    probs, labels, paths = predict_probs(model, loader, device, desc="clean")
    # ROC-AUC is the primary metric and is threshold-independent. Pick the
    # ROC-optimal decision threshold (Youden's J) for all threshold-dependent
    # metrics and error counts so they reflect the best operating point.
    best_thr = optimal_threshold(labels, probs)
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
    ap.add_argument("--weights", default=str(BEST_WEIGHTS))
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--threshold", type=float, default=EVAL_CFG.threshold)
    ap.add_argument("--severity", type=float, default=ROBUSTNESS_CFG.severity)
    ap.add_argument("--device", type=str, default=EVAL_CFG.device)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    EVAL_CFG.threshold = a.threshold
    EVAL_CFG.device = a.device
    ROBUSTNESS_CFG.severity = a.severity
    main(Path(a.weights), Path(a.data_root), EVAL_CFG, ROBUSTNESS_CFG)
