"""Test the best trained model against real-world data in real/ and ai/ folders.

Unlike evaluate.py (which expects data/test/{real,ai}), this lets you point at
any folder that contains `real/` and `ai/` subfolders (e.g. the project root).
It loads models/best_model.pth (config.BEST_WEIGHTS) and reports the same
diagnostics used during training: ROC-AUC, accuracy, F1, and a false
positive / false negative error analysis.

Usage:
    python3 test_realworld.py --root . --weights models/best_model.pth
"""

import argparse
import json

import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

from config import BEST_WEIGHTS, MODEL_CFG, ModelConfig, weights_path
from model import build_model, load_best_weights
from metrics import classification_metrics, optimal_threshold
from evaluate import error_analysis

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resolve_device(pref):
    # Same device-selection helper as training/inference.
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gather(root, cls, label):
    # Collect (path, label) for every image in root/<cls>.
    p = Path(root) / cls
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [(q, label) for q in sorted(p.iterdir()) if q.suffix.lower() in exts]


# Preprocessing must match training exactly: resize -> ToTensor -> ImageNet norm.
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    ap = argparse.ArgumentParser(description="Test best model on real-world real/ai folders.")
    ap.add_argument("--root", default=".", help="Folder containing real/ and ai/ subfolders")
    ap.add_argument("--weights", default=str(BEST_WEIGHTS), help="Path to best_model.pth")
    ap.add_argument("--model", default=MODEL_CFG.name,
                    help="TorchVision EfficientNet variant (e.g. efficientnet_b2)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision boundary")
    ap.add_argument("--batch_size", type=int, default=64, help="Images per forward pass")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="Cap samples per class (e.g. 2000 for real, all for ai)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="realworld_eval.json", help="Report output path")
    a = ap.parse_args()

    # If no explicit weights path was given, fall back to the variant-specific
    # checkpoint (models/best_model_<model>.pth) instead of the b0 default.
    if a.weights == str(BEST_WEIGHTS):
        a.weights = str(weights_path(a.model))

    device = resolve_device(a.device)
    model_cfg = ModelConfig(name=a.model, pretrained=MODEL_CFG.pretrained,
                            num_classes=MODEL_CFG.num_classes, dropout=MODEL_CFG.dropout)
    print(f"Device: {device} | model: {model_cfg.name}")
    model = build_model(model_cfg, device)
    print(f"Loading weights from {a.weights} ...")
    load_best_weights(model, a.weights, device)  # populates models/best_model.pth
    print("Model ready.")

    real_items = gather(a.root, "real", 0)
    ai_items = gather(a.root, "ai", 1)
    if a.max_per_class is not None:
        # Cap both classes (first N by sorted filename) so each contributes at
        # most --max_per_class samples (e.g. 2000 each).
        real_items = real_items[:a.max_per_class]
        ai_items = ai_items[:a.max_per_class]
    items = real_items + ai_items
    if not items:
        raise SystemExit(f"No images found under {a.root}/real and {a.root}/ai")
    n_real = sum(1 for _, y in items if y == 0)
    n_ai = len(items) - n_real
    print(f"Loaded {len(items)} images: {n_real} real, {n_ai} ai")

    probs, labels, paths = [], [], []
    with torch.no_grad():
        # Iterate over the dataset in batches with a progress bar.
        n_batches = (len(items) + a.batch_size - 1) // a.batch_size
        for start in tqdm(range(0, len(items), a.batch_size), total=n_batches,
                          desc="scoring", unit="batch"):
            batch = items[start:start + a.batch_size]
            # Stack the batch into a single tensor for one forward pass.
            x = torch.stack([TF(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
            p = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y for _, y in batch)
            paths.extend(str(pp) for pp, _ in batch)
    print(f"Scored all {len(probs)} images")
    probs, labels = np.array(probs), np.array(labels)

    # ROC-AUC is the primary metric; use the ROC-optimal threshold for hard metrics.
    best_thr = optimal_threshold(labels, probs)
    m, _ = classification_metrics(labels, probs, best_thr)
    m["threshold"] = best_thr
    err = error_analysis(probs, labels, paths, best_thr)
    report = {"threshold": best_thr, "n": len(labels), "metrics": m, "error_analysis": err}
    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Samples: {len(labels)} | ROC-AUC: {m['roc_auc']:.4f} | "
          f"thr: {best_thr:.3f} | acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}")
    print(f"FP(real->AI): {err['false_positives']} | "
          f"FN(AI->real): {err['false_negatives']}")
    print(f"{err['interpretation']}")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
