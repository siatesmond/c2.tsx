"""
evaluate.py

Loads a trained checkpoint and evaluates it against EVERY transform in
get_eval_transforms() (clean + all JPEG/blur/noise/resize/crop/jitter
variants from the challenge spec). Produces:

  1. A robustness table (CSV) -- accuracy, precision, recall, F1 per
     transform -- this is required deliverable #4.
  2. A list of misclassified examples (false positives / false negatives)
     per transform -- feeds directly into deliverable #5 (error analysis).

Usage:
    python3 -m src.evaluate --data_dir data/test --checkpoint outputs/checkpoints/best.pt

    # limit to a subset for a fast check:
    python3 -m src.evaluate --data_dir data/test --checkpoint outputs/checkpoints/best.pt --limit 1000
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AIGCDataset, get_eval_transforms
from src.model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness of a trained AIGC detector")
    parser.add_argument("--data_dir", type=str, default="data/test",
                         help="Path to evaluation data (must contain REAL/ and FAKE/ subfolders)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                         help="If set, only evaluate this many samples (for a fast check)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Sigmoid threshold for classifying fake vs real")
    parser.add_argument("--output_dir", type=str, default="outputs/results")
    parser.add_argument("--max_error_examples", type=int, default=20,
                         help="Max false positive / false negative examples to save per transform")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate_transform(model, data_dir, transform, device, batch_size, num_workers,
                        threshold, limit=None, max_error_examples=20, seed=42):
    """
    Runs the model over data_dir under a single transform pipeline.
    Returns a dict of metrics plus lists of false positive / false negative filepaths.
    """
    dataset = AIGCDataset(data_dir, transform=transform)

    if limit:
        from torch.utils.data import Subset
        import random
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(limit, len(dataset)))
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    tp = fp = tn = fn = 0
    false_positives = []  # real images predicted as fake
    false_negatives = []  # fake images predicted as real

    for images, labels, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu()
        preds = (probs > threshold).long()

        for pred, label, path, prob in zip(preds.tolist(), labels.tolist(), paths, probs.tolist()):
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
                if len(false_positives) < max_error_examples:
                    false_positives.append({"path": path, "confidence": round(prob, 4)})
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1
                if len(false_negatives) < max_error_examples:
                    false_negatives.append({"path": path, "confidence": round(prob, 4)})

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model = build_model(args.backbone, pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Run every eval transform ---
    eval_transforms = get_eval_transforms()
    print(f"Evaluating across {len(eval_transforms)} transforms on {args.data_dir}...\n")

    results = {}
    for name, transform in eval_transforms.items():
        metrics = evaluate_transform(
            model, args.data_dir, transform, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
            threshold=args.threshold, limit=args.limit,
            max_error_examples=args.max_error_examples, seed=args.seed,
        )
        results[name] = metrics
        print(
            f"{name:<20} acc={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
            f"f1={metrics['f1']:.4f}  (n={metrics['total_samples']})"
        )

    # --- Save robustness table (CSV) -- required deliverable #4 ---
    csv_path = output_dir / "robustness_table.csv"
    clean_acc = results["clean"]["accuracy"]
    with open(csv_path, "w") as f:
        f.write("transform,n_samples,accuracy,accuracy_drop_vs_clean,precision,recall,f1\n")
        for name, m in results.items():
            drop = clean_acc - m["accuracy"]
            f.write(
                f"{name},{m['total_samples']},{m['accuracy']:.4f},{drop:.4f},"
                f"{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f}\n"
            )
    print(f"\nRobustness table saved to: {csv_path}")

    # --- Save error analysis (false positives / negatives per transform) -- deliverable #5 ---
    errors_path = output_dir / "error_examples.json"
    error_summary = {
        name: {
            "false_positives": m["false_positives"],
            "false_negatives": m["false_negatives"],
        }
        for name, m in results.items()
    }
    with open(errors_path, "w") as f:
        json.dump(error_summary, f, indent=2)
    print(f"Error examples saved to: {errors_path}")

    # --- Quick summary ---
    print("\n=== Summary ===")
    print(f"Clean accuracy: {clean_acc:.4f}")
    worst = min(results.items(), key=lambda kv: kv[1]["accuracy"])
    print(f"Worst transform: {worst[0]} (accuracy={worst[1]['accuracy']:.4f}, "
          f"drop={clean_acc - worst[1]['accuracy']:.4f})")


if __name__ == "__main__":
    main()