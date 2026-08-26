"""
infer.py

REQUIRED DELIVERABLE SCRIPT (per challenge spec 5.5.2):
Takes a directory of images and outputs a JSON file containing, for each
image, its path and a confidence score ("pred") indicating the likelihood
that it is AI-generated (0 = confidently real, 1 = confidently fake).

Usage:
    python3 -m src.infer --input_dir path/to/images --output preds.json

    # with a specific checkpoint / backbone:
    python3 -m src.infer --input_dir path/to/images --checkpoint outputs/checkpoints/best.pt --backbone efficientnet_b0

Output format (preds.json):
    [
      {"image_path": "path/to/images/img1.jpg", "pred": 0.9812},
      {"image_path": "path/to/images/img2.jpg", "pred": 0.0421},
      ...
    ]
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.dataset import get_val_transform, IMG_SIZE
from src.model import build_model, get_device

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AIGC detection inference on a directory of images"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing images (searched recursively)")
    parser.add_argument("--output", type=str, default="outputs/predictions/preds.json",
                         help="Path to write the output JSON file")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


class InferenceDataset(Dataset):
    """
    Loads every image found (recursively) under a directory -- no labels,
    no REAL/FAKE subfolder assumption. Used purely for prediction.
    """

    def __init__(self, input_dir, transform=None):
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.filepaths = sorted(
            str(p) for p in self.input_dir.rglob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        if len(self.filepaths) == 0:
            raise RuntimeError(f"No images found under {self.input_dir}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"Failed to read image (corrupt or unsupported format): {filepath}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, filepath


@torch.no_grad()
def run_inference(model, loader, device):
    results = []
    for images, filepaths in tqdm(loader, desc="infer"):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()

        for filepath, prob in zip(filepaths, probs):
            results.append({"image_path": filepath, "pred": round(float(prob), 4)})

    return results


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # --- Load model ---
    model = build_model(args.backbone, pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Load data ---
    dataset = InferenceDataset(args.input_dir, transform=get_val_transform(IMG_SIZE))
    print(f"Found {len(dataset)} images under {args.input_dir}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    # --- Run inference ---
    results = run_inference(model, loader, device)

    # --- Save output ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} predictions to {output_path}")

    # Quick sanity summary
    n_predicted_fake = sum(1 for r in results if r["pred"] > 0.5)
    print(f"Predicted fake: {n_predicted_fake} / {len(results)} "
          f"({100 * n_predicted_fake / len(results):.1f}%)")


if __name__ == "__main__":
    main()