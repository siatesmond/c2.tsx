# Standard library imports
import argparse
import json
from pathlib import Path

# Third-party imports
import torch
from PIL import Image, ImageFile

# Allow loading of truncated images without crashing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project-local modules
from config import MODEL_CFG, weights_path
from datasets import build_transform
from model import build_model, load_best_weights


def resolve_device(pref):
    # Same device-selection helper as training: prefer GPU, then Apple MPS, then CPU.
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_tensor(path, image_size, model_name=None):
    # Open an image and run it through the same preprocessing used in training
    # for this model (ImageNet-normalised 3-channel, or the hybrid detector's
    # 4-channel RGB+FFT tensor). Add a batch dimension.
    img = Image.open(path).convert("RGB")
    tf = build_transform(image_size, train=False, model_name=model_name)
    return tf(img).unsqueeze(0)


@torch.no_grad()
def predict(model, paths, device, image_size, threshold, model_name=None):
    # Run the detector over a list of image paths and return human-readable results.
    model.eval()  # evaluation mode
    results = []
    for p in paths:
        x = load_image_tensor(p, image_size, model_name).to(device)
        # Sigmoid of the single logit -> probability that the image is AI-generated.
        prob = float(torch.sigmoid(model(x)).squeeze(-1).cpu().item())
        label = "ai" if prob >= threshold else "real"
        results.append({
            # Required deliverable fields (Section 5.5): "a JSON file containing
            # image_path and pred for each image" -- pred is the AI-likelihood score.
            "image_path": str(p),
            "pred": round(prob, 4),
            # Extra fields kept for human-readable / debugging convenience.
            "label": label,
            "is_ai": bool(prob >= threshold),
            "threshold": threshold,
        })
    return results


def gather_paths(input_path):
    # If the input is a directory, collect all image files; otherwise treat it as a single file.
    p = Path(input_path)
    if p.is_dir():
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        return sorted(q for q in p.iterdir() if q.suffix.lower() in exts)
    return [p]


def main():
    # CLI for running inference on one image or a folder of images.
    ap = argparse.ArgumentParser(description="Inference for real-vs-AI detector.")
    ap.add_argument("input", help="Image file or directory")
    ap.add_argument("--model", type=str, default=MODEL_CFG.name,
                    help="Model variant to build (must match the checkpoint): "
                         "an EfficientNet name, or hybrid_clip / hybrid_effb0 / hybrid_effb1.")
    ap.add_argument("--weights", default=None,
                    help="Checkpoint path. Defaults to the standard path for --model.")
    ap.add_argument("--threshold", type=float, default=0.5)  # decision boundary
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output", type=str, default="predictions.json",
                    help="Path to write the JSON predictions file (Section 5.5 deliverable)")
    ap.add_argument("--quiet", action="store_true", help="Suppress the per-image console log")
    args = ap.parse_args()

    # Set the shared model name before build_model() / transform construction.
    MODEL_CFG.name = args.model
    weights = args.weights or str(weights_path(args.model))

    device = resolve_device(args.device)
    model = build_model(MODEL_CFG, device)
    load_best_weights(model, weights, device)  # populate model from checkpoint

    paths = gather_paths(args.input)
    if not paths:
        raise SystemExit(f"No images found at {args.input}")
    results = predict(model, paths, device, args.image_size, args.threshold,
                      model_name=args.model)

    # Required deliverable: "The output should be a JSON file containing
    # image_path and pred for each image." Always written, regardless of flags.
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if not args.quiet:
        for r in results:
            print(f"{r['image_path']}  ->  {r['label'].upper()}  (pred {r['pred']})")
    print(f"Wrote {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()
