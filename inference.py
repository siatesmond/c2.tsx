# Standard library imports
import argparse
import json
from pathlib import Path

# Third-party imports
import torch
from PIL import Image, ImageFile
from torchvision import transforms

# Allow loading of truncated images without crashing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project-local modules
from config import BEST_WEIGHTS, MODEL_CFG
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


def load_image_tensor(path, image_size):
    # Open an image and run it through the same preprocessing used in training:
    # resize -> ToTensor -> ImageNet normalization. Add a batch dimension.
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def predict(model, paths, device, image_size, threshold):
    # Run the detector over a list of image paths and return human-readable results.
    model.eval()  # evaluation mode
    results = []
    for p in paths:
        x = load_image_tensor(p, image_size).to(device)
        # Sigmoid of the single logit -> probability that the image is AI-generated.
        prob = float(torch.sigmoid(model(x)).squeeze(-1).cpu().item())
        label = "ai" if prob >= threshold else "real"
        results.append({
            "path": str(p),
            "label": label,
            "is_ai": bool(prob >= threshold),
            "confidence": round(prob, 4),  # AI probability, rounded
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
    ap.add_argument("--weights", default=str(BEST_WEIGHTS))  # checkpoint to load
    ap.add_argument("--threshold", type=float, default=0.5)  # decision boundary
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model = build_model(MODEL_CFG, device)
    load_best_weights(model, args.weights, device)  # populate model from checkpoint

    paths = gather_paths(args.input)
    if not paths:
        raise SystemExit(f"No images found at {args.input}")
    results = predict(model, paths, device, args.image_size, args.threshold)

    if args.json:
        # Machine-readable output.
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output line per image.
        for r in results:
            print(f"{r['path']}  ->  {r['label'].upper()}  (confidence {r['confidence']})")


if __name__ == "__main__":
    main()
