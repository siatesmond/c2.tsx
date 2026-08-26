"""
dataset.py

Loads CIFAKE-style real/fake image data and builds augmentation pipelines
that match the hackathon's robustness transform spec:

  Transform            Parameters
  --------------------------------------------------------------
  JPEG Compression      quality = 90, 70, 50, 30
  Gaussian Blur         kernel sigma = 0.5, 1.0, 2.0
  Resize                scale 0.5x / 0.25x then upscale
  Gaussian Noise        sigma = 0.02, 0.05, 0.10
  Color Jitter          brightness/contrast/saturation +/- 20%
  Center Crop           crop 80%

Expected folder layout (standard CIFAKE Kaggle structure):

  data/raw/
      train/
          REAL/*.jpg
          FAKE/*.jpg
      test/
          REAL/*.jpg
          FAKE/*.jpg

Usage:
    from src.dataset import AIGCDataset, get_train_transform, get_val_transform

    train_ds = AIGCDataset("data/raw/train", transform=get_train_transform())
    val_ds   = AIGCDataset("data/raw/test",  transform=get_val_transform())
"""

import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224  # standard input size for EfficientNet-B0 / most timm backbones

# Label convention used everywhere in this project: 0 = REAL, 1 = FAKE (AI-generated)
LABEL_MAP = {"REAL": 0, "FAKE": 1, "real": 0, "fake": 1}


class AIGCDataset(Dataset):
    """
    Generic real/fake image dataset.

    Expects a root directory containing two subfolders (case-insensitive):
    REAL/ (or real/) and FAKE/ (or fake/), each containing images.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (filepath, label)

        for subdir in self.root_dir.iterdir():
            if not subdir.is_dir():
                continue
            label = LABEL_MAP.get(subdir.name)
            if label is None:
                continue  # skip unrelated folders
            for fp in subdir.glob("*"):
                if fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(fp), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                f"Expected subfolders named REAL/ and FAKE/ (or real/ fake/)."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label, filepath


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def get_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """
    Training-time pipeline: resize + normalize + RANDOMLY apply the
    challenge's robustness transforms so the model learns to be invariant
    to them. This is the key trick for the robustness requirement -- if we
    only trained on clean images, accuracy would collapse under transforms.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),

            # --- Robustness augmentations (applied with some probability each) ---
            A.ImageCompression(quality_range=(30, 90), p=0.5),
            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=0.3),
            # simulate downscale-then-upscale (thumbnail generation)
            A.Downscale(scale_range=(0.25, 0.5), p=0.3),
            # GaussNoise std_range is a fraction of max pixel value (0-1), matching sigma spec
            A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.3),
            A.OneOf(
                [
                    A.CenterCrop(int(img_size * 0.8), int(img_size * 0.8), p=1.0),
                ],
                p=0.2,
            ),
            A.Resize(img_size, img_size),  # re-resize in case crop changed dims

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Clean, no-augmentation pipeline -- used for validation and 'clean' eval."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# Named, SINGLE-transform pipelines for the robustness evaluation table.
# Each one applies exactly one transform at a fixed, specific parameter value,
# so evaluate.py can report clean vs. each individual transform/severity.
# ---------------------------------------------------------------------------

def get_eval_transforms(img_size: int = IMG_SIZE) -> dict:
    """
    Returns a dict of {name: A.Compose} for robustness evaluation.
    Includes 'clean' plus every transform x parameter combo from the spec.
    """
    normalize_and_tensor = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    transforms = {
        "clean": A.Compose([A.Resize(img_size, img_size), *normalize_and_tensor]),
    }

    # JPEG Compression: quality = 90, 70, 50, 30
    for q in [90, 70, 50, 30]:
        transforms[f"jpeg_q{q}"] = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.ImageCompression(quality_range=(q, q), p=1.0),
                *normalize_and_tensor,
            ]
        )

    # Gaussian Blur: sigma = 0.5, 1.0, 2.0
    for sigma in [0.5, 1.0, 2.0]:
        transforms[f"blur_sigma{sigma}"] = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.GaussianBlur(sigma_limit=(sigma, sigma), p=1.0),
                *normalize_and_tensor,
            ]
        )

    # Resize: scale 0.5x / 0.25x then upscale (thumbnail simulation)
    for scale in [0.5, 0.25]:
        transforms[f"resize_scale{scale}"] = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Downscale(scale_range=(scale, scale), p=1.0),
                *normalize_and_tensor,
            ]
        )

    # Gaussian Noise: sigma = 0.02, 0.05, 0.10 (as fraction of max pixel value)
    for sigma in [0.02, 0.05, 0.10]:
        transforms[f"noise_sigma{sigma}"] = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.GaussNoise(std_range=(sigma, sigma), p=1.0),
                *normalize_and_tensor,
            ]
        )

    # Color Jitter: brightness/contrast/saturation +/- 20%
    transforms["color_jitter_20pct"] = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0),
            *normalize_and_tensor,
        ]
    )

    # Center Crop: crop 80%
    crop_size = int(img_size * 0.8)
    transforms["center_crop_80pct"] = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.CenterCrop(crop_size, crop_size, p=1.0),
            A.Resize(img_size, img_size),
            *normalize_and_tensor,
        ]
    )

    return transforms


if __name__ == "__main__":
    # Quick sanity check -- run `python -m src.dataset data/raw/train`
    # after downloading CIFAKE into data/raw/train and data/raw/test.
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train"
    ds = AIGCDataset(root, transform=get_val_transform())
    print(f"Loaded {len(ds)} samples from {root}")
    img, label, path = ds[0]
    print(f"Sample 0: shape={img.shape}, label={label}, path={path}")

    eval_transforms = get_eval_transforms()
    print(f"\nAvailable eval transforms ({len(eval_transforms)}):")
    for name in eval_transforms:
        print(f"  - {name}")