# Standard library imports
import random
from pathlib import Path

# Third-party imports
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Project-local augmentation helpers (random, family-based training augmentation)
from augmentations import apply_training_augmentation

# Allow PIL to load images that are truncated/incomplete (common with scraped data).
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allowed image file extensions when scanning directories.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_base_transform(image_size, train):
    # Construct the common image pipeline applied to every image:
    # resize -> ToTensor -> ImageNet-style normalization.
    # (The `train` flag is accepted for interface symmetry; augmentation is applied separately.)
    tf = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225]),   # ImageNet std
    ]
    return transforms.Compose(tf)


class RealAIDataset(Dataset):
    def __init__(self, root, split, image_size=224, augment_level=0, seed=42):
        # Store configuration. Augmentation only applies to the "train" split.
        # `augment_level` is now just an on/off switch (0 disables, >0 enables);
        # the actual mix of how many transforms and which families are sampled
        # per image comes from config.TRAIN_AUG_CFG (see augmentations.py).
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = bool(augment_level) and split == "train"
        # Seeded RNG so augmentation choices are deterministic per run.
        self.rng = random.Random(seed)
        self.transform = build_base_transform(image_size, split == "train")

        # Build the list of (path, label, class_name) samples by scanning folders.
        # NOTE: any leftover pre-generated augmentation files (named like
        # "<stem>_l1_*" / "<stem>_l2_*" from an older pipeline) are still
        # excluded here so they can never accidentally get counted as extra
        # train/val/test samples. Augmentation itself is now generated fresh
        # on the fly every epoch instead of relying on those static files.
        self.samples = []
        for cls in ["real", "ai", "full_synthetic_part2"]:
            # Label encoding: real -> 0, everything AI-generated -> 1.
            cls_idx = 0 if cls == "real" else 1
            cls_dir = self.root / split / cls
            if not cls_dir.exists():
                continue  # skip missing class folders
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue  # ignore non-image files
                if "_l1_" in p.stem or "_l2_" in p.stem:
                    # Skip old-style pre-generated augmentation files entirely.
                    continue
                self.samples.append((p, cls_idx, cls))

    def __len__(self):
        # Required by PyTorch Dataset: number of samples.
        return len(self.samples)

    def _load_sample(self, idx):
        # Load one sample (image + label), applying augmentation when appropriate.
        path, label, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")  # force 3-channel RGB

        if self.augment:
            img = apply_training_augmentation(img, self.rng)

        if self.transform is not None:
            img = self.transform(img)  # resize / to-tensor / normalize
        # Return (tensor_image, float_label) as expected by the training loop.
        return img, torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):
        # Required by PyTorch Dataset. Wraps _load_sample with a retry-on-error guard.
        for _ in range(10):
            try:
                return self._load_sample(idx)
            except Exception:
                # If a file is unreadable, fall back to a random other sample.
                idx = self.rng.randrange(len(self.samples))
        # If everything keeps failing, return a blank (black) image with label 0.
        img = torch.zeros((3, self.image_size, self.image_size))
        return img, torch.tensor(0.0, dtype=torch.float32)
