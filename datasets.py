# Standard library imports
import random
from pathlib import Path

# Third-party imports
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

# Project-local augmentation helpers (random/level-based transforms)
from augmentations import apply_level_augmentation

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


def find_variants(folder, stem):
    # Look for pre-generated augmented versions of an image named like "<stem>_l1_*" and "<stem>_l2_*".
    # Returns a dict {level: [matching_paths]}. Level 0 (original) is always empty here.
    variants = {0: [], 1: [], 2: []}
    for lvl in (1, 2):
        variants[lvl] = sorted(
            p for p in Path(folder).glob(f"{stem}_l{lvl}_*")
            if p.suffix.lower() in IMAGE_EXTS
        )
    return variants


class RealAIDataset(Dataset):
    def __init__(self, root, split, image_size=224, augment_level=0,
                 use_stored_aug=True, use_onfly_fallback=True, seed=42):
        # Store configuration. Augmentation only applies to the "train" split.
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment_level = augment_level if split == "train" else 0
        self.use_stored_aug = use_stored_aug and split == "train"
        self.use_onfly_fallback = use_onfly_fallback
        # Seeded RNG so augmentation choices are deterministic per run.
        self.rng = random.Random(seed)
        self.transform = build_base_transform(image_size, split == "train")

        # Build the list of (path, label, class_name) samples by scanning folders.
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
                    # Skip augmentation files; they are picked via _pick_path instead.
                    continue
                self.samples.append((p, cls_idx, cls))

    def __len__(self):
        # Required by PyTorch Dataset: number of samples.
        return len(self.samples)

    def _pick_path(self, path, cls_dir):
        # Decide which version of an image to use: original (level 0) or a stored augmentation.
        if not self.use_stored_aug or self.augment_level <= 0:
            return path, 0
        variants = find_variants(cls_dir, path.stem)
        # Levels we are allowed to sample, e.g. level up to augment_level.
        allowed = [0]
        for lvl in range(1, self.augment_level + 1):
            if variants[lvl]:
                allowed.append(lvl)
        if len(allowed) == 1:
            # No augmentations available -> always use the original.
            return path, 0
        # Randomly pick a level, then randomly pick a file at that level.
        choice = self.rng.choice(allowed)
        if choice == 0:
            return path, 0
        return self.rng.choice(variants[choice]), choice

    def _load_sample(self, idx):
        # Load one sample (image + label), applying augmentation when appropriate.
        path, label, cls = self.samples[idx]
        cls_dir = path.parent
        chosen, chosen_level = self._pick_path(path, cls_dir)
        img = Image.open(chosen).convert("RGB")  # force 3-channel RGB

        if chosen_level == 0 and self.augment_level > 0 and self.use_onfly_fallback:
            # No stored augmentation was chosen, so generate one on the fly.
            img = apply_level_augmentation(img, self.augment_level, self.rng)

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
