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
from config import HYBRID_CFG, is_hybrid
from frequency import stack_image_and_spectrogram

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


class HybridTransform:
    """Transform for the hybrid detector: resize -> ToTensor([0,1]) -> append
    the per-patch FFT spectrogram as a 4th channel.

    The model applies CLIP's own normalisation and the high-pass residual, so
    this stage intentionally leaves the RGB channels un-normalised. Written as
    a picklable class (not a lambda) so it works inside DataLoader workers.
    """

    def __init__(self, image_size, freq_grid=HYBRID_CFG.freq_grid):
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        self.freq_grid = freq_grid

    def __call__(self, img):
        rgb01 = self.to_tensor(self.resize(img))          # (3, H, W) in [0, 1]
        return stack_image_and_spectrogram(rgb01, grid=self.freq_grid)  # (4, H, W)


def build_transform(image_size, train, model_name=None):
    """Pick the right image pipeline for the configured model."""
    if is_hybrid(model_name):
        return HybridTransform(image_size)
    return build_base_transform(image_size, train)


class RealAIDataset(Dataset):
    def __init__(self, root, split, image_size=224, augment_level=0, seed=42,
                 model_name=None):
        # Store configuration. Augmentation only applies to the "train" split.
        # `augment_level` is now just an on/off switch (0 disables, >0 enables);
        # the actual mix of how many transforms and which families are sampled
        # per image comes from config.TRAIN_AUG_CFG (see augmentations.py).
        # `model_name` selects the image pipeline: the hybrid detector needs the
        # 4-channel (RGB + FFT spectrogram) tensor, everything else the
        # ImageNet-normalised 3-channel one.
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = bool(augment_level) and split == "train"
        # Seeded RNG so augmentation choices are deterministic per run.
        self.rng = random.Random(seed)
        self.transform = build_transform(image_size, split == "train", model_name)

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
        # If everything keeps failing, return a blank image with label 0.
        # Match the channel count the transform would have produced (4 for the
        # hybrid RGB+spectrogram tensor, 3 otherwise).
        n_ch = 4 if isinstance(self.transform, HybridTransform) else 3
        img = torch.zeros((n_ch, self.image_size, self.image_size))
        return img, torch.tensor(0.0, dtype=torch.float32)
