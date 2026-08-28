"""Image transformation primitives and augmentation sets.

Provides:
  * Reusable PIL-based transforms (JPEG, blur, resize, noise, color jitter, crop).
  * `AUGMENT_POOL`  - the pool sampled by `apply_level_augmentation` for on-the-fly
                       training augmentation (levels 0/1/2).
  * `get_robustness_transforms()` - the fixed 15-transform set used by the
                       evaluation robustness score (matches the hackathon spec).
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import ROBUSTNESS_CFG


def jpeg_compress(img, quality):
    """Re-encode to JPEG at the given quality (simulates social/messaging re-upload)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img, sigma):
    """Gaussian blur with the given kernel sigma (simulates out-of-focus)."""
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_down_up(img, scale):
    """Downscale by `scale` then upscale back to the original size (thumbnail generation)."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))


def gaussian_noise(img, sigma):
    """Add zero-mean Gaussian noise with the given sigma in [0,1] (low-light sensor noise)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def color_jitter(img, factor):
    """Shift brightness, contrast and saturation together by `factor` (filter/auto-enhance)."""
    img = ImageEnhance.Brightness(img).enhance(factor)
    img = ImageEnhance.Contrast(img).enhance(factor)
    img = ImageEnhance.Color(img).enhance(factor)
    return img


def center_crop(img, frac):
    """Crop to the central `frac` fraction of the image (profile-picture framing)."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def hflip(img):
    """Horizontal flip."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """Vertical flip."""
    return img.transpose(Image.FLIP_TOP_BOTTOM)


# Training augmentation pool (sampled by apply_level_augmentation for levels 1/2).
# Covers the same transform families used in the robustness evaluation.
AUGMENT_POOL = [
    ("horizontal_flip", hflip),
    ("vertical_flip", vflip),
    ("jpeg_70", lambda im: jpeg_compress(im, 70)),
    ("blur_10", lambda im: gaussian_blur(im, 1.0)),
    ("resize_05", lambda im: resize_down_up(im, 0.5)),
    ("noise_005", lambda im: gaussian_noise(im, 0.05)),
    ("color_jitter_up", lambda im: color_jitter(im, 1.2)),
    ("color_jitter_down", lambda im: color_jitter(im, 0.8)),
    ("center_crop_80", lambda im: center_crop(im, 0.8)),
]


def get_augment_names():
    """Return the names of the training augmentation pool."""
    return [n for n, _ in AUGMENT_POOL]


def apply_level_augmentation(img, level, rng=None):
    """Apply `level` random transforms from AUGMENT_POOL (level 0 = original)."""
    if level <= 0:
        return img
    rng = rng or random.Random()
    chosen = rng.sample(range(len(AUGMENT_POOL)), k=min(level, len(AUGMENT_POOL)))
    for idx in chosen:
        _, fn = AUGMENT_POOL[idx]
        img = fn(img)
    return img


def get_robustness_transforms(severity=None):
    """Return the fixed 15-transform robustness evaluation set.

    Keys are transform names; values are (family, callable) pairs. The families
    and parameters exactly match the hackathon robustness spec:
      JPEG q in {90,70,50,30}, Blur sigma in {0.5,1.0,2.0},
      Resize scale in {0.5,0.25}, Noise sigma in {0.02,0.05,0.10},
      Color jitter +/-20%, Center crop 80%.
    """
    t = {}
    for q in (90, 70, 50, 30):
        t[f"jpeg_q{q}"] = ("compression", lambda im, q=q: jpeg_compress(im, q))
    for s in (0.5, 1.0, 2.0):
        t[f"blur_s{s}"] = ("blur", lambda im, s=s: gaussian_blur(im, s))
    for sc in (0.5, 0.25):
        t[f"resize_{sc}"] = ("resolution", lambda im, sc=sc: resize_down_up(im, sc))
    for s in (0.02, 0.05, 0.10):
        t[f"noise_s{s}"] = ("noise", lambda im, s=s: gaussian_noise(im, s))
    t["color_jitter_up"] = ("photometric", lambda im: color_jitter(im, 1.2))
    t["color_jitter_down"] = ("photometric", lambda im: color_jitter(im, 0.8))
    t["center_crop_80"] = ("spatial", lambda im: center_crop(im, 0.8))
    return t
