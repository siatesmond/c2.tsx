"""Image transformation primitives and augmentation sets.

Provides:
  * Reusable PIL-based transforms (JPEG, blur, resize, noise, color jitter, crop, ...).
  * `apply_training_augmentation()` - the sampler used on-the-fly during training.
  * `get_robustness_transforms()` - the FIXED 15-transform set used by the
    evaluation robustness score (matches the hackathon spec exactly, unchanged).

Design goals for the training sampler (see config.TRAIN_AUG_CFG):
  1. Cover the same 6 families the robustness eval measures (JPEG, blur, resize,
     noise, color jitter, crop) so training exposure roughly matches the eval space.
  2. Sample a *random* severity per family per image, instead of one fixed value,
     and deliberately keep those draws away from the exact discrete points the
     eval harness tests at (see config.EVAL_SEVERITY_POINTS) -- training on the
     exact eval grid would make the robustness score partly measure memorization
     of those specific parameters rather than genuine generalization.
  3. Apply AT MOST ONE family per image -- transforms are never stacked. The
     robustness spec evaluates each transform individually, and a typical
     real-world re-upload applies a single degradation step, so training
     matches that rather than mixing/matching corruptions.
  4. Keep a separate "generalization" pool (small rotation, grayscale) that
     is NOT part of the evaluated families at all. This adds useful invariances
     without ever overlapping the eval space, so it can't inflate/deflate the
     robustness score in either direction. Horizontal/vertical flips are
     deliberately excluded -- for AI-detection, orientation can itself be a
     useful signal (asymmetric artifacts, mirrored text), so we don't want
     the model to become invariant to it.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from config import EVAL_SEVERITY_POINTS, TRAIN_AUG_CFG

# ---------------------------------------------------------------------------
# Core transform primitives (shared by training augmentation AND the fixed
# robustness eval set below -- keep these signatures stable).
# ---------------------------------------------------------------------------


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


def slight_rotate(img, degrees):
    """Small rotation, resample-filled, size preserved (camera-tilt realism)."""
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(127, 127, 127))


def grayscale(img):
    """Desaturate fully (some re-shares/filters strip color)."""
    return ImageOps.grayscale(img).convert("RGB")


# ---------------------------------------------------------------------------
# Fixed robustness evaluation set -- DO NOT change 
#   JPEG q in {90,70,50,30}, Blur sigma in {0.5,1.0,2.0},
#   Resize scale in {0.5,0.25}, Noise sigma in {0.02,0.05,0.10},
#   Color jitter +/-20%, Center crop 80%.
# ---------------------------------------------------------------------------


def get_robustness_transforms(severity=None):
    """Return the fixed 15-transform robustness evaluation set."""
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


# ---------------------------------------------------------------------------
# Training-time augmentation sampler.
# ---------------------------------------------------------------------------

# The 6 robustness families, mapped to (callable, severity-range-attr-name,
# eval-grid key). Keeping this table means adding/removing a family only
# requires one line here, everything else (sampling, weighting) follows.
FAMILY_FUNCS = {
    "jpeg": jpeg_compress,
    "blur": gaussian_blur,
    "resize": resize_down_up,
    "noise": gaussian_noise,
    "color": color_jitter,
    "crop": center_crop,
}
FAMILIES = list(FAMILY_FUNCS.keys())

GENERALIZATION_POOL = [
    ("slight_rotation", lambda im, rng: slight_rotate(im, rng.uniform(-15, 15))),
    ("grayscale", lambda im, rng: grayscale(im)),
]


def _sample_away_from_grid(rng, lo, hi, grid_points, margin):
    """Uniformly sample in [lo, hi], retrying if the draw lands within `margin`
    of any exact value the robustness eval will test at. This is what keeps
    training severities "eval-adjacent but not identical" -- covering the same
    realistic range without literally training on the eval grid points.
    """
    for _ in range(25):
        val = rng.uniform(lo, hi)
        if all(abs(val - g) >= margin for g in grid_points):
            return val
    return val  # fall back to last draw if we couldn't avoid the grid (rare)


def _draw_severity(family, rng, cfg):
    grid = EVAL_SEVERITY_POINTS[family]
    if family == "jpeg":
        lo, hi = cfg.jpeg_quality_range
        q = _sample_away_from_grid(rng, lo, hi, grid, margin=4)
        return int(round(q))
    if family == "blur":
        lo, hi = cfg.blur_sigma_range
        return _sample_away_from_grid(rng, lo, hi, grid, margin=0.12)
    if family == "resize":
        lo, hi = cfg.resize_scale_range
        return _sample_away_from_grid(rng, lo, hi, grid, margin=0.04)
    if family == "noise":
        lo, hi = cfg.noise_sigma_range
        return _sample_away_from_grid(rng, lo, hi, grid, margin=0.008)
    if family == "color":
        lo, hi = cfg.color_factor_range
        return _sample_away_from_grid(rng, lo, hi, grid, margin=0.05)
    if family == "crop":
        lo, hi = cfg.crop_frac_range
        return _sample_away_from_grid(rng, lo, hi, grid, margin=0.03)
    raise ValueError(f"Unknown family: {family}")


def apply_family_transform(img, family, rng, cfg):
    fn = FAMILY_FUNCS[family]
    severity = _draw_severity(family, rng, cfg)
    return fn(img, severity)


def apply_training_augmentation(img, rng, aug_cfg=None):
    """Sample one realistic training augmentation for a single image.

    At most ONE transform is ever applied -- corruptions are never stacked.

    - With probability `aug_cfg.generalization_prob`, draw a single transform
      from GENERALIZATION_POOL (rotate/grayscale) -- never overlaps the
      evaluated families.
    - Otherwise, with probability `aug_cfg.apply_prob`, draw exactly one
      robustness family (uniformly from FAMILIES) and apply it at a randomly
      sampled severity that avoids the exact eval grid points. With the
      remaining probability the image is left clean.
    """
    aug_cfg = aug_cfg or TRAIN_AUG_CFG

    if rng.random() < aug_cfg.generalization_prob:
        _, fn = rng.choice(GENERALIZATION_POOL)
        return fn(img, rng)

    if rng.random() >= aug_cfg.apply_prob:
        return img  # leave a clean image so the undistorted signal is kept

    fam = rng.choice(FAMILIES)
    return apply_family_transform(img, fam, rng, aug_cfg)


def get_augment_names():
    """Names of everything the training sampler can draw from (for logging)."""
    return FAMILIES + [n for n, _ in GENERALIZATION_POOL]