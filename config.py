"""Project configuration and shared paths.

Centralises all hyperparameters (model, training, evaluation, robustness) and
filesystem locations so every module reads from one place. The robustness spec
below documents the exact transform families/parameters used for evaluation.
"""

from dataclasses import dataclass
from pathlib import Path

# Root of this project (the folder containing this file).
PROJECT_ROOT = Path(__file__).resolve().parent

# Standard data layout: data/{train,val,test}/{real,ai,...}
DATA_ROOT = PROJECT_ROOT / "data"
# Raw dump location used by preprocess.py before splitting.
RAW_ROOT = PROJECT_ROOT / "raw"
# Where trained model weights are stored.
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def weights_path(model_name):
    """Per-model-variant checkpoint path, e.g. models/best_model_efficientnet_b1.pth.

    Use this (not a single hardcoded filename) whenever more than one model
    variant might get trained -- otherwise training efficientnet_b0 and then
    efficientnet_b1 would silently overwrite the same best_model.pth, and
    you'd lose the ability to compare variants afterwards.
    """
    safe = model_name.replace("/", "_")
    return MODELS_DIR / f"best_model_{safe}.pth"

# Train / val / test split proportions used by preprocess.py.
SPLIT_RATIOS = (0.7, 0.15, 0.15)


@dataclass
class ModelConfig:
    # TorchVision EfficientNet variant to use as the backbone.
    name: str = "efficientnet_b0"
    # Load ImageNet pretrained weights.
    pretrained: bool = True
    # 1 -> single logit + sigmoid (binary real-vs-AI); 2 -> softmax.
    num_classes: int = 1
    # Dropout applied before the classification head.
    dropout: float = 0.3


@dataclass
class TrainConfig:
    # Input resolution (images are resized to this square).
    image_size: int = 224
    batch_size: int = 32
    # Total training epochs (early stopping may end sooner).
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    # Multiplicative decay applied to weight_decay each epoch.
    wd_decay_factor: float = 0.95
    optimizer: str = "adamw"
    # LR scheduler: cosine_warm_restarts (in/decreases LR), reduce_on_plateau, step.
    scheduler: str = "cosine_warm_restarts"
    t_max: int = 8
    t_mult: int = 1
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    # Stop training after this many epochs without improvement on `monitor`.
    early_stopping_patience: int = 8
    # Validation metric used to pick the best model and for early stopping.
    monitor: str = "f1"
    # Decision threshold for converting probabilities to class predictions.
    threshold: float = 0.5
    # Master on/off switch for training-time augmentation. Which family gets
    # applied (and at what severity) is controlled by TRAIN_AUG_CFG below.
    # Note: at most ONE transform is ever applied per image -- see
    # TrainAugConfig.apply_prob / augmentations.apply_training_augmentation.
    augment_level: int = 1
    num_workers: int = 4
    seed: int = 42
    # "auto" picks cuda > mps > cpu.
    device: str = "auto"
    # Use automatic mixed precision (only effective on cuda).
    amp: bool = True


@dataclass
class EvalConfig:
    image_size: int = 224
    batch_size: int = 32
    threshold: float = 0.5
    num_workers: int = 4
    device: str = "auto"


@dataclass
class RobustnessConfig:
    # Legacy knob retained for API compatibility; the 15-transform set is fixed.
    severity: float = 0.5
    # False -> robustness score is the plain mean accuracy over all 15 transforms.
    weight_by_severity: bool = False


@dataclass
class TrainAugConfig:
    """Controls the on-the-fly training augmentation sampler
    (augmentations.apply_training_augmentation).

    Priorities encoded here (see augmentations.py docstring for the full
    rationale):
      * Cover the same 6 families the robustness eval uses, at RANDOM
        severities per image rather than one fixed value each.
      * Keep those random severities away from the exact eval grid points
        (config.EVAL_SEVERITY_POINTS) so the robustness score measures
        generalization, not memorization of the eval parameters.
      * Apply AT MOST ONE transform per image -- never stacked. The brief
        evaluates robustness against each transform individually ("a subset
        of the following augmentations"), so training mirrors that: one
        degradation per image, matching a single real-world re-upload step.
      * Route a separate slice of samples to transforms that are NOT part of
        the evaluated families at all (rotation/grayscale), so those never
        touch -- and can't leak into -- the eval space.
    """

    # Probability that a robustness-family transform is applied at all. With
    # probability (1 - apply_prob) the image is left clean, so the model keeps
    # seeing the undistorted signal. When a transform IS applied, exactly one
    # family is drawn (uniformly) -- transforms are never combined.
    apply_prob: float = 0.85

    # Probability of instead drawing a single transform from the
    # generalization-only pool (rotate/grayscale) -- mutually exclusive with
    # the robustness-family draw above.
    generalization_prob: float = 0.15

    # Per-family severity ranges sampled at TRAINING time. These intentionally
    # span (and go a bit beyond) the eval grid so the model sees the general
    # shape of each corruption family, while _sample_away_from_grid() in
    # augmentations.py keeps individual draws off the exact eval values.
    jpeg_quality_range: tuple = (20, 95)
    blur_sigma_range: tuple = (0.3, 2.5)
    resize_scale_range: tuple = (0.2, 0.6)
    noise_sigma_range: tuple = (0.01, 0.12)
    color_factor_range: tuple = (0.7, 1.3)
    crop_frac_range: tuple = (0.65, 0.95)


# The exact robustness evaluation spec (matches the TikTok TechJam brief):
# 6 transform families with the specified parameter levels -> 15 transforms total.
ROBUSTNESS_SPEC = {
    "JPEG Compression": [("quality", [90, 70, 50, 30])],
    "Gaussian Blur": [("sigma", [0.5, 1.0, 2.0])],
    "Resize (downscale then upscale)": [("scale", [0.5, 0.25])],
    "Gaussian Noise": [("sigma", [0.02, 0.05, 0.10])],
    "Color Jitter (brightness/contrast/sat)": [("factor", [1.2, 0.8])],
    "Center Crop": [("crop", [0.8])],
}

# Same information as ROBUSTNESS_SPEC, keyed the way augmentations.py's family
# names are, and used ONLY to steer training severities away from these exact
# points (see TrainAugConfig / _sample_away_from_grid). This is the guard
# against accidentally "training on the eval set" in parameter space.
EVAL_SEVERITY_POINTS = {
    "jpeg": [90, 70, 50, 30],
    "blur": [0.5, 1.0, 2.0],
    "resize": [0.5, 0.25],
    "noise": [0.02, 0.05, 0.10],
    "color": [1.2, 0.8],
    "crop": [0.8],
}


# Singleton config instances imported across the project.
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()
ROBUSTNESS_CFG = RobustnessConfig()
TRAIN_AUG_CFG = TrainAugConfig()

# Default checkpoint path for the currently configured model variant (used as
# the default --weights in evaluate.py/inference.py). If you train a
# different variant via `train.py --model efficientnet_b1`, that run computes
# its own path via weights_path() rather than using this constant, so it
# won't overwrite a different variant's checkpoint.
BEST_WEIGHTS = weights_path(MODEL_CFG.name)