"""Project configuration and shared paths.

Centralises all hyperparameters (model, training, evaluation, robustness) and
filesystem locations so every module reads from one place. The robustness spec
below documents the exact transform families/parameters used for evaluation.
"""

from dataclasses import dataclass, field
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

# Filename for the best checkpoint (chosen by the monitored validation metric).
BEST_WEIGHTS = MODELS_DIR / "best_model.pth"

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
    # Augmentation strength during training: 0 = none, 1 = one mod, 2 = two mods.
    augment_level: int = 2
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


# Singleton config instances imported across the project.
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
EVAL_CFG = EvalConfig()
ROBUSTNESS_CFG = RobustnessConfig()
